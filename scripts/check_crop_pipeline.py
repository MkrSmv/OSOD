"""Замер конвейера «вырезки на CPU параллельно проходу GPU» на калибровочных сценах → `experiments/crop_pipeline_check.json`.

Конвейер — `src.encode.pipeline`: процессы готовят вырезки батчами, основной процесс кодирует их
на GPU **в том же порядке и тем же составом батчей**, что последовательный путь, поэтому эмбеддинги обязаны
совпасть побитово — это и проверяется, для обоих энкодеров, вместе с ускорением. Первая версия скрипта несла
собственный прототип конвейера (DINOv2, 2, 4 и 6 процессов); нынешняя меряет код пайплайна.

Только калибровочные сцены (три сцены сверки), маски — из кеша масок.

    python scripts/check_crop_pipeline.py [--workers 4]
    python scripts/check_crop_pipeline.py --scan --workers 4 6 8   # только DINOv2: сколько процессов брать; пишет
                                                                   # блок `workers_scan` в ту же запись
    python scripts/check_crop_pipeline.py --paired                 # вырезки один раз на оба энкодера, пять парных вариантов: блок `paired_encoders`

`--paired`: сначала раздельные прогоны конвейером — модель в памяти GPU одна, как в прогоне сетки, — затем обе модели
в памяти и общий проход (`encode_variant_multi`). Критерий записан до замера: эмбеддинги каждого энкодера в парном
проходе побитово (`np.array_equal`) совпадают с его раздельным прогоном на каждом варианте; иначе парные прогоны
не вводятся.
"""

from __future__ import annotations

import argparse
import datetime
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import env
from src.encode import pipeline as PL

SCENES = ["easy/meeting_room_001/rgb_001", "easy/meeting_room_001/rgb_016", "easy/sink/rgb_010"]  # сцены сверки, cal
VARIANTS = [("0", 1.0), ("blur", 1.5)]  # самый дешёвый и самый дорогой по CPU вариант
# варианты, которые в сетке идут парными проходами: все вырезки, кроме C(0, 1,0)
PAIRED_VARIANTS = [("0", 1.5), ("mean", 1.0), ("mean", 1.5), ("blur", 1.0), ("blur", 1.5)]
ENCODERS = ("encoder_dinov2", "encoder_dinov3")
OUT = Path("experiments/crop_pipeline_check.json")
_STAMP = env.code_stamp()  # версия кода — на момент запуска процесса


def _stamp() -> dict:
    return {**env.code_stamp(), "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"]}


def paired(images, feeder, M, torch) -> None:
    def timed(models, b, alpha):
        t = time.perf_counter()
        out = [z for _, z in PL.encode_variant_multi(models, feeder, images, b, alpha)]
        torch.cuda.synchronize()
        return {n: np.concatenate([z[n] for z in out]) for n in models}, time.perf_counter() - t

    alone, sec_alone = {}, {}
    for enc in ENCODERS:
        model = M.load(enc)
        M.encode_crops(model, np.zeros((PL.BATCH, 448, 448, 3), np.uint8))  # прогрев
        for v in PAIRED_VARIANTS:
            z, sec = timed({enc: model}, *v)
            alone[enc, v], sec_alone[enc, v] = z[enc], sec
        del model
        torch.cuda.empty_cache()
    models = {enc: M.load(enc) for enc in ENCODERS}
    for m in models.values():
        M.encode_crops(m, np.zeros((PL.BATCH, 448, 448, 3), np.uint8))
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for v in PAIRED_VARIANTS:
        z, sec = timed(models, *v)
        row = {"variant": f"C({v[0]}|{v[1]})", "n_masks": int(len(z[ENCODERS[0]])),
               "separate_sec": {e: round(sec_alone[e, v], 1) for e in ENCODERS},
               "separate_sec_sum": round(sum(sec_alone[e, v] for e in ENCODERS), 1), "paired_sec": round(sec, 1),
               "bitwise_identical_to_separate": {e: bool(np.array_equal(z[e], alone[e, v])) for e in ENCODERS},
               "max_abs_diff": {e: float(np.abs(z[e] - alone[e, v]).max()) for e in ENCODERS}}
        row["saving"] = round(1 - sec / row["separate_sec_sum"], 2)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    rec = json.loads(OUT.read_text())
    rec["paired_encoders"] = {
        "written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(), **_stamp(),
        "encoders": list(ENCODERS),
        "criterion": "побитовое совпадение с раздельным прогоном (модель в памяти GPU одна), все пять парных "
                     "вариантов, оба энкодера; замер двух крайних вариантов, включая C(0|1.0), — коммит 6642c61",
        "crop_workers": feeder.workers, "blur_work_side": PL.C.BLUR_WORK_SIDE,
        "vram_peak_allocated_mib": torch.cuda.max_memory_allocated() // 2**20,
        "passed": all(all(r["bitwise_identical_to_separate"].values()) for r in rows), "rows": rows}
    feeder.close()
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT)
    print("ПРОЙДЕНО" if rec["paired_encoders"]["passed"] else "НЕ ПРОЙДЕНО", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[PL.WORKERS])
    ap.add_argument("--scan", action="store_true",
                    help="подбор числа процессов: один энкодер, результат — блок workers_scan существующей записи")
    ap.add_argument("--paired", action="store_true", help="парный проход двух энкодеров против раздельных прогонов")
    args = ap.parse_args()
    encoders = ENCODERS[:1] if args.scan else ENCODERS

    import cv2
    import torch

    from src.data import hr_insdet
    from src.encode import model as M
    from src.segment import cache as MC

    sp = json.loads(Path("splits/hr_insdet.json").read_text())
    assert all(s in sp["cal"] for s in SCENES), "только калибровочные сцены"
    cnl = json.loads(Path("experiments/runs/4_2_hr_insdet.json").read_text())["selection"]["selected_crop_n_layers"]
    mc = MC.MaskCache("hr_insdet", MC.auto_key(cnl))
    images = [(PL.image_ref("hr_insdet", s, hr_insdet.ROOT / "Scenes" / f"{s}.jpg", mc.key), len(mc.load(s)))
              for s in SCENES]

    # процессы создаются до инициализации CUDA и способом spawn
    feeders = {w: PL.CropFeeder(workers=w) for w in args.workers}
    seq = PL.CropFeeder(workers=0)
    cpu_threads = cv2.getNumThreads()
    if args.paired:
        paired(images, feeders[args.workers[0]], M, torch)
        return
    rows = []
    for enc in encoders:
        model = M.load(enc)
        M.encode_crops(model, np.zeros((PL.BATCH, 448, 448, 3), np.uint8))  # прогрев
        for b, alpha in VARIANTS:
            def run(feeder):
                PL._state.clear()
                t = time.perf_counter()
                z = np.concatenate([v for _, v in PL.encode_variant(model, feeder, images, b, alpha)])
                torch.cuda.synchronize()
                return z, time.perf_counter() - t

            ref, sec_seq = run(seq)
            row = {"encoder": enc, "variant": f"C({b}|{alpha})", "n_masks": int(len(ref)),
                   "sequential_sec": round(sec_seq, 1), "pipelined": {}}
            for w, feeder in feeders.items():
                got, sec = run(feeder)
                row["pipelined"][str(w)] = {"sec": round(sec, 1), "speedup": round(sec_seq / sec, 2),
                                            "cv2_threads_per_worker": feeder.cv2_threads,
                                            "bitwise_identical_to_sequential": bool(np.array_equal(got, ref)),
                                            "max_abs_diff": float(np.abs(got - ref).max())}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            rows.append(row)
        del model
        torch.cuda.empty_cache()
    for f in feeders.values():
        f.close()
    # пик ОЗУ одного процесса подготовки (наибольший среди завершённых): квадрат 12 288² — сотни МБ на буфер
    worker_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    main_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

    if args.scan:
        rec = json.loads(OUT.read_text())
        rec["workers_scan"] = {"written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                               **_stamp(),
                               "encoder": encoders[0], "workers": args.workers, "max_in_flight": PL.MAX_IN_FLIGHT,
                               "blur_work_side": PL.C.BLUR_WORK_SIDE, "rows": rows,
                               "ram_peak_mib": {"one_crop_worker_max": worker_rss, "main_process": main_rss}}
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(OUT)
        return

    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    rec = {"written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(), **_stamp(),
           "split": "cal", "scenes": SCENES,
           "crop_n_layers": cnl, "encoders": list(ENCODERS), **seq.journal(), "sequential_cv2_threads": cpu_threads,
           "note": "код конвейера — src/encode/pipeline.py; первая сцена каждого пути включает чтение снимка; "
                   "у конвейера снимок читает каждый процесс; замер прототипа (DINOv2; 2, 4, 6 процессов) — "
                   "эта же запись в коммите f07eb48",
           "rows": rows}
    rec["crop_workers"] = args.workers
    rec.update({k: old[k] for k in ("workers_scan", "paired_encoders") if k in old})  # блоки других режимов сохраняются
    rec["ram_peak_mib"] = {"one_crop_worker_max": worker_rss, "main_process": main_rss,
                           "note": "сцена meeting_room_001/rgb_001 содержит маски со стороной рамки 8 192 пикс. — "
                                   "крупнейшие на 10 сценах сверки; оценка сверху для конвейера — main + workers × max"}
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT)


if __name__ == "__main__":
    main()
