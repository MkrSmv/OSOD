"""Индексация эталонов: галерея одного прогона сетки → `gallery/<encoder>/<variant>/<dataset>/`.

    python scripts/build_gallery.py --config configs/dinov2_c_0_10_hr_insdet.yaml [--status]
    python scripts/build_gallery.py --config … --verify   # пересборка с нуля в cache/checks/gallery_verify/ и побайтная
                                                          # сверка с действующей галереей → experiments/gallery_verify_<run_id>.json

Маски эталонов — из кеша масок по промпту-рамке (`scripts/segment.py --mode box`); сегментатор не загружается.
Прерываемо: эмбеддинги эталонов копятся в `cache/run_emb/<run_id>__gallery/`, повторный запуск продолжает с места,
`emb.npy` собирается из этих файлов. Контроль — IoU маски $S$ с маской GrabCut по всем эталонам (один раз на ключ
кеша масок, от варианта не зависит) — в метаданные и паспорт галереи. Вывод — ещё и в `logs/gallery_<run_id>.log`.
HR-InsDet и PKU-Market-PCB (у PCB контроля по GrabCut нет — в паспорт идёт сводка масок эталонов по типам дефекта, а у первой
галереи `dinov2_c_mean_10_pcb` — критерий осмысленности галереи PCB); галереи $P$ собирает прогон сетки (`scripts/run_4_3.py`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as CFG
from src import env
from src.gallery import build as GB
from src.gallery import store
from src.segment import cache as MC

EVERY = 100  # строка прогресса — раз в столько эталонов


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--status", action="store_true", help="сколько эталонов закодировано, и выйти")
    ap.add_argument("--verify", action="store_true",
                    help="закодировать эталоны заново в отдельный каталог и сверить галерею побайтно с действующей")
    args = ap.parse_args()
    cfg = CFG.load(args.config)
    stamp = env.code_stamp()  # версия кода — на момент запуска
    Path("logs").mkdir(exist_ok=True)
    log = open(Path("logs") / f"gallery_{cfg.run_id}.log", "a")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] [gallery {cfg.run_id}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    mc = MC.MaskCache(cfg.dataset, MC.box_key(cfg.seg_long_side))
    refs = GB.references(cfg.dataset)
    absent = [r["id"] for r in refs if not mc.has(r["id"], [r["box"]])]
    if absent:
        raise SystemExit(f"в кеше масок нет {len(absent)} эталонов (первый — {absent[0]}): "
                         f"python scripts/segment.py --dataset {cfg.dataset} --mode box")
    vroot = Path("cache/checks/gallery_verify") if args.verify else None
    work = GB.work_store(cfg, mc.key, root=None if vroot is None else vroot / "run_emb")
    done = sum(work.has(r["id"]) for r in refs)
    say(f"эталонов {len(refs)}, закодировано {done}, осталось {len(refs) - done}")
    if args.status:
        return

    t0 = time.perf_counter()

    def progress(n: int, total: int) -> None:
        if n % EVERY == 0 or n == total:
            sec = time.perf_counter() - t0
            say(f"закодировано {n}/{total}, {n / sec:.1f} эталона/с, осталось ~{(total - n) / n * sec / 60:.1f} мин")

    n_encoded, feeder_info = 0, None
    if done < len(refs):
        if cfg.phi.kind == "crop":
            from src.encode import pipeline as PL

            with PL.CropFeeder() as feeder:  # процессы — до инициализации CUDA
                from src.encode import model as M

                model = M.load(cfg.model_key)
                feeder_info = feeder.journal(cfg.encoder_batch)
                n_encoded = GB.encode_crop({cfg.run_id: cfg}, {cfg.run_id: model}, {cfg.run_id: work}, feeder, refs,
                                           mc, progress)
        else:
            raise SystemExit("галереи P и P⊥ собирает общий проход прогона сетки (`src.gallery.build.encode_pool`)")
    sec_encode = time.perf_counter() - t0

    t1 = time.perf_counter()
    ious = None  # у PCB масок GrabCut нет: в паспорт идёт сводка масок эталонов по типам (`src.gallery.build.ref_mask_summary`)
    if cfg.dataset == "hr_insdet":
        say("контроль: IoU маски S с маской GrabCut по всем эталонам")
        ious = GB.control_ious(cfg.dataset, refs, mc)
    sec_iou = time.perf_counter() - t1
    g = GB.assemble(cfg, refs, mc, work, ious, root=None if vroot is None else vroot / "gallery", build_info={
        **stamp, "n_encoded_this_session": n_encoded, "sec_encode_this_session": round(sec_encode, 1),
        "sec_control_iou": round(sec_iou, 1), "feeder": feeder_info})
    rows = g.rows()
    s = g.passport["build"]["control_iou"]
    say(f"ГОТОВО: {g.path} — N = {len(g)}, меток {len(g.labels)}, n_max = {g.n_max(rows)}, k = {g.k(rows)}"
        + ("" if s is None else f"; IoU с GrabCut: медиана {s['median']:.4f}, минимум {s['min']:.4f}, ниже 0,9 — "
                                f"{s['n_below_0.9']}, ниже 0,5 — {s['n_below_0.5']}"))
    crit = g.passport["build"].get("pcb_gallery_criterion")
    if crit is not None:
        say("КРИТЕРИЙ ОСМЫСЛЕННОСТИ ГАЛЕРЕИ PCB: top-1 по типам — "
            + ", ".join(f"{t} {v:.3f}" for t, v in crit["top1_by_type"].items())
            + f"; среднее {crit['mean_top1']:.4f} при пороге {crit['mean_top1_min']}; типов не ниже {crit['type_top1_min']} — "
              f"{crit['n_types_ge_min']} при пороге {crit['types_min']}; критерий "
              f"{'ВЫПОЛНЕН' if crit['run_4_5_on_pcb'] else 'НЕ ВЫПОЛНЕН'}")
    if args.verify:
        import hashlib
        import json

        live = store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
        rec = {"run_id": cfg.run_id, **stamp, "what": "эталоны закодированы заново закоммиченным кодом; сверка побайтная",
               "n_encoded": n_encoded, "live": str(live), "rebuilt": str(g.path),
               **{f"{n}_sha256": {"live": sha(live / n), "rebuilt": sha(g.path / n)} for n in ("emb.npy", "meta.jsonl")}}
        rec["identical"] = all(rec[f"{n}_sha256"]["live"] == rec[f"{n}_sha256"]["rebuilt"] for n in ("emb.npy", "meta.jsonl"))
        out = Path("experiments") / f"gallery_verify_{cfg.run_id}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(out)
        say(f"СВЕРКА: {'побайтно совпадает' if rec['identical'] else 'НЕ СОВПАДАЕТ'} — {out}")
    if store.Gallery.open(g.path, g.passport).emb.shape != g.emb.shape:
        raise SystemExit(f"{g.path}: записанная галерея не читается такой, какой собрана")


if __name__ == "__main__":
    main()
