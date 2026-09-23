"""Диагностика после результата сетки HR-InsDet: усреднение под маской по патч-токенам **вырезки**.

    python scripts/check_pool_crop.py            # → experiments/pool_crop_diagnostic.json (прерываемо)
    python scripts/check_pool_crop.py --status

Вопрос: проигрывает ли вырезке усреднение под маской как таковое или конструкция $P$ сетки — карта целого кадра на
входе 1536×2048, где маска искомого объекта накрывает десятки патчей, а маска эталона — тысячи. Протокол, константы и
трактовка записаны до этого кода отдельным коммитом. Это диагностика, не прогон сетки: запись не идёт в `experiments/runs/` и в таблицы,
лучший $\\varphi$ по ней не выбирается, галерея в `gallery/` не пишется. Читаются только калибровочные сцены (сторож).

Отображение одно для эталонов и запросов: квадрат $Q$ при $\\alpha=1{,}0$ из исходного снимка, фон не
заполняется, за кадром — $\\bar x$; 448×448; патч-токены вырезки после финальной нормировки; окно маски приводится к
448×448 и переводится на сетку патчей вырезки как доля пикселей патча под маской; среднее с этими весами, нормировка.
Вырезка от энкодера не зависит — батч готовится один раз и кодируется обоими энкодерами, как в парном проходе сетки.

До счёта сверяется путь оценки: `selection_cal` прогонов $C(0,1{,}0)$ и $P$ пересчитывается по их рабочим файлам
кодом этого скрипта (только калибровочные сцены) и обязан совпасть с записью журнала.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as CF  # noqa: E402
from src import env  # noqa: E402
from src.encode import crop as C  # noqa: E402
from src.encode import pipeline as PL  # noqa: E402
from src.encode import run_emb as RE  # noqa: E402
from src.eval import protocol as PR  # noqa: E402
from src.eval import recall as R  # noqa: E402
from src.eval import rules as RU  # noqa: E402
from src.gallery import build as GB  # noqa: E402
from src.segment import cache as MC  # noqa: E402

OUT = Path("experiments/pool_crop_diagnostic.json")
WORK = Path("cache/checks/pool_crop")
RUNS = Path("experiments/runs")
DATASET = RU.POOL_CROP_DATASET
BATCH = PL.BATCH
MASK_LEVELS = 255  # окно маски приводится к 448×448 в uint8: доля пикселя под маской с шагом 1/255


# ---------------------------------------------------------------------- подготовка вырезок (процессы)


def make_pool_crop(img: np.ndarray, box, mask_window) -> tuple[np.ndarray, np.ndarray]:
    """Вырезка 448×448 без заполнения фона и окно маски того же размера (uint8, 0…255 — доля пикселя под маской)."""
    import cv2

    size = RU.POOL_CROP_SIZE
    qx, qy, side = C.square(box, RU.POOL_CROP_ALPHA)
    q = C._paste(img, qx, qy, qx + side, qy + side, C.MEAN_RGB)
    m = mask_window(qx, qy, qx + side, qy + side).astype(np.uint8) * MASK_LEVELS
    if side == size:
        return q, m
    return (cv2.resize(q, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_CUBIC),
            cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR))


_state: dict = {}


def make_batch(task: tuple[PL.ImageRef, int, int]) -> tuple[np.ndarray, np.ndarray]:
    from src.segment import sam2 as S

    ref, start, stop = task
    if _state.get("ref") != ref:
        _state.clear()
        _state.update(ref=ref, entry=PL.load_entry(ref), img=S.read_rgb(ref.path))
    e, img = _state["entry"], _state["img"]
    if tuple(img.shape[:2]) != tuple(e.hw):
        raise ValueError(f"{ref.image_id}: размер снимка {img.shape[:2]} расходится с записью кеша {e.hw}")
    pairs = [make_pool_crop(img, e.records[i]["box"], e.windower(i)) for i in range(start, stop)]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


# ---------------------------------------------------------------------- кодирование


def grid_weights(m448, patch: int, device: str = "cuda"):
    """$\\tilde m_p$ на сетке патчей вырезки: (n, 448, 448) uint8 → (n, P) float32; патчи построчно, как у модели."""
    import torch

    n, size = m448.shape[0], RU.POOL_CROP_SIZE
    if size % patch:
        raise ValueError(f"вырезка {size} не кратна шагу патча {patch}")
    g = size // patch
    w = torch.from_numpy(m448).to(device).float() / MASK_LEVELS
    return w.reshape(n, g, patch, g, patch).mean((2, 4)).reshape(n, g * g), g


def encode_batch(model, crops: np.ndarray, m448: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """$z$ усреднением патч-токенов вырезки под маской: (n, d) float32, флаги запасного правила, патчей под маской."""
    import torch

    from src.encode import model as M

    with torch.no_grad():
        f = model(pixel_values=M._pixel_values(crops, model)).last_hidden_state[:, M.N_PREFIX:].float()
        w, g = grid_weights(m448, M.patch_size(model))
        if f.shape[1] != g * g:
            raise ValueError(f"патч-токенов {f.shape[1]}, сетка {g}×{g}")
        s = w.sum(1)
        empty = s == 0
        if empty.any():  # маска не попала ни в один патч вырезки — центральный патч с весом 1
            w[empty, (g // 2) * g + g // 2] = 1.0
        z = torch.nn.functional.normalize(torch.einsum("np,npd->nd", w, f) / w.sum(1)[:, None], dim=-1)
        if not torch.isfinite(z).all():
            raise FloatingPointError("не-конечный эмбеддинг")
    return z.cpu().numpy(), empty.cpu().numpy(), s.cpu().numpy()


def work_store(encoder: str, what: str, mask_key: dict) -> RE.RunEmb:
    key = {"what": f"pool_crop_{what}", "encoder": encoder, "dataset": DATASET, "mask_key": mask_key, "batch": BATCH,
           "batch_rule": PL.BATCH_RULE, "mask_levels": MASK_LEVELS, "precision": env.ENCODER_DTYPE[f"encoder_{encoder}"],
           "protocol": {k: getattr(RU, k) for k in dir(RU) if k.startswith("POOL_CROP_")}}
    return RE.RunEmb(f"{encoder}_{what}", key, root=WORK, dtype=np.float32 if what == "gallery" else np.float16)


def encode(images: list[tuple[PL.ImageRef, int]], stores: dict[str, RE.RunEmb], models: dict, feeder, say, label: str) -> float:
    """Снимки, ещё не лежащие в рабочих файлах, — общим проходом обоих энкодеров. Возвращает секунды."""
    todo = [(ref, n) for ref, n in images if not all(s.has(ref.image_id) for s in stores.values())]
    tasks = [(ref, k, min(k + BATCH, n)) for ref, n in todo for k in range(0, n, BATCH)]
    t0 = time.perf_counter()
    stream = _batches(feeder, tasks)
    for done, (ref, n) in enumerate(todo, 1):
        parts = {name: ([], [], []) for name in models}
        for _ in range(0, n, BATCH):
            task, (crops, m448) = next(stream)
            assert task[0] == ref
            for name, model in models.items():
                for acc, v in zip(parts[name], encode_batch(model, crops, m448)):
                    acc.append(v)
        for name, store_ in stores.items():
            d = models[name].config.hidden_size  # снимок без масок — пустые массивы нужной формы
            z, empty, s = (np.concatenate(v) if v else np.zeros((0, d) if k == 0 else (0,), np.float32)
                           for k, v in enumerate(parts[name]))
            store_.save(ref.image_id, z, empty=empty.astype(bool), patches_under_mask=s.astype(np.float32))
        if done % 50 == 0 or done == len(todo):
            say(f"{label}: {done} из {len(todo)} снимков, {time.perf_counter() - t0:.0f} с")
    return time.perf_counter() - t0


def _batches(feeder: PL.CropFeeder, tasks: list):
    """Батчи строго в порядке задач — то же правило, что `CropFeeder.batches`, со своей функцией подготовки."""
    import collections

    if feeder._pool is None:
        for t in tasks:
            yield t, make_batch(t)
        return
    pending: collections.deque = collections.deque()
    it = iter(tasks)
    while True:
        while len(pending) < feeder.max_in_flight and (t := next(it, None)) is not None:
            pending.append((t, feeder._pool.apply_async(make_batch, (t,))))
        if not pending:
            return
        t, res = pending.popleft()
        yield t, res.get()


# ---------------------------------------------------------------------- оценка


class DiagGallery:
    """Строки галереи диагностики с тем, что читает `protocol.search`; в `gallery/` не пишется."""

    def __init__(self, emb: np.ndarray, labels: list[str]):
        self.emb, self._labels = np.ascontiguousarray(emb, np.float32), list(labels)
        self.labels = sorted(set(labels))
        idx = {y: i for i, y in enumerate(self.labels)}
        self.label_ids = np.array([idx[y] for y in labels], np.int64)
        self.deleted = np.zeros(len(labels), bool)

    def rows(self, protocol: str = "full") -> np.ndarray:
        if protocol != RU.POOL_CROP_GALLERY:
            raise ValueError(protocol)
        return np.arange(len(self.emb))

    def n_max(self, rows: np.ndarray) -> int:
        return int(np.bincount(self.label_ids[rows]).max())


def evaluate_cal(scenes, labels, edges, gt, gallery, emb: list[np.ndarray], cfg) -> dict:
    y, s, info = PR.search(emb, gallery, RU.POOL_CROP_GALLERY)
    m = PR.evaluate_oracle(scenes, labels, edges, gt, gallery.labels, y, s, cfg.max_dets, cfg.bootstrap, cfg.seed,
                           names=(PR.SELECT_SUBSET,))
    return {"search": info, **m["subsets"][PR.SELECT_SUBSET]}


def self_check(scenes, labels, edges, gt, refs_cache, scenes_cache, encoder: str, say) -> dict:
    """`selection_cal` прогонов сетки, пересчитанный этим путём оценки по одним калибровочным сценам, — против записи."""
    from src.gallery import store

    out = {}
    for variant in RU.POOL_CROP_COMPARE.values():
        cfg = CF.default(encoder, variant, DATASET)
        rec = json.loads((RUNS / f"{cfg.run_id}.json").read_text())["selection_cal"]
        st = RE.scene_store(cfg, scenes_cache.key)
        emb = [st.load(s.id)["z"] for s in scenes]
        g = store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset), store.expected(cfg.to_dict(), refs_cache.key))
        a = evaluate_cal(scenes, labels, edges, gt, g, emb, cfg)["by_area"]["all"]
        same = abs(a["ap"] - rec["ap"]) < 1e-12 and np.allclose(a["ci95"]["ap"], rec["ci95"]["ap"], atol=1e-12)
        say(f"сверка пути оценки, {cfg.run_id}: AP {100 * a['ap']:.4f} против записи {100 * rec['ap']:.4f} — "
            f"{'совпало' if same else 'НЕ СОВПАЛО'}")
        if not same:
            raise SystemExit(f"{cfg.run_id}: путь оценки диагностики не воспроизводит selection_cal записи журнала")
        out[variant] = {"run_id": cfg.run_id, "ap": rec["ap"], "ap50": rec["ap50"], "ci95_ap": rec["ci95"]["ap"],
                        "recomputed_matches_record": True}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=PL.WORKERS)
    args = ap.parse_args()

    def say(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    split = json.loads(Path(f"splits/{DATASET}.json").read_text())
    if RU.POOL_CROP_SPLIT != "cal" or set(split["cal"]) & set(split["test"]):
        raise SystemExit("диагностика принимает только калибровочные сцены")
    cfg0 = CF.default(RU.POOL_CROP_ENCODERS[0], RU.POOL_CROP_COMPARE["crop"], DATASET)
    scenes_cache = MC.MaskCache(DATASET, MC.auto_key(cfg0.crop_n_layers, cfg0.seg_long_side, cfg0.points_per_batch))
    refs_cache = MC.MaskCache(DATASET, MC.box_key(cfg0.seg_long_side))
    scenes, labels, edges, _ = PR.load_scenes(DATASET, scenes_cache, splits=(RU.POOL_CROP_SPLIT,))
    if [s.id for s in scenes] != split["cal"] or any(s.split != "cal" for s in scenes):
        raise SystemExit("состав сцен расходится с калибровочной частью разбиения")
    refs = GB.references(DATASET)
    root = GB.data_root(DATASET)
    stores = {enc: {"scenes": work_store(enc, "scenes", scenes_cache.key), "gallery": work_store(enc, "gallery", refs_cache.key)}
              for enc in RU.POOL_CROP_ENCODERS}
    if args.status:
        for enc, st in stores.items():
            print(enc, "сцен", sum(st["scenes"].has(s.id) for s in scenes), "из", len(scenes),
                  "эталонов", sum(st["gallery"].has(r["id"]) for r in refs), "из", len(refs))
        return

    stamp = env.code_stamp()  # версия кода — на момент запуска; время записи — отдельно, в конце
    stamp["started_utc"] = stamp.pop("written_utc")
    gt = R.coco_gt([{"id": s.id, "wh": s.wh, "gt": s.gt} for s in scenes], labels)
    say(f"калибровочных сцен {len(scenes)}, масок {sum(len(s.boxes) for s in scenes)}, эталонов {len(refs)}")
    check = {enc: self_check(scenes, labels, edges, gt, refs_cache, scenes_cache, enc, say) for enc in RU.POOL_CROP_ENCODERS}

    from src.data import hr_insdet as H

    cache_root = scenes_cache.dir.parents[1]
    scene_imgs = [(PL.image_ref(DATASET, s.id, H.ROOT / "Scenes" / f"{s.id}.jpg", scenes_cache.key, None, cache_root),
                   len(s.boxes)) for s in scenes]
    ref_imgs = [(PL.image_ref(DATASET, r["id"], root / r["image"], refs_cache.key, [r["box"]], cache_root), 1) for r in refs]
    sec = {}
    with PL.CropFeeder(workers=args.workers) as feeder:  # процессы — до инициализации CUDA
        from src.encode import model as M

        models = {enc: M.load(f"encoder_{enc}") for enc in RU.POOL_CROP_ENCODERS}
        sec["gallery"] = encode(ref_imgs, {e: stores[e]["gallery"] for e in models}, models, feeder, say, "эталоны")
        sec["scenes"] = encode(scene_imgs, {e: stores[e]["scenes"] for e in models}, models, feeder, say, "сцены")

    result = {}
    for enc in RU.POOL_CROP_ENCODERS:
        cfg = CF.default(enc, RU.POOL_CROP_COMPARE["pool"], DATASET)
        gfiles = [stores[enc]["gallery"].load(r["id"]) for r in refs]
        gal = DiagGallery(np.concatenate([x["z"] for x in gfiles]), [r["label"] for r in refs])
        if gal.labels != labels:
            raise SystemExit("метки галереи расходятся с категориями истины")
        sfiles = [stores[enc]["scenes"].load(s.id) for s in scenes]
        m = evaluate_cal(scenes, labels, edges, gt, gal, [x["z"] for x in sfiles], cfg)
        a = m["by_area"]["all"]
        oracle_patches = np.concatenate([x["patches_under_mask"][s.oracle[s.oracle >= 0]] for x, s in zip(sfiles, scenes)])
        ref_patches = np.concatenate([x["patches_under_mask"] for x in gfiles])
        verdict = RU.pool_crop_verdict(100 * a["ap"], 100 * check[enc][RU.POOL_CROP_COMPARE["crop"]]["ap"],
                                       100 * check[enc][RU.POOL_CROP_COMPARE["pool"]]["ap"])
        result[enc] = {"n_gallery": len(gal.emb), "n_masks": int(sum(len(x["z"]) for x in sfiles)),
                       "n_center_patch_fallback": {"scenes": int(sum(x["empty"].sum() for x in sfiles)),
                                                   "gallery": int(sum(x["empty"].sum() for x in gfiles))},
                       "patches_under_mask_median": {"oracle_masks": float(np.median(oracle_patches)),
                                                     "references": float(np.median(ref_patches))},
                       "selection_cal_of_grid_runs": check[enc], "metrics_cal": m, "verdict": verdict}
        say(f"{enc}: AP {100 * a['ap']:.2f} [{100 * a['ci95']['ap'][0]:.1f}; {100 * a['ci95']['ap'][1]:.1f}], top-1 "
            f"{100 * a['top1']:.2f}, AUROC {a['auroc']:.4f}; C(0,1.0) {verdict['ap_crop']:.2f}, P {verdict['ap_pool']:.2f} "
            f"→ {verdict['verdict']}")

    rec = {"what": "диагностика после результата сетки HR-InsDet: усреднение под маской по патч-токенам вырезки; "
                   "не прогон сетки — в таблицы 4.3 не идёт, лучший φ по ней не выбирается",
           "post_hoc": True, **stamp, "dataset": DATASET, "split": RU.POOL_CROP_SPLIT, "n_scenes": len(scenes),
           "protocol": {k: getattr(RU, k) for k in dir(RU) if k.startswith("POOL_CROP_")},
           "protocol_commit": "74b9d81", "mask_levels": MASK_LEVELS, "encoder_batch": BATCH, "batch_rule": PL.BATCH_RULE,
           "features": "last_hidden_state вырезки после финальной нормировки, без CLS и 4 регистров, fp32",
           "precision": {e: env.ENCODER_DTYPE[f"encoder_{e}"] for e in RU.POOL_CROP_ENCODERS},
           "mask_cache": {"scenes": scenes_cache.dir.name, "references": refs_cache.dir.name},
           "evaluation": "protocol.evaluate_oracle по подмножеству cal/all: оракульные маски и дистракторы, точный перебор, "
                         "полная галерея, те же повторы бутстрэпа, что у selection_cal прогонов сетки",
           "encode_sec_this_launch_both_encoders": {k: round(v, 1) for k, v in sec.items()}, "encoders": result}
    rec["written_utc"] = env.code_stamp()["written_utc"]
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT)
    say(f"запись — {OUT}")


if __name__ == "__main__":
    main()
