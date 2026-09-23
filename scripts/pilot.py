"""Пилотный замер → `experiments/pilot.json`.

Что меряется: время по этапам и число масок на 10 сценах HR-InsDet и 10 снимках PCB; recall SAM 2 по
размеченным дефектам PCB (`crop_n_layers=1`, IoU ≥ 0,5) и стоп-критерий PCB; распределение масок авторежима
на фоне (`Background/`) против сцен пилота; по замерам — бюджет в часах по прогонам и ступеням.

Состав выборок задаётся здесь, до замера, по seed из `splits/*.json` и от результатов не зависит: сцены и снимки PCB — только из калибровочной части,
потому что маски авторежима тестовой части до построения кеша масок не строятся. Это замер, а не кеш: маски пилота
лежат в `cache/checks/pilot/` и в кеш масок `cache/masks/` не переносятся. Этапы — отдельные процессы ($S$ и $E$ в одном
процессе не держатся), каждый пишет построчный промежуточный итог и продолжается с места повторным запуском:

    python scripts/pilot.py segment
    python scripts/pilot.py crops
    python scripts/pilot.py blurcheck
    python scripts/pilot.py encode --encoder encoder_dinov2     # затем encoder_dinov3
    python scripts/pilot.py owlv2
    python scripts/pilot.py evalcost
    python scripts/pilot.py report
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import env
from src.data import hr_insdet, pcb

OUT = Path("experiments/pilot.json")
WORK = Path("cache/checks/pilot")

# --- записано до замера, по результатам не пересматривается ---
PCB_RECALL_MIN = 0.3
PCB_MIN_PER_TYPE = 5
RECALL_IOU = 0.5
CROP_N_LAYERS_RECALL = 1

N_SCENES = 10
N_PCB = 10
N_BACKGROUND = 20
N_REFS = 10
DISTRACTOR_IOU_MAX = 0.1
POOL_LONG_SIDE_FALLBACK = 1368  # запасное разрешение входа E: 1024×1368 для HR-InsDet
ENCODERS = ("encoder_dinov2", "encoder_dinov3")
VARIANTS = [(b, a) for b in ("0", "mean", "blur") for a in (1.0, 1.5)]
AREA_BINS_HR = [0, 200**2, 400**2, float("inf")]


# Визуальная разметка 20 снимков фона из выборки пилота:
# same — узнаётся помещение сцен; type — помещение того же типа (переговорная, офис, зона отдыха), но не то же;
# other — спортзал, студия, коридор, холл, лаунж. Оценка на глаз, не измерение.
BACKGROUND_VISUAL = {
    "same": ["069", "093", "141", "172"],
    "type": ["039", "047", "061", "071", "075", "100", "168", "179"],
    "other": ["008", "024", "026", "109", "149", "183", "184", "192"],
}


# ---------------------------------------------------------------- выборки


def samples() -> dict:
    """Состав пилота: детерминирован seed разбиений, задаётся до любого замера."""
    hr = json.loads(Path("splits/hr_insdet.json").read_text())
    pc = json.loads(Path("splits/pcb.json").read_text())
    rng = np.random.default_rng(hr["seed"])
    scenes = []
    for room in hr["rooms"]["cal"]:
        ids = sorted(i for i in hr["cal"] if i.startswith(room + "/"))
        scenes += sorted(rng.choice(ids, N_SCENES // len(hr["rooms"]["cal"]), replace=False).tolist())
    background = sorted(rng.choice([f"{k:03d}" for k in range(200)], N_BACKGROUND, replace=False).tolist())
    hr_refs = sorted(rng.choice([r["id"] for r in hr["gallery"]], N_REFS, replace=False).tolist())

    rng = np.random.default_rng(pc["seed"])
    info = {i["id"]: i for i in pcb.images()}
    types = list(rng.permutation(pcb.CLASSES))
    # плата 09 (60 снимков) — по снимку на каждый из 6 типов; плата 10 (32 снимка) — на первые 4 типа перестановки
    pcb_imgs = []
    for board, tt in (("09", types), ("10", types[:N_PCB - len(types)])):
        for t in tt:
            ids = sorted(i for i in pc["cal"] if info[i]["board"] == board and info[i]["label"] == t)
            pcb_imgs.append(str(rng.choice(ids)))
    pcb_refs = sorted(rng.choice(pc["gallery_images"], N_REFS, replace=False).tolist())
    return {"seed_hr_insdet": hr["seed"], "seed_pcb": pc["seed"], "hr_scenes": scenes, "hr_background": background,
            "hr_refs": hr_refs, "pcb_images": sorted(pcb_imgs), "pcb_ref_images": pcb_refs}


def _paths(s: dict) -> dict[str, dict[str, Path]]:
    hr = json.loads(Path("splits/hr_insdet.json").read_text())
    ref = {r["id"]: r for r in hr["gallery"]}
    info = {i["id"]: i for i in pcb.images()}
    return {
        "hr_scenes": {i: hr_insdet.ROOT / "Scenes" / f"{i}.jpg" for i in s["hr_scenes"]},
        "hr_background": {i: hr_insdet.ROOT / "Background" / f"{i}.jpg" for i in s["hr_background"]},
        "hr_refs": {i: hr_insdet.ROOT / ref[i]["image"] for i in s["hr_refs"]},
        "pcb_images": {i: pcb.ROOT / info[i]["image"] for i in s["pcb_images"]},
        "pcb_ref_images": {i: pcb.ROOT / info[i]["image"] for i in s["pcb_ref_images"]},
    }


def _ref_boxes(s: dict) -> dict[str, dict[str, list]]:
    hr = json.loads(Path("splits/hr_insdet.json").read_text())
    pc = json.loads(Path("splits/pcb.json").read_text())
    ref = {r["id"]: r for r in hr["gallery"]}
    out = {"hr_refs": {i: [ref[i]["box"]] for i in s["hr_refs"]}, "pcb_ref_images": {}}
    for i in s["pcb_ref_images"]:
        out["pcb_ref_images"][i] = [r["box"] for r in pc["gallery"] if r["id"].rsplit("/", 1)[0] == i]
    return out


def _gt(group: str, iid: str) -> np.ndarray:
    if group == "hr_scenes":
        labels = hr_insdet.objects()
        sc = {x["id"]: x for x in hr_insdet.scenes()}[iid]
        g = hr_insdet.scene_gt(sc, labels)
    elif group == "pcb_images":
        g = pcb.image_gt({i["id"]: i for i in pcb.images()}[iid])
    else:
        g = []
    return np.array([x["box"] for x in g], float).reshape(-1, 4)


# ---------------------------------------------------------------- служебное


class Partial:
    """Построчный промежуточный итог: прерванный этап продолжается с места."""

    def __init__(self, name: str):
        WORK.mkdir(parents=True, exist_ok=True)
        self.path = WORK / f"{name}.jsonl"
        self.rows = [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []
        self.keys = {r["key"] for r in self.rows}

    def add(self, key: str, **row) -> None:
        r = {"key": key, **row}
        with self.path.open("a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.rows.append(r)
        self.keys.add(key)
        print({k: v for k, v in r.items() if k != "masks"}, flush=True)


def _mask_file(group: str, iid: str) -> Path:
    return WORK / "masks" / group / (iid.replace("/", "__") + ".json")


def _save_masks(group: str, iid: str, recs: list, info: dict) -> None:
    p = _mask_file(group, iid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"info": info, "masks": recs}))
    tmp.replace(p)


def _load_masks(group: str, iid: str) -> tuple[list, dict]:
    d = json.loads(_mask_file(group, iid).read_text())
    return d["masks"], d["info"]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def _timed_read(path: Path):
    from src.segment import sam2 as S

    t = time.perf_counter()
    img = S.read_rgb(path)
    return img, time.perf_counter() - t


# ---------------------------------------------------------------- этап: сегментация


def stage_segment() -> None:
    import torch

    from src.segment import sam2 as S

    s = samples()
    paths = _paths(s)
    part = Partial("segment")
    for cnl in (1, 0):
        groups = ["hr_scenes", "pcb_images"] + (["hr_background"] if cnl == 1 else [])
        todo = [(g, i) for g in groups for i in paths[g] if f"{g}|{i}|auto{cnl}" not in part.keys]
        if not todo:
            continue
        gen = S.build_generator(cnl)
        for g, i in todo:
            img, t_read = _timed_read(paths[g][i])
            recs, info = S.generate(gen, img)
            if cnl == 1:
                _save_masks(g, i, recs, info)
            part.add(f"{g}|{i}|auto{cnl}", group=g, image=i, mode="auto", crop_n_layers=cnl, hw=list(img.shape[:2]),
                     n_masks=len(recs), sec_read=t_read, **info)
        del gen
        torch.cuda.empty_cache()
    boxes = _ref_boxes(s)
    todo = [(g, i) for g in ("hr_refs", "pcb_ref_images") for i in paths[g] if f"{g}|{i}|box" not in part.keys]
    if todo:
        pred = S.build_predictor()
        for g, i in todo:
            img, t_read = _timed_read(paths[g][i])
            recs, info = S.predict_boxes(pred, img, boxes[g][i])
            for r, b in zip(recs, boxes[g][i]):
                r["prompt_box"] = b
            _save_masks(g, i, recs, info)
            part.add(f"{g}|{i}|box", group=g, image=i, mode="box", hw=list(img.shape[:2]), n_boxes=len(recs),
                     sec_read=t_read, **info)


# ---------------------------------------------------------------- этап: подготовка вырезок (CPU)


def _windower(m_seg: np.ndarray, scale_xy):
    from src.segment.sam2 import mask_window

    return lambda x0, y0, x1, y1: mask_window(m_seg, scale_xy, x0, y0, x1, y1)


def stage_crops() -> None:
    """Время построения вырезок по вариантам; от энкодера не зависит ($\\bar x$ и нормировка у обоих одни)."""
    from src.encode import crop as C
    from src.segment import sam2 as S

    s = samples()
    paths = _paths(s)
    part = Partial("crops")
    for g in ("hr_scenes", "pcb_images", "hr_refs", "pcb_ref_images"):
        for i in paths[g]:
            if f"{g}|{i}" in part.keys:
                continue
            recs, info = _load_masks(g, i)
            img, t_read = _timed_read(paths[g][i])
            sec = {}
            for b, a in VARIANTS:
                t = time.perf_counter()
                for r in recs:
                    # в прогоне вариант — отдельный прогон, поэтому декодирование маски входит в его время
                    C.make_crop(img, r["box"], _windower(S.decode(r["rle"]), info["scale_xy"]), b, a)
                sec[f"{b}|{a}"] = time.perf_counter() - t
            side = [max(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) for r in recs]
            part.add(f"{g}|{i}", group=g, image=i, n_masks=len(recs), sec_read=t_read, sec_by_variant=sec,
                     box_side_median=float(np.median(side)), box_side_max=float(np.max(side)),
                     cv2_threads=cv2.getNumThreads())


# ---------------------------------------------------------------- сверка размытия

def stage_blurcheck() -> None:
    """Размытие на уменьшенном окне против буквального (в исходном разрешении): пиксели вырезки и CLS обоих энкодеров."""
    import torch

    from src.encode import crop as C
    from src.encode import model as M
    from src.segment import sam2 as S

    s = samples()
    paths = _paths(s)
    part = Partial("blurcheck")
    if "crops" not in part.keys:
        pairs, meta = [], []
        for g, i in (("hr_scenes", s["hr_scenes"][0]), ("hr_scenes", s["hr_scenes"][5]), ("hr_refs", s["hr_refs"][0])):
            recs, info = _load_masks(g, i)
            img = S.read_rgb(paths[g][i])
            side = np.array([max(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) for r in recs])
            # только там, где пути различаются (сторона квадрата > рабочей) и буквальный ещё считается за секунды
            ok = [k for k in np.argsort(side) if C.BLUR_WORK_SIDE < side[k] and 1.5 * side[k] <= 2600]
            for k in ok[::max(1, len(ok) // 6)][:6]:
                win = _windower(S.decode(recs[k]["rle"]), info["scale_xy"])
                for a in C.ALPHAS:
                    t0 = time.perf_counter()
                    fast = C.make_crop(img, recs[k]["box"], win, "blur", a)
                    t1 = time.perf_counter()
                    exact = C.make_crop(img, recs[k]["box"], win, "blur", a, exact_blur=True)
                    t2 = time.perf_counter()
                    d = np.abs(fast.astype(int) - exact.astype(int))
                    pairs.append((fast, exact))
                    meta.append({"image": i, "side": float(side[k]), "alpha": a, "sec_fast": t1 - t0,
                                 "sec_exact": t2 - t1, "abs_diff_max": int(d.max()), "abs_diff_mean": float(d.mean())})
        np.savez_compressed(WORK / "blurcheck_crops.npz", fast=np.stack([p[0] for p in pairs]),
                            exact=np.stack([p[1] for p in pairs]))
        part.add("crops", pairs=meta)
    z = np.load(WORK / "blurcheck_crops.npz")
    for e in ENCODERS:
        if e in part.keys:
            continue
        model = M.load(e)
        cos = []
        for k in range(0, len(z["fast"]), 16):
            a, b = M.encode_crops(model, z["fast"][k:k + 16]), M.encode_crops(model, z["exact"][k:k + 16])
            cos += (a * b).sum(1).cpu().tolist()
        # масштаб для сравнения: сходство вырезок разных масок между собой
        a = M.encode_crops(model, z["exact"][:16])
        off = (a @ a.T)[~torch.eye(len(a), dtype=bool, device=a.device)]
        part.add(e, cls_cos_fast_vs_exact_min=float(np.min(cos)), cls_cos_fast_vs_exact_median=float(np.median(cos)),
                 cls_cos_between_different_masks_median=float(off.median()))
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- этап: энкодер (GPU)


def stage_encode(name: str) -> None:
    import torch

    from src.encode import crop as C
    from src.encode import model as M
    from src.encode.pool import patch_grid_mask, pool
    from src.segment import sam2 as S

    s = samples()
    paths = _paths(s)
    part = Partial(f"encode_{name}")
    model = M.load(name)
    p = M.patch_size(model)

    # пропускная способность на вырезках 448: содержимое на время не влияет, вариант один — C(0, 1,0)
    if "throughput" not in part.keys:
        g, i = "hr_scenes", s["hr_scenes"][0]
        recs, info = _load_masks(g, i)
        img = S.read_rgb(paths[g][i])
        crops = np.stack([C.make_crop(img, r["box"], _windower(S.decode(r["rle"]), info["scale_xy"]), "0", 1.0)
                          for r in recs[:192]])
        del img
        res = {}
        for bs in (16, 32, 64):
            M.encode_crops(model, crops[:bs])  # прогрев
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            n = 0
            for k in range(0, len(crops) - bs + 1, bs):
                M.encode_crops(model, crops[k:k + bs]).cpu()
                n += bs
            torch.cuda.synchronize()
            res[str(bs)] = {"crops_per_sec": n / (time.perf_counter() - t),
                            "vram_peak_mib": torch.cuda.max_memory_allocated() // 2**20}
        part.add("throughput", n_crops=len(crops), by_batch=res)

    # P: проход полного изображения при входе S (по умолчанию) и при запасном разрешении, привязка масок, усреднение
    for g in ("hr_scenes", "pcb_images", "hr_refs", "pcb_ref_images"):
        for i in paths[g]:
            if f"{g}|{i}" in part.keys:
                continue
            recs, info = _load_masks(g, i)
            x = S.seg_input(S.read_rgb(paths[g][i]))
            row = {}
            for tag, long_side in (("default", None), ("fallback", POOL_LONG_SIDE_FALLBACK)):
                xe = x if long_side is None else S.seg_input(x, long_side)
                xe_c = M.crop_to_patch_multiple(xe, p)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                f, grid = M.encode_full(model, xe_c)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                vram = torch.cuda.max_memory_allocated() // 2**20
                mg = np.stack([patch_grid_mask(S.decode(r["rle"]), xe.shape[:2], p).reshape(-1) for r in recs])
                t2 = time.perf_counter()
                mg_t = torch.from_numpy(mg).cuda()
                centers = torch.zeros(len(recs), dtype=torch.long, device="cuda")
                for k, r in enumerate(recs):  # патч под центром рамки — для масок, не попавших ни в один блок
                    cx = (r["box"][0] + r["box"][2]) / 2 / info["scale_xy"][0] * xe.shape[1] / x.shape[1]
                    cy = (r["box"][1] + r["box"][3]) / 2 / info["scale_xy"][1] * xe.shape[0] / x.shape[0]
                    centers[k] = min(int(cy // p), grid[0] - 1) * grid[1] + min(int(cx // p), grid[1] - 1)
                z, empty = pool(f, mg_t, centers)
                z = z.cpu()
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                patches = mg.sum(1)
                row[tag] = {"input_hw": list(xe_c.shape[:2]), "tokens": grid[0] * grid[1], "sec_forward": t1 - t0,
                            "vram_peak_mib": int(vram), "sec_mask_grid": t2 - t1, "sec_pool": t3 - t2,
                            "n_empty": int(empty.sum()), "finite": bool(torch.isfinite(z).all()),
                            "patches_under_mask_q10_median": [float(np.quantile(patches, 0.1)),
                                                              float(np.median(patches))]}
                del f, mg_t, z
            part.add(f"{g}|{i}", group=g, image=i, n_masks=len(recs), **row)


def stage_owlv2() -> None:
    """Время одного прохода зрительной башни OWLv2 — только для оценки ступени «полная»; метод не реализуется."""
    import torch
    from transformers import Owlv2ForObjectDetection

    part = Partial("owlv2")
    if "forward" in part.keys:
        return
    model = Owlv2ForObjectDetection.from_pretrained(env.MODEL_IDS["owlv2"], dtype=torch.float16,
                                                    attn_implementation="eager").eval().cuda()
    x = torch.randn(1, 3, 1008, 1008, device="cuda", dtype=torch.float16)
    secs = []
    with torch.no_grad():
        for k in range(6):
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.image_embedder(pixel_values=x)
            torch.cuda.synchronize()
            secs.append(time.perf_counter() - t)
    part.add("forward", sec_image_embedder_median=float(np.median(secs[1:])),
             vram_peak_mib=torch.cuda.max_memory_allocated() // 2**20,
             note="случайный вход 1×3×1008×1008, fp16, eager; только зрительная башня")


# ---------------------------------------------------------------- этап: стоимость поиска и оценки


def stage_evalcost() -> None:
    """Поиск и оценка на синтетике реального объёма: сами эти этапы на момент замера ещё не написаны."""
    import faiss
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    part = Partial("evalcost")
    seg = Partial("segment").rows
    mean_masks = {g: float(np.mean([r["n_masks"] for r in seg if r["group"] == g and r.get("crop_n_layers") == 1]))
                  for g in ("hr_scenes", "pcb_images")}
    hr = json.loads(Path("splits/hr_insdet.json").read_text())["counts"]
    pc = json.loads(Path("splits/pcb.json").read_text())["counts"]
    boxes_pcb_test = 1770  # порядок величины; точное число — бины (549 + 596 + 625)
    cases = {"hr_insdet": (120, hr["gt_boxes_test"], 100, hr["gallery"], mean_masks["hr_scenes"], 8192, 6144),
             "pcb": (pc["test_images"], boxes_pcb_test, 6, pc["gallery"], mean_masks["pcb_images"], 3000, 2400)}
    rng = np.random.default_rng(0)
    for ds, (n_img, n_gt, n_cat, n_gal, mm, w, h) in cases.items():
        if ds in part.keys:
            continue
        nq = int(n_img * mm)
        gal = rng.standard_normal((n_gal, 1024)).astype(np.float32)
        gal /= np.linalg.norm(gal, axis=1, keepdims=True)
        q = rng.standard_normal((nq, 1024)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        labels = rng.integers(0, n_cat, n_gal)
        index = faiss.IndexFlatIP(1024)
        index.add(gal)
        t = time.perf_counter()
        best = np.full((nq, n_cat), -2.0, np.float32)
        for k in range(0, nq, 4096):  # точный перебор: s_y — максимум по всем эталонам метки
            sim = q[k:k + 4096] @ gal.T
            for c in range(n_cat):
                best[k:k + 4096, c] = sim[:, labels == c].max(1)
        t_search = time.perf_counter() - t
        t = time.perf_counter()
        index.search(q, min(48, n_gal))
        t_faiss = time.perf_counter() - t

        def boxes(n, w=w, h=h):
            xy = rng.random((n, 2)) * [w * 0.8, h * 0.8]
            wh = rng.random((n, 2)) * [w * 0.1, h * 0.1] + 8
            return np.concatenate([xy, wh], 1)

        gt_b, dt_b = boxes(n_gt), boxes(nq)
        gt = {"images": [{"id": k, "width": w, "height": h} for k in range(n_img)],
              "categories": [{"id": c} for c in range(n_cat)],
              "annotations": [{"id": k + 1, "image_id": int(rng.integers(n_img)), "category_id": int(rng.integers(n_cat)),
                               "bbox": b.tolist(), "area": float(b[2] * b[3]), "iscrowd": 0}
                              for k, b in enumerate(gt_b)]}
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        dets = [{"image_id": int(k // mm) % n_img, "category_id": int(best[k].argmax()), "bbox": b.tolist(),
                 "score": float(best[k].max())} for k, b in enumerate(dt_b)]
        E = COCOeval(coco, coco.loadRes(dets), "bbox")
        E.params.maxDets = [1, 10, 100, 1000]
        t = time.perf_counter()
        E.evaluate()
        t_eval = time.perf_counter() - t
        t = time.perf_counter()
        E.accumulate()
        t_acc = time.perf_counter() - t
        part.add(ds, n_images=n_img, n_queries=nq, n_gallery=n_gal, sec_exact_search=t_search,
                 sec_faiss_flat_k48=t_faiss, sec_coco_evaluate=t_eval, sec_coco_accumulate=t_acc,
                 note="синтетические эмбеддинги и рамки реального объёма; детекции — все маски")


# ---------------------------------------------------------------- отчёт


def _q(v, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)) -> list[float]:
    return [round(float(x), 1) for x in np.quantile(np.asarray(v, float), qs)] if len(v) else []


def _med(rows, key) -> float:
    return float(np.median([r[key] for r in rows]))


def _mask_stats(group: str, ids: list[str]) -> dict:
    """Число масок и площади (пикс.² исходного снимка): все маски и дистракторы (IoU рамки < 0,1 ко всем GT)."""
    n_all, n_dis, a_all, a_dis, per_image = [], [], [], [], {}
    for i in ids:
        recs, _ = _load_masks(group, i)
        b = np.array([r["box"] for r in recs], float).reshape(-1, 4)
        a = np.array([r["area"] for r in recs], float)
        gt = _gt(group, i)
        dis = box_iou(b, gt).max(1) < DISTRACTOR_IOU_MAX if len(gt) else np.ones(len(b), bool)
        n_all.append(len(b))
        n_dis.append(int(dis.sum()))
        a_all += a.tolist()
        a_dis += a[dis].tolist()
        per_image[i] = {"masks": len(b), "distractors": int(dis.sum()), "gt_boxes": len(gt),
                        "area_median": float(np.median(a)) if len(a) else None}

    def bins(v):
        h = np.histogram(v, AREA_BINS_HR)[0]
        return [round(float(x), 3) for x in h / max(1, h.sum())]

    return {"n_images": len(ids), "masks_per_image_min_median_mean_max":
            [int(np.min(n_all)), float(np.median(n_all)), round(float(np.mean(n_all)), 1), int(np.max(n_all))],
            "distractors_per_image_min_median_mean_max":
            [int(np.min(n_dis)), float(np.median(n_dis)), round(float(np.mean(n_dis)), 1), int(np.max(n_dis))],
            "area_quantiles_0_10_25_50_75_90_100": {"all": _q(a_all), "distractors": _q(a_dis)},
            "area_share_lt200sq_200to400sq_gt400sq": {"all": bins(a_all), "distractors": bins(a_dis)},
            "_areas": {"all": a_all, "distractors": a_dis}, "per_image": per_image}


def _ks(a, b) -> float:
    """Статистика Колмогорова — Смирнова двух выборок (наибольшее расхождение эмпирических функций распределения)."""
    a, b = np.sort(a), np.sort(b)
    v = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b)).max())


def _pcb_recall(s: dict) -> dict:
    info = {i["id"]: i for i in pcb.images()}
    rows = []
    for i in s["pcb_images"]:
        recs, _ = _load_masks("pcb_images", i)
        b = np.array([r["box"] for r in recs], float).reshape(-1, 4)
        for g in _gt("pcb_images", i):
            iou = box_iou(g[None], b)[0]
            # справочно, не критерий: есть ли маска, рамка которой лежит внутри рамки разметки (≥ 0,9 своей площади)
            # и не мельче 5 % её площади, — дефект выделен, но рамка разметки свободнее маски
            ix = np.clip(np.minimum(b[:, 2], g[2]) - np.maximum(b[:, 0], g[0]), 0, None)
            iy = np.clip(np.minimum(b[:, 3], g[3]) - np.maximum(b[:, 1], g[1]), 0, None)
            ab, ag = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), (g[2] - g[0]) * (g[3] - g[1])
            inside = (ix * iy >= 0.9 * ab) & (ab >= 0.05 * ag)
            rows.append({"image": i, "board": info[i]["board"], "type": info[i]["label"],
                         "best_iou": float(iou.max()) if len(iou) else 0.0, "mask_inside_box": bool(inside.any()),
                         "box_area": float(ag)})
    best = np.array([r["best_iou"] for r in rows])

    def rec(sel) -> dict:
        v = best[sel]
        return {"n": len(v), "recall_050": round(float((v >= 0.5).mean()), 4),
                "recall_075": round(float((v >= 0.75).mean()), 4)}

    n, k = len(best), int((best >= RECALL_IOU).sum())
    z = 1.96  # интервал Уилсона, 95 %
    c = (k + z * z / 2) / (n + z * z)
    hw = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    per_type = json.loads(Path("splits/pcb.json").read_text())["counts"]["gallery_per_type"]
    recall = round(k / n, 4)
    return {"definition": "доля размеченных рамок, для которых наибольший IoU с box(m) по всем маскам M(I) ≥ 0,5 "
                          "(как в 4.2); crop_n_layers=1",
            "n_boxes": n, "recall": recall, "wilson95": [round(float(c - hw), 3), round(float(c + hw), 3)],
            "recall_075": rec(np.ones(n, bool))["recall_075"],
            "best_iou_quantiles_0_10_25_50_75_90_100": [round(float(x), 3) for x in
                                                        np.quantile(best, [0, .1, .25, .5, .75, .9, 1])],
            "reference_not_criterion": {
                "recall_by_iou": {str(t): round(float((best >= t).mean()), 4) for t in (0.1, 0.25, 0.5, 0.75)},
                "share_with_mask_inside_box": round(float(np.mean([r["mask_inside_box"] for r in rows])), 4),
                "share_with_mask_inside_box_by_type": {t: round(float(np.mean(
                    [r["mask_inside_box"] for r in rows if r["type"] == t])), 4) for t in pcb.CLASSES},
                "note": "рамки разметки свободнее дефекта: маска, точно выделившая дефект, даёт IoU рамок 0,2–0,35; "
                        "стоп-критерий пилотного замера — только recall при IoU ≥ 0,5"},
            "by_type": {t: rec(np.array([r["type"] == t for r in rows])) for t in pcb.CLASSES},
            "by_board": {bd: rec(np.array([r["board"] == bd for r in rows])) for bd in ("09", "10")},
            "stop_criterion": {"recall_min": PCB_RECALL_MIN, "min_refs_per_type": PCB_MIN_PER_TYPE,
                               "refs_per_type": per_type,
                               "fired_by_recall": bool(recall < PCB_RECALL_MIN),
                               "fired_by_refs": bool(min(per_type.values()) < PCB_MIN_PER_TYPE),
                               "fired": bool(recall < PCB_RECALL_MIN or min(per_type.values()) < PCB_MIN_PER_TYPE)},
            "boxes": rows}


def _budget(seg: list, crops: list, enc: dict, owl: list, ev: list, stats: dict) -> dict:
    """Бюджет в часах. Все множители — из замера; числа изображений и эталонов — из `splits/`."""
    hr = json.loads(Path("splits/hr_insdet.json").read_text())["counts"]
    pc = json.loads(Path("splits/pcb.json").read_text())["counts"]
    H = 3600.0

    def seg_rows(g, **kw):
        return [r for r in seg if r["group"] == g and all(r.get(k) == v for k, v in kw.items())]

    def per_image_auto(g, cnl):
        rr = seg_rows(g, mode="auto", crop_n_layers=cnl)
        return _med(rr, "sec_read") + _med(rr, "sec_resize") + _med(rr, "sec_generate")

    def per_image_box(g):
        rr = seg_rows(g, mode="box")
        return _med(rr, "sec_read") + _med(rr, "sec_set_image") + _med(rr, "sec_predict")

    def crop_cpu(g):
        """Секунд CPU на одну маску по вариантам (сумма по снимкам / число масок) и чтение снимка."""
        rr = [r for r in crops if r["group"] == g]
        n = sum(r["n_masks"] for r in rr)
        return {v: sum(r["sec_by_variant"][v] for r in rr) / n for v in rr[0]["sec_by_variant"]}, _med(rr, "sec_read")

    def gpu_rate(e):
        by = next(r for r in enc[e] if r["key"] == "throughput")["by_batch"]
        bs = max(by, key=lambda k: by[k]["crops_per_sec"])
        return by[bs]["crops_per_sec"], int(bs)

    def pool_img(e, g, tag):
        rr = [r for r in enc[e] if r.get("group") == g]
        return float(np.median([r[tag]["sec_forward"] + r[tag]["sec_mask_grid"] + r[tag]["sec_pool"] for r in rr]))

    evd = {r["key"]: r for r in ev}
    ds_cfg = {
        "hr_insdet": {"scenes": "hr_scenes", "refs": "hr_refs", "n_auto": hr["cal_scenes"] + hr["test_scenes"],
                      "n_test": hr["test_scenes"], "n_ref_images": hr["gallery"], "n_refs": hr["gallery"]},
        "pcb": {"scenes": "pcb_images", "refs": "pcb_ref_images",
                "n_auto": pc["cal_images"] + pc["test_images"] + 2,  # + 2 бездефектных снимка калибровочных плат
                "n_test": pc["test_images"], "n_ref_images": pc["gallery_images"], "n_refs": pc["gallery"]},
    }
    out = {"assumptions": [
        "время на изображение — медиана по изображениям пилота; число масок на изображение — среднее по пилоту",
        ("сцены пилота HR-InsDet — два калибровочных помещения уровня easy; на hard-сценах масок может быть больше — "
        "кодирование вырезок пересчитывается пропорционально числу масок после построения кеша масок"),
        ("вырезки: «последовательно» — подготовка на CPU и проход GPU по очереди; «конвейер» — подготовка в отдельном "
        "процессе параллельно GPU, время = max(CPU, GPU) + чтение; в сумму ступеней идёт «последовательно» (верхняя оценка)"),
        ("поиск и оценка — синтетика реального объёма; бутстрэп 1 000 повторов оценён как 1 000 × accumulate — верхняя "
        "граница; сам бутстрэп на момент замера ещё не был написан"),
        ("P и P⊥ на DINOv3 — один проход; оценка U_r (один проход fp32) и сверки сегментатора и кодирования — минуты, в "
         "бюджет не входят"),
    ], "per_dataset": {}}
    for ds, c in ds_cfg.items():
        m_mean = stats[c["scenes"]]["masks_per_image_min_median_mean_max"][2]
        cpu_q, read_q = crop_cpu(c["scenes"])
        cpu_r, read_r = crop_cpu(c["refs"])
        d = {"masks_per_image_mean": m_mean,
             "segmentation_auto_h": {f"crop_n_layers={k}": round(c["n_auto"] * per_image_auto(c["scenes"], k) / H, 2)
                                     for k in (0, 1)},
             "segmentation_box_refs_h": round(c["n_ref_images"] * per_image_box(c["refs"]) / H, 2),
             "crop_run_h": {}, "pool_run_h": {}}
        for e in ENCODERS:
            rate, bs = gpu_rate(e)
            n_q, n_r = c["n_test"] * m_mean, c["n_refs"]
            gpu = (n_q + n_r) / rate
            read = c["n_test"] * read_q + c["n_ref_images"] * read_r
            for v in cpu_q:
                cpu = n_q * cpu_q[v] + n_r * cpu_r[v]
                d["crop_run_h"][f"{e}|C({v})"] = {"sequential": round((cpu + gpu + read) / H, 2),
                                                  "pipelined": round((max(cpu, gpu) + read) / H, 2),
                                                  "cpu_h": round(cpu / H, 2), "gpu_h": round(gpu / H, 2),
                                                  "gallery_share_h": round((n_r * cpu_r[v] + n_r / rate
                                                                            + c["n_ref_images"] * read_r) / H, 2)}
            for tag in ("default", "fallback"):
                t = c["n_test"] * (pool_img(e, c["scenes"], tag) + read_q) \
                    + c["n_ref_images"] * (pool_img(e, c["refs"], tag) + read_r)
                d["pool_run_h"][f"{e}|{tag}"] = {"total": round(t / H, 2), "gallery_share_h": round(
                    c["n_ref_images"] * (pool_img(e, c["refs"], tag) + read_r) / H, 2)}
            d.setdefault("gpu_crops_per_sec", {})[e] = {"rate": round(rate, 1), "batch": bs}
        e_ = evd[ds]
        d["search_eval_per_run_h"] = round((e_["sec_exact_search"] + e_["sec_coco_evaluate"]
                                            + e_["sec_coco_accumulate"]) / H, 3)
        d["bootstrap_1000_per_run_h_upper"] = round(1000 * e_["sec_coco_accumulate"] / H, 2)
        out["per_dataset"][ds] = d

    def tier(ds, pool_tag, encoders=ENCODERS, bootstrap=True):
        d = out["per_dataset"][ds]
        seg_h = sum(d["segmentation_auto_h"].values()) + d["segmentation_box_refs_h"]
        runs = {k: v for k, v in d["crop_run_h"].items() if k.split("|")[0] in encoders}
        crop_h = sum(v["sequential"] for v in runs.values())
        crop_p = sum(v["pipelined"] for v in runs.values())
        pool_h = sum(v["total"] for k, v in d["pool_run_h"].items() if k.endswith(pool_tag) and k.split("|")[0] in encoders)
        n_grid = len(runs) + len(encoders) + ("encoder_dinov3" in encoders)  # вырезки + P на энкодер + P⊥ на DINOv3
        n_base = len(encoders)  # контрольный прогон по протоколу baseline — лучший φ каждого энкодера
        # поиск и оценка: сетка + контрольные + протокол «один эталон на класс»; бутстрэп — для сетки и контрольных
        ev_h = (2 * n_grid + n_base) * d["search_eval_per_run_h"] \
            + (n_grid + n_base) * d["bootstrap_1000_per_run_h_upper"] * bootstrap
        return {"n_grid_runs": n_grid, "segmentation": round(seg_h, 1),
                f"crop_runs_{len(runs)}_sequential": round(crop_h, 1), f"crop_runs_{len(runs)}_pipelined": round(crop_p, 1),
                f"pool_runs_{len(encoders)}": round(pool_h, 1), "search_eval_bootstrap_upper": round(ev_h, 1),
                "total_sequential": round(seg_h + crop_h + pool_h + ev_h, 1),
                "total_pipelined": round(seg_h + crop_p + pool_h + ev_h, 1)}

    tiers = {}
    for tag in ("default", "fallback"):
        t_min, t_plan = tier("hr_insdet", tag), tier("pcb", tag)
        # калибровка на лучшем φ: D_cal кодируется одним вариантом на энкодер — 40 сцен HR-InsDet из 120
        hr_d = out["per_dataset"]["hr_insdet"]
        worst = {e: max(v["sequential"] - v["gallery_share_h"] for k, v in hr_d["crop_run_h"].items()
                        if k.startswith(e)) for e in ENCODERS}
        calib = sum(worst.values()) * hr["cal_scenes"] / hr["test_scenes"]
        owl_t = owl[0]["sec_image_embedder_median"] if owl else None
        owl_h = None if owl_t is None else (hr["gallery"] + hr["test_scenes"] + pc["gallery"] + pc["test_images"]) * owl_t / H
        full_extra = {
            "section_2_3_analytics_cal_scenes": round(calib, 1),
            "exp_4_4_4_5_search_only": round(10 * (hr_d["search_eval_per_run_h"] + hr_d["bootstrap_1000_per_run_h_upper"]), 1),
            "owlv2_forward_passes": None if owl_h is None else round(owl_h, 1),
            "caveat": "оценка до того, как заданы правила §2.3: 4.4 и 4.5 считаются по уже сохранённым эмбеддингам всех масок (поиск и оценка, "
                      "принято 10 прогонов на оба эксперимента); аналитика §2.3 — кодирование 40 калибровочных сцен "
                      "одним вариантом на энкодер, при нескольких вариантах — кратно; OWLv2 — только проходы зрительной "
                      "башни (эталоны + сцены + снимки PCB), без постобработки; стресс-тест полного разрешения "
                      "(crop_n_layers=2, опция 4.4) не входит",
        }
        full_sum = sum(v for v in full_extra.values() if isinstance(v, float))
        # стоп-критерий PCB: статус «стресс-тест» — 4.2 и 4.3 только на DINOv2, без интервалов по платам
        t_stress = tier("pcb", tag, encoders=("encoder_dinov2",), bootstrap=False)
        tiers[f"pool_input_{tag}"] = {
            "minimum": t_min, "plan_increment_pcb": t_plan, "plan_increment_pcb_stress_test_dinov2_only": t_stress,
            "cumulative_sequential_pcb_stress_test": {
                "minimum": t_min["total_sequential"],
                "plan": round(t_min["total_sequential"] + t_stress["total_sequential"] + calib, 1),
                "full": round(t_min["total_sequential"] + t_stress["total_sequential"] + calib + full_sum, 1)},
            "plan_increment_step6_calibration": round(calib, 1),
            "full_increment": full_extra,
            "cumulative_sequential": {"minimum": t_min["total_sequential"],
                                      "plan": round(t_min["total_sequential"] + t_plan["total_sequential"] + calib, 1),
                                      "full": round(t_min["total_sequential"] + t_plan["total_sequential"] + calib
                                                    + full_sum, 1)},
            "cumulative_pipelined": {"minimum": t_min["total_pipelined"],
                                     "plan": round(t_min["total_pipelined"] + t_plan["total_pipelined"] + calib, 1),
                                     "full": round(t_min["total_pipelined"] + t_plan["total_pipelined"] + calib
                                                   + full_sum, 1)}}
    out["tiers_h"] = tiers
    # вариант Б: 30 снимков фона — сегментация и кодирование одним вариантом на энкодер;
    # 40 сцен докодируются лучшим φ каждого энкодера для контрольного прогона на 160 сценах
    per_scene = sum(worst.values()) / hr["test_scenes"]
    out["variant_B_increment_h"] = {"background_30_images": round(30 * (per_image_auto("hr_background", 1) / H + per_scene), 1),
                                    "baseline_control_40_scenes": round(40 * per_scene, 1)}
    # Итог: вход E — вход сегментатора;
    # PCB — «стресс-тест» (DINOv2, только тестовые платы, без бутстрэпа, без 4.4; OWLv2 — на обеих областях); D — калибровочные
    # сцены; контрольный прогон HR-InsDet — на 160 сценах.
    t = tiers["pool_input_default"]
    cal40 = round(40 * per_scene, 1)
    owl_hr = None if not owl else round((hr["gallery"] + hr["test_scenes"]) * owl[0]["sec_image_embedder_median"] / H, 1)
    hr_rows = {"segmentation_160_scenes_and_2400_refs": t["minimum"]["segmentation"],
               "crop_runs_12": t["minimum"]["crop_runs_12_sequential"], "pool_runs_P_and_Pperp": t["minimum"]["pool_runs_2"],
               "search_eval_bootstrap_upper": t["minimum"]["search_eval_bootstrap_upper"],
               "encode_40_cal_scenes_best_variant_per_encoder": cal40,
               "exp_4_4_4_5_search_only": t["full_increment"]["exp_4_4_4_5_search_only"], "owlv2_forward_passes": owl_hr}
    pd_ = out["per_dataset"]["pcb"]
    n_auto_all = pc["cal_images"] + pc["test_images"] + 2
    seg_pcb = round(sum(pd_["segmentation_auto_h"].values()) * pc["test_images"] / n_auto_all
                    + pd_["segmentation_box_refs_h"], 1)
    st = t["plan_increment_pcb_stress_test_dinov2_only"]
    pcb_rows = {"segmentation_361_test_images_and_240_ref_images": seg_pcb, "crop_runs_6_dinov2": st["crop_runs_6_sequential"],
                "pool_run_P_dinov2": st["pool_runs_1"], "search_eval": st["search_eval_bootstrap_upper"],
                # верхняя граница: проход на каждый эталон; нижняя — проход на снимок эталонной платы
                "owlv2_forward_passes_upper": None if not owl else round(
                    (pc["gallery"] + pc["test_images"]) * owl[0]["sec_image_embedder_median"] / H, 1)}
    out["final_by_dataset_h"] = {
        "hr_insdet": {**hr_rows, "total": round(sum(v for v in hr_rows.values() if v), 1)},
        "pcb": {**pcb_rows, "total": round(sum(v for v in pcb_rows.values() if v), 1)},
        "note": "верхняя оценка (подготовка вырезок и GPU по очереди). Кодирование 40 калибровочных сцен лучшим φ каждого "
                "энкодера посчитано один раз: одни и те же эмбеддинги нужны контрольному прогону на 160 сценах, калибровке "
                "порога и аналитике §2.3; если аналитике понадобятся ещё варианты — кратно. OWLv2 — только проходы зрительной "
                "башни. Оговорка о hard-сценах — в assumptions.",
    }
    return out


def _pcb_ref_masks(s: dict) -> dict:
    """Маски эталонов PCB по промпту-рамке: какую долю свободной рамки разметки занимает маска."""
    info = {i["id"]: i for i in pcb.images()}
    rows = []
    for i in s["pcb_ref_images"]:
        recs, _ = _load_masks("pcb_ref_images", i)
        for r in recs:
            b = r["prompt_box"]
            rows.append({"image": i, "type": info[i]["label"], "predicted_iou": round(r["predicted_iou"], 3),
                         "mask_area_to_box_area": round(r["area"] / ((b[2] - b[0]) * (b[3] - b[1])), 3)})
    share = [r["mask_area_to_box_area"] for r in rows]
    by_type = {}
    for t in pcb.CLASSES:
        rr = [r for r in rows if r["type"] == t]
        if rr:
            by_type[t] = {"n": len(rr), "mask_area_to_box_area_median": float(np.median([r["mask_area_to_box_area"] for r in rr])),
                          "predicted_iou_median": float(np.median([r["predicted_iou"] for r in rr]))}
    return {"n": len(rows), "mask_area_to_box_area_min_median_max": [min(share), float(np.median(share)), max(share)],
            "by_type": by_type, "rows": rows,
            "note": "рамка разметки свободнее дефекта; у типов, где дефект — изменение формы дорожки, маска по рамке — "
                    "фрагмент платы внутри рамки, а не дефект"}


def _blur_equivalence() -> dict:
    rows = {r["key"]: r for r in Partial("blurcheck").rows}
    pairs = rows["crops"]["pairs"]
    return {"what": "размытие на уменьшенном окне против буквального в исходном разрешении; вырезки 448×448",
            "n_pairs": len(pairs), "square_side_min_max": [min(p["side"] * p["alpha"] for p in pairs),
                                                           max(p["side"] * p["alpha"] for p in pairs)],
            "abs_diff_max_uint8": max(p["abs_diff_max"] for p in pairs),
            "abs_diff_mean_max": round(max(p["abs_diff_mean"] for p in pairs), 3),
            "sec_exact_max": round(max(p["sec_exact"] for p in pairs), 2),
            "sec_fast_max": round(max(p["sec_fast"] for p in pairs), 2),
            "literal_blur_observed": "квадрат 3 300 пикс.: 20 с при α=1,0 и 56 с при α=1,5 на вырезку (при параллельной "
                                     "загрузке CPU); у квадрата 8 192–12 288 пикс. — минуты",
            **{e: {k: round(v, 5) for k, v in rows[e].items() if k != "key"} for e in ENCODERS}}


def _background_rooms(s: dict, stats: dict) -> dict:
    """Снят ли фон в помещениях сцен: метаданные съёмки (EXIF) и визуальная разметка выборки пилота."""
    from PIL import Image

    def shot(path):
        e = Image.open(path).getexif()
        g = e.get_ifd(0x8825)
        deg = lambda v: round(float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600, 4) if v else None
        return {"model": e.get(272), "date": (e.get_ifd(0x8769).get(36867) or e.get(306) or "")[:10].replace(":", "-"),
                "lat": deg(g.get(2)), "lon": deg(g.get(4))}

    bg = [shot(p) for p in sorted((hr_insdet.ROOT / "Background").glob("*.jpg"))]
    sc = {}
    for x in hr_insdet.scenes():
        sc.setdefault(x["room"], []).append(shot(hr_insdet.ROOT / x["image"]))
    count = lambda rows, k: {v: sum(r[k] == v for r in rows) for v in sorted({r[k] for r in rows})}
    span = lambda rows, k: [min(r[k] for r in rows if r[k]), max(r[k] for r in rows if r[k])]
    per = stats["hr_background"]["per_image"]
    assert sorted(i for v in BACKGROUND_VISUAL.values() for i in v) == s["hr_background"]
    return {
        "exif": {"background": {"n": len(bg), "camera": count(bg, "model"), "dates": count(bg, "date"),
                                "lat_span": span(bg, "lat"), "lon_span": span(bg, "lon")},
                 "scenes": {room: {"n": len(v), "camera": count(v, "model"), "dates": span(v, "date"),
                                   "lat_span": span(v, "lat"), "lon_span": span(v, "lon")} for room, v in sc.items()}},
        "visual_pilot_sample": {k: {"images": v, "masks_per_image_median": float(np.median([per[i]["masks"] for i in v])),
                                    "area_median_of_medians": float(np.median([per[i]["area_median"] for i in v]))}
                                for k, v in BACKGROUND_VISUAL.items()},
        "visual_all_200": "по контактным листам всех 200 снимков: фон снят тем же аппаратом в том же здании, что и сцены, "
                          "но охватывает здание целиком — спортзал (около 30 снимков), зал, студию, библиотеку, коридоры, "
                          "санузлы, холлы; помещения сцен среди них узнаются (переговорная с круглым столом, обе кухни, зона "
                          "отдыха, офис с перегородками, умывальник), но составляют меньшую часть — порядка трети; санузел "
                          "сцен sink/rgb_010–019 и оранжевая стена pantry_room_002 на фоне не найдены. Кадр фона — общий план "
                          "помещения с пустыми поверхностями; кадр сцены — крупный план стола, заставленного мелкими "
                          "предметами, в том числе посторонними. Оценка на глаз, не измерение",
    }


def stage_report() -> None:
    s = samples()
    seg = Partial("segment").rows
    crops = Partial("crops").rows
    enc = {e: Partial(f"encode_{e}").rows for e in ENCODERS}
    owl = Partial("owlv2").rows
    ev = Partial("evalcost").rows
    stats = {g: _mask_stats(g, s[g]) for g in ("hr_scenes", "hr_background", "pcb_images")}

    bg, sc = stats["hr_background"]["_areas"], stats["hr_scenes"]["_areas"]
    compare = {
        "ks_log_area": {"background_vs_scene_all_masks": round(_ks(np.log(bg["all"]), np.log(sc["all"])), 3),
                        "background_vs_scene_distractors": round(_ks(np.log(bg["all"]), np.log(sc["distractors"])), 3)},
        "note": "фон: каждая маска — дистрактор; сцены: дистрактор — маска с IoU рамки < 0,1 ко всем рамкам истины",
    }
    for v in stats.values():
        v.pop("_areas")

    def timing(g, **kw):
        rr = [r for r in seg if r["group"] == g and all(r.get(k) == x for k, x in kw.items())]
        keys = [k for k in ("sec_read", "sec_resize", "sec_generate", "sec_set_image", "sec_predict") if k in rr[0]]
        return {"n": len(rr), **{k: round(_med(rr, k), 2) for k in keys},
                "vram_peak_mib_max": int(max(r["vram_peak_mib"] for r in rr))}

    def pool_summary(e, g):
        rr = [r for r in enc[e] if r.get("group") == g]
        return {tag: {"input_hw": rr[0][tag]["input_hw"], "tokens": rr[0][tag]["tokens"],
                      "sec_forward_median": round(float(np.median([r[tag]["sec_forward"] for r in rr])), 2),
                      "sec_mask_grid_median": round(float(np.median([r[tag]["sec_mask_grid"] for r in rr])), 2),
                      "sec_pool_median": round(float(np.median([r[tag]["sec_pool"] for r in rr])), 3),
                      "vram_peak_mib_max": int(max(r[tag]["vram_peak_mib"] for r in rr)),
                      "n_empty_masks": int(sum(r[tag]["n_empty"] for r in rr)),
                      "all_finite": all(r[tag]["finite"] for r in rr),
                      "patches_under_mask_median_of_medians": round(float(np.median(
                          [r[tag]["patches_under_mask_q10_median"][1] for r in rr])), 1)}
                for tag in ("default", "fallback")}

    def crop_summary(g):
        rr = [r for r in crops if r["group"] == g]
        n = sum(r["n_masks"] for r in rr)
        return {"n_masks": n, "ms_per_mask": {v: round(1000 * sum(r["sec_by_variant"][v] for r in rr) / n, 1)
                                              for v in rr[0]["sec_by_variant"]},
                "cv2_threads": rr[0]["cv2_threads"]}

    report = {
        "written": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "config": {"long_side": 2048, "points_per_batch": 16, "crop_size": 448,
                   "pool_long_side_fallback": POOL_LONG_SIDE_FALLBACK, "distractor_iou_max": DISTRACTOR_IOU_MAX,
                   "sam2_commit": env.sam2_install_info()["commit"], "sam2_autocast": "bfloat16",
                   "encoder_dtype": env.ENCODER_DTYPE, "gpu": "RTX 3050 6 ГБ"},
        "samples": {**s, "rule": "сцены HR-InsDet — по 5 из каждого калибровочного помещения; PCB — калибровочные платы: "
                                 "09 — по снимку на тип, 10 — на 4 типа; фон — 20 из 200; эталоны — 10 и 10; всё по seed "
                                 "разбиений, до замера; тестовые сцены и платы не затрагиваются"},
        "masks": stats,
        "background_vs_scenes": {**compare, "rooms": _background_rooms(s, stats)},
        "pcb_recall": _pcb_recall(s),
        "timing": {
            "segmentation": {g: {f"crop_n_layers={k}": timing(g, mode="auto", crop_n_layers=k)
                                 for k in ((0, 1) if g != "hr_background" else (1,))}
                             for g in ("hr_scenes", "pcb_images", "hr_background")},
            "segmentation_box": {g: timing(g, mode="box") for g in ("hr_refs", "pcb_ref_images")},
            "masks_per_image_by_crop_n_layers": {g: {str(k): _q([r["n_masks"] for r in seg if r["group"] == g
                                                                 and r.get("crop_n_layers") == k], (0, 0.5, 1))
                                                     for k in (0, 1)} for g in ("hr_scenes", "pcb_images")},
            "crops_cpu": {g: crop_summary(g) for g in ("hr_scenes", "pcb_images", "hr_refs", "pcb_ref_images")},
            "crops_gpu": {e: next(r for r in enc[e] if r["key"] == "throughput")["by_batch"] for e in ENCODERS},
            "pool": {e: {g: pool_summary(e, g) for g in ("hr_scenes", "pcb_images", "hr_refs", "pcb_ref_images")}
                     for e in ENCODERS},
            "owlv2": owl[0] if owl else None,
            "search_eval": {r["key"]: {k: v for k, v in r.items() if k != "key"} for r in ev},
        },
        "pcb_ref_masks": _pcb_ref_masks(s),
        "blur_equivalence": _blur_equivalence(),
        "budget": _budget(seg, crops, enc, owl, ev, stats),
        "rows": {"segment": seg, "crops": crops, **{f"encode_{e}": enc[e] for e in ENCODERS}},
    }
    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT)
    print(json.dumps({k: report[k] for k in ("background_vs_scenes",)}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report["pcb_recall"].items() if k != "boxes"}, ensure_ascii=False, indent=1))
    print(json.dumps(report["budget"]["tiers_h"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["samples", "segment", "crops", "blurcheck", "encode", "owlv2", "evalcost", "report"])
    ap.add_argument("--encoder", choices=ENCODERS)
    a = ap.parse_args()
    if a.stage == "samples":
        print(json.dumps(samples(), ensure_ascii=False, indent=1))
    elif a.stage == "encode":
        stage_encode(a.encoder)
    else:
        {"segment": stage_segment, "crops": stage_crops, "blurcheck": stage_blurcheck, "owlv2": stage_owlv2, "evalcost": stage_evalcost,
         "report": stage_report}[a.stage]()
