"""Эксперимент 4.2 — предел полноты сегментатора → `experiments/runs/4_2_<dataset>.json`.

По кешу масок (`scripts/segment.py`), без GPU: для каждой размеченной рамки — наибольший IoU с $\\mathrm{box}(m)$ по всем маскам
$M(I)$; recall при IoU ≥ 0,5 / 0,75 / 0,5:0,95 и AR@100 по протоколу COCO (оценка предложения — predicted IoU
SAM 2), при `crop_n_layers` 0 и 1, отдельно на калибровочных, тестовых и всех сценах, по уровню трудности и бинам
площади; интервалы — бутстрэп по сценам. `crop_n_layers=2` не считается.

Разрешение входа для последующих прогонов выбирается правилом, записанным до счёта: только по калибровочным сценам. Из масок берутся рамка и оценка модели — ничего больше.

PKU-Market-PCB: только тестовые платы — калибровочные не сегментируются; блок
выбора отключён явно, `crop_n_layers` перенесён с HR-InsDet; отчёт при обоих значениях, с разбивкой по платам и типам
дефекта; зоны игнорирования — в истине с `iscrowd=1`, в полноту не входят. Рядом пишется recall пилотного замера на
калибровочных платах (стоп-критерий области); роль области по числу 4.2 не пересматривается.

    python scripts/run_4_2.py
    python scripts/run_4_2.py --dataset pcb
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import env
from src.data import hr_insdet, pcb
from src.eval import recall as R
from src.segment import cache as MC

# --- записано до счёта, по результату не пересматривается ---
CROP_N_LAYERS = (0, 1)
SELECT_SPLIT = "cal"    # тестовые сцены в выборе не участвуют
SELECT_IOU = 0.5
SELECT_DEFAULT = 1      # остаётся, если recall на калибровочных сценах при 1 не ниже, чем при 0; иначе — 0

N_BOOT = 1000           # бутстрэп по сценам; seed — seed разбиения
SCORE = "predicted_iou"
RUNS = Path("experiments/runs")


PILOT = Path("experiments/pilot.json")  # recall пилота на калибровочных платах PCB — рядом с 4.2, не критерий 4.2
REFERENCE_IOUS = (0.1, 0.25, 0.5, 0.75)  # справочно, не критерий


def select(recall_cal: dict[int, float]) -> int:
    return SELECT_DEFAULT if recall_cal[1] >= recall_cal[0] else 0


def load_hr_insdet(sp: dict) -> tuple[list[dict], list[str], list[float], dict[str, np.ndarray]]:
    labels = hr_insdet.objects()
    by_id = {s["id"]: s for s in hr_insdet.scenes()}
    ids = sp["cal"] + sp["test"]
    scenes = [{"id": i, "wh": hr_insdet.SCENE_WH, "level": by_id[i]["level"],
               "gt": hr_insdet.scene_gt(by_id[i], labels)} for i in ids]
    split_of = np.array(["cal"] * len(sp["cal"]) + ["test"] * len(sp["test"]))
    level_of = np.array([s["level"] for s in scenes])
    every = np.ones(len(ids), bool)
    subsets = {f"{s}/{lv}": np.flatnonzero((every if s == "all" else split_of == s) & (every if lv == "all" else level_of == lv))
               for s in ("cal", "test", "all") for lv in ("all", "easy", "hard")}
    return scenes, labels, hr_insdet.AREA_EDGES, subsets


def load_pcb(sp: dict) -> tuple[list[dict], list[str], list[float], dict[str, np.ndarray]]:
    """Только тестовые платы: калибровочные в авторежиме не сегментируются."""
    by_id = {i["id"]: i for i in pcb.images()}
    zones = pcb.ignore_zones()
    scenes = [{"id": i, "wh": pcb.image_wh(by_id[i]), "board": by_id[i]["board"], "type": by_id[i]["label"],
               "gt": pcb.gt_with_zones(by_id[i], zones)} for i in sp["test"]]
    board_of, type_of = (np.array([s[k] for s in scenes]) for k in ("board", "type"))
    subsets = {"test/all": np.arange(len(scenes))}
    subsets |= {f"test/board_{b}": np.flatnonzero(board_of == b) for b in pcb.BOARDS["test"]}
    subsets |= {f"test/type_{t}": np.flatnonzero(type_of == t) for t in pcb.CLASSES}
    return scenes, pcb.CLASSES, pcb.AREA_EDGES, subsets


def inside_box(gt_boxes: np.ndarray, mask_boxes: np.ndarray) -> np.ndarray:
    """Справочно, не критерий: у рамки разметки есть маска, рамка которой лежит
    внутри неё (не меньше 0,9 своей площади) и не мельче 5 % её площади, — дефект выделен, но рамка разметки свободнее."""
    g, b = gt_boxes[:, None], mask_boxes[None]
    ix = np.clip(np.minimum(b[..., 2], g[..., 2]) - np.maximum(b[..., 0], g[..., 0]), 0, None)
    iy = np.clip(np.minimum(b[..., 3], g[..., 3]) - np.maximum(b[..., 1], g[..., 1]), 0, None)
    ab, ag = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]), (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return ((ix * iy >= 0.9 * ab) & (ab >= 0.05 * ag)).any(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hr_insdet", choices=["hr_insdet", "pcb"])
    args = ap.parse_args()
    t_start = time.perf_counter()
    is_pcb = args.dataset == "pcb"

    sp = json.loads(Path(f"splits/{args.dataset}.json").read_text())
    scenes, labels, area_edges, subsets = (load_pcb if is_pcb else load_hr_insdet)(sp)
    ids = [s["id"] for s in scenes]
    # полнота — по размеченным рамкам; зоны игнорирования PCB (`iscrowd=1`) остаются только в истине COCOeval
    gt_boxes = [np.array([g["box"] for g in s["gt"] if not g["iscrowd"]], float).reshape(-1, 4) for s in scenes]
    bins = [R.area_bin((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]), area_edges) for g in gt_boxes]
    gt = R.coco_gt(scenes, labels)

    results, caches, seg_time, best_by_cnl, reference = {}, {}, {}, {}, {}
    for cnl in CROP_N_LAYERS:
        mc = MC.MaskCache(args.dataset, MC.auto_key(cnl))
        missing = [i for i in ids if not mc.has(i)]
        if missing:
            sys.exit(f"crop_n_layers={cnl}: в кеше нет {len(missing)} сцен — сначала "
                     f"python scripts/segment.py --dataset {'pcb' if is_pcb else 'hr_insdet'} --crop-n-layers 1 и 0")
        entries = [mc.load(i) for i in ids]
        boxes = [e.boxes for e in entries]
        scores = [np.array([r[SCORE] for r in e.records]) for e in entries]
        best = R.best_ious(gt_boxes, boxes)
        counts = R.proposal_counts(gt, boxes, scores, area_edges)
        for a in range(3):  # бины площади в recall и в COCOeval — одни и те же рамки
            assert sum(int((b == a).sum()) for b in bins) == counts["n_gt"][:, a + 1].sum()
        n_masks = np.array([len(e) for e in entries])
        results[str(cnl)] = {}
        for name, sel in subsets.items():
            if not len(sel):
                continue
            rng = np.random.default_rng([sp["seed"], cnl, sorted(subsets).index(name)])
            results[str(cnl)][name] = {
                "n_scenes": int(len(sel)), "n_masks": int(n_masks[sel].sum()),
                "masks_per_scene_mean_median_max": [float(n_masks[sel].mean()), float(np.median(n_masks[sel])),
                                                    int(n_masks[sel].max())],
                "by_area": R.summarize(best, bins, counts, sel, N_BOOT, rng)}
        caches[str(cnl)] = {"dir": mc.dir.name, "key": mc.key}
        seg_time[str(cnl)] = {k: round(float(sum(e.info[k] for e in entries)), 1)
                              for k in ("sec_read", "sec_resize", "sec_generate")}
        seg_time[str(cnl)]["attempts_max"] = int(max(e.info.get("attempts", 1) for e in entries))
        best_by_cnl[cnl] = best
        if is_pcb:
            flat = np.concatenate(best)
            inside = np.concatenate([inside_box(g, b) for g, b in zip(gt_boxes, boxes)])
            type_of_box = np.concatenate([[s["type"]] * len(g) for s, g in zip(scenes, gt_boxes)])
            reference[str(cnl)] = {
                "recall_by_iou": {str(t): float((flat >= t).mean()) for t in REFERENCE_IOUS},
                "share_with_mask_inside_box": float(inside.mean()),
                "share_with_mask_inside_box_by_type": {t: float(inside[type_of_box == t].mean()) for t in pcb.CLASSES},
                "best_iou_quantiles_0_10_25_50_75_90_100": [float(v) for v in
                                                            np.quantile(flat, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])]}

    if is_pcb:
        extra = pcb_blocks(reference)
    else:
        extra = {"selection": selection_block(sp, subsets, results, best_by_cnl, gt_boxes)}

    record = {
        "run_id": f"4_2_{args.dataset}", "experiment": "4.2", "dataset": args.dataset,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": {"crop_n_layers": list(CROP_N_LAYERS), "long_side": MC.S.LONG_SIDE,
                   "points_per_batch": MC.S.POINTS_PER_BATCH, "proposal_score": SCORE,
                   "iou_thresholds": [round(float(t), 2) for t in R.IOU_THRS], "max_dets": R.MAX_DETS,
                   "use_cats": 0, "area_edges_px2": area_edges[:2], "area_of": "рамка разметки, w·h, исходный снимок",
                   "bootstrap": {"unit": "image" if is_pcb else "scene", "n": N_BOOT, "ci": "percentile 95 %"},
                   "splits_file": f"splits/{args.dataset}.json", "splits_written": sp["written"]},
        "seed": sp["seed"],
        "code": env.code_stamp(),
        "versions": {"packages": env.package_versions(), "sam2": env.sam2_install_info(),
                     "segmenter": json.loads(env.ENV_JSON.read_text())["models"]["segmenter"]["revision"],
                     "lock_sha256": env.lock_sha256()},
        "mask_cache": caches,
        "timing": {"segmentation_from_cache_info_sec": seg_time, "evaluation_sec": round(time.perf_counter() - t_start, 1)},
        "n_gt_boxes": {s: int(sum(len(gt_boxes[k]) for k in subsets[f"{s}/all"]))
                       for s in ("cal", "test") if f"{s}/all" in subsets},
        "results": results, **extra,
    }
    if is_pcb:
        record["n_ignore_zones"] = int(sum(g["iscrowd"] for s in scenes for g in s["gt"]))
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{record['run_id']}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(out)

    shown = [n for n in subsets if is_pcb or n in ("cal/all", "test/all", "all/all", "all/easy", "all/hard")]
    for c in CROP_N_LAYERS:
        for name in shown:
            m = results[str(c)][name]["by_area"]["all"]
            print(f"crop_n_layers={c} {name:26s} recall50 {m['recall_50']:.4f} recall75 {m['recall_75']:.4f} "
                  f"recall50:95 {m['recall_50_95']:.4f} AR@100 {m['ar_100']:.4f} AR@1000 {m['ar_1000']:.4f}")
    print(json.dumps(extra, ensure_ascii=False, indent=1))
    print(f"запись: {out}")


def pcb_blocks(reference: dict) -> dict:
    """Блоки записи PCB: выбор отключён (значение перенесено с HR-InsDet) и recall пилота на калибровочных платах."""
    hr = json.loads((RUNS / "4_2_hr_insdet.json").read_text())["selection"]
    pilot = json.loads(PILOT.read_text())["pcb_recall"]
    return {
        "selection": {
            "enabled": False,
            "reason": ("калибровочные платы PCB в авторежиме не сегментируются; crop_n_layers перенесён с HR-InsDet; "
                       "тестовые платы в выборе не участвуют"),
            "carried_over_crop_n_layers": hr["selected_crop_n_layers"], "carried_over_from": "experiments/runs/4_2_hr_insdet.json"},
        "pilot_cal_reference": {
            "source": "experiments/pilot.json, pcb_recall", "boards": pcb.BOARDS["cal"], "crop_n_layers": 1,
            "n_boxes": pilot["n_boxes"], "recall_50": pilot["recall"], "recall_75": pilot["recall_075"],
            "wilson95": pilot["wilson95"], "by_type": pilot["by_type"], "by_board": pilot["by_board"],
            "stop_criterion_recall_min": pilot["stop_criterion"]["recall_min"],
            "note": ("статус «стресс-тест на границе применимости» задан стоп-критерием пилотного замера на калибровочных платах "
                     "и при любом recall 4.2 на тестовых платах не пересматривается")},
        "reference_not_criterion": reference}


def selection_block(sp: dict, subsets: dict, results: dict, best_by_cnl: dict, gt_boxes: list) -> dict:
    # выбор разрешения входа — правилом, записанным до счёта; парная разность — справочно, в выборе не участвует
    cal = subsets[f"{SELECT_SPLIT}/all"]
    recall_cal = {c: results[str(c)][f"{SELECT_SPLIT}/all"]["by_area"]["all"]["recall_50"] for c in CROP_N_LAYERS}
    hit = {c: np.array([(best_by_cnl[c][k] >= SELECT_IOU).sum() for k in cal]) for c in CROP_N_LAYERS}
    n = np.array([len(gt_boxes[k]) for k in cal])
    idx = np.random.default_rng([sp["seed"], 99]).integers(0, len(cal), (N_BOOT, len(cal)))
    diff = (hit[1][idx].sum(1) - hit[0][idx].sum(1)) / n[idx].sum(1)
    selection = {
        "rule": (f"crop_n_layers={SELECT_DEFAULT}, если recall при IoU ≥ {SELECT_IOU} на сценах split «{SELECT_SPLIT}» "
                 "при 1 не ниже, чем при 0; иначе 0"),
        "split": SELECT_SPLIT, "iou": SELECT_IOU, "recall": {str(c): recall_cal[c] for c in CROP_N_LAYERS},
        "diff_1_minus_0": recall_cal[1] - recall_cal[0],
        "diff_ci95_bootstrap_scenes_reference_only": [float(v) for v in np.quantile(diff, [0.025, 0.975])],
        "selected_crop_n_layers": select(recall_cal)}
    return selection


if __name__ == "__main__":
    main()
