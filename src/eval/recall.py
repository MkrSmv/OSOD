"""Предел полноты сегментатора: recall по порогам IoU и AR@maxDets по протоколу COCO.

Две величины, обе по рамкам $\\mathrm{box}(m)$ (исключающим, в пикселях исходного снимка):

- recall — доля размеченных рамок, для которых среди **всех** масок $M(I)$ есть маска с IoU рамки не ниже порога;
  сопоставление не взаимно-однозначное, предела числа масок нет — это и есть предел полноты до этапа поиска;
- AR@maxDets — полнота предложений по COCOeval без категорий (`useCats=0`): предложения ранжируются оценкой модели,
  берутся лучшие `maxDets` на снимок, сопоставление с рамками жадное и взаимно-однозначное, среднее по порогам
  0,5:0,95.

COCOeval вызывается один раз на набор снимков; из `evalImgs` берутся счётчики по снимкам (сколько рамок найдено
при каждом пороге, сколько рамок всего), так что любое подмножество сцен — калибровочные, тестовые, easy / hard —
и бутстрэп по сценам считаются суммированием, без повторных вызовов. Для полного набора сумма обязана совпасть
с `accumulate()` — это проверяется при каждом вызове.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval.boxes import best_iou_per_gt

IOU_THRS = np.linspace(0.5, 0.95, 10)  # как в COCOeval
MAX_DETS = [1, 10, 100, 1000]
# метки COCOeval остаются штатными: `summarize()` ищет по ним
AREA_LABELS = ["all", "small", "medium", "large"]


def area_ranges(edges: list[float]) -> list[list[float]]:
    """Бины площади — полуинтервалы $[a,b)$; pycocotools включает обе границы, поэтому верхняя — $b-0{,}5$.

    Площади рамок целые (`area = w·h` по целым пикселям разметки), так что сдвиг на 0,5 ничего не теряет.
    """
    lo, mid, hi = edges
    return [[0, 1e10], [0, lo - 0.5], [lo, mid - 0.5], [mid, hi]]


def area_bin(area: np.ndarray, edges: list[float]) -> np.ndarray:
    """Номер бина площади рамки: 0 — small, 1 — medium, 2 — large (полуинтервалы, как в `area_ranges`)."""
    return np.digitize(area, edges[:2])


def coco_gt(scenes: list[dict], categories: list[str]) -> COCO:
    """Истина в формате COCO. `scenes`: `{"id", "wh", "gt": [{"box", "category_id", "iscrowd"}]}`.

    Идентификаторы аннотаций — с 1: COCOeval пишет идентификатор найденной рамки в `dtMatches` и считает нулевое
    значение отсутствием пары. `area = w·h`.
    """
    images, anns = [], []
    for k, s in enumerate(scenes, 1):
        images.append({"id": k, "file_name": s["id"], "width": s["wh"][0], "height": s["wh"][1]})
        for g in s["gt"]:
            x0, y0, x1, y1 = g["box"]
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": g["category_id"],
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "area": float((x1 - x0) * (y1 - y0)),
                         "iscrowd": g.get("iscrowd", 0)})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns,
                    "categories": [{"id": i, "name": n} for i, n in enumerate(categories)]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def proposal_counts(gt: COCO, boxes: list[np.ndarray], scores: list[np.ndarray], edges: list[float]) -> dict:
    """Счётчики COCOeval по снимкам для предложений без категорий.

    Возвращает `tp` формы (снимки, бины площади, maxDets, пороги IoU) — число найденных рамок — и `n_gt` формы
    (снимки, бины площади); порядок снимков — как в `gt` (и в `boxes`, `scores`).
    """
    img_ids = sorted(gt.imgs)
    cat0 = min(gt.cats)  # при `useCats=0` категория предложения не используется, но обязана существовать
    dets = [{"image_id": i, "category_id": cat0, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(s)}
            for i, bb, ss in zip(img_ids, boxes, scores) for b, s in zip(np.asarray(bb, float).reshape(-1, 4), ss)]
    n_img, n_area, n_det, n_thr = len(img_ids), len(AREA_LABELS), len(MAX_DETS), len(IOU_THRS)
    tp = np.zeros((n_img, n_area, n_det, n_thr))
    n_gt = np.zeros((n_img, n_area))
    ranges = area_ranges(edges)
    for a, rng in enumerate(ranges):
        n_gt[:, a] = [sum(rng[0] <= x["area"] <= rng[1] and not x.get("iscrowd", 0) for x in gt.imgToAnns[i])
                      for i in img_ids]  # рамки с `iscrowd=1` (зоны игнорирования PCB) COCOeval в полноту не считает
    if not dets:  # `loadRes([])` падает
        return {"tp": tp, "n_gt": n_gt}
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), "bbox")
        E.params.useCats = 0
        E.params.maxDets = MAX_DETS
        E.params.iouThrs = IOU_THRS
        E.params.areaRng, E.params.areaRngLbl = ranges, AREA_LABELS
        E.evaluate()
        E.accumulate()
    pos = {i: k for k, i in enumerate(img_ids)}
    for e in E.evalImgs:
        if e is None:
            continue
        a, k = ranges.index(e["aRng"]), pos[e["image_id"]]
        hit = (np.asarray(e["dtMatches"]) > 0) & ~np.asarray(e["dtIgnore"], bool)  # (пороги, предложения по убыванию оценки)
        for m, md in enumerate(MAX_DETS):
            tp[k, a, m] = hit[:, :md].sum(1)
        if int((np.asarray(e["gtIgnore"]) == 0).sum()) != n_gt[k, a]:
            raise AssertionError(f"бин площади {AREA_LABELS[a]}: число рамок расходится с COCOeval")
    ours = tp.sum(0) / np.maximum(n_gt.sum(0), 1)[:, None, None]            # (бины, maxDets, пороги)
    theirs = np.moveaxis(E.eval["recall"][:, 0], 0, -1)                      # [T, K, A, M] → (бины, maxDets, пороги)
    if not np.allclose(np.where(theirs < 0, 0, theirs), ours):
        raise AssertionError("счётчики по снимкам расходятся с COCOeval.accumulate()")
    return {"tp": tp, "n_gt": n_gt}


def best_ious(gt_boxes: list[np.ndarray], mask_boxes: list[np.ndarray]) -> list[np.ndarray]:
    return [best_iou_per_gt(g, m) for g, m in zip(gt_boxes, mask_boxes)]


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def summarize(best: list[np.ndarray], bins: list[np.ndarray], counts: dict, sel: np.ndarray,
              n_boot: int, rng: np.random.Generator) -> dict:
    """Метрики по подмножеству сцен `sel` (индексы) с бутстрэпом по сценам (перцентильный интервал 95 %).

    По бинам площади: recall при 0,5 / 0,75 / 0,5:0,95 и AR при каждом `maxDets`.
    """
    # по сценам: число рамок, найденных при каждом пороге, — (сцены, бины площади, пороги); бин 0 — все рамки
    hits = np.zeros((len(best), len(AREA_LABELS), len(IOU_THRS)))
    for k, (b, ab) in enumerate(zip(best, bins)):
        ok = b[:, None] >= IOU_THRS[None, :]
        hits[k, 0] = ok.sum(0)
        for a in range(3):
            hits[k, a + 1] = ok[ab == a].sum(0)
    tp, n_gt = counts["tp"], counts["n_gt"]

    def metrics(idx: np.ndarray) -> np.ndarray:
        """(…, бины, 3 + maxDets): recall50, recall75, recall50:95, AR@maxDets; `idx` — (…, сцены)."""
        den = n_gt[idx].sum(-2)                                  # (…, бины)
        r = _ratio(hits[idx].sum(-3), den[..., None])            # (…, бины, пороги)
        ar = _ratio(tp[idx].sum(-4), den[..., None, None]).mean(-1)  # (…, бины, maxDets)
        return np.concatenate([r[..., [0]], r[..., [5]], r.mean(-1, keepdims=True), ar], -1)

    names = ["recall_50", "recall_75", "recall_50_95"] + [f"ar_{m}" for m in MAX_DETS]
    point = metrics(sel)
    boot = metrics(sel[rng.integers(0, len(sel), (n_boot, len(sel)))])
    with warnings.catch_warnings():  # бин без рамок в подмножестве: все повторы — NaN, в записи будет null
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
    out = {}
    for a, lab in enumerate(AREA_LABELS):
        out[lab] = {"n_gt": int(n_gt[sel, a].sum()),
                    **{n: None if np.isnan(point[a, j]) else float(point[a, j]) for j, n in enumerate(names)},
                    "ci95": {n: None if np.isnan(point[a, j]) else [float(lo[a, j]), float(hi[a, j])]
                             for j, n in enumerate(names)}}
    return out
