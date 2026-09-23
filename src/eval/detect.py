"""Метрики детекции 4.3: AP / AP50 / AP75 и AR по протоколу COCO, top-1, AUROC — с бутстрэпом по сценам.

COCOeval вызывается один раз на набор снимков (`evaluate`); сопоставление детекций с рамками в нём идёт внутри
снимка (и категории), поэтому из `evalImgs` берутся записи по снимкам, а любое подмножество сцен и любой повтор
бутстрэпа считаются по ним заново своим `accumulate` с весами снимков — без повторных вызовов COCOeval.

- Точечная оценка (`_ap_exact`) повторяет `COCOeval.accumulate()` операция в операцию; для полного набора снимков
  она сверяется с ним при каждом вызове `evaluate` (до порядка суммирования, 1e-12), как счётчики 4.2.
- Повторы бутстрэпа (`_ap_weighted`) — тот же алгоритм, векторизованный по повторам: вес снимка — сколько раз он
  попал в повтор. Поиск порогов полноты идёт одним `searchsorted` по строкам, сдвинутым на номер строки; сдвиг
  округляет сравнение `rc >= thr`, и полнота, меньшая порога на 1 ulp (7/100 против 0,07000000000000001 у `linspace`),
  считается равной ему. Ошибка односторонняя: AP повтора завышен в среднем на ~1e-4, границы интервалов AP — примерно
  на 0,01 п.; точечные оценки так не считаются.

AP берётся из своих таблиц при `maxDets` из конфигурации, а не из `summarize()`: тот считает только при 100. Предел `maxDets` в COCOeval действует на пару «снимок — категория» при `useCats=1` и на
снимок при `useCats=0`.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval import recall as R

IOU_THRS = R.IOU_THRS
REC_THRS = np.linspace(0.0, 1.0, 101)  # как в `pycocotools.cocoeval.Params`; сверяется в `evaluate`
MAX_DETS = R.MAX_DETS
AREA_LABELS = R.AREA_LABELS
T50, T75 = 0, 5
_CHUNK = 4_000_000  # элементов (повторы × пороги × детекции) в одном куске векторизованного счёта


@dataclass
class Matches:
    """Итог `COCOeval.evaluate()` по снимкам. Группа — пара (категория, бин площади); при `use_cats=False`
    категория одна. В группе детекции идут в порядке снимков, внутри снимка — по убыванию оценки (`rank`)."""
    n_img: int
    cat_ids: list[int]
    use_cats: bool
    n_gt: np.ndarray                                   # (снимки, категории, бины)
    groups: dict[tuple[int, int], dict] = field(default_factory=dict)

    def tp_img(self, max_det: int) -> np.ndarray:
        """Число найденных рамок: (снимки, категории, бины, пороги IoU) — для AR."""
        out = np.zeros((self.n_img, len(self.cat_ids), len(AREA_LABELS), len(IOU_THRS)))
        for (k, a), g in self.groups.items():
            keep = g["rank"] < max_det
            hit = (g["match"] & ~g["ignore"])[:, keep]                     # (пороги, детекции)
            np.add.at(out[:, k, a], g["img"][keep], hit.T)
        return out


def _n_gt(gt: COCO, img_ids: list[int], cat_ids: list[int], ranges: list[list[float]], use_cats: bool) -> np.ndarray:
    """Неигнорируемые рамки истины по снимкам, категориям и бинам — как их считает COCOeval (обе границы включены)."""
    pos_i, pos_k = {i: n for n, i in enumerate(img_ids)}, {c: n for n, c in enumerate(cat_ids)}
    out = np.zeros((len(img_ids), len(cat_ids) if use_cats else 1, len(ranges)))
    for ann in gt.dataset["annotations"]:
        if ann.get("iscrowd", 0) or ann.get("ignore", 0):
            continue
        for a, (lo, hi) in enumerate(ranges):
            if lo <= ann["area"] <= hi:
                out[pos_i[ann["image_id"]], pos_k[ann["category_id"]] if use_cats else 0, a] += 1
    return out


def evaluate(gt: COCO, dets: list[dict], edges: list[float], use_cats: bool = True,
             score_min: float | None = None, check: bool = True) -> Matches:
    """`dets` — по снимку в порядке `sorted(gt.imgs)`: `{"box": (n, 4) исключающие, "label_id": (n,), "score": (n,)}`.

    `score_min` — порог на оценку; AP считается без порога.
    """
    img_ids, cat_ids = sorted(gt.imgs), sorted(gt.cats)
    if len(dets) != len(img_ids):
        raise ValueError(f"детекции — на {len(dets)} снимков, в истине {len(img_ids)}")
    ranges = R.area_ranges(edges)
    rows = []
    for i, d in zip(img_ids, dets):
        box, lab, sc = np.asarray(d["box"], float).reshape(-1, 4), np.asarray(d["label_id"]), np.asarray(d["score"], float)
        if not len(box) == len(lab) == len(sc):
            raise ValueError("рамки, метки и оценки детекций — разной длины")
        if len(lab) and not set(lab.tolist()) <= set(cat_ids):  # при чужом `category_id` COCOeval молча теряет детекцию
            raise ValueError("метка детекции вне категорий истины")
        keep = np.ones(len(sc), bool) if score_min is None else sc >= score_min
        rows += [{"image_id": i, "category_id": int(c), "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(s)}
                 for b, c, s in zip(box[keep], lab[keep], sc[keep])]
    m = Matches(len(img_ids), cat_ids if use_cats else [-1], use_cats, _n_gt(gt, img_ids, cat_ids, ranges, use_cats))
    if not rows:  # `loadRes([])` падает; без детекций AP = 0 без вызова COCOeval
        return m
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(rows), "bbox")
        E.params.useCats = int(use_cats)
        E.params.maxDets = MAX_DETS
        E.params.iouThrs = IOU_THRS
        E.params.areaRng, E.params.areaRngLbl = ranges, AREA_LABELS
        if not np.array_equal(E.params.recThrs, REC_THRS):
            raise AssertionError("пороги полноты установленной версии pycocotools — не 0:0,01:1")
        E.evaluate()
    n_i, n_a = len(img_ids), len(ranges)
    for k in range(len(m.cat_ids)):
        for a in range(n_a):
            img, score, rank, match, ignore = [], [], [], [], []
            for i in range(n_i):
                e = E.evalImgs[k * n_a * n_i + a * n_i + i]  # порядок `evaluate()`: категории × бины × снимки
                if e is None:
                    continue
                if int((np.asarray(e["gtIgnore"]) == 0).sum()) != m.n_gt[i, k, a]:
                    raise AssertionError(f"бин площади {AREA_LABELS[a]}: число рамок расходится с COCOeval")
                n = len(e["dtScores"])
                if n:
                    img.append(np.full(n, i)), score.append(np.asarray(e["dtScores"], float)), rank.append(np.arange(n))
                    match.append(np.asarray(e["dtMatches"]) > 0), ignore.append(np.asarray(e["dtIgnore"], bool))
            if img:
                m.groups[k, a] = {"img": np.concatenate(img), "score": np.concatenate(score), "rank": np.concatenate(rank),
                                  "match": np.concatenate(match, 1), "ignore": np.concatenate(ignore, 1)}
    if check:
        _check_against_cocoeval(m, E)
    return m


def _check_against_cocoeval(m: Matches, E: COCOeval) -> None:
    """Свой `accumulate` на полном наборе снимков обязан совпасть с `COCOeval.accumulate()`."""
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate()
    every = np.ones(m.n_img, bool)
    for j, md in enumerate(MAX_DETS):
        ap = ap_table(m, every, md)                                         # (пороги, категории, бины)
        theirs = E.eval["precision"][:, :, :, :, j].mean(1)                 # [T, R, K, A, M] → среднее по порогам полноты
        theirs = np.where(E.eval["precision"][:, 0, :, :, j] < 0, np.nan, theirs)
        if not np.allclose(np.nan_to_num(ap, nan=-1.0), np.nan_to_num(theirs, nan=-1.0), rtol=0, atol=1e-12):
            raise AssertionError(f"AP по записям снимков расходится с COCOeval.accumulate() при maxDets={md}")
        rc = recall_table(m, every[None].astype(float), md)[0]              # (категории, бины, пороги)
        theirs = np.moveaxis(E.eval["recall"][:, :, :, j], 0, -1)
        if not np.allclose(np.nan_to_num(rc, nan=-1.0), theirs, rtol=0, atol=1e-12):
            raise AssertionError(f"полнота по записям снимков расходится с COCOeval.accumulate() при maxDets={md}")


# ---------------------------------------------------------------------- accumulate


def _ap_exact(match: np.ndarray, ignore: np.ndarray, npig: float) -> np.ndarray:
    """AP по порогам IoU для детекций одной группы, уже отсортированных по оценке, — как в `COCOeval.accumulate()`."""
    tp = np.cumsum(match & ~ignore, axis=1).astype(float)
    fp = np.cumsum(~match & ~ignore, axis=1).astype(float)
    out = np.zeros(len(tp))
    for t in range(len(tp)):
        rc = tp[t] / npig
        pr = tp[t] / (fp[t] + tp[t] + np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]                          # огибающая точности
        inds = np.searchsorted(rc, REC_THRS, side="left")
        q = np.zeros(len(REC_THRS))
        ok = inds < len(pr)
        q[ok] = pr[inds[ok]]
        out[t] = q.mean()
    return out


def ap_table(m: Matches, sel: np.ndarray, max_det: int) -> np.ndarray:
    """Точечная оценка: AP (пороги IoU, категории, бины) по снимкам `sel` (булева маска); без рамок истины — NaN."""
    npig = m.n_gt[sel].sum(0)
    out = np.where(npig > 0, 0.0, np.nan)[None].repeat(len(IOU_THRS), 0)
    for (k, a), g in m.groups.items():
        if npig[k, a] == 0:
            continue
        keep = np.flatnonzero((g["rank"] < max_det) & sel[g["img"]])
        keep = keep[np.argsort(-g["score"][keep], kind="mergesort")]        # как в COCOeval: по снимкам, затем устойчиво
        if len(keep):
            out[:, k, a] = _ap_exact(g["match"][:, keep], g["ignore"][:, keep], npig[k, a])
    return out


def _ap_weighted(match: np.ndarray, ignore: np.ndarray, w: np.ndarray, npig: np.ndarray) -> np.ndarray:
    """То же для B повторов сразу: `w` — (B, детекции) веса снимков, `npig` — (B,). Возвращает (B, пороги IoU)."""
    n_b, n_d = w.shape
    n_t = len(match)
    tp = np.cumsum(w[:, None, :] * (match & ~ignore)[None], axis=2)         # (B, T, D)
    fp = np.cumsum(w[:, None, :] * (~match & ~ignore)[None], axis=2)
    rc = tp / np.maximum(npig, 1)[:, None, None]
    pr = tp / (fp + tp + np.spacing(1))
    pr = np.maximum.accumulate(pr[..., ::-1], axis=2)[..., ::-1].reshape(-1, n_d)
    rows = np.arange(n_b * n_t)[:, None]
    flat = (rc.reshape(-1, n_d) + 2.0 * rows).ravel()                       # rc ∈ [0, 1]: строки не перекрываются
    inds = np.searchsorted(flat, (REC_THRS[None] + 2.0 * rows).ravel(), side="left").reshape(-1, len(REC_THRS)) - rows * n_d
    ok = inds < n_d
    q = np.where(ok, np.take_along_axis(pr, np.minimum(inds, n_d - 1), 1), 0.0)
    return np.where(npig[:, None] > 0, q.mean(1).reshape(n_b, n_t), np.nan)


def ap_boot(m: Matches, W: np.ndarray, max_det: int) -> np.ndarray:
    """AP для повторов бутстрэпа: `W` — (B, снимки) веса. Возвращает (B, пороги IoU, категории, бины)."""
    npig = np.einsum("bi,ika->bka", W, m.n_gt)
    out = np.where(npig > 0, 0.0, np.nan)[:, None].repeat(len(IOU_THRS), 1)
    for (k, a), g in m.groups.items():
        keep = np.flatnonzero(g["rank"] < max_det)
        keep = keep[np.argsort(-g["score"][keep], kind="mergesort")]
        if not len(keep):
            continue
        step = max(1, _CHUNK // (len(IOU_THRS) * len(keep)))
        for b0 in range(0, len(W), step):
            w = W[b0:b0 + step][:, g["img"][keep]].astype(float)
            out[b0:b0 + step, :, k, a] = _ap_weighted(g["match"][:, keep], g["ignore"][:, keep], w, npig[b0:b0 + step, k, a])
    return out


def recall_table(m: Matches, W: np.ndarray, max_det: int) -> np.ndarray:
    """Полнота итоговых детекций: (B, категории, бины, пороги IoU); без рамок истины — NaN."""
    tp = np.einsum("bi,ikat->bkat", W, m.tp_img(max_det))
    npig = np.einsum("bi,ika->bka", W, m.n_gt)
    return np.where(npig[..., None] > 0, tp / np.maximum(npig, 1)[..., None], np.nan)


# ---------------------------------------------------------------------- сводки с бутстрэпом по сценам


def boot_weights(sel: np.ndarray, n_img: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Веса снимков в повторах бутстрэпа по сценам `sel` (индексы): (n_boot, n_img), сколько раз снимок взят."""
    idx = sel[rng.integers(0, len(sel), (n_boot, len(sel)))]
    W = np.zeros((n_boot, n_img), np.int32)
    np.add.at(W, (np.arange(n_boot)[:, None], idx), 1)
    return W


def _ci(boot: np.ndarray) -> list[float] | None:
    boot = boot[~np.isnan(boot)]
    return [float(v) for v in np.quantile(boot, [0.025, 0.975])] if len(boot) else None


def _nanmean(x: np.ndarray, axis) -> np.ndarray:
    ok = ~np.isnan(x)
    n = ok.sum(axis)
    return np.where(n > 0, np.where(ok, x, 0).sum(axis) / np.maximum(n, 1), np.nan)


def _val(x) -> float | None:
    return None if np.isnan(x) else float(x)


def summarize_ap(m: Matches, sel: np.ndarray, W: np.ndarray | None, max_det: int, ref_max_det: int = 100,
                 cat_subsets: dict[str, list[int]] | None = None, keep_boot: bool = False,
                 keep_boot_metrics: bool = False) -> dict:
    """AP, AP50, AP75 при `max_det` по снимкам `sel` (индексы) и бинам площади; интервал 95 % — по повторам `W`.

    Справочно — AP при `ref_max_det`. `cat_subsets` — имя → номера категорий: те же величины как среднее по части
    категорий (при `useCats=1` категории оцениваются независимо, так что это тот же COCOeval с `catIds`).
    `keep_boot` — оставить в строке `all` значения AP по повторам (`boot_ap`); `keep_boot_metrics` — там же, без
    округления, повторы AP, AP50 и AP75 (`boot`) — для парных разностей 4.4.
    """
    mask = np.zeros(m.n_img, bool)
    mask[sel] = True
    point, ref = ap_table(m, mask, max_det), ap_table(m, mask, ref_max_det)
    boot = None if W is None else ap_boot(m, W, max_det)
    subsets = {"": list(range(len(m.cat_ids))), **(cat_subsets or {})}
    out: dict = {}
    for name, cats in subsets.items():
        block = out if not name else out.setdefault(name, {})
        for a, lab in enumerate(AREA_LABELS):
            p = _nanmean(point[:, cats, a], 1)                              # (пороги IoU,)
            row = {"n_gt": int(m.n_gt[sel][:, cats, a].sum()), "n_categories": int((~np.isnan(point[0, cats, a])).sum()),
                   "ap": _val(_nanmean(p, 0)), "ap50": _val(p[T50]), "ap75": _val(p[T75]),
                   f"ap_maxdets_{ref_max_det}": _val(_nanmean(_nanmean(ref[:, cats, a], 1), 0))}
            if boot is not None:
                b = _nanmean(boot[:, :, cats, a], 2)                        # (B, пороги IoU)
                row["ci95"] = {"ap": _ci(_nanmean(b, 1)), "ap50": _ci(b[:, T50]), "ap75": _ci(b[:, T75])}
                if keep_boot and not name and a == 0:  # повторы AP по всем категориям и размерам — для парных разностей
                    row["boot_ap"] = [round(float(v), 6) for v in _nanmean(b, 1)]
                if keep_boot_metrics and not name and a == 0:
                    row["boot"] = {"ap": _nanmean(b, 1), "ap50": b[:, T50], "ap75": b[:, T75]}
            block[lab] = row
    return out


def ap_by_category(m: Matches, sel: np.ndarray, max_det: int) -> list[dict]:
    """Точечные AP, AP50, AP75 каждой категории по снимкам `sel` (индексы), все размеры; без рамок истины — None."""
    mask = np.zeros(m.n_img, bool)
    mask[sel] = True
    point = ap_table(m, mask, max_det)[:, :, 0]                             # (пороги IoU, категории)
    return [{"n_gt": int(m.n_gt[sel][:, k, 0].sum()), "ap": _val(_nanmean(point[:, k], 0)), "ap50": _val(point[T50, k]),
             "ap75": _val(point[T75, k])} for k in range(len(m.cat_ids))]


def summarize_ar(m: Matches, sel: np.ndarray, W: np.ndarray | None, max_dets: tuple[int, ...]) -> dict:
    """AR@maxDets итоговых детекций по бинам площади: среднее по порогам IoU и, при `use_cats`, по категориям."""
    point_w = np.zeros((1, m.n_img))
    point_w[0, sel] = 1
    out: dict = {lab: {"n_gt": int(m.n_gt[sel][:, :, a].sum())} for a, lab in enumerate(AREA_LABELS)}
    for md in max_dets:
        tables = [recall_table(m, point_w, md)] + ([] if W is None else [recall_table(m, W.astype(float), md)])
        ar = [_nanmean(_nanmean(t, 1), 2) for t in tables]                  # (B, бины)
        for a, lab in enumerate(AREA_LABELS):
            out[lab][f"ar_{md}"] = _val(ar[0][0, a])
            if W is not None:
                out[lab].setdefault("ci95", {})[f"ar_{md}"] = _ci(ar[1][:, a])
    return out


def summarize_top1(correct: list[np.ndarray], bins: list[np.ndarray], sel: np.ndarray, W: np.ndarray | None) -> dict:
    """Top-1 на оракульных масках: `correct` и `bins` — по снимку, по назначенной рамке (бин — по площади рамки)."""
    n_img = len(correct)
    hit, n = np.zeros((n_img, len(AREA_LABELS))), np.zeros((n_img, len(AREA_LABELS)))
    for i, (c, b) in enumerate(zip(correct, bins)):
        hit[i, 0], n[i, 0] = c.sum(), len(c)
        for a in range(3):
            hit[i, a + 1], n[i, a + 1] = c[b == a].sum(), (b == a).sum()
    out = {}
    for a, lab in enumerate(AREA_LABELS):
        den = n[sel, a].sum()
        out[lab] = {"n_oracle": int(den), "top1": None if den == 0 else float(hit[sel, a].sum() / den)}
        if W is not None and den > 0:
            d = W @ n[:, a]
            out[lab]["ci95"] = {"top1": _ci(np.where(d > 0, (W @ hit[:, a]) / np.maximum(d, 1), np.nan))}
    return out


def auroc(score: np.ndarray, known: np.ndarray, img: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUROC «известный / неизвестный» по оценке, с весами снимков: (B,). Равные оценки считаются за половину.

    `score`, `known`, `img` — по маске; `W` — (B, снимки).
    """
    if not len(score):
        return np.full(len(W), np.nan)
    order = np.argsort(score, kind="mergesort")
    s, pos, im = score[order], known[order], img[order]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])                    # начала групп равных оценок
    gid = np.cumsum(np.r_[True, s[1:] != s[:-1]]) - 1
    out = np.empty(len(W))
    step = max(1, _CHUNK // len(s))
    for b0 in range(0, len(W), step):
        w = W[b0:b0 + step][:, im].astype(float)
        wn, wp = w * ~pos, w * pos
        cn = np.cumsum(wn, 1)
        below = np.where(start[gid] > 0, cn[:, np.maximum(start[gid] - 1, 0)], 0.0)   # вес неизвестных строго ниже
        tie = np.add.reduceat(wn, start, 1)[:, gid]                                   # вес неизвестных с той же оценкой
        den = wp.sum(1) * wn.sum(1)
        out[b0:b0 + step] = np.where(den > 0, (wp * (below + 0.5 * tie)).sum(1) / np.maximum(den, 1e-300), np.nan)
    return out


def summarize_auroc(score: np.ndarray, known: np.ndarray, img: np.ndarray, abin: np.ndarray, n_img: int,
                    sel: np.ndarray, W: np.ndarray | None) -> dict:
    """AUROC по $s^*$: известные — оракульные маски (бин — по площади рамки разметки), неизвестные — дистракторы
    (бин — по площади собственной рамки $\\mathrm{box}(m)$)."""
    point_w = np.zeros((1, n_img))
    point_w[0, sel] = 1
    in_sel = point_w[0, img] > 0
    out = {}
    for a, lab in enumerate(AREA_LABELS):
        keep = np.ones(len(score), bool) if a == 0 else abin == a - 1
        sc, kn, im = score[keep], known[keep], img[keep]
        out[lab] = {"n_known": int((kn & in_sel[keep]).sum()), "n_unknown": int((~kn & in_sel[keep]).sum()),
                    "auroc": _val(auroc(sc, kn, im, point_w)[0])}
        if W is not None and out[lab]["auroc"] is not None:
            out[lab]["ci95"] = {"auroc": _ci(auroc(sc, kn, im, W))}
    return out
