"""Протоколы оценки 4.3: прогон сетки с оракульным отбором и контрольный прогон baseline.

Оба считаются по одним и тем же эмбеддингам масок $M(I)$ из рабочих файлов прогона и одному точному поиску:

- прогон сетки (`oracle`): $M^*$ — оракульные маски (взаимно-однозначное жадное назначение, IoU ≥ 0,5) и дистракторы
  (IoU рамки < 0,1 ко всем размеченным рамкам); маски, не попавшие ни туда ни туда, в оценке не участвуют — так 4.3
  измеряет только $\\varphi$. Метрики — AP / AP50 / AP75, top-1 на оракульных масках, AUROC по $s^*$;
- контрольный прогон (`baseline`): все маски $M(I)$, независимые решения, штатный NMS SAM 2 — AP и AR итоговых
  детекций в четырёх вариантах (без учёта и с учётом категорий × без порога и с порогом на $s^*$).

Порог $\\tau$ не применяется (калибровка описана в §2.4): каждая маска получает $\\hat y=\\arg\\max_y s_y$, оценка
детекции — $s^*$. Поиск — только точный перебор. Решения по маскам независимы.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eval import detect as DT
from src.eval import recall as R
from src.eval import rules as RU
from src.eval.oracle import distractors, oracle_assign

SPLITS = ("cal", "test", "all")
LEVELS = ("all", "easy", "hard")
GRID_SUBSETS = ("test/all", "test/easy", "test/hard")            # метрики 4.3 — по тестовым сценам
BASELINE_SUBSETS = ("all/all", "all/easy", "all/hard", "test/all", "test/easy", "test/hard")  # 160 сцен — набор [5]
SELECT_SUBSET = f"{RU.SELECT_SPLIT}/all"


def levels(dataset: str) -> tuple[str, ...]:
    """Вторая ось подмножеств: easy / hard у HR-InsDet, плата у PCB. Состав постоянен —
    от числа загруженных сцен не зависит (от порядка подмножеств зависят повторы бутстрэпа, `_weights`)."""
    if dataset == "hr_insdet":
        return LEVELS
    from src.data import pcb

    return ("all", *(f"board_{b}" for b in pcb.BOARDS["test"]))


def grid_subsets(dataset: str) -> tuple[str, ...]:
    """Подмножества метрик 4.3: тестовые сцены; у PCB оцениваются только тестовые платы, разбивка — по платам."""
    return GRID_SUBSETS if dataset == "hr_insdet" else tuple(f"test/{lv}" for lv in levels(dataset))


def baseline_subsets(dataset: str) -> tuple[str, ...]:
    """Контрольный прогон: у HR-InsDet — 160 сцен и 120 тестовых, у PCB — те же тестовые платы, что у прогона сетки."""
    return BASELINE_SUBSETS if dataset == "hr_insdet" else grid_subsets(dataset)


@dataclass
class Scene:
    id: str
    split: str
    level: str
    wh: tuple[int, int]
    gt: list[dict]                 # `{"label", "category_id", "box", "iscrowd"}`
    boxes: np.ndarray              # рамки масок $M(I)$, (n, 4), исключающие, пиксели исходного снимка
    oracle: np.ndarray             # для каждой размеченной рамки — номер маски либо −1 (пропуск сегментатора)
    distractor: np.ndarray         # (n,) bool
    ignore: np.ndarray | None = None               # зоны игнорирования (k, 4) — только тестовые платы PCB
    distractor_no_zones: np.ndarray | None = None  # дистракторы, как если бы зон не было, — для справочного AP без зон

    @property
    def zones(self) -> np.ndarray:
        return np.zeros((0, 4)) if self.ignore is None else self.ignore

    @property
    def gt_boxes(self) -> np.ndarray:
        return np.array([g["box"] for g in self.gt], float).reshape(-1, 4)

    @property
    def gt_labels(self) -> np.ndarray:
        return np.array([g["category_id"] for g in self.gt], np.int64)


def load_scenes(dataset: str, mask_cache, splits: tuple[str, ...] = ("cal", "test")
                ) -> tuple[list[Scene], list[str], list[float], dict]:
    """Сцены по ролям из `splits/<dataset>.json` (калибровочные, затем тестовые), истина, рамки масок из кеша.

    `splits=("cal",)` — только калибровочные сцены: кеш масок тестовых сцен не читается (диагностики после результата).
    """
    if dataset == "pcb":
        return _load_pcb(mask_cache, splits)
    if dataset != "hr_insdet":
        raise ValueError(dataset)
    from src.data import hr_insdet as H

    sp = json.loads(Path(f"splits/{dataset}.json").read_text())
    if set(sp["cal"]) & set(sp["test"]):
        raise ValueError("калибровочные и тестовые сцены пересекаются")
    labels = H.objects()
    if labels != sp["labels"]:
        raise ValueError("метки раздачи расходятся с разбиением")
    by_id = {s["id"]: s for s in H.scenes()}
    out = []
    if not set(splits) <= {"cal", "test"}:
        raise ValueError(f"роли сцен {splits} вне («cal», «test»)")
    for split in (r for r in ("cal", "test") if r in splits):
        for i in sp[split]:
            gt = H.scene_gt(by_id[i], labels)
            boxes = mask_cache.load(i).boxes
            gb = np.array([g["box"] for g in gt], float).reshape(-1, 4)
            out.append(Scene(i, split, by_id[i]["level"], H.SCENE_WH, gt, boxes, oracle_assign(gb, boxes),
                             distractors(gb, boxes)))
    return out, labels, H.AREA_EDGES, sp


def _load_pcb(mask_cache, splits: tuple[str, ...]) -> tuple[list[Scene], list[str], list[float], dict]:
    """Снимки тестовых плат PCB: истина — рамки разметки, зоны игнорирования — отдельно (в истину COCOeval их вносит
    `coco_truth`); маска, чья рамка пересекает зону, в дистракторы не входит. Калибровочные платы в авторежиме не
    сегментируются и не оцениваются."""
    from src.data import pcb as P

    if tuple(splits) not in (("test",), ("cal", "test")):  # по умолчанию — роли прогона сетки; у PCB это только тест
        raise ValueError(f"на PCB оцениваются только тестовые платы, запрошено {splits}")
    sp = json.loads(Path("splits/pcb.json").read_text())
    if sp["labels"] != P.CLASSES:
        raise ValueError("метки раздачи расходятся с разбиением")
    by_id = {i["id"]: i for i in P.images()}
    zones = P.ignore_zones()
    if set(zones) - set(sp["test"]):
        raise ValueError("зоны игнорирования вне тестовых снимков")
    out = []
    for i in sp["test"]:
        img = by_id[i]
        if img["board"] not in P.BOARDS["test"]:
            raise ValueError(f"{i}: плата {img['board']} не тестовая")
        gt = P.image_gt(img)
        boxes = mask_cache.load(i).boxes
        gb = np.array([g["box"] for g in gt], float).reshape(-1, 4)
        z = np.array(zones.get(i, []), float).reshape(-1, 4)
        out.append(Scene(i, "test", f"board_{img['board']}", tuple(P.image_wh(img)), gt, boxes, oracle_assign(gb, boxes),
                         distractors(gb, boxes, ignore_boxes=z), ignore=z, distractor_no_zones=distractors(gb, boxes)))
    return out, list(P.CLASSES), P.AREA_EDGES, sp


def coco_truth(scenes: list[Scene], labels: list[str], zones: bool = True):
    """Истина COCOeval. Зона игнорирования входит с `iscrowd=1` в каждой категории: детекция в зоне ни верная, ни
    ложная, каким бы типом дефекта она ни была названа. `zones=False` — истина без
    зон, для справочного AP без зон."""
    rows = []
    for s in scenes:
        crowd = [{"category_id": c, "box": [float(v) for v in z], "iscrowd": 1}
                 for z in (s.zones if zones else []) for c in range(len(labels))]
        rows.append({"id": s.id, "wh": s.wh, "gt": list(s.gt) + crowd})
    return R.coco_gt(rows, labels)


def subsets(scenes: list[Scene], levels_: tuple[str, ...] = LEVELS) -> dict[str, np.ndarray]:
    split = np.array([s.split for s in scenes])
    level = np.array([s.level for s in scenes])
    if set(level) - set(levels_):
        raise ValueError(f"уровни сцен {sorted(set(level) - set(levels_))} вне {levels_}")
    every = np.ones(len(scenes), bool)
    return {f"{sp}/{lv}": np.flatnonzero((every if sp == "all" else split == sp) & (every if lv == "all" else level == lv))
            for sp in SPLITS for lv in levels_}


def counts(scenes: list[Scene], sel: np.ndarray, edges: list[float]) -> dict:
    """Состав оценки по сценам `sel`: рамки, оракульные маски, пропуски сегментатора (по бинам площади), дистракторы."""
    ss = [scenes[k] for k in sel]
    gb = np.concatenate([s.gt_boxes for s in ss]) if ss else np.zeros((0, 4))
    miss = np.concatenate([s.oracle < 0 for s in ss]) if ss else np.zeros(0, bool)
    abin = R.area_bin((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]), edges)
    n_masks, n_distr = sum(len(s.boxes) for s in ss), int(sum(s.distractor.sum() for s in ss))
    zones = {} if all(s.ignore is None for s in ss) else {
        "n_ignore_zones": int(sum(len(s.zones) for s in ss)),
        "n_masks_excluded_from_distractors_by_zones": int(sum((s.distractor_no_zones & ~s.distractor).sum() for s in ss))}
    return {"n_scenes": len(ss), "n_masks": n_masks, "n_gt": len(gb), "n_oracle": int((~miss).sum()), **zones,
            "n_segmenter_miss": int(miss.sum()),
            "n_segmenter_miss_by_area": {lab: int(miss[abin == a].sum()) for a, lab in enumerate(R.AREA_LABELS[1:])},
            "n_distractors": n_distr, "n_masks_not_evaluated_in_oracle_protocol": n_masks - int((~miss).sum()) - n_distr}


def search(scene_emb: list[np.ndarray], gallery, protocol: str) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Точный поиск и правило решения без порога: по сцене — $\\hat y$ (номер метки галереи) и $s^*$."""
    from src.search import decide as DE
    from src.search import index as IX

    rows = gallery.rows(protocol)
    index = IX.ExactIndex(gallery.emb, rows)
    label_ids, deleted = gallery.label_ids, gallery.deleted
    y_hat, s_star = [], []
    for z in scene_emb:
        if len(z):
            y, s = DE.decide(*index.search(z), label_ids, deleted=deleted)  # k = N: $s_y$ по всей галерее (§2.5)
        else:
            y, s = np.zeros(0, np.int64), np.zeros(0, np.float32)
        if (y == DE.UNKNOWN).any():
            raise AssertionError("без порога каждая маска обязана получить метку")
        y_hat.append(y), s_star.append(s)
    return y_hat, s_star, {"index": "IndexFlatIP", "k": "N", "N": int(len(rows)), "n_labels": int(len(set(label_ids[rows]))),
                           "n_max": gallery.n_max(rows), "tau": None}


def _dets(scenes: list[Scene], y_hat, s_star, mode: str, zones: bool = True, keep_masks=None) -> list[dict]:
    """`keep_masks` — по сцене номера масок, дающих детекции (4.4: $M^*$ либо отбор после поиска); только `baseline`."""
    if keep_masks is not None and (mode != "baseline" or len(keep_masks) != len(scenes)):
        raise ValueError("подмножество масок задаётся по каждой сцене и только в протоколе baseline")
    out = []
    for k, (s, y, sc) in enumerate(zip(scenes, y_hat, s_star)):
        if mode == "oracle":
            keep = (s.distractor if zones or s.distractor_no_zones is None else s.distractor_no_zones).copy()
            keep[s.oracle[s.oracle >= 0]] = True
        elif mode == "baseline":
            keep = np.ones(len(s.boxes), bool)
            if keep_masks is not None:
                keep[:] = False
                keep[np.asarray(keep_masks[k], np.int64)] = True
        else:
            raise ValueError(mode)
        out.append({"box": s.boxes[keep], "label_id": y[keep], "score": sc[keep]})
    return out


def _part_mask_cats(labels: list[str]) -> dict[str, list[int]]:
    """Метки с маской-частью у эталона (HR-InsDet, `RU.PART_MASK_LABELS`); у датасета без таких меток — пусто."""
    if not set(RU.PART_MASK_LABELS) & set(labels):
        return {}
    missing = set(RU.PART_MASK_LABELS) - set(labels)
    if missing:
        raise ValueError(f"меток {sorted(missing)} нет среди категорий")
    return {"without_part_mask_labels": [i for i, n in enumerate(labels) if n not in RU.PART_MASK_LABELS]}


def _weights(sub: dict[str, np.ndarray], names, n_img: int, n_boot: int, seed: int) -> dict[str, np.ndarray | None]:
    """Повторы бутстрэпа по сценам; зависят только от seed и подмножества — одни и те же у всех прогонов сетки,
    протоколов галереи и метрик, так что разности между прогонами можно считать парно. `n_boot=0` (PCB) — без повторов."""
    if n_boot == 0:
        return {n: None for n in names}
    order = sorted(sub)
    return {n: DT.boot_weights(sub[n], n_img, n_boot, np.random.default_rng([seed, order.index(n)])) for n in names}


def _by_type(m, scenes: list[Scene], labels: list[str], sel: np.ndarray, max_det: int, y_hat=None) -> dict:
    """AP категории по снимкам `sel` — для PCB, где категория есть тип дефекта: ложные детекции категории считаются
    на всех снимках подмножества, а не только на снимках своего типа. С `y_hat` — ещё
    и top-1 на оракульных масках типа."""
    rows = DT.ap_by_category(m, sel, max_det)
    for c, name in enumerate(labels):
        gl = np.concatenate([scenes[k].gt_labels for k in sel]) if len(sel) else np.zeros(0, np.int64)
        ok = np.concatenate([scenes[k].oracle >= 0 for k in sel]) if len(sel) else np.zeros(0, bool)
        rows[c]["n_oracle"] = int((ok & (gl == c)).sum())
        if y_hat is not None:
            hit = np.concatenate([y_hat[k][scenes[k].oracle[scenes[k].oracle >= 0]] == scenes[k].gt_labels[scenes[k].oracle >= 0]
                                  for k in sel]) if len(sel) else np.zeros(0, bool)
            of_type = gl[ok] == c
            rows[c]["top1"] = float(hit[of_type].mean()) if of_type.any() else None
    return {name: rows[c] for c, name in enumerate(labels)}


def evaluate_oracle(scenes: list[Scene], labels: list[str], edges: list[float], gt, gallery_label_names: list[str],
                    y_hat, s_star, max_det: int, n_boot: int, seed: int, names: tuple[str, ...] | None = None,
                    levels_: tuple[str, ...] = LEVELS, zones: bool = True, by_type: bool = False) -> dict:
    """Метрики прогона сетки при одном протоколе галереи: по тестовым сценам (`GRID_SUBSETS`) и `selection_cal`.

    `names` — иной состав подмножеств (диагностика по одним калибровочным сценам — `(SELECT_SUBSET,)`); повторы
    бутстрэпа подмножества от состава `names` и от числа загруженных сцен не зависят.
    """
    if gallery_label_names != labels:
        raise ValueError("номера меток галереи расходятся с категориями истины")
    sub = subsets(scenes, levels_)
    names = (*GRID_SUBSETS, SELECT_SUBSET) if names is None else names
    W = _weights(sub, names, len(scenes), n_boot, seed)
    m = DT.evaluate(gt, _dets(scenes, y_hat, s_star, "oracle", zones), edges, use_cats=True)

    # top-1 на оракульных масках и AUROC «известный / неизвестный» по s*
    correct, gbins, score, known, img, abin = [], [], [], [], [], []
    for k, (s, y, sc) in enumerate(zip(scenes, y_hat, s_star)):
        ok = s.oracle >= 0
        gb = s.gt_boxes
        garea = R.area_bin((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]), edges)
        correct.append(y[s.oracle[ok]] == s.gt_labels[ok]), gbins.append(garea[ok])
        d = np.flatnonzero(s.distractor if zones or s.distractor_no_zones is None else s.distractor_no_zones)
        if np.intersect1d(d, s.oracle[ok]).size:
            raise AssertionError(f"{s.id}: маска одновременно оракульная и дистрактор")
        darea = R.area_bin((s.boxes[d, 2] - s.boxes[d, 0]) * (s.boxes[d, 3] - s.boxes[d, 1]), edges)
        score += [sc[s.oracle[ok]], sc[d]]
        known += [np.ones(ok.sum(), bool), np.zeros(len(d), bool)]
        abin += [garea[ok], darea]
        img.append(np.full(ok.sum() + len(d), k))
    score, known, img, abin = map(np.concatenate, (score, known, img, abin))

    out: dict = {"subsets": {}}
    for n in names:
        ap = DT.summarize_ap(m, sub[n], W[n], max_det, cat_subsets=_part_mask_cats(labels),
                             keep_boot=n == RU.BOOT_AP_SUBSET)
        boot_ap = ap["all"].pop("boot_ap", None)
        top1 = DT.summarize_top1(correct, gbins, sub[n], W[n])
        au = DT.summarize_auroc(score.astype(float), known, img, abin, len(scenes), sub[n], W[n])
        part = ap.pop("without_part_mask_labels", None)
        by_area = {}
        for lab in R.AREA_LABELS:
            ci = {**ap[lab].pop("ci95", {}), **top1[lab].pop("ci95", {}), **au[lab].pop("ci95", {})}
            by_area[lab] = {**ap[lab], **top1[lab], **au[lab], **({"ci95": ci} if ci else {})}
        out["subsets"][n] = {"n_scenes": int(len(sub[n])), "by_area": by_area}
        if part is not None:
            out["subsets"][n]["ap_without_part_mask_labels"] = part["all"]
        if by_type:
            out["subsets"][n]["by_type"] = _by_type(m, scenes, labels, sub[n], max_det, y_hat)
        if boot_ap is not None:
            out["boot_ap"] = {"subset": n, "values": boot_ap}
    return out


def evaluate_baseline(scenes: list[Scene], labels: list[str], edges: list[float], gt, gallery_label_names: list[str],
                      y_hat, s_star, max_det: int, n_boot: int, seed: int, names: tuple[str, ...] = BASELINE_SUBSETS,
                      levels_: tuple[str, ...] = LEVELS, by_type: bool = False, with_ar: bool = True,
                      keep_masks=None, boot_subsets: tuple[str, ...] = (), dets: list[dict] | None = None,
                      ar_thresholds: tuple = (None, RU.AR_SCORE_THRESHOLD)) -> dict:
    """Контрольный прогон при одном протоколе галереи: AP и AR итоговых детекций в четырёх вариантах.

    4.4: `keep_masks` — по сцене номера масок, дающих детекции ($M^*$ либо отбор после поиска): те же
    независимые решения и та же оценка по подмножеству масок; `boot_subsets` — подмножества сцен, у которых в ответ
    идут повторы бутстрэпа AP, AP50, AP75 (`boot`, массивы; в запись журнала не пишутся) — для парных разностей.

    Метод сравнения OWLv2: `dets` — готовые детекции по сценам (формат `detect.evaluate`) вместо
    детекций по маскам, `y_hat` и `s_star` тогда не передаются; `ar_thresholds=(None,)` — AR только без порога
    (порог 0,4 — шкала косинуса метода, к оценкам OWLv2 не относится)."""
    if gallery_label_names != labels:
        raise ValueError("номера меток галереи расходятся с категориями истины")
    sub = subsets(scenes, levels_)
    W = _weights(sub, names, len(scenes), n_boot, seed)
    if set(boot_subsets) - set(names):
        raise ValueError(f"повторы запрошены для подмножеств вне {names}")
    if dets is None:
        dets = _dets(scenes, y_hat, s_star, "baseline", keep_masks=keep_masks)
    elif y_hat is not None or s_star is not None or keep_masks is not None or len(dets) != len(scenes):
        raise ValueError("готовые детекции — по одной записи на сцену и без решений по маскам")
    if set(ar_thresholds) - {None, RU.AR_SCORE_THRESHOLD}:
        raise ValueError(f"пороги AR {ar_thresholds} вне (None, {RU.AR_SCORE_THRESHOLD})")
    m = DT.evaluate(gt, dets, edges, use_cats=True)
    ar_variants = {}
    for use_cats in ((False, True) if with_ar else ()):  # `with_ar=False` — только AP (справочный AP без зон на PCB)
        for thr in ar_thresholds:
            name = f"{'with' if use_cats else 'no'}_cats__{'no_threshold' if thr is None else f'threshold_{thr:g}'}"
            mm = m if (use_cats and thr is None) else DT.evaluate(gt, dets, edges, use_cats=use_cats, score_min=thr)
            ar_variants[name] = (mm, {"use_cats": use_cats, "score_threshold": thr})
    out: dict = {"subsets": {}, "n_detections": int(sum(len(d["score"]) for d in dets))}
    if RU.AR_SCORE_THRESHOLD in ar_thresholds:
        out["n_detections_score_ge_threshold"] = int(sum((d["score"] >= RU.AR_SCORE_THRESHOLD).sum() for d in dets))
    for n in names:
        ap = DT.summarize_ap(m, sub[n], W[n], max_det, cat_subsets=_part_mask_cats(labels),
                             keep_boot_metrics=n in boot_subsets)
        if n in boot_subsets:
            out.setdefault("boot", {})[n] = ap["all"].pop("boot")
        part = ap.pop("without_part_mask_labels", None)
        ar = {name: {**info, "by_area": DT.summarize_ar(mm, sub[n], W[n], RU.AR_MAX_DETS)}
              for name, (mm, info) in ar_variants.items()}
        out["subsets"][n] = {"n_scenes": int(len(sub[n])), "by_area": ap, "ar_final_detections": ar}
        if part is not None:
            out["subsets"][n]["ap_without_part_mask_labels"] = part["all"]
        if by_type:
            out["subsets"][n]["by_type"] = _by_type(m, scenes, labels, sub[n], max_det)
    return out
