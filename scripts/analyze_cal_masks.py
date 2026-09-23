"""Аналитика §2.3 по 40 калибровочным сценам HR-InsDet — перечень величин записан до счёта.

    python scripts/analyze_cal_masks.py            # только split cal; иной split — отказ

Описание данных, по которому пишется §2.3: вложенность масок $M(I)$ при нескольких $\\theta$ без выбора одного,
устойчивость эмбеддинга между уровнями вложенности на лучших $\\varphi$ обоих энкодеров, оценки сегментатора, площади
и покрытие, близость эталонов внутри экземпляра. Критериев и порогов не содержит. Тестовые сцены и их кеш масок не
читаются; GPU не нужен: маски — кеш `crop_n_layers=1`, эмбеддинги — рабочие файлы прогонов сетки лучших $\\varphi$,
галереи этих прогонов, точный перебор. Вывод — `experiments/analysis/<блок>.json` и сводка `summary.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as MU
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as CFG
from src import env
from src.encode import run_emb as RE
from src.encode.variants import grid
from src.eval import protocol as PR
from src.eval import rules as RU
from src.gallery import store
from src.segment import cache as MC

SPLIT = "cal"                                   # единственный допустимый split
DATASET = "hr_insdet"
N_SCENES = 40
THETAS = (0.80, 0.85, 0.90, 0.95, 0.99)         # сетка для описания; одно значение не выбирается
ENCODERS = ("dinov2", "dinov3")
AREA_RATIO_EDGES = (0.0, 0.25, 0.5, 0.75, 1.0 + 1e-9)
COVER_LEVELS = tuple(round(0.50 + 0.05 * k, 2) for k in range(10))   # 0,50 … 0,95 — описание, уровень не выбирается
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
OUT = Path("experiments/analysis")
RUNS = Path("experiments/runs")
_STAMP = env.code_stamp()


def check_split(split: str) -> str:
    """Аналитика §2.3 разрабатывается только на калибровочных сценах: любой иной split — отказ."""
    if split != SPLIT:
        raise SystemExit(f"аналитика §2.3 — только split «{SPLIT}»; запрошен «{split}»")
    return split


# ---------------------------------------------------------------------- сводки распределений


def dist(x, edges=None) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    out = {"n": int(len(x))}
    if len(x):
        out.update(mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
                   q={str(q): float(v) for q, v in zip(QS, np.quantile(x, QS))})
        if edges is not None:
            out["hist"] = {"edges": [float(e) for e in edges], "counts": np.histogram(x, edges)[0].tolist()}
    return out


def by_group(x, groups, edges=None) -> dict:
    x, groups = np.asarray(x, float), np.asarray(groups)
    return {str(gv): dist(x[groups == gv], edges) for gv in sorted(set(groups.tolist()), key=str)}


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return None if ok.sum() < 3 else float(spearmanr(a[ok], b[ok]).statistic)


def area_bin_name(box_area: np.ndarray, edges) -> np.ndarray:
    return np.array(["<200^2", "200^2-400^2", ">400^2"])[np.digitize(box_area, edges[:2])]


# ---------------------------------------------------------------------- маски сцены и отношение вложенности


class SceneMasks:
    """Маски одной калибровочной сцены: площади и пересечения — точно по RLE в разрешении входа $S$."""

    def __init__(self, scene: PR.Scene, entry: MC.MaskEntry):
        self.id, self.room = scene.id, scene.id.split("/")[1]
        self.entry = entry
        rles = [r["rle"] for r in entry.records]
        self.rles = rles
        self.n = len(rles)
        self.area_seg = MU.area(rles).astype(np.int64) if rles else np.zeros(0, np.int64)
        self.area = np.array([r["area"] for r in entry.records], float)             # пиксели снимка
        self.box = entry.boxes
        self.box_area = (self.box[:, 2] - self.box[:, 0]) * (self.box[:, 3] - self.box[:, 1])
        self.piou = np.array([r["predicted_iou"] for r in entry.records], float)
        self.stab = np.array([r["stability_score"] for r in entry.records], float)
        h, w = entry.seg_hw
        self.layer = np.array(["frame" if list(r["crop_box"]) == [0, 0, w, h] else "window" for r in entry.records])
        role = np.full(self.n, "near", dtype=object)
        role[scene.distractor] = "distractor"
        obj = scene.oracle[scene.oracle >= 0]
        role[obj] = "object"
        self.role = role.astype(str)
        self.gt_of_object = {int(m): g for g, m in enumerate(scene.oracle) if m >= 0}   # маска «объект» → номер рамки
        self.gt_labels = scene.gt_labels
        self.gt_boxes = scene.gt_boxes
        # пересечения пар с пересекающимися рамками (в разрешении S)
        bb = MU.toBbox(rles) if rles else np.zeros((0, 4))
        x0, y0, x1, y1 = bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]
        self.inter: dict[tuple[int, int], int] = {}
        for i in range(self.n):
            j = np.flatnonzero((x0[i] < x1) & (x0 < x1[i]) & (y0[i] < y1) & (y0 < y1[i]))
            for k in j[j > i]:
                v = int(MU.area(MU.merge([rles[i], rles[k]], intersect=True)))
                if v > 0:
                    self.inter[i, int(k)] = v

    def near_gt(self) -> dict[int, int]:
        """Маска «около объекта» → размеченная рамка с наибольшим IoU рамок (не ниже 0,1)."""
        from src.eval.boxes import box_iou

        near = np.flatnonzero(self.role == "near")
        if not len(near) or not len(self.gt_boxes):
            return {}
        iou = box_iou(self.box[near], self.gt_boxes)
        return {int(m): int(g) for m, g in zip(near, iou.argmax(1))}


class Nesting:
    """Отношение $m_a\\preceq_\\theta m_b$: $|m_a\\cap m_b|\\ge\\theta|m_a|$ и $|m_a|<|m_b|$; при равных площадях —
    «дубликаты». Лес «наименьшего содержащего», антицепи, транзитивность."""

    def __init__(self, sm: SceneMasks, theta: float):
        self.sm, self.theta, n = sm, theta, sm.n
        a = sm.area_seg
        self.contains = [set() for _ in range(n)]     # contains[b] — маски, вложенные в b
        self.inside = [set() for _ in range(n)]       # inside[a] — маски, содержащие a
        self.n_nested = self.n_crossing = self.n_duplicate = 0
        for (i, k), v in sm.inter.items():
            small, big = (i, k) if (a[i], i) < (a[k], k) else (k, i)
            if v >= theta * a[small]:
                if a[small] == a[big]:
                    self.n_duplicate += 1
                    continue
                self.inside[small].add(big)
                self.contains[big].add(small)
                self.n_nested += 1
            else:
                self.n_crossing += 1
        self.parent = np.full(n, -1)
        for i in range(n):
            if self.inside[i]:
                self.parent[i] = min(self.inside[i], key=lambda b: (a[b], b))
        self.children = [[] for _ in range(n)]
        for i, p in enumerate(self.parent):
            if p >= 0:
                self.children[p].append(i)
        self.depth = np.zeros(n, int)
        for i in np.argsort(-a, kind="stable"):       # родитель всегда больше ребёнка по площади
            if self.parent[i] >= 0:
                self.depth[i] = self.depth[self.parent[i]] + 1

    def position(self) -> np.ndarray:
        root, leaf = self.parent < 0, np.array([not c for c in self.children])
        return np.where(root & leaf, "isolated", np.where(root, "root", np.where(leaf, "leaf", "internal")))

    def transitivity(self) -> tuple[int, int]:
        triples = viol = 0
        for a_ in range(self.sm.n):
            for b in self.inside[a_]:
                for c in self.inside[b]:
                    triples += 1
                    viol += c not in self.inside[a_]
        return triples, viol

    def width(self) -> int:
        """Наибольшая антицепь транзитивного замыкания: $n$ минус наибольшее паросочетание (Дилуорс)."""
        n = self.sm.n
        if n == 0:
            return 0
        reach = np.zeros((n, n), bool)
        for i in np.argsort(-self.sm.area_seg, kind="stable"):    # сначала большие: их замыкание уже готово
            for b in self.inside[i]:
                reach[i, b] = True
                reach[i] |= reach[b]
        m = maximum_bipartite_matching(csr_matrix(reach), perm_type="column")
        return int(n - (m >= 0).sum())

    def descendants(self, i: int) -> dict[int, int]:
        """Потомки в лесу: маска → число поколений вниз."""
        out, stack = {}, [(c, 1) for c in self.children[i]]
        while stack:
            c, d = stack.pop()
            out[c] = d
            stack += [(g, d + 1) for g in self.children[c]]
        return out

    def ancestors(self, i: int) -> dict[int, int]:
        out, p, d = {}, self.parent[i], 1
        while p >= 0:
            out[int(p)] = d
            p, d = self.parent[p], d + 1
        return out


# ---------------------------------------------------------------------- блоки


def nesting_block(scenes: list[SceneMasks], nest: dict) -> dict:
    ratio, iou = [], []
    for sm in scenes:
        for (i, k), v in sm.inter.items():
            ratio.append(v / min(sm.area_seg[i], sm.area_seg[k]))
            iou.append(v / (sm.area_seg[i] + sm.area_seg[k] - v))
    out = {"n_scenes": len(scenes), "n_masks": int(sum(s.n for s in scenes)),
           "n_pairs_overlapping": len(ratio),
           "overlap_fraction_of_smaller": dist(ratio, np.linspace(0, 1, 21)),
           "overlap_iou": dist(iou, np.linspace(0, 1, 21)), "by_theta": {}}
    for th in THETAS:
        rows = {"nested": 0, "crossing": 0, "duplicate": 0, "triples": 0, "transitivity_violations": 0}
        depth_all, pos_all, nkids, ratio_e, cross_layer, per_scene = [], [], [], [], [], []
        obj_rows, near_rel = [], {"inside_object": 0, "contains_object": 0, "duplicate_of_object": 0, "crossing_object": 0,
                                  "disjoint_from_object": 0, "no_object_mask_for_its_box": 0}
        for sm in scenes:
            ne = nest[sm.id][th]
            t, v = ne.transitivity()
            rows["nested"] += ne.n_nested; rows["crossing"] += ne.n_crossing; rows["duplicate"] += ne.n_duplicate
            rows["triples"] += t; rows["transitivity_violations"] += v
            pos = ne.position()
            depth_all += ne.depth.tolist(); pos_all += pos.tolist()
            nkids += [len(c) for c in ne.children if c]
            for i, p in enumerate(ne.parent):
                if p >= 0:
                    ratio_e.append(sm.area_seg[i] / sm.area_seg[p]); cross_layer.append(sm.layer[i] != sm.layer[p])
            n_max = sum(not x for x in ne.inside); n_min = sum(not x for x in ne.contains)
            per_scene.append({"scene": sm.id, "room": sm.room, "n_masks": sm.n, "forest_depth": int(ne.depth.max(initial=0)),
                              "n_roots": int((ne.parent < 0).sum()), "n_leaves": int(sum(not c for c in ne.children)),
                              "n_isolated": int((pos == "isolated").sum()), "n_maximal": int(n_max), "n_minimal": int(n_min),
                              "width": ne.width()})
            for m, g in sm.gt_of_object.items():
                obj_rows.append({"depth": int(ne.depth[m]), "position": str(pos[m]), "n_ancestors": len(ne.ancestors(m)),
                                 "n_descendants": len(ne.descendants(m)), "n_containers": len(ne.inside[m]),
                                 "n_contained": len(ne.contains[m])})
            obj_of_gt = {g: m for m, g in sm.gt_of_object.items()}
            for m, g in sm.near_gt().items():
                o = obj_of_gt.get(g)
                if o is None:
                    near_rel["no_object_mask_for_its_box"] += 1
                elif o in ne.inside[m]:
                    near_rel["inside_object"] += 1
                elif m in ne.inside[o]:
                    near_rel["contains_object"] += 1
                else:
                    key = (min(m, o), max(m, o))
                    if key not in sm.inter:
                        near_rel["disjoint_from_object"] += 1
                    elif sm.area_seg[m] == sm.area_seg[o] and sm.inter[key] >= th * sm.area_seg[m]:
                        near_rel["duplicate_of_object"] += 1
                    else:
                        near_rel["crossing_object"] += 1
        depth_all, pos_all = np.array(depth_all), np.array(pos_all)
        ps = {k: [r[k] for r in per_scene] for k in ("n_masks", "forest_depth", "n_roots", "n_leaves", "n_isolated",
                                                     "n_maximal", "n_minimal", "width")}
        rooms = sorted({r["room"] for r in per_scene})
        out["by_theta"][str(th)] = {
            "pairs": rows, "transitivity_violation_share": rows["transitivity_violations"] / max(rows["triples"], 1),
            "masks_by_depth": {str(d): int((depth_all == d).sum()) for d in range(int(depth_all.max()) + 1)},
            "masks_by_position": {p: int((pos_all == p).sum()) for p in ("root", "internal", "leaf", "isolated")},
            "children_per_internal_node": dist(nkids, np.arange(0.5, 21.5, 1)),
            "child_parent_area_ratio": dist(ratio_e, np.linspace(0, 1, 21)),
            "edges_across_generator_layers_share": float(np.mean(cross_layer)) if cross_layer else None,
            "per_scene_summary": {k: dist(v) for k, v in ps.items()},
            "width_over_n_masks": dist(np.array(ps["width"]) / np.maximum(ps["n_masks"], 1)),
            "by_room": {rm: {k: dist([r[k] for r in per_scene if r["room"] == rm]) for k in ("n_masks", "forest_depth", "width")}
                        for rm in rooms},
            "object_masks": {"n": len(obj_rows),
                             "by_position": {p: sum(r["position"] == p for r in obj_rows) for p in ("root", "internal", "leaf", "isolated")},
                             "depth": dist([r["depth"] for r in obj_rows], np.arange(-0.5, 8.5, 1)),
                             "n_ancestors": dist([r["n_ancestors"] for r in obj_rows], np.arange(-0.5, 8.5, 1)),
                             "n_descendants": dist([r["n_descendants"] for r in obj_rows], np.arange(-0.5, 30.5, 1))},
            "near_object_masks_relative_to_object_mask_of_their_box": near_rel,
            "per_scene": per_scene}
    return out


def _cos(z, i, k) -> float:
    return float(z[i] @ z[k])


def stability_block(scenes, nest, emb, dec) -> dict:
    """`emb[e][scene]` — нормированные эмбеддинги масок; `dec[e][scene]` — (ŷ, s*) по полной галерее."""
    out = {}
    for e in ENCODERS:
        blk = {"reference": {}, "by_theta": {}}
        cross, allp = [], []
        for sm in scenes:
            z = emb[e][sm.id]
            ne = nest[sm.id][THETAS[2]]
            for (i, k) in sm.inter:
                if k not in ne.inside[i] and i not in ne.inside[k]:
                    cross.append(_cos(z, i, k))
            G = z @ z.T
            allp += G[np.triu_indices(sm.n, 1)].tolist()
        edges_c = np.linspace(-0.2, 1.0, 25)
        blk["reference"] = {"crossing_pairs_theta_0.9": dist(cross, edges_c), "all_pairs_in_scene": dist(allp, edges_c)}
        for th in THETAS:
            ce, rat, rc, rp, dep, same_y, ds, ca = [], [], [], [], [], [], [], []
            obj = {d: {"cos": [], "s_star": [], "correct": []} for d in range(-3, 4)}
            argmax_place = {"self": 0, "ancestor": 0, "descendant": 0}
            for sm in scenes:
                z, (y, s) = emb[e][sm.id], dec[e][sm.id]
                ne = nest[sm.id][th]
                for i, p in enumerate(ne.parent):
                    if p < 0:
                        continue
                    ce.append(_cos(z, i, p)); rat.append(sm.area_seg[i] / sm.area_seg[p])
                    rc.append(sm.role[i]); rp.append(sm.role[p]); dep.append(min(int(ne.depth[i]), 4))
                    same_y.append(y[i] == y[p]); ds.append(float(s[p] - s[i]))
                for i in range(sm.n):
                    ca += [_cos(z, i, b) for b in ne.inside[i]]
                for m, g in sm.gt_of_object.items():
                    lab = sm.gt_labels[g]
                    chain = {m: 0, **{a: -d for a, d in ne.descendants(m).items()}, **ne.ancestors(m)}
                    for k, d in chain.items():
                        dd = max(-3, min(3, d))
                        obj[dd]["cos"].append(_cos(z, m, k)); obj[dd]["s_star"].append(float(s[k]))
                        obj[dd]["correct"].append(bool(y[k] == lab))
                    best = max(chain, key=lambda k: (s[k], -k))
                    argmax_place["self" if best == m else "ancestor" if chain[best] > 0 else "descendant"] += 1
            ce, rat, same_y, ds = map(np.asarray, (ce, rat, same_y, ds))
            pair = np.array([f"{a}->{b}" for a, b in zip(rc, rp)])
            rbin = np.digitize(rat, AREA_RATIO_EDGES[1:-1])
            edges_c = np.linspace(-0.2, 1.0, 25)
            blk["by_theta"][str(th)] = {
                "forest_edges_cos": dist(ce, edges_c),
                "all_nested_pairs_cos": dist(ca, edges_c),
                "forest_edges_cos_by_area_ratio": {f"{AREA_RATIO_EDGES[b]:g}-{min(AREA_RATIO_EDGES[b + 1], 1):g}": dist(ce[rbin == b])
                                                   for b in range(len(AREA_RATIO_EDGES) - 1)},
                "forest_edges_cos_by_roles_child_to_parent": by_group(ce, pair),
                "forest_edges_cos_by_child_depth": by_group(ce, np.array([str(d) if d < 4 else "4+" for d in dep])),
                "forest_edges_same_y_hat_share": float(same_y.mean()) if len(same_y) else None,
                "forest_edges_same_y_hat_share_by_roles": {k: float(same_y[pair == k].mean()) for k in sorted(set(pair.tolist()))},
                "forest_edges_s_star_parent_minus_child": dist(ds, np.linspace(-0.6, 0.6, 25)),
                "object_mask_chain": {
                    "levels": "0 — маска «объект»; +d — предок через d поколений, −d — потомок (±3 — 3 и дальше)",
                    "by_level": {str(d): {"n": len(v["cos"]), "cos_with_object_mask": dist(v["cos"]),
                                          "s_star": dist(v["s_star"]),
                                          "y_hat_equals_box_label_share": float(np.mean(v["correct"])) if v["correct"] else None}
                                 for d, v in obj.items()},
                    "place_of_max_s_star_in_chain": argmax_place}}
        out[e] = blk
    return out


def scores_block(scenes, nest, refs) -> dict:
    cat = lambda f: np.concatenate([f(s) for s in scenes])
    piou, stab = cat(lambda s: s.piou), cat(lambda s: s.stab)
    role, layer = cat(lambda s: s.role), cat(lambda s: s.layer)
    abin = cat(lambda s: area_bin_name(s.box_area, [200.0 ** 2, 400.0 ** 2]))
    area = cat(lambda s: s.area)
    ep, es = np.linspace(0.80, 1.0, 21), np.linspace(0.95, 1.0, 21)
    out = {"n_masks": int(len(piou)),
           "predicted_iou": {"all": dist(piou, ep), "by_role": by_group(piou, role, ep), "by_layer": by_group(piou, layer, ep),
                             "by_area_bin": by_group(piou, abin, ep)},
           "stability_score": {"all": dist(stab, es), "by_role": by_group(stab, role, es), "by_layer": by_group(stab, layer, es),
                               "by_area_bin": by_group(stab, abin, es)},
           "spearman": {"predicted_iou_vs_stability": spearman(piou, stab),
                        "predicted_iou_vs_log_area": spearman(piou, np.log10(area)),
                        "stability_vs_log_area": spearman(stab, np.log10(area))},
           "by_forest_position": {}}
    for th in THETAS:
        pos = np.concatenate([nest[s.id][th].position() for s in scenes])
        out["by_forest_position"][str(th)] = {"predicted_iou": by_group(piou, pos, ep), "stability_score": by_group(stab, pos, es)}
    rp, rs, ci = (np.array([r[k] for r in refs], float) for k in ("predicted_iou", "stability_score", "control_iou"))
    worst = sorted(refs, key=lambda r: r["control_iou"])[:10]
    out["references"] = {"n": len(refs), "predicted_iou": dist(rp, ep), "stability_score": dist(rs, es),
                         "control_iou_grabcut": dist(ci, np.linspace(0, 1, 21)),
                         "spearman": {"predicted_iou_vs_control_iou": spearman(rp, ci), "stability_vs_control_iou": spearman(rs, ci)},
                         "lowest_control_iou": [{k: r[k] for k in ("id", "control_iou", "predicted_iou", "stability_score")} for r in worst]}
    return out


def areas_block(scenes, nest) -> dict:
    cat = lambda f: np.concatenate([f(s) for s in scenes])
    area, box_area = cat(lambda s: s.area), cat(lambda s: s.box_area)
    role, layer = cat(lambda s: s.role), cat(lambda s: s.layer)
    abin = cat(lambda s: area_bin_name(s.box_area, [200.0 ** 2, 400.0 ** 2]))
    frame = float(np.prod(scenes[0].entry.hw))
    la, fill = np.log10(area), area / np.maximum(box_area, 1)
    el, ef = np.arange(1.0, 8.01, 0.25), np.linspace(0, 1, 21)
    out = {"frame_px": frame,
           "log10_area_px": {"all": dist(la, el), "by_role": by_group(la, role, el), "by_layer": by_group(la, layer, el)},
           "area_fraction_of_frame": {"all": dist(area / frame), "by_role": by_group(area / frame, role)},
           "masks_by_area_bin": {b: int((abin == b).sum()) for b in ("<200^2", "200^2-400^2", ">400^2")},
           "masks_by_area_bin_and_role": {b: {r: int(((abin == b) & (role == r)).sum()) for r in ("object", "near", "distractor")}
                                          for b in ("<200^2", "200^2-400^2", ">400^2")},
           "box_fill": {"all": dist(fill, ef), "by_role": by_group(fill, role, ef), "by_area_bin": by_group(fill, abin, ef)},
           "masks_per_scene": dist([s.n for s in scenes]),
           "masks_per_scene_by_room": {rm: dist([s.n for s in scenes if s.room == rm]) for rm in sorted({s.room for s in scenes})},
           "roles": {r: int((role == r).sum()) for r in ("object", "near", "distractor")},
           "layers": {v: int((layer == v).sum()) for v in ("frame", "window")}}
    cov = {"union": [], "roots": {str(th): [] for th in THETAS}, "multiplicity": {str(k): 0 for k in (1, 2, 3, 4, 5)},
           "mean_multiplicity_covered": []}
    gt_cov, gt_iou = [], []
    from src.eval.boxes import box_iou

    for sm in scenes:
        h, w = sm.entry.seg_hw
        cnt = np.zeros((h, w), np.uint16)
        for i in range(sm.n):
            cnt += sm.entry.decode(i)
        covered = cnt > 0
        cov["union"].append(float(covered.mean()))
        cov["mean_multiplicity_covered"].append(float(cnt[covered].mean()) if covered.any() else 0.0)
        for k in (1, 2, 3, 4):
            cov["multiplicity"][str(k)] += int((cnt == k).sum())
        cov["multiplicity"]["5"] += int((cnt >= 5).sum())
        for th in THETAS:
            roots = np.flatnonzero(nest[sm.id][th].parent < 0)
            u = np.zeros((h, w), bool)
            for i in roots:
                u |= sm.entry.decode(int(i))
            cov["roots"][str(th)].append(float(u.mean()))
        for m, g in sm.gt_of_object.items():
            x0, y0, x1, y1 = (int(round(v)) for v in sm.gt_boxes[g])
            win = sm.entry.window(m, x0, y0, x1, y1)
            gt_cov.append(float(win.sum()) / max((x1 - x0) * (y1 - y0), 1))
            gt_iou.append(float(box_iou(sm.box[m:m + 1], sm.gt_boxes[g:g + 1])[0, 0]))
    tot = sum(cov["multiplicity"].values())
    out["frame_coverage"] = {"union_share_per_scene": dist(cov["union"]),
                             "roots_union_share_per_scene": {th: dist(v) for th, v in cov["roots"].items()},
                             "covered_pixels_by_multiplicity_share": {("5+" if k == "5" else k): v / max(tot, 1)
                                                                      for k, v in cov["multiplicity"].items()},
                             "mean_multiplicity_over_covered_pixels_per_scene": dist(cov["mean_multiplicity_covered"])}
    out["object_masks"] = {"gt_box_covered_by_mask_share": dist(gt_cov, ef), "box_iou_with_gt": dist(gt_iou, np.linspace(0.5, 1, 11))}
    return out


def redundancy_block(galleries, scenes, emb) -> dict:
    out = {}
    for e in ENCODERS:
        g = galleries[e]
        z = np.asarray(g.emb, np.float64)
        labs = np.array([m["label"] for m in g.meta])
        view = np.array([int(m["id"].rsplit("/", 1)[1]) for m in g.meta])
        C = z @ z.T
        np.fill_diagonal(C, -np.inf)
        same = labs[:, None] == labs[None, :]
        nn_same = np.where(same, C, -np.inf).max(1)
        nn_other = np.where(~same, C, -np.inf).max(1)
        pair_cos, dview, ring, per_label = [], [], [], {}
        for y in sorted(set(labs)):
            idx = np.flatnonzero(labs == y)
            iu = np.triu_indices(len(idx), 1)
            v = C[np.ix_(idx, idx)][iu]
            d = np.abs(view[idx][:, None] - view[idx][None, :])[iu]
            pair_cos += v.tolist(); dview += d.tolist(); ring += np.minimum(d, 24 - d).tolist()
            covers = {}
            for c in COVER_LEVELS:
                A = C[np.ix_(idx, idx)] >= c
                np.fill_diagonal(A, True)
                left, picks = np.ones(len(idx), bool), 0
                while left.any():
                    k = int(np.argmax((A & left[None, :]).sum(1)))
                    left &= ~A[k]; picks += 1
                covers[str(c)] = picks
            per_label[y] = {"median_pair_cos": float(np.median(v)), "min_pair_cos": float(v.min()), "greedy_cover": covers}
        pair_cos, dview, ring = map(np.asarray, (pair_cos, dview, ring))
        ec = np.linspace(-0.2, 1.0, 25)
        part = {y: per_label[y] for y in RU.PART_MASK_LABELS}
        # маски «объект» калибровочных сцен: какой ракурс своей метки ближайший
        winners, n_obj = {}, 0
        for sm in scenes:
            zz = emb[e][sm.id]
            for m, gi in sm.gt_of_object.items():
                y = g.labels[sm.gt_labels[gi]]
                idx = np.flatnonzero(labs == y)
                k = idx[int(np.argmax(zz[m] @ z[idx].T))]
                winners.setdefault(y, []).append(int(view[k])); n_obj += 1
        distinct = {y: len(set(v)) for y, v in winners.items()}
        out[e] = {"gallery": str(g.path), "n_refs": int(len(z)), "n_labels": int(len(set(labs))),
                  "within_label_pair_cos": dist(pair_cos, ec),
                  "within_label_median_pair_cos_per_label": dist([v["median_pair_cos"] for v in per_label.values()], ec),
                  "nn_same_label_cos": dist(nn_same, ec), "nn_other_label_cos": dist(nn_other, ec),
                  "nn_same_minus_nn_other": dist(nn_same - nn_other, np.linspace(-0.4, 0.8, 25)),
                  "share_refs_nn_other_above_nn_same": float((nn_other > nn_same).mean()),
                  "pair_cos_by_view_index_distance": {str(d): dist(pair_cos[dview == d]) for d in range(1, 24)},
                  "pair_cos_by_ring_distance": {str(d): dist(pair_cos[ring == d]) for d in range(1, 13)},
                  "greedy_cover_refs_per_label": {str(c): dist([v["greedy_cover"][str(c)] for v in per_label.values()])
                                                  for c in COVER_LEVELS},
                  "greedy_cover_total_refs": {str(c): int(sum(v["greedy_cover"][str(c)] for v in per_label.values()))
                                              for c in COVER_LEVELS},
                  "part_mask_labels": part,
                  "cal_object_masks_nearest_view_of_own_label": {
                      "n_object_masks": n_obj, "n_labels": len(winners),
                      "distinct_nearest_views_per_label": dist(list(distinct.values())),
                      "object_masks_per_label": dist([len(v) for v in winners.values()]),
                      "per_label": {y: {"n": len(v), "views": sorted(set(v))} for y, v in sorted(winners.items())}},
                  "per_label": per_label}
    return out


# ---------------------------------------------------------------------- вход


def best_variants() -> dict[str, str]:
    """Лучший вариант энкодера по `selection_cal` журнала."""
    out = {}
    for e in ENCODERS:
        names = [v for enc, v, d in grid() if enc == e and d == DATASET]
        cal = {v: json.loads((RUNS / f"{e}_{v}_{DATASET}.json").read_text())["selection_cal"] for v in names}
        out[e] = RU.best_variant({v: {"ap": c["ap"], "ap50": c["ap50"]} for v, c in cal.items()})
    return out


def load(split: str):
    check_split(split)
    best = best_variants()
    cfgs = {e: CFG.load(CFG.CONFIG_DIR / f"{e}_{best[e]}_{DATASET}.yaml") for e in ENCODERS}
    c0 = cfgs[ENCODERS[0]]
    mc = MC.MaskCache(DATASET, MC.auto_key(c0.crop_n_layers, c0.seg_long_side, c0.points_per_batch))
    scenes, labels, edges, sp = PR.load_scenes(DATASET, mc, splits=(split,))   # кеш тестовых сцен не читается
    if len(scenes) != N_SCENES or [s.id for s in scenes] != list(sp[SPLIT]) or any(s.split != SPLIT for s in scenes) \
            or set(sp[SPLIT]) & set(sp["test"]):
        raise SystemExit("ожидаются ровно 40 калибровочных сцен в порядке splits")
    sms = [SceneMasks(s, mc.load(s.id)) for s in scenes]
    galleries, emb, dec = {}, {}, {}
    from src.search import decide as DE
    from src.search import index as IX

    for e, cfg in cfgs.items():
        g = store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset),
                               store.expected(cfg.to_dict(), MC.box_key(cfg.seg_long_side)))
        if g.labels != labels:
            raise SystemExit("метки галереи расходятся с категориями истины")
        galleries[e] = g
        work = RE.scene_store(cfg, mc.key)
        rows = g.rows("full")
        index = IX.ExactIndex(g.emb, rows)
        emb[e], dec[e] = {}, {}
        for sm in sms:
            z = work.load(sm.id)["z"]
            if len(z) != sm.n:
                raise SystemExit(f"{sm.id}: эмбеддингов {len(z)}, масок {sm.n}")
            y, s = DE.decide(*index.search(z), g.label_ids, deleted=g.deleted)
            dec[e][sm.id] = (y, s)
            emb[e][sm.id] = z / np.linalg.norm(z, axis=1, keepdims=True)
    refs_key = MC.box_key(c0.seg_long_side)
    rmc = MC.MaskCache(DATASET, refs_key)
    meta = {m["id"]: m for m in galleries[ENCODERS[0]].meta}
    refs = []
    for r in sp["gallery"]:
        rec = rmc.load(r["id"], [r["box"]]).records[0]
        refs.append({"id": r["id"], "predicted_iou": rec["predicted_iou"], "stability_score": rec["stability_score"],
                     "control_iou": meta[r["id"]]["control_iou"]})
    return best, cfgs, mc, sms, galleries, emb, dec, refs


def _write(name: str, block: dict, head: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{name}.json.tmp"
    tmp.write_text(json.dumps({**head, name: block}, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT / f"{name}.json")


def summary_md(head: dict, nb: dict, sb: dict, sc: dict, ar: dict, rd: dict) -> str:
    f = lambda d, q="0.5": f"{d['q'][q]:.3f}" if d.get("n") else "—"
    L = ["# Аналитика §2.3 — сводка (40 калибровочных сцен HR-InsDet)", "",
         "Сводка по JSON-файлам этого каталога, посчитанным `scripts/analyze_cal_masks.py`. Только split `cal`; "
         "описание данных без критериев и порогов; полные числа — JSON-файлы этого каталога.", "",
         f"Масок: {nb['n_masks']}; ролей: объект {ar['roles']['object']}, около объекта {ar['roles']['near']}, "
         f"дистрактор {ar['roles']['distractor']}; слоёв генератора: кадр целиком {ar['layers']['frame']}, окна {ar['layers']['window']}. "
         f"Перекрывающихся пар: {nb['n_pairs_overlapping']}.", "",
         "## Вложенность", "",
         "| θ | вложенных пар | пересечений без вложения | дубликатов | нарушений транзитивности | глубина леса (медиана / макс. по сценам) | корней на сцену (медиана) | ширина / масок (медиана) | маска «объект»: корень / внутр. / лист / изол. |",
         "|---|---|---|---|---|---|---|---|---|"]
    for th in THETAS:
        b = nb["by_theta"][str(th)]
        p, o = b["pairs"], b["object_masks"]["by_position"]
        L.append(f"| {th:.2f} | {p['nested']} | {p['crossing']} | {p['duplicate']} | {b['transitivity_violation_share']:.3f} | "
                 f"{f(b['per_scene_summary']['forest_depth'])} / {b['per_scene_summary']['forest_depth']['max']:.0f} | "
                 f"{f(b['per_scene_summary']['n_roots'])} | {f(b['width_over_n_masks'])} | "
                 f"{o['root']} / {o['internal']} / {o['leaf']} / {o['isolated']} |")
    L += ["", "## Устойчивость эмбеддинга «ребёнок — родитель» (рёбра леса), медиана косинуса", "",
          "| θ | " + " | ".join(f"{e}: все рёбра | {e}: тот же ŷ" for e in ENCODERS) + " |",
          "|---|" + "---|---|" * len(ENCODERS)]
    for th in THETAS:
        row = []
        for e in ENCODERS:
            b = sb[e]["by_theta"][str(th)]
            row.append(f"{f(b['forest_edges_cos'])} | {b['forest_edges_same_y_hat_share']:.3f}")
        L.append(f"| {th:.2f} | " + " | ".join(row) + " |")
    L += ["", "Для сравнения (θ = 0,90 для пересечений): " + "; ".join(
        f"{e} — пары с пересечением без вложения {f(sb[e]['reference']['crossing_pairs_theta_0.9'])}, "
        f"все пары сцены {f(sb[e]['reference']['all_pairs_in_scene'])}" for e in ENCODERS) + ".", "",
          "Где в цепочке через маску «объект» наибольшее s* (θ = 0,90): " + "; ".join(
        f"{e} — {sb[e]['by_theta']['0.9']['object_mask_chain']['place_of_max_s_star_in_chain']}" for e in ENCODERS) + ".", "",
          "## Оценки сегментатора, медианы", "",
          "| роль | predicted IoU | stability score |", "|---|---|---|"]
    for r in ("object", "near", "distractor"):
        L.append(f"| {r} | {f(sc['predicted_iou']['by_role'][r])} | {f(sc['stability_score']['by_role'][r])} |")
    L += [f"| эталоны (рамка) | {f(sc['references']['predicted_iou'])} | {f(sc['references']['stability_score'])} |", "",
          f"Ранговая корреляция predicted IoU и stability — {sc['spearman']['predicted_iou_vs_stability']:.3f}; "
          f"у эталонов с IoU маски GrabCut — {sc['references']['spearman']['predicted_iou_vs_control_iou']:.3f} и "
          f"{sc['references']['spearman']['stability_vs_control_iou']:.3f}.", "",
          "## Площади и покрытие", "",
          f"Масок на сцену — медиана {f(ar['masks_per_scene'])}; покрытие кадра объединением масок — медиана "
          f"{f(ar['frame_coverage']['union_share_per_scene'])}; кратность покрытия (доли покрытых пикселей): "
          + ", ".join(f"{k} — {v:.3f}" for k, v in ar['frame_coverage']['covered_pixels_by_multiplicity_share'].items())
          + f"; у масок «объект» доля рамки разметки под маской — медиана {f(ar['object_masks']['gt_box_covered_by_mask_share'])}.", "",
          "| роль | log10 площади, медиана | заполнение рамки, медиана |", "|---|---|---|"]
    for r in ("object", "near", "distractor"):
        L.append(f"| {r} | {f(ar['log10_area_px']['by_role'][r])} | {f(ar['box_fill']['by_role'][r])} |")
    L += ["", "## Близость эталонов внутри экземпляра", "",
          "| энкодер | косинус ракурсов метки, медиана | ближайший своей метки, медиана | ближайший чужой, медиана | доля эталонов, у которых чужой ближе | эталонов при жадном покрытии, всего: " + ", ".join(f"{c:.2f}" for c in COVER_LEVELS[::2]) + " |",
          "|---|---|---|---|---|---|"]
    for e in ENCODERS:
        b = rd[e]
        L.append(f"| {e} | {f(b['within_label_pair_cos'])} | {f(b['nn_same_label_cos'])} | {f(b['nn_other_label_cos'])} | "
                 f"{b['share_refs_nn_other_above_nn_same']:.3f} | "
                 + ", ".join(str(b['greedy_cover_total_refs'][str(c)]) for c in COVER_LEVELS[::2]) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=SPLIT, help="только «cal»; иное — отказ")
    args = ap.parse_args()
    t0 = time.time()
    best, cfgs, mc, sms, galleries, emb, dec, refs = load(args.split)
    print(f"сцен {len(sms)}, масок {sum(s.n for s in sms)}; лучшие φ — {best}; {time.time() - t0:.0f} с", flush=True)
    nest = {sm.id: {th: Nesting(sm, th) for th in THETAS} for sm in sms}
    head = {"what": "аналитика §2.3: описание данных без критериев и порогов",
            "dataset": DATASET, "split": SPLIT, "scenes": [s.id for s in sms], "thetas": list(THETAS),
            "best_variants": best, "runs": {e: c.run_id for e, c in cfgs.items()}, "mask_cache": mc.dir.name,
            **_STAMP}
    nb = nesting_block(sms, nest); _write("nesting", nb, head); print("nesting", flush=True)
    sb = stability_block(sms, nest, emb, dec); _write("embedding_stability", sb, head); print("stability", flush=True)
    sc = scores_block(sms, nest, refs); _write("scores", sc, head); print("scores", flush=True)
    ar = areas_block(sms, nest); _write("areas_coverage", ar, head); print("areas", flush=True)
    rd = redundancy_block(galleries, sms, emb); _write("gallery_redundancy", rd, head); print("redundancy", flush=True)
    (OUT / "summary.md").write_text(summary_md(head, nb, sb, sc, ar, rd))
    print(f"ГОТОВО: {OUT}/ за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
