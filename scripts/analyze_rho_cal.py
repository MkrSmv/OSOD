"""Аналитика §2.3, часть 2 — кандидатные правила $\\rho$ и дедупликации по 40 калибровочным сценам HR-InsDet.

    python scripts/analyze_rho_cal.py              # только split cal; иной split — отказ до чтения данных

Перечень величин и правила выбора записаны до кода и до счёта. Маски, отношение $\\preceq_\\theta$, лес, эмбеддинги и галереи — те же, что в
части 1 (`scripts/analyze_cal_masks.py`, загрузка переиспользуется). Вывод — `experiments/analysis/rho_rules.json`,
`rho_embedding.json`, `gallery_dedup.json` и сводка `summary_rho.md`.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as MU

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.boxes import box_iou  # noqa: E402
from src.eval import rules as RU  # noqa: E402
from src.search import decide as DE  # noqa: E402
from src.search import index as IX  # noqa: E402

_spec = importlib.util.spec_from_file_location("analyze_cal_masks", ROOT / "scripts" / "analyze_cal_masks.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

SPLIT, THETAS, ENCODERS, OUT = A.SPLIT, A.THETAS, A.ENCODERS, A.OUT
THETA_SEL = 0.90                                          # правило выбора (1): по части 1, до счёта
SPLIT_C = (0.9, 0.8, 0.7, 0.6, 0.5)                       # порядок простоты для правила (2)
MIN_GAIN_PP = 1.0                                         # правило (2): выигрыш полноты при 0,5, п. п.
AUROC_MARGIN = 0.15                                       # правило (3)
DEDUP_T = tuple(round(0.90 + 0.01 * k, 2) for k in range(10))
DEDUP_MAX_LOSS = 3                                        # правило (4): верных ŷ из 764
RATIO_BINS = (0.0, 0.25, 0.5, 0.75, 1.0 + 1e-9)
IOU_THRS = np.linspace(0.5, 0.95, 10)


def check_split(split: str) -> str:
    return A.check_split(split)


# ---------------------------------------------------------------------- правила ρ


def rule_maximal(ne) -> list[int]:
    return [i for i in range(ne.sm.n) if not ne.inside[i]]


def rule_minimal(ne) -> list[int]:
    return [i for i in range(ne.sm.n) if not ne.contains[i]]


def _cover(sm, v: int, kids: list[int]) -> float:
    """Доля площади вершины `v`, накрытая объединением детей (точно по RLE, в разрешении входа $S$)."""
    u = MU.merge([sm.rles[k] for k in kids], intersect=False)
    return float(MU.area(MU.merge([u, sm.rles[v]], intersect=True))) / float(sm.area_seg[v])


def rule_split(ne, c: float) -> tuple[list[int], int]:
    """Срез леса сверху: вершина заменяется детьми, если их не меньше двух и они покрывают ≥ c её площади; затем
    удаляются отобранные маски, у которых среди отобранных есть содержащая. Возвращает (отбор, число удалённых)."""
    sm, keep, stack = ne.sm, [], [i for i in range(ne.sm.n) if ne.parent[i] < 0]
    while stack:
        v = stack.pop()
        kids = ne.children[v]
        if len(kids) >= 2 and _cover(sm, v, kids) >= c:
            stack += kids
        else:
            keep.append(v)
    ks = set(keep)
    out = [i for i in keep if not (ne.inside[i] & ks)]
    return sorted(out), len(keep) - len(out)


NEARDUP_A = (0.80, 0.85, 0.90, 0.95, 0.98)                # часть 3, правило выбора (5)


def rule_neardup(ne, a: float) -> list[int]:
    """Часть 3: вложение $m\\preceq_\\theta m'$ существенное при $|m|<a|m'|$, иначе $m$ — почти-дубликат $m'$.
    Группы почти-дубликатов — связные компоненты; группа отбирается, если ни одна её маска не вложена в маску вне
    группы; от группы остаётся наименьшая по площади маска (при равной — меньший номер)."""
    n, ar = ne.sm.n, ne.sm.area_seg
    comp = list(range(n))

    def find(x):
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x

    for m in range(n):
        for b in ne.inside[m]:
            if ar[m] >= a * ar[b]:
                comp[find(m)] = find(b)
    groups: dict[int, list[int]] = {}
    for m in range(n):
        groups.setdefault(find(m), []).append(m)
    out = []
    for g in groups.values():
        gs = set(g)
        if all(ne.inside[m] <= gs for m in g):
            out.append(min(g, key=lambda m: (ar[m], m)))
    return sorted(out)


def rule_sets_part3(ne) -> dict[str, tuple[list[int], int]]:
    out = {"all": (list(range(ne.sm.n)), 0), "maximal": (rule_maximal(ne), 0)}
    for a in NEARDUP_A:
        out[f"neardup_{a}"] = (rule_neardup(ne, a), 0)
    return out


def rule_sets(ne) -> dict[str, tuple[list[int], int]]:
    out = {"all": (list(range(ne.sm.n)), 0), "maximal": (rule_maximal(ne), 0), "minimal": (rule_minimal(ne), 0)}
    for c in SPLIT_C:
        out[f"split_{c}"] = rule_split(ne, c)
    return out


def rule_metrics(sms, nest, th, sets_fn=None) -> dict:
    per_rule: dict[str, dict] = {}
    for sm in sms:
        sets = (sets_fn or rule_sets)(nest[sm.id][th])
        gt_area = (sm.gt_boxes[:, 2] - sm.gt_boxes[:, 0]) * (sm.gt_boxes[:, 3] - sm.gt_boxes[:, 1])
        abin = np.digitize(gt_area, [200.0 ** 2, 400.0 ** 2])
        iou_all = box_iou(sm.gt_boxes, sm.box)                           # (рамки, маски)
        found_all = iou_all.max(1) >= 0.5 if iou_all.shape[1] else np.zeros(len(gt_area), bool)
        best_mask_iou = iou_all.max(0) if iou_all.shape[0] else np.zeros(sm.n)
        objs = set(sm.gt_of_object)
        for name, (sel, repaired) in sets.items():
            r = per_rule.setdefault(name, {"n_kept": [], "roles": {"object": 0, "near": 0, "distractor": 0},
                                           "best": [], "bin": [], "lost": 0, "objects_kept": 0, "mult2": 0,
                                           "near_parts_groups": 0, "repaired": 0, "n_all": 0})
            sel = np.asarray(sel, int)
            r["n_kept"].append(len(sel)); r["n_all"] += sm.n; r["repaired"] += repaired
            for ro in ("object", "near", "distractor"):
                r["roles"][ro] += int((sm.role[sel] == ro).sum())
            iou = iou_all[:, sel] if len(sel) else np.zeros((len(gt_area), 0))
            best = iou.max(1) if iou.shape[1] else np.zeros(len(gt_area))
            r["best"] += best.tolist(); r["bin"] += abin.tolist()
            r["lost"] += int((found_all & (best < 0.5)).sum())
            r["objects_kept"] += len(objs & set(sel.tolist()))
            r["mult2"] += int(((iou >= 0.5).sum(1) >= 2).sum())
            b = best_mask_iou[sel]
            r["near_parts_groups"] += int(((b >= 0.1) & (b < 0.5)).sum())
    out = {}
    for name, r in per_rule.items():
        best, abin = np.asarray(r["best"]), np.asarray(r["bin"])
        rec = lambda m: {"recall_50": float((best[m] >= 0.5).mean()), "recall_75": float((best[m] >= 0.75).mean()),
                         "recall_50_95": float((best[m][:, None] >= IOU_THRS[None]).mean()), "n_gt": int(m.sum())}
        nk = np.asarray(r["n_kept"])
        out[name] = {"n_kept": int(nk.sum()), "share_of_M": float(nk.sum() / r["n_all"]),
                     "per_scene_median": float(np.median(nk)), "by_role": r["roles"],
                     **rec(np.ones(len(best), bool)),
                     "by_gt_area_bin": {n: rec(abin == k) for k, n in enumerate(("<200^2", "200^2-400^2", ">400^2"))},
                     "gt_found_in_M_lost_at_50": r["lost"], "object_masks_kept": r["objects_kept"],
                     "gt_with_2plus_masks_at_50_share": float(r["mult2"] / len(best)),
                     "kept_masks_best_gt_iou_in_0.1_0.5": r["near_parts_groups"],
                     "removed_by_antichain_repair": r["repaired"]}
    return out


def objects_with_parent(sms, nest) -> dict:
    ratio, piou, nobj = [], [], []
    for sm in sms:
        ne = nest[sm.id][THETA_SEL]
        objs = set(sm.gt_of_object)
        for m, g in sm.gt_of_object.items():
            p = ne.parent[m]
            if p < 0:
                continue
            ratio.append(sm.area_seg[m] / sm.area_seg[p])
            piou.append(float(box_iou(sm.box[[p]], sm.gt_boxes[[g]])[0, 0]))
            nobj.append(len(objs & set(ne.descendants(int(p)))))
    piou = np.asarray(piou)
    return {"n": len(ratio), "area_ratio_object_to_parent": A.dist(ratio, np.linspace(0, 1, 11)),
            "parent_box_iou_with_gt": A.dist(piou, np.linspace(0, 1, 11)),
            "parent_box_iou_with_gt_ge_0.5_share": float((piou >= 0.5).mean()) if len(piou) else None,
            "object_masks_among_parent_descendants": A.dist(nobj, np.arange(0.5, 11.5, 1))}


def select_rule(res: dict) -> dict:
    """Правило выбора (2), записанное до счёта."""
    r = lambda n: 100.0 * res[n]["recall_50"]
    cur = "maximal" if r("maximal") >= r("minimal") else "minimal"
    steps = [f"исходное — {cur} ({r(cur):.2f} %; maximal {r('maximal'):.2f}, minimal {r('minimal'):.2f})"]
    for c in SPLIT_C:
        n = f"split_{c}"
        if r(n) >= r(cur) + MIN_GAIN_PP:
            steps.append(f"{n}: {r(n):.2f} ≥ {r(cur):.2f} + {MIN_GAIN_PP} — принято"); cur = n
        else:
            steps.append(f"{n}: {r(n):.2f} < {r(cur):.2f} + {MIN_GAIN_PP} — нет")
    return {"theta": THETA_SEL, "selected": cur, "steps": steps,
            "recall_50_selected": res[cur]["recall_50"], "recall_50_all": res["all"]["recall_50"],
            "loss_vs_all_pp": 100.0 * (res["all"]["recall_50"] - res[cur]["recall_50"])}


def select_part3(res: dict) -> dict:
    """Правило выбора (5), записанное до счёта части 3."""
    r = lambda n: 100.0 * res[n]["recall_50"]
    best = None
    for a in sorted(NEARDUP_A, reverse=True):                   # при разнице < 0,1 п. п. — большее a
        n = f"neardup_{a}"
        if best is None or r(n) > r(best) + 0.1:
            best = n
    take = r(best) >= r("maximal") + MIN_GAIN_PP
    return {"theta": THETA_SEL, "best_neardup": best, "recall_50_best_neardup": res[best]["recall_50"],
            "recall_50_maximal": res["maximal"]["recall_50"], "replaces_maximal": take,
            "selected": best if take else "maximal",
            "loss_vs_all_pp": 100.0 * (res["all"]["recall_50"] - res[best if take else "maximal"]["recall_50"])}


def antichain_ok(sms, nest, a: float) -> int:
    """Число пар отобранных масок, связанных $\\preceq_\\theta$ (должно быть 0)."""
    bad = 0
    for sm in sms:
        ne = nest[sm.id][THETA_SEL]
        ks = set(rule_neardup(ne, a))
        bad += sum(len(ne.inside[i] & ks) for i in ks)
    return bad


def summary_part3(head, rr, sel, bad) -> str:
    L = ["# Аналитика §2.3, часть 3 — правило ρ с почти-дубликатами (40 калибровочных сцен HR-InsDet)", "",
         "Сводка по `rho_rules_part3.json`, посчитанному `scripts/analyze_rho_cal.py --part3`. Правило выбора (5) записано до счёта.", "",
         f"## θ = {THETA_SEL:.2f}", "",
         "| правило | масок | доля M(I) | объект / около / дистр. | recall@0,5 | @0,75 | 0,5:0,95 | <200² | 200²–400² | >400² | потеряно из найденных в M(I) | масок «объект» | рамок с ≥2 масками | частей и групп рядом |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in rr[str(THETA_SEL)].items():
        bb = r["by_gt_area_bin"]
        L.append(f"| {n} | {r['n_kept']} | {r['share_of_M']:.3f} | {r['by_role']['object']} / {r['by_role']['near']} / "
                 f"{r['by_role']['distractor']} | {100 * r['recall_50']:.2f} | {100 * r['recall_75']:.2f} | "
                 f"{100 * r['recall_50_95']:.2f} | " + " | ".join(f"{100 * bb[k]['recall_50']:.1f}" for k in bb)
                 + f" | {r['gt_found_in_M_lost_at_50']} | {r['object_masks_kept']} | {r['gt_with_2plus_masks_at_50_share']:.3f} | "
                 f"{r['kept_masks_best_gt_iou_in_0.1_0.5']} |")
    L += ["", f"Выбор по правилу (5): лучшее neardup — {sel['best_neardup']} ({100 * sel['recall_50_best_neardup']:.2f} % против "
          f"{100 * sel['recall_50_maximal']:.2f} у maximal); заменяет maximal — {'да' if sel['replaces_maximal'] else 'нет'}; "
          f"**выбрано {sel['selected']}**; потеря полноты при 0,5 относительно M(I) — {sel['loss_vs_all_pp']:.2f} п. п. "
          f"Пар отобранных масок, связанных вложением, у выбранного a: {bad}.", "",
          "Чувствительность к θ (recall@0,5, %; число масок):", "",
          "| θ | " + " | ".join(rr[str(THETA_SEL)].keys()) + " |", "|---|" + "---|" * len(rr[str(THETA_SEL)])]
    for th in THETAS:
        x = rr[str(th)]
        L.append(f"| {th:.2f} | " + " | ".join(f"{100 * x[n]['recall_50']:.2f}; {x[n]['n_kept']}" for n in x) + " |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- косинус против отношения площадей


def auroc(score, label) -> float | None:
    """P(score_+ > score_-) + ½P(=), по рангам."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return None
    from scipy.stats import rankdata
    rk = rankdata(score)
    return float((rk[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def embedding_block(sms, nest, emb) -> dict:
    out = {}
    for e in ENCODERS:
        cos, rat, lab, cos_all, rat_all = [], [], [], [], []
        for sm in sms:
            z, ne = emb[e][sm.id], nest[sm.id][THETA_SEL]
            for i, p in enumerate(ne.parent):
                if p < 0:
                    continue
                c, q = float(z[i] @ z[p]), sm.area_seg[i] / sm.area_seg[p]
                cos_all.append(c); rat_all.append(q)
                ci, cp = sm.role[i] == "object", sm.role[p] == "object"
                if ci != cp:
                    cos.append(c); rat.append(q); lab.append(bool(cp))
        cos, rat, lab = map(np.asarray, (cos, rat, lab))
        bins = np.digitize(rat, RATIO_BINS[1:-1])
        per_bin, num, den = {}, 0.0, 0.0
        for k in range(len(RATIO_BINS) - 1):
            m = bins == k
            npos, nneg = int(lab[m].sum()), int((~lab[m]).sum())
            a = auroc(cos[m], lab[m])
            per_bin[f"{RATIO_BINS[k]:.2f}-{min(RATIO_BINS[k + 1], 1.0):.2f}"] = {"n_parent_object": npos, "n_child_object": nneg,
                                                                                "auroc_cos": a}
            if a is not None:
                num += a * npos * nneg; den += npos * nneg
        strat = num / den if den else None
        out[e] = {"n_edges": int(len(lab)), "n_parent_object": int(lab.sum()), "n_child_object": int((~lab).sum()),
                  "auroc_area_ratio": auroc(rat, lab), "auroc_cos": auroc(cos, lab),
                  "spearman_cos_area_ratio_all_forest_edges": A.spearman(cos_all, rat_all),
                  "spearman_cos_area_ratio_these_edges": A.spearman(cos, rat),
                  "auroc_cos_by_area_ratio_bin": per_bin, "stratified_auroc_cos": strat}
    keep = all(out[e]["stratified_auroc_cos"] is not None and abs(out[e]["stratified_auroc_cos"] - 0.5) >= AUROC_MARGIN
               for e in ENCODERS)
    out["verdict"] = {"rule": f"|стратифицированный AUROC − 0,5| ≥ {AUROC_MARGIN} у обоих энкодеров",
                      "embedding_stability_remains_candidate": keep}
    return out


# ---------------------------------------------------------------------- дедупликация галереи


def dedup_keep(z: np.ndarray, labs: np.ndarray, order: np.ndarray, t: float) -> np.ndarray:
    keep = np.zeros(len(z), bool)
    for j in order:
        same = np.flatnonzero(keep & (labs == labs[j]))
        if len(same) == 0 or float((z[same] @ z[j]).max()) < t:
            keep[j] = True
    return keep


def dedup_block(sms, galleries, emb) -> dict:
    out = {}
    for e in ENCODERS:
        g = galleries[e]
        z = np.asarray(g.emb, np.float32)
        lab_names = np.array([m["label"] for m in g.meta])
        labs = np.asarray(g.label_ids)
        order = np.argsort([m["insert_no"] for m in g.meta], kind="stable")
        rows_full = g.rows("full")
        qz, qy, qd = [], [], []
        for sm in sms:
            zz = emb[e][sm.id].astype(np.float32)
            for m, gi in sm.gt_of_object.items():
                qz.append(zz[m]); qy.append(int(sm.gt_labels[gi]))
            qd.append(zz[sm.role == "distractor"])
        qz, qy, qd = np.stack(qz), np.asarray(qy), np.concatenate(qd)

        def run(rows):
            ix = IX.ExactIndex(g.emb, rows)
            y, _ = DE.decide(*ix.search(qz), g.label_ids, deleted=g.deleted)
            _, sd = DE.decide(*ix.search(qd), g.label_ids, deleted=g.deleted)
            sub = np.asarray(rows)
            S = qz @ z[sub].T
            s_true = np.array([S[k, labs[sub] == qy[k]].max() if (labs[sub] == qy[k]).any() else -np.inf
                               for k in range(len(qy))])
            return y, sd, s_true

        y0, sd0, st0 = run(rows_full)
        part = np.isin(lab_names, RU.PART_MASK_LABELS)
        part_q = np.isin(np.asarray(g.labels)[qy], RU.PART_MASK_LABELS)
        res = {"n_refs": int(len(z)), "n_object_masks": int(len(qy)), "n_distractors": int(len(qd)),
               "full": {"correct": int((y0 == qy).sum()), "distractor_s_star_q95": float(np.quantile(sd0, 0.95)),
                        "distractor_s_star_q99": float(np.quantile(sd0, 0.99))}, "by_t": {}}
        full_set = set(int(r) for r in rows_full)
        for t in DEDUP_T:
            keep = dedup_keep(z.astype(np.float64), labs, order, t)
            rows = np.asarray([r for r in rows_full if keep[r]], np.int64)
            assert set(rows) <= full_set
            y1, sd1, st1 = run(rows)
            per = np.bincount(labs[keep], minlength=len(g.labels))
            drop = st0 - st1
            res["by_t"][str(t)] = {
                "n_kept": int(keep.sum()), "per_label_min": int(per.min()), "per_label_median": float(np.median(per)),
                "correct": int((y1 == qy).sum()), "correct_loss": int((y0 == qy).sum() - (y1 == qy).sum()),
                "y_hat_changed": int((y1 != y0).sum()),
                "true_label_sim_drop": A.dist(drop, np.linspace(0, 0.2, 21)),
                "distractor_s_star_q95": float(np.quantile(sd1, 0.95)),
                "distractor_s_star_q99": float(np.quantile(sd1, 0.99)),
                "part_mask_labels": {"n_kept": int((keep & part).sum()), "n_refs": int(part.sum()),
                                     "object_masks": int(part_q.sum()),
                                     "correct_full": int(((y0 == qy) & part_q).sum()),
                                     "correct": int(((y1 == qy) & part_q).sum())}}
        ok = [t for t in DEDUP_T if res["by_t"][str(t)]["correct_loss"] <= DEDUP_MAX_LOSS]
        res["selection"] = {"rule": f"наименьшее t сетки с потерей верных ŷ ≤ {DEDUP_MAX_LOSS}",
                            "t_star": min(ok) if ok else None}
        out[e] = res
    return out


# ---------------------------------------------------------------------- сводка


def summary_md(head, rr, sel, owp, eb, dd) -> str:
    L = ["# Аналитика §2.3, часть 2 — сводка (40 калибровочных сцен HR-InsDet)", "",
         "Сводка по JSON-файлам этого каталога, посчитанным `scripts/analyze_rho_cal.py`. Только split `cal`; "
         "правила выбора записаны до счёта.", "",
         f"## Кандидатные правила ρ при θ = {THETA_SEL:.2f}", "",
         "| правило | масок | доля M(I) | объект / около / дистр. | recall@0,5 | @0,75 | 0,5:0,95 | <200² | 200²–400² | >400² | потеряно из найденных в M(I) | масок «объект» | рамок с ≥2 масками | частей и групп рядом |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    b = rr[str(THETA_SEL)]
    for n, r in b.items():
        bb = r["by_gt_area_bin"]
        L.append(f"| {n} | {r['n_kept']} | {r['share_of_M']:.3f} | {r['by_role']['object']} / {r['by_role']['near']} / "
                 f"{r['by_role']['distractor']} | {100 * r['recall_50']:.2f} | {100 * r['recall_75']:.2f} | "
                 f"{100 * r['recall_50_95']:.2f} | " + " | ".join(f"{100 * bb[k]['recall_50']:.1f}" for k in bb)
                 + f" | {r['gt_found_in_M_lost_at_50']} | {r['object_masks_kept']} | {r['gt_with_2plus_masks_at_50_share']:.3f} | "
                 f"{r['kept_masks_best_gt_iou_in_0.1_0.5']} |")
    L += ["", f"Выбор по правилу (2): **{sel['selected']}**; " + "; ".join(sel["steps"])
          + f". Потеря полноты при 0,5 относительно M(I) — {sel['loss_vs_all_pp']:.2f} п. п.", "",
          "Чувствительность к θ (recall@0,5, %; число масок):", "",
          "| θ | " + " | ".join(b.keys()) + " |", "|---|" + "---|" * len(b)]
    for th in THETAS:
        x = rr[str(th)]
        L.append(f"| {th:.2f} | " + " | ".join(f"{100 * x[n]['recall_50']:.2f}; {x[n]['n_kept']}" for n in b) + " |")
    L += ["", f"Маски «объект» с предком (θ = {THETA_SEL:.2f}): {owp['n']}; отношение площадей к родителю — медиана "
          f"{owp['area_ratio_object_to_parent']['q']['0.5']:.3f}; IoU рамки родителя с разметкой ≥ 0,5 — "
          f"{owp['parent_box_iou_with_gt_ge_0.5_share']:.3f}; масок «объект» среди потомков родителя — медиана "
          f"{owp['object_masks_among_parent_descendants']['q']['0.5']:.1f}.", "",
          "## Косинус против отношения площадей (рёбра леса с маской «объект» на одном конце)", "",
          "| энкодер | рёбер (родитель — объект / ребёнок — объект) | AUROC отношения площадей | AUROC косинуса | стратифицированный AUROC косинуса | ρ Спирмена косинус—отношение |",
          "|---|---|---|---|---|---|"]
    for e in ENCODERS:
        x = eb[e]
        L.append(f"| {e} | {x['n_parent_object']} / {x['n_child_object']} | {x['auroc_area_ratio']:.3f} | {x['auroc_cos']:.3f} | "
                 f"{x['stratified_auroc_cos']:.3f} | {x['spearman_cos_area_ratio_all_forest_edges']:.3f} |")
    L += ["", f"Вердикт по правилу (3): устойчивость эмбеддинга остаётся кандидатом — "
          f"{'да' if eb['verdict']['embedding_stability_remains_candidate'] else 'нет'}.", "",
          "## Дедупликация галереи", "",
          "| энкодер | t | эталонов | верных ŷ (полная — сокращённая) | изменилось ŷ | q0,99 s* дистракторов (полная → сокращённая) |",
          "|---|---|---|---|---|---|"]
    for e in ENCODERS:
        x = dd[e]
        for t in DEDUP_T:
            y = x["by_t"][str(t)]
            L.append(f"| {e} | {t:.2f} | {y['n_kept']} | {x['full']['correct']} — {y['correct']} | {y['y_hat_changed']} | "
                     f"{x['full']['distractor_s_star_q99']:.4f} → {y['distractor_s_star_q99']:.4f} |")
    L += ["", "Выбор по правилу (4): " + "; ".join(f"{e} — t* = {dd[e]['selection']['t_star']}" for e in ENCODERS) + "."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=SPLIT, help="только «cal»; иное — отказ")
    ap.add_argument("--part3", action="store_true", help="часть 3: правило с почти-дубликатами")
    args = ap.parse_args()
    check_split(args.split)
    t0 = time.time()
    best, cfgs, mc, sms, galleries, emb, dec, refs = A.load(args.split)
    nest = {sm.id: {th: A.Nesting(sm, th) for th in THETAS} for sm in sms}
    if args.part3:
        head = {"what": "аналитика §2.3, часть 3: правило ρ с почти-дубликатами; правило выбора (5) записано до счёта",
                "dataset": A.DATASET, "split": SPLIT, "scenes": [s.id for s in sms], "thetas": list(THETAS),
                "mask_cache": mc.dir.name, **A._STAMP}
        rr = {str(th): rule_metrics(sms, nest, th, rule_sets_part3) for th in THETAS}
        sel = select_part3(rr[str(THETA_SEL)])
        a = float(sel["best_neardup"].split("_")[1])
        bad = antichain_ok(sms, nest, a)
        A._write("rho_rules_part3", {"by_theta": rr, "selection": sel, "antichain_violations_best": bad}, head)
        (OUT / "summary_rho_part3.md").write_text(summary_part3(head, rr, sel, bad))
        print(f"ГОТОВО за {time.time() - t0:.0f} с")
        return
    head = {"what": "аналитика §2.3, часть 2: кандидатные правила ρ и дедупликации; правила выбора записаны до счёта",
            "dataset": A.DATASET, "split": SPLIT, "scenes": [s.id for s in sms], "thetas": list(THETAS),
            "best_variants": best, "runs": {e: c.run_id for e, c in cfgs.items()}, "mask_cache": mc.dir.name,
            **A._STAMP}
    rr = {str(th): rule_metrics(sms, nest, th) for th in THETAS}
    sel = select_rule(rr[str(THETA_SEL)])
    owp = objects_with_parent(sms, nest)
    A._write("rho_rules", {"by_theta": rr, "selection": sel, "objects_with_parent": owp}, head); print("rules", flush=True)
    eb = embedding_block(sms, nest, emb); A._write("rho_embedding", eb, head); print("embedding", flush=True)
    dd = dedup_block(sms, galleries, emb); A._write("gallery_dedup", dd, head); print("dedup", flush=True)
    (OUT / "summary_rho.md").write_text(summary_md(head, rr, sel, owp, eb, dd))
    print(f"ГОТОВО за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
