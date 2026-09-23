"""Эксперимент 4.5 — отклонение и калибровка порога при росте галереи: счёт. Протокол заморожен; константы —
`src.eval.rules`.

Всё считается по сходствам одного точного поиска (`IndexFlatIP`, $k=N$) по полной галерее: матрица «маска — строка
галереи» в порядке строк (`dense`); подвыборка галереи — подмножество столбцов, $\\hat y$ и $s^*$ при ней — тем же
`search.decide`, что везде. Калибровка подвыборки — теми же функциями `src.calib.threshold`, что `calib.json`
(`tau_q`, `Tail`, `solve_kappa`), поэтому при 100 экземплярах она обязана совпасть с записью калибровки галереи точно.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.calib import threshold as TH
from src.eval import detect as DT
from src.eval import rules as RU
from src.gallery import store
from src.search import decide as DE


def dense(sim: np.ndarray, idx: np.ndarray, n_rows: int) -> np.ndarray:
    """Выход точного поиска при $k=N$ (сходства по убыванию и строки галереи) → матрица (запросы × строки галереи)."""
    sim, idx = np.asarray(sim, np.float32), np.asarray(idx, np.int64)
    if sim.shape != idx.shape or sim.ndim != 2 or sim.shape[1] != n_rows \
            or not (np.sort(idx, 1) == np.arange(n_rows)).all():
        raise ValueError(f"поиск при k = N по всей галерее: сходства {sim.shape}, строк {n_rows}")
    out = np.empty((len(sim), n_rows), np.float32)
    np.put_along_axis(out, idx, sim, 1)
    return out


def answers(S: np.ndarray, rows: np.ndarray, label_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """$\\hat y$ (без порога) и $s^*$ при галерее из строк `rows`: то же правило решения, что в прогонах."""
    rows = np.asarray(rows, np.int64)
    if not len(S):
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    return DE.decide(S[:, rows], np.broadcast_to(rows, (len(S), len(rows))), label_ids)


@dataclass
class Img:
    """Снимок оценки: маски, по которым принимаются решения ($M^*$)."""
    id: str
    group: str               # easy / hard
    S: np.ndarray            # (n, N) — сходства со всеми строками полной галереи
    mask_no: np.ndarray      # (n,) номера масок в записи кеша — развязка равных $s^*$ у top-K
    distractor: np.ndarray   # (n,) bool — IoU рамки < 0,1 ко всем размеченным рамкам
    gt_label: np.ndarray     # (n,) номер метки рамки, которой маска назначена оракулом среди этих масок; иначе −1


# ---------------------------------------------------------------------- калибровка подвыборки


def calibrate_rows(S_cal: np.ndarray, rows: np.ndarray, eps_levels, delta: float) -> dict:
    """$\\tau^{\\mathrm q}$, хвост $F$ и $\\kappa$ по $\\mathcal D_{\\mathrm{cal}}$ при галерее из строк `rows` ($N_0$ = их
    число). $\\kappa$ вне $[10^{-3},1]$ — записываемый исход ячейки (`kappa_out_of_range`), не ошибка."""
    sim = np.ascontiguousarray(S_cal[:, np.asarray(rows, np.int64)])
    s_star = sim.max(1)
    tail = TH.Tail.from_pairs(sim)
    n0 = int(sim.shape[1])
    by_eps = {}
    for eps in eps_levels:
        tq = TH.tau_q(s_star, eps)
        lo, hi = TH.tau_m(tail, eps, TH.KAPPA_MIN, n0)[0], TH.tau_m(tail, eps, TH.KAPPA_MAX, n0)[0]
        rec = {"tau_q": tq, "check_at_N0": TH.control_check(s_star, tq, eps, delta)}
        if lo <= tq <= hi:
            sol = TH.solve_kappa(tail, eps, n0, tq)
            rec.update(kappa=sol["kappa"], tau_m=sol["tau_m"], kappa_N0=sol["kappa"] * n0, residual=sol["residual"],
                       f_inv_path=sol["f_inv_path"], kappa_out_of_range=False)
        else:
            rec.update(kappa=None, tau_m=None, kappa_N0=None, kappa_out_of_range=True, tau_m_at_bounds=[lo, hi])
        by_eps[store.eps_key(eps)] = rec
    return {"N0": n0, "tail": tail, "s_star_cal": s_star, "by_eps": by_eps}


def tail_summary(tail: TH.Tail) -> dict:
    return {"n_pairs": tail.n, "max": float(tail.top[0]),
            "quantiles_0.5_0.9_0.99_0.999": [float(np.interp(q, tail.levels, tail.q)) for q in (0.5, 0.9, 0.99, 0.999)]}


def model_tau(cal: dict, eps: float, n_rows: int) -> float | None:
    """$\\tau^{\\mathrm m}_N=F_{n_0}^{-1}\\big(1-(1-\\varepsilon)^{1/(\\kappa_{n_0}N)}\\big)$; без $\\kappa$ — `None`."""
    r = cal["by_eps"][store.eps_key(eps)]
    return None if r["kappa_out_of_range"] else TH.tau_m(cal["tail"], eps, r["kappa"], n_rows)[0]


def predicted_ratio_frozen(eps: float, n_rows: int, n0_rows: int) -> float:
    """Предсказанное моделью §2.4 отношение $\\mathrm{FPR}(N)/\\mathrm{FPR}(N_0)$ порога, поставленного при $N_0$ на уровне
    $\\varepsilon$ и не пересчитанного: $(1-(1-\\varepsilon)^{N/N_0})/\\varepsilon$ ($\\kappa$ сокращается)."""
    return float(-np.expm1(np.log1p(-eps) * n_rows / n0_rows) / eps)


def new_exemplar_tail(S_cal: np.ndarray, rows_n0: np.ndarray, rows_n: np.ndarray, levels: dict[str, float | None]) -> dict:
    """Хвост пар «дистрактор — новый эталон» ($G_n\\setminus G_{n_0}$) против пар при $G_{n_0}$: отношение долей пар не
    ниже уровня. Описательно — проверка допущения «новые эталоны имеют тот же хвост $F$»."""
    new = np.setdiff1d(rows_n, rows_n0)
    a, b = S_cal[:, new], S_cal[:, np.asarray(rows_n0, np.int64)]
    out = {"n_new_rows": int(len(new))}
    for name, t in levels.items():
        if t is None or not len(new):
            out[name] = None
            continue
        fa, fb = float((a >= t).mean()), float((b >= t).mean())
        out[name] = {"level": float(t), "share_new": fa, "share_n0": fb, "ratio": fa / fb if fb > 0 else None}
    return out


# ---------------------------------------------------------------------- измерение на снимках оценки


def top_k_answered(s_star: np.ndarray, mask_no: np.ndarray, k: int) -> np.ndarray:
    """Ответ получают `k` масок снимка с наибольшим $s^*$; при равных $s^*$ — меньший номер маски."""
    order = np.lexsort((np.asarray(mask_no), -np.asarray(s_star, np.float64)))
    out = np.zeros(len(s_star), bool)
    out[order[:k]] = True
    return out


def _ci(point_num: np.ndarray, point_den: np.ndarray, W: np.ndarray | None) -> list[float] | None:
    if W is None:
        return None
    num, den = W @ point_num, W @ point_den
    if (den <= 0).any():
        raise ValueError("повтор бутстрэпа без дистракторов")
    lo, hi = np.quantile(num / den, [0.025, 0.975])
    return [float(lo), float(hi)]


def measure(imgs: list[Img], ans: list[tuple[np.ndarray, np.ndarray]], in_subset: np.ndarray, tau: float | None,
            top_k: int | None = None, W: np.ndarray | None = None, sel: np.ndarray | None = None) -> dict:
    """Величины одной ячейки на снимках `sel` (по умолчанию — все): частота ложных срабатываний на дистракторах (с
    интервалом, если даны веса `W` — (повторы × снимки `imgs`)), доля пропусков и доля верных ответов на известных
    масках, доля ответов на масках объектов, чьих меток в подвыборке нет. `in_subset` — (|Y|,) bool: метка в подвыборке.
    Ответ $\\hat y\\ne\\varnothing$: $s^*\\ge\\tau$ либо, у top-K, попадание в `top_k` масок снимка."""
    if (tau is None) == (top_k is None):
        raise ValueError("ячейка задаётся порогом либо числом K")
    n = len(imgs)
    fp, nd = np.zeros(n), np.zeros(n)
    n_known = n_miss = n_correct = n_unseen = n_unseen_answered = 0
    for k, (im, (y, s)) in enumerate(zip(imgs, ans)):
        answered = s >= tau if top_k is None else top_k_answered(s, im.mask_no, top_k)
        fp[k], nd[k] = (answered & im.distractor).sum(), im.distractor.sum()
        if sel is not None and k not in sel:
            continue
        has = im.gt_label >= 0
        known = has & in_subset[np.maximum(im.gt_label, 0)]
        unseen = has & ~known
        n_known += int(known.sum())
        n_miss += int((known & ~answered).sum())
        n_correct += int((known & answered & (y == im.gt_label)).sum())
        n_unseen += int(unseen.sum())
        n_unseen_answered += int((unseen & answered).sum())
    keep = np.ones(n, bool) if sel is None else np.isin(np.arange(n), sel)
    out = {"n_distractors": int(nd[keep].sum()), "n_false_positive": int(fp[keep].sum()),
           "fpr": float(fp[keep].sum() / nd[keep].sum()) if nd[keep].sum() else None,
           "n_known": n_known, "miss_rate": n_miss / n_known if n_known else None,
           "correct_rate": n_correct / n_known if n_known else None,
           "unseen_objects": {"n": n_unseen, "answered_rate": n_unseen_answered / n_unseen if n_unseen else None}}
    if W is not None:
        if sel is not None:
            raise ValueError("интервал считается по всем снимкам оценки")
        out["fpr_ci95"] = _ci(fp, nd, W)
    return out


def auroc_known_unknown(imgs: list[Img], ans, in_subset: np.ndarray, sel: np.ndarray | None = None) -> dict:
    """AUROC «известный — неизвестный» по $s^*$: известные — оракульные маски меток подвыборки, неизвестные — дистракторы."""
    score, known = [], []
    for k, (im, (_, s)) in enumerate(zip(imgs, ans)):
        if sel is not None and k not in sel:
            continue
        kn = (im.gt_label >= 0) & in_subset[np.maximum(im.gt_label, 0)]
        score += [s[kn], s[im.distractor]]
        known += [np.ones(int(kn.sum()), bool), np.zeros(int(im.distractor.sum()), bool)]
    score, known = np.concatenate(score).astype(float), np.concatenate(known)
    if not known.any() or known.all():
        return {"auroc": None, "n_known": int(known.sum()), "n_unknown": int((~known).sum())}
    a = DT.auroc(score, known, np.zeros(len(score), np.int64), np.ones((1, 1)))[0]
    return {"auroc": float(a), "n_known": int(known.sum()), "n_unknown": int((~known).sum())}


# ---------------------------------------------------------------------- сбор вердикта


def collect_verdict(holds: dict) -> dict:
    """`holds[eps][rule][r][(n0, n)]` → у каждого ε и правила: по цепочкам (`rules.chain_holds`) и по большинству."""
    out = {}
    for eps, by_rule in holds.items():
        out[eps] = {}
        for rule, by_chain in by_rule.items():
            flags = [RU.chain_holds(by_chain[r]) for r in sorted(by_chain)]
            out[eps][rule] = {**RU.majority_holds(flags), "by_chain": flags}
    return out


def pair_key(n0: int, n: int) -> str:
    return f"{n0}->{n}"
