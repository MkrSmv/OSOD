"""Правила калибровки порога — §2.4.

Правило по квантили: $\\tau^{\\mathrm q}_N$ — $(1-\\varepsilon)$-квантиль $s^*$ по $\\mathcal D_{\\mathrm{cal}}$ при текущей
галерее. Правило по модели: $\\tau^{\\mathrm m}_N=F^{-1}\\big(1-(1-\\varepsilon)^{1/(\\kappa N)}\\big)$, где
$F(\\tau)=\\Pr\\{s(z,z_j)\\ge\\tau\\}$ — хвост (функция выживания, убывает по $\\tau$) сходства «дистрактор — эталон» по
всем парам при $N_0$, а $\\kappa\\in(0,1]$ решает $\\tau^{\\mathrm m}_{N_0}=\\tau^{\\mathrm q}_{N_0}$. Контрольная проверка —
доля элементов $\\mathcal D_{\\mathrm{chk}}$ (вся $\\mathcal D_{\\mathrm{cal}}$, те же маски, по которым поставлен порог) с
$s^*\\ge\\tau$ против $\\varepsilon(1+\\delta)$; для правила по квантили — ещё $N/N_0>1+\\delta$.

Обе квантили — линейная интерполяция по порядковым номерам (`np.quantile`, метод по умолчанию): $F^{-1}(p)$ есть
квантиль уровня $1-p$ пар, $\\tau^{\\mathrm q}$ — квантиль уровня $1-\\varepsilon$ значений $s^*$; при таком определении
$\\tau^{\\mathrm m}(\\kappa)$ непрерывна и не убывает по $\\kappa$, и $\\kappa$ определён однозначно.
Хранение $F$ — `src.gallery.store.check_calib`: наибольшие сходства точно и сетка квантилей.
"""

from __future__ import annotations

import math

import numpy as np

from src.gallery import store

KAPPA_MIN, KAPPA_MAX = 1e-3, 1.0  # $\kappa\in[10^{-3},1]$
KAPPA_LOG_TOL = 1e-12             # бисекция по $\log\kappa$ — до относительной ширины $10^{-12}$
# контрольная проверка: доля против ε(1+δ), без критерия
CHECK_RULE = {"set": "D_cal", "test": "fraction_vs_limit", "recalibrate_if": "k / |D_chk| > eps * (1 + delta), k = #{s* >= tau}"}


def mask_id(scene_id: str, mask_no: int) -> str:
    """Идентификатор маски $\\mathcal D$: `<сцена>#<номер маски в записи кеша>`."""
    return f"{scene_id}#{int(mask_no)}"


def parse_mask_id(mid: str) -> tuple[str, int]:
    scene, _, k = mid.rpartition("#")
    if not scene or not k.isdigit():
        raise ValueError(f"идентификатор маски {mid!r}: ожидается «<сцена>#<номер>»")
    return scene, int(k)


class Tail:
    """Хвост $F$ по всем парам «дистрактор — эталон»: `top` — наибольшие сходства по убыванию, `quantiles` — сетка
    квантилей уровней `levels` по возрастанию; `n_pairs` — число пар."""

    def __init__(self, n_pairs: int, top: np.ndarray, quantiles: np.ndarray):
        self.n = int(n_pairs)
        self.top = np.asarray(top, np.float64)
        self.q = np.asarray(quantiles, np.float64)
        self.levels = np.linspace(0.0, 1.0, len(self.q))

    @classmethod
    def from_pairs(cls, pairs: np.ndarray) -> "Tail":
        x = np.sort(np.asarray(pairs, np.float64).ravel())
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("пары «дистрактор — эталон»: пусто либо не-конечные значения")
        q = np.quantile(x, np.linspace(0.0, 1.0, store.F_QUANTILES))
        return cls(len(x), x[::-1][:store.F_TOP].copy(), q)

    @classmethod
    def from_dict(cls, d: dict) -> "Tail":
        return cls(d["n_pairs"], d["top"], d["quantiles"])

    def to_dict(self) -> dict:
        return {"n_pairs": self.n, "top": [float(v) for v in self.top], "quantiles": [float(v) for v in self.q]}

    def _asc(self, i: int) -> float:
        """Значение с порядковым номером `i` по возрастанию; только для номеров, хранимых в `top`."""
        return float(self.top[self.n - 1 - i])

    def inverse(self, p: float) -> tuple[float, str]:
        """$F^{-1}(p)$ — квантиль уровня $1-p$ пар с линейной интерполяцией; второе значение — путь: `top` (точно,
        оба соседних порядковых номера в хранимых наибольших) либо `quantiles` (интерполяция по сетке)."""
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"F^-1: p = {p} вне [0, 1]")
        q = 1.0 - p
        h = (self.n - 1) * q
        lo = min(int(math.floor(h)), self.n - 1)
        g = h - lo
        if lo >= self.n - len(self.top):
            a = self._asc(lo)
            return (a if lo == self.n - 1 else a + g * (self._asc(lo + 1) - a)), "top"
        t = q * (len(self.q) - 1)
        i = min(int(math.floor(t)), len(self.q) - 2)
        w = t - i
        return float(self.q[i] + w * (self.q[i + 1] - self.q[i])), "quantiles"

    def survival(self, tau: float) -> float:
        """$F(\\tau)=\\Pr\\{s\\ge\\tau\\}$: точно, если $\\tau$ выше наименьшего хранимого из `top`; иначе — по сетке
        квантилей (справочно; калибровка обращается к $F$ только через `inverse`)."""
        if tau > self.top[-1] or len(self.top) == self.n:
            return float(np.count_nonzero(self.top >= tau)) / self.n
        return float(1.0 - np.interp(tau, self.q, self.levels))


def tau_q(s_star: np.ndarray, eps: float) -> float:
    """Правило по квантили: $(1-\\varepsilon)$-квантиль $s^*$ по $\\mathcal D_{\\mathrm{cal}}$, линейная интерполяция."""
    s = np.asarray(s_star, np.float64)
    if not len(s) or not np.isfinite(s).all():
        raise ValueError("s* дистракторов: пусто либо не-конечные значения")
    return float(np.quantile(s, 1.0 - eps))


def p_single(eps: float, kappa: float, n: int) -> float:
    """$1-(1-\\varepsilon)^{1/(\\kappa N)}$ — уровень хвоста одного сравнения; через `expm1`/`log1p` без потери точности."""
    return float(-math.expm1(math.log1p(-eps) / (kappa * n)))


def tau_m(tail: Tail, eps: float, kappa: float, n: int) -> tuple[float, str]:
    """Правило по модели: $\\tau^{\\mathrm m}_N=F^{-1}\\big(1-(1-\\varepsilon)^{1/(\\kappa N)}\\big)$."""
    return tail.inverse(p_single(eps, kappa, n))


def solve_kappa(tail: Tail, eps: float, n0: int, target: float) -> dict:
    """$\\kappa$ из $\\tau^{\\mathrm m}_{N_0}(\\kappa)=\\tau^{\\mathrm q}_{N_0}$ бисекцией по $\\log\\kappa$ на
    $[10^{-3},1]$. $\\tau^{\\mathrm m}$ не убывает по $\\kappa$; берётся правый конец итогового отрезка, то есть
    $\\tau^{\\mathrm m}_{N_0}\\ge\\tau^{\\mathrm q}_{N_0}$. Корня на отрезке нет — ошибка: $\\kappa$ не обрезается."""
    lo, hi = math.log(KAPPA_MIN), math.log(KAPPA_MAX)
    t_lo, s_lo = tau_m(tail, eps, KAPPA_MIN, n0)
    t_hi, s_hi = tau_m(tail, eps, KAPPA_MAX, n0)
    if not t_lo <= target <= t_hi:
        raise ValueError(f"ε = {eps}: τ_q = {target:.6f} вне [τ_m(κ={KAPPA_MIN:g}) = {t_lo:.6f}, "
                         f"τ_m(κ={KAPPA_MAX:g}) = {t_hi:.6f}] — κ на отрезке не определён")
    n_iter = 0
    while hi - lo > KAPPA_LOG_TOL:
        mid = 0.5 * (lo + hi)
        if tau_m(tail, eps, math.exp(mid), n0)[0] < target:
            lo = mid
        else:
            hi = mid
        n_iter += 1
    kappa = min(math.exp(hi), KAPPA_MAX)
    t, src = tau_m(tail, eps, kappa, n0)
    return {"kappa": kappa, "tau_m": t, "residual": t - target, "n_iter": n_iter, "f_inv_path": src,
            "f_inv_path_at_bounds": [s_lo, s_hi], "tau_m_at_bounds": [t_lo, t_hi],
            "p_single": p_single(eps, kappa, n0)}


def check_set(n_cal: int) -> np.ndarray:
    """$\\mathcal D_{\\mathrm{chk}}$ — вся $\\mathcal D_{\\mathrm{cal}}$ в её порядке."""
    if n_cal <= 0:
        raise ValueError("D_cal пуста")
    return np.arange(n_cal)


def control_check(s_star_chk: np.ndarray, tau: float, eps: float, delta: float) -> dict:
    """Контрольная проверка после пополнения: $k$ — число элементов с $s^*\\ge\\tau$; перекалибровка,
    если $k/n>\\varepsilon(1+\\delta)$. Набор — те же маски, по которым поставлен порог: при $N_0$ доля равна
    $\\varepsilon$ по построению, и проверка измеряет сдвиг от пополнения. Сравнение — в рациональных числах;
    `k_trigger` — наименьшее $k$, при котором проверка сработала бы."""
    from fractions import Fraction

    s = np.asarray(s_star_chk, np.float64)
    n, n_ge = int(len(s)), int(np.count_nonzero(s >= tau))
    lim = Fraction(str(eps)) * (1 + Fraction(str(delta)))
    k_trigger = int(lim * n) + 1                      # наименьшее k с k/n > lim
    return {"n": n, "n_ge_tau": n_ge, "frac": n_ge / n, "limit": float(lim), "k_trigger": k_trigger,
            "recalibrate": bool(Fraction(n_ge, n) > lim)}


def growth_requires_recalibration(n: int, n0: int, delta: float) -> bool:
    """Условие перекалибровки правила по квантили: $N/N_0>1+\\delta$ (§2.4)."""
    return bool(n / n0 > 1 + delta)


def calibrate(sim: np.ndarray, s_star: np.ndarray, ids: list[str], eps_levels: tuple[float, ...],
              delta: float) -> tuple[dict, dict]:
    """Калибровка при $N_0$ по матрице сходств `sim` ($|\\mathcal D_{\\mathrm{cal}}|\\times N_0$, точный перебор, все
    пары) и $s^*$ дистракторов. Возвращает запись для `calib.json` (без `format`, `D_source` и полей галереи) и
    сводку: невязки, пути $F^{-1}$, контрольную проверку при $N_0$."""
    sim = np.asarray(sim)
    if sim.ndim != 2 or len(sim) != len(ids) or len(s_star) != len(ids):
        raise ValueError(f"матрица сходств {sim.shape} при {len(ids)} дистракторах и {len(s_star)} значениях s*")
    if not np.array_equal(np.asarray(s_star, np.float64), sim.max(1).astype(np.float64)):
        raise AssertionError("s* дистрактора не равен максимуму его строки сходств")
    n0 = int(sim.shape[1])
    tail = Tail.from_pairs(sim)
    chk = check_set(len(ids))
    by_eps, summary = {}, {}
    for eps in eps_levels:
        tq = tau_q(s_star, eps)
        sol = solve_kappa(tail, eps, n0, tq)
        by_eps[store.eps_key(eps)] = {"tau_q": tq, "tau_m": sol["tau_m"], "kappa": sol["kappa"]}
        summary[store.eps_key(eps)] = {
            **sol, "tau_q": tq, "n_cal_ge_tau_q": int(np.count_nonzero(np.asarray(s_star, np.float64) >= tq)),
            "check_at_N0": {rule: control_check(np.asarray(s_star)[chk], t, eps, delta)
                            for rule, t in (("tau_q", tq), ("tau_m", sol["tau_m"]))}}
    rec = {"N0": n0, "F": tail.to_dict(), "delta": float(delta), "check_set_ids": [ids[i] for i in chk],
           "check_rule": CHECK_RULE, "by_eps": by_eps}
    return rec, summary
