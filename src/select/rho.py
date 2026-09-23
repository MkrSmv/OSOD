"""Отбор гранулярности $\\rho$ — §2.3: $M^*=\\rho(M(I))$ по маскам авторежима одного изображения.

Параметры заморожены: $\\theta=9/10$, $\\gamma=4/5$ — рациональные константы, сравнения
целочисленные, одни для калибровочных и тестовых сцен. Площади и пересечения — по RLE кеша, в
разрешении входа $S$; поле `area` записи кеша (пиксели исходного снимка, число с плавающей точкой) не используется.

- вложенность: $m_a\\preceq_\\theta m_b \\iff |m_a|<|m_b|$ и $10|m_a\\cap m_b|\\ge 9|m_a|$; пара равной площади не
  вложена ни в какую сторону, обе маски проходят отбор независимо;
- вложение — почти-дубликат, если $5|m_a|\\ge 4|m_b|$, иначе существенное; проверяется для всех вложенных пар;
- группы — связные компоненты по вложениям почти-дубликатов; группа отбирается, если ни одна её маска не вложена в
  маску вне группы; от группы остаётся маска наименьшей площади, при равенстве — с меньшим номером в записи кеша.

На PKU-Market-PCB в экспериментах не применяется (`DATASETS`). $M^*$ от $\\varphi$ не зависит и хранится производным
кешем `cache/rho/` (`RhoCache`): один раз на изображение и ключ кеша масок.
"""

from __future__ import annotations

import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from pycocotools import mask as MU

from src.segment import cache as MC

THETA = Fraction(9, 10)
GAMMA = Fraction(4, 5)
RHO_RULE = "neardup_min_area_v1"
DATASETS = ("hr_insdet",)      # на PCB ρ в экспериментах не применяется
ROOT = Path("cache/rho")
FORMAT = 1
_AREA_CHUNK = 255


def geometry(rles: list[dict]) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    """Площади $|m|$ и ненулевые пересечения $|m_i\\cap m_k|$, $i<k$, — целые, по RLE входа $S$. Пересечение считается
    для пар с пересекающимися описывающими рамками RLE: у остальных оно равно нулю — отсев, а не приближение."""
    n = len(rles)
    if n == 0:
        return np.zeros(0, np.int64), {}
    # `mask.area` установленной версии pycocotools на списке длиннее 255 RLE падает (внутри — массив uint8 от числа
    # масок) — площади считаются порциями, значения те же.
    area = np.concatenate([MU.area(rles[k:k + _AREA_CHUNK]) for k in range(0, n, _AREA_CHUNK)]).astype(np.int64)
    bb = MU.toBbox(rles)
    x0, y0, x1, y1 = bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]
    inter: dict[tuple[int, int], int] = {}
    for i in range(n):
        j = np.flatnonzero((x0[i] < x1) & (x0 < x1[i]) & (y0[i] < y1) & (y0 < y1[i]))
        for k in j[j > i]:
            v = int(MU.area(MU.merge([rles[i], rles[int(k)]], intersect=True)))
            if v > 0:
                inter[i, int(k)] = v
    return area, inter


def nesting(area: np.ndarray, inter: dict[tuple[int, int], int], theta: Fraction = THETA) -> tuple[list[set[int]], int]:
    """`inside[a]` — маски, в которые вложена `a` по $\\preceq_\\theta$; второе значение — число пар равной площади с
    долей пересечения не ниже $\\theta$ (вложенными не считаются)."""
    inside: list[set[int]] = [set() for _ in range(len(area))]
    n_equal = 0
    for (i, k), v in inter.items():
        small, big = (i, k) if (int(area[i]), i) < (int(area[k]), k) else (k, i)
        if theta.denominator * v >= theta.numerator * int(area[small]):
            if area[small] == area[big]:
                n_equal += 1
            else:
                inside[small].add(big)
    return inside, n_equal


def select(area: np.ndarray, inside: list[set[int]], gamma: Fraction = GAMMA) -> tuple[list[int], dict]:
    """Отбор по группам почти-дубликатов. Возвращает номера масок $M^*$ по возрастанию и счётчики."""
    n = len(area)
    comp = list(range(n))

    def find(x: int) -> int:
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x

    n_neardup = 0
    for a in range(n):
        for b in inside[a]:
            if gamma.denominator * int(area[a]) >= gamma.numerator * int(area[b]):
                n_neardup += 1
                comp[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for m in range(n):
        groups.setdefault(find(m), []).append(m)
    out = []
    for g in groups.values():
        gs = set(g)
        if all(inside[m] <= gs for m in g):
            out.append(min(g, key=lambda m: (int(area[m]), m)))
    return sorted(out), {"n_nested_pairs": sum(len(s) for s in inside), "n_neardup_pairs": n_neardup,
                         "n_groups": len(groups)}


def rho(entry: MC.MaskEntry) -> dict:
    """$M^*$ изображения: `selected` — номера отобранных масок в порядке записи кеша — и счётчики записи отбора."""
    t0 = time.perf_counter()
    area, inter = geometry([r["rle"] for r in entry.records])
    inside, n_equal = nesting(area, inter)
    sel, stats = select(area, inside)
    return {"selected": sel, "n_masks": len(area), "n_selected": len(sel), **stats,
            "n_equal_area_duplicates": n_equal, "sec": round(time.perf_counter() - t0, 4)}


def antichain_violations(inside: list[set[int]], selected: list[int]) -> int:
    """Число пар отобранных масок, связанных $\\preceq_\\theta$ (должно быть 0)."""
    ks = set(selected)
    return sum(len(inside[i] & ks) for i in ks)


def _frac(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


class RhoCache:
    """`cache/rho/<dataset>/<каталог ключа кеша масок>__theta9-10_gamma4-5/<изображение>.json`. Запись несёт ключ кеша
    масок и sha256 файла масок, по которому посчитана; расхождение при чтении — ошибка."""

    def __init__(self, mask_cache: MC.MaskCache, root: Path | None = None):
        if mask_cache.dataset not in DATASETS:
            raise ValueError(f"ρ применяется только к {DATASETS}: на {mask_cache.dataset} M* = M(I)")
        if mask_cache.key["mode"] != "auto":
            raise ValueError("ρ применяется к маскам авторежима; эталон, заданный рамкой, имеет одну маску")
        self.mc = mask_cache
        self.key = {"format": FORMAT, "mask_key": mask_cache.key, "theta": _frac(THETA), "gamma": _frac(GAMMA),
                    "rule": RHO_RULE}
        name = f"{mask_cache.dir.name}__theta{THETA.numerator}-{THETA.denominator}_gamma{GAMMA.numerator}-{GAMMA.denominator}"
        self.dir = (ROOT if root is None else Path(root)) / mask_cache.dataset / name

    def path(self, image_id: str) -> Path:
        return self.dir / self.mc.path(image_id).name

    def has(self, image_id: str) -> bool:
        return self.path(image_id).is_file()

    def _masks_sha256(self, image_id: str) -> str:
        return hashlib.sha256(self.mc.path(image_id).read_bytes()).hexdigest()

    def compute(self, image_id: str) -> dict:
        rec = {"image_id": image_id, **rho(self.mc.load(image_id)), "key": self.key,
               "masks_sha256": self._masks_sha256(image_id)}
        self.dir.mkdir(parents=True, exist_ok=True)
        kf = self.dir / "key.json"
        if not kf.exists():
            MC._atomic_write(kf, json.dumps(self.key, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        elif json.loads(kf.read_text()) != self.key:
            raise ValueError(f"{kf}: ключ каталога не совпадает с запрошенным")
        MC._atomic_write(self.path(image_id), json.dumps(rec, ensure_ascii=False))
        return rec

    def load(self, image_id: str) -> dict:
        p = self.path(image_id)
        rec = json.loads(p.read_text())
        if rec["key"] != self.key or rec["image_id"] != image_id:
            raise ValueError(f"{p}: запись отбора сделана с другим ключом или для другого снимка")
        if rec["masks_sha256"] != self._masks_sha256(image_id):
            raise ValueError(f"{p}: файл масок изменился после отбора")
        sel = rec["selected"]
        if len(sel) != rec["n_selected"] or sel != sorted(set(sel)) or (sel and not 0 <= sel[0] <= sel[-1] < rec["n_masks"]):
            raise ValueError(f"{p}: номера отобранных масок повреждены")
        return rec

    def selected(self, image_id: str) -> np.ndarray:
        return np.asarray(self.load(image_id)["selected"], np.int64)
