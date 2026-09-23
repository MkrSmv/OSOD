"""Индексы Faiss по внутреннему произведению; §2.5.

Точный перебор (`IndexFlatIP`) — единственный источник метрик качества; HNSW — только задержка и полнота
относительно точного. Оба строятся по строкам галереи `rows` и возвращают номера строк галереи, а не индекса.
Точный индекс строится без помеченных `deleted` (`Gallery.rows`); HNSW удаления не поддерживает, поэтому помеченные
после построения эталоны остаются в графе и исключаются при вычислении $s_y$ (`decide`, аргумент `deleted`).
"""

from __future__ import annotations

import faiss
import numpy as np


def as_faiss(x: np.ndarray, d: int | None = None) -> np.ndarray:
    """Вход Faiss — только float32, C-contiguous, (n, d); из fp16 приводится явно."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2 or (d is not None and x.shape[1] != d):
        raise ValueError(f"вход Faiss формы {x.shape}, ожидается (n, {d})")
    if not np.isfinite(x).all():  # на NaN Faiss молча возвращает бессмысленных соседей
        raise FloatingPointError("не-конечные значения на входе Faiss")
    return x


class _Index:
    index: faiss.Index

    def __init__(self, emb: np.ndarray, rows: np.ndarray | None):
        self.rows = np.arange(len(emb), dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
        self.d = emb.shape[1]

    def _map(self, i: np.ndarray) -> np.ndarray:
        """Номера индекса → номера строк галереи; `−1` (нехватка соседей) остаётся `−1`."""
        return np.where(i >= 0, self.rows[np.maximum(i, 0)], -1)

    def __len__(self) -> int:
        return self.index.ntotal


class ExactIndex(_Index):
    def __init__(self, emb: np.ndarray, rows: np.ndarray | None = None):
        super().__init__(emb, rows)
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(as_faiss(emb[self.rows], self.d))

    def search(self, q: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Сходства по убыванию и строки галереи. По умолчанию $k=N$: при точном переборе $s_y$ вычисляется
        по всей галерее (§2.5)."""
        k = len(self) if k is None else min(k, len(self))
        dist, i = self.index.search(as_faiss(q, self.d), k)
        return dist, self._map(i)


class HnswIndex(_Index):
    def __init__(self, emb: np.ndarray, rows: np.ndarray | None = None, M: int = 32, ef_construction: int = 200,
                 ef_search_min: int = 64):
        super().__init__(emb, rows)
        # третий аргумент обязателен: без него метрика молча L2
        self.index = faiss.IndexHNSWFlat(self.d, M, faiss.METRIC_INNER_PRODUCT)
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise RuntimeError("HNSW построен не по внутреннему произведению")
        self.index.hnsw.efConstruction = ef_construction  # до add: после него значение на граф уже не влияет
        self.ef_search_min = ef_search_min
        self.index.add(as_faiss(emb[self.rows], self.d))

    def prepare(self, k: int) -> int:
        """`efSearch = max(k, ef_search_min)` — перед `search`; возвращает фактическое $k\\le N$."""
        k = min(k, len(self))
        self.index.hnsw.efSearch = max(k, self.ef_search_min)
        return k

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = self.prepare(k)
        dist, i = self.index.search(as_faiss(q, self.d), k)
        return dist, self._map(i)
