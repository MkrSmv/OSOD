"""Правило решения — §2.1.

$s_y(z)=\\max_{j\\in\\mathcal N_k(z):\\,y_j=y}s(z,z_j)$, $s^*(z)=\\max_y s_y(z)$, $\\hat y=\\arg\\max_y s_y(z)$ при
$s^*\\ge\\tau$, иначе $\\varnothing$. Соседи — выход индекса (`src.search.index`): при точном переборе — вся
галерея, у HNSW — $k$ кандидатов. Решение по каждой маске независимо: строки запросов не взаимодействуют,
взаимно-однозначного сопоставления масок с эталонами нет.
"""

from __future__ import annotations

import numpy as np

UNKNOWN = -1  # ответ $\varnothing$; номер метки — индекс в `Gallery.labels`


def _valid(idx: np.ndarray, deleted: np.ndarray | None) -> np.ndarray:
    """Кандидаты, участвующие в $s_y$: не `−1` (нехватка соседей) и не помеченные `deleted` (§2.5)."""
    valid = idx >= 0
    if deleted is not None:
        valid &= ~np.asarray(deleted, bool)[np.maximum(idx, 0)]
    return valid


def s_by_label(sim: np.ndarray, idx: np.ndarray, label_ids: np.ndarray, n_labels: int,
               deleted: np.ndarray | None = None) -> np.ndarray:
    """$s_y(z)$, форма (n, |Y|). Если ни один эталон метки не вошёл в соседи, $s_y$ не определена — `−inf`:
    метка на ответ не претендует (§2.1)."""
    valid = _valid(idx, deleted)
    out = np.full((len(sim), n_labels), -np.inf, np.float32)
    q, c = np.nonzero(valid)
    np.maximum.at(out, (q, label_ids[idx[q, c]]), sim[q, c])
    return out


def decide(sim: np.ndarray, idx: np.ndarray, label_ids: np.ndarray, tau: float = -np.inf,
           deleted: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """$(\\hat y, s^*)$ для каждого запроса; $\\hat y$ = `UNKNOWN` при $s^*<\\tau$ и при отсутствии кандидатов.

    `sim`, `idx` — (n, k): сходства и строки галереи; `label_ids` — метка каждой строки галереи. При равных
    сходствах у разных меток берётся метка с меньшим номером: ответ не зависит от порядка выдачи индекса.
    """
    valid = _valid(idx, deleted)
    s = np.where(valid, sim, -np.inf)
    s_star = s.max(1) if s.shape[1] else np.full(len(s), -np.inf, np.float32)
    lab = np.where(valid & (s == s_star[:, None]), label_ids[np.maximum(idx, 0)], np.iinfo(np.int64).max)
    y = lab.min(1) if lab.shape[1] else np.full(len(s), UNKNOWN, np.int64)
    y_hat = np.where(np.isfinite(s_star) & (s_star >= tau), y, UNKNOWN)
    return y_hat.astype(np.int64), s_star.astype(np.float32)


def detections(boxes: np.ndarray, y_hat: np.ndarray, s_star: np.ndarray) -> dict[str, np.ndarray]:
    """$D(I)$: рамки $\\mathrm{box}(m)$, метки и оценки $s^*$ масок с $\\hat y\\ne\\varnothing$."""
    keep = y_hat != UNKNOWN
    return {"box": np.asarray(boxes, float).reshape(-1, 4)[keep], "label_id": y_hat[keep], "score": s_star[keep],
            "mask_no": np.flatnonzero(keep)}
