"""Ориентиры отбора после поиска для 4.4, блок `post_search`.

Оба правила работают по эмбеддингам всех масок $M(I)$ — после кодирования и поиска, в отличие от $\\rho$; вердикта по
ним нет. Свободных параметров, кроме замороженного $\\theta$ и константы `rules.POST_NMS_IOU`, нет; метки $\\hat y$ в
правилах не участвуют.
"""

from __future__ import annotations

import numpy as np

from src.eval.boxes import box_iou


def chain_max(inside: list[set[int]], s_star: np.ndarray) -> np.ndarray:
    """Выбор уровня по наибольшему $s^*$ в цепочке вложенности. Маски сравнимы, если одна вложена в другую по
    $\\preceq_\\theta$ (`inside` — из `src.select.rho.nesting`, то же отношение, что у $\\rho$). Маска остаётся, если среди
    сравнимых с ней нет маски с большим $s^*$; при равных $s^*$ остаётся маска с меньшим номером в записи кеша."""
    n = len(inside)
    if len(s_star) != n:
        raise ValueError(f"масок {n}, оценок {len(s_star)}")
    comparable: list[set[int]] = [set(s) for s in inside]
    for a, outer in enumerate(inside):
        for b in outer:
            comparable[b].add(a)
    keep = [m for m in range(n)
            if not any(s_star[c] > s_star[m] or (s_star[c] == s_star[m] and c < m) for c in comparable[m])]
    return np.asarray(keep, np.int64)


def box_nms(boxes: np.ndarray, s_star: np.ndarray, iou_thr: float) -> np.ndarray:
    """Подавление немаксимумов по описывающим рамкам: жадно по убыванию $s^*$ (при равенстве — меньший номер), без учёта
    меток; маска подавляется при IoU рамки с уже оставленной выше `iou_thr`. Возвращает номера по возрастанию."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if len(s_star) != len(boxes):
        raise ValueError(f"рамок {len(boxes)}, оценок {len(s_star)}")
    if not len(boxes):
        return np.zeros(0, np.int64)
    iou = box_iou(boxes, boxes)
    order = sorted(range(len(boxes)), key=lambda i: (-float(s_star[i]), i))
    keep: list[int] = []
    for i in order:
        if not keep or not (iou[i, keep] > iou_thr).any():
            keep.append(i)
    return np.asarray(sorted(keep), np.int64)
