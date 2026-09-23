"""Оракульный отбор масок и дистракторы (§4.1)."""

from __future__ import annotations

import numpy as np

from src.eval.boxes import box_iou

ORACLE_IOU = 0.5
DISTRACTOR_IOU_MAX = 0.1


def oracle_assign(gt_boxes: np.ndarray, mask_boxes: np.ndarray, iou_min: float = ORACLE_IOU) -> np.ndarray:
    """Для каждой размеченной рамки — номер маски либо −1 (пропуск сегментатора).

    Назначение взаимно-однозначное, жадное по убыванию IoU: каждая маска назначается не более одного раза,
    рамка получает лучшую из ещё не занятых масок при IoU ≥ `iou_min`. При равных IoU порядок — по номеру
    рамки, затем маски (устойчивая сортировка), то есть детерминирован.
    """
    iou = box_iou(gt_boxes, mask_boxes)
    out = np.full(len(iou), -1, int)
    g, m = np.nonzero(iou >= iou_min)
    used = set()
    for k in np.argsort(-iou[g, m], kind="stable"):
        if out[g[k]] < 0 and m[k] not in used:
            out[g[k]] = m[k]
            used.add(m[k])
    return out


def distractors(gt_boxes: np.ndarray, mask_boxes: np.ndarray, iou_max: float = DISTRACTOR_IOU_MAX,
                ignore_boxes: np.ndarray | None = None) -> np.ndarray:
    """Булев признак дистрактора: IoU описывающей рамки с каждой размеченной рамкой < `iou_max` (§2.4).

    `ignore_boxes` — зоны игнорирования: маска, чья рамка
    пересекает зону, в дистракторы не входит.
    """
    iou = box_iou(mask_boxes, gt_boxes)
    out = iou.max(1) < iou_max if iou.shape[1] else np.ones(len(iou), bool)
    if ignore_boxes is not None and len(ignore_boxes):
        out &= ~(box_iou(mask_boxes, ignore_boxes) > 0).any(1)
    return out
