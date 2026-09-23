"""Рамки: IoU и полнота по размеченным рамкам. Рамки исключающие, (x0, y0, x1, y1)."""

from __future__ import annotations

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU всех пар рамок: форма (len(a), len(b))."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def best_iou_per_gt(gt_boxes: np.ndarray, mask_boxes: np.ndarray) -> np.ndarray:
    """Для каждой размеченной рамки — наибольший IoU с $\\mathrm{box}(m)$ по всем маскам $M(I)$; без масок — нули."""
    iou = box_iou(gt_boxes, mask_boxes)
    return iou.max(1) if iou.shape[1] else np.zeros(len(iou))
