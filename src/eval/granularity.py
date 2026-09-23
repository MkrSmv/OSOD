"""Описательные величины 4.4 рядом с AP: полнота после отбора в определении 4.2 и кратные
детекции. Всё — точечные оценки по подмножеству масок каждой сцены; вердикта по ним нет."""

from __future__ import annotations

import numpy as np

from src.eval import recall as R
from src.eval import rules as RU
from src.eval.boxes import box_iou

MULT_IOU = RU.RHO_MULT_IOU


def _iou(scene, keep) -> np.ndarray:
    """IoU (размеченные рамки × отобранные маски) сцены; `keep` — номера масок либо `None` (все $M(I)$)."""
    boxes = scene.boxes if keep is None else scene.boxes[np.asarray(keep, np.int64)]
    gb = scene.gt_boxes
    return box_iou(gb, boxes) if len(gb) and len(boxes) else np.zeros((len(gb), len(boxes)))


def recall_after_selection(scenes: list, keep_masks: list | None, sel: np.ndarray) -> dict:
    """Полнота в определении 4.2: доля размеченных рамок, для которых среди масок есть маска с IoU
    рамок не ниже порога, — при 0,5, 0,75 и в среднем по 0,5:0,95; сцены — `sel` (индексы)."""
    best = []
    for k in sel:
        iou = _iou(scenes[k], None if keep_masks is None else keep_masks[k])
        best.append(iou.max(1) if iou.shape[1] else np.zeros(iou.shape[0]))
    best = np.concatenate(best) if best else np.zeros(0)
    if not len(best):
        return {"n_gt": 0, "recall_50": None, "recall_75": None, "recall_50_95": None}
    return {"n_gt": int(len(best)), "recall_50": float((best >= 0.5).mean()), "recall_75": float((best >= 0.75).mean()),
            "recall_50_95": float((best[:, None] >= R.IOU_THRS[None]).mean())}


def multiplicity(scenes: list, keep_masks: list | None, y_hat: list, sel: np.ndarray) -> dict:
    """Кратные детекции по сценам `sel`: (а) доля размеченных рамок, которым соответствуют две маски и больше с IoU
    рамок ≥ 0,5 — определение §2.3, без меток, основное; (б) доля рамок с двумя и больше детекциями, у которых
    $\\hat y$ равен метке рамки, при IoU ≥ 0,5 — справочно."""
    n_gt = a = b = 0
    for k in sel:
        s = scenes[k]
        keep = None if keep_masks is None else np.asarray(keep_masks[k], np.int64)
        hit = _iou(s, keep) >= MULT_IOU
        y = y_hat[k] if keep is None else y_hat[k][keep]
        n_gt += len(s.gt)
        a += int((hit.sum(1) >= 2).sum())
        b += int(((hit & (y[None, :] == s.gt_labels[:, None])).sum(1) >= 2).sum())
    return {"n_gt": n_gt, "n_gt_with_2plus_masks": a, "share_gt_with_2plus_masks": a / n_gt if n_gt else None,
            "n_gt_with_2plus_detections_of_its_label": b,
            "share_gt_with_2plus_detections_of_its_label": b / n_gt if n_gt else None}
