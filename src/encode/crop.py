"""Вырезка $C(b,\\alpha)$: заполнение в исходном разрешении, затем масштабирование до 448."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

CROP_SIZE = 448
SIGMA_FRAC = 0.05  # $\sigma_b$ — доля стороны квадрата до масштабирования
# $\bar x$ — среднее нормировки энкодера в uint8; у DINOv2 и DINOv3 нормировка одна.
MEAN_RGB = (124, 116, 104)
FILLERS = ("0", "mean", "blur")
ALPHAS = (1.0, 1.5)


def square(box, alpha: float) -> tuple[int, int, int]:
    """Квадрат $Q$: левый верхний угол и сторона в целых пикселях исходного снимка."""
    x0, y0, x1, y1 = box
    side = max(1, round(alpha * max(x1 - x0, y1 - y0)))
    return round((x0 + x1) / 2 - side / 2), round((y0 + y1) / 2 - side / 2), side


def _paste(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill) -> np.ndarray:
    """Окно снимка; часть за кадром — `fill`."""
    h, w = img.shape[:2]
    out = np.empty((y1 - y0, x1 - x0, 3), np.uint8)
    out[:] = fill
    a0, a1, b0, b1 = max(x0, 0), min(x1, w), max(y0, 0), min(y1, h)
    if a1 > a0 and b1 > b0:
        out[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = img[b0:b1, a0:a1]
    return out


# Размытие фона считается на окне, уменьшенном до этой стороны квадрата:
# после приведения к 448 параметр размытия — 22,4 пикс. при любом размере объекта; буквальный путь на квадрате
# 3 000–12 000 пикс. стоит от двух до пяти минут на вырезку. Значение — по сверке кодирования на 8 крупнейших масках
# калибровочных сцен пилота (`env.json`, `blur_large_masks`): при 2 × 448 порог косинуса CLS 0,998 не выдержан
# (минимум 0,9973), по записанному заранее правилу сторона удвоена, при 4 × 448 — выдержан (минимум 0,9987).
BLUR_WORK_SIDE = 4 * CROP_SIZE


def _blurred_background(img: np.ndarray, qx: int, qy: int, side: int, exact: bool,
                        sigma_frac: float = SIGMA_FRAC, work_side: int = BLUR_WORK_SIDE) -> np.ndarray:
    """Фон квадрата $Q$, размытый с $\\sigma_b=0{,}05\\cdot$`side` (в исходных пикселях); за кадром — $\\bar x$."""
    h, w = img.shape[:2]
    sigma = sigma_frac * side
    # размытие считается на окне, расширенном на 3σ внутрь кадра: граница квадрата не отражается в фоне
    r = int(np.ceil(3 * sigma))
    ex0, ey0, ex1, ey1 = max(qx - r, 0), max(qy - r, 0), min(qx + side + r, w), min(qy + side + r, h)
    if ex1 <= ex0 or ey1 <= ey0:
        return _paste(img, qx, qy, qx + side, qy + side, MEAN_RGB)
    ext = img[ey0:ey1, ex0:ex1]
    k = work_side / side
    if exact or k >= 1:
        blurred = cv2.GaussianBlur(ext, (0, 0), sigma)
    else:
        small = cv2.resize(ext, (max(1, round(ext.shape[1] * k)), max(1, round(ext.shape[0] * k))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma * k)
        blurred = cv2.resize(small, (ext.shape[1], ext.shape[0]), interpolation=cv2.INTER_LINEAR)
    # расширенное окно покрывает всю часть квадрата внутри кадра; вне его — за кадром, там $\\bar x$
    return _paste(blurred, qx - ex0, qy - ey0, qx - ex0 + side, qy - ey0 + side, MEAN_RGB)


def make_crop(img: np.ndarray, box, mask_window: Callable[[int, int, int, int], np.ndarray],
              b: str, alpha: float, size: int = CROP_SIZE, exact_blur: bool = False,
              sigma_frac: float = SIGMA_FRAC, work_side: int = BLUR_WORK_SIDE) -> np.ndarray:
    """Вырезка `size`×`size` uint8 RGB. `mask_window(x0, y0, x1, y1)` — окно маски в исходных координатах.

    Порядок: заполнение в исходном разрешении, затем масштабирование до `size`.
    `exact_blur=True` — размытие буквально в исходном разрешении (только для сверки).
    """
    if b not in FILLERS:
        raise ValueError(f"заполнитель {b!r} вне сетки {FILLERS}")
    qx, qy, side = square(box, alpha)
    m = mask_window(qx, qy, qx + side, qy + side)
    if b == "blur":
        # исходные пиксели под маской ставятся в размытый фон на месте: третьего буфера размера квадрата нет
        # (квадрат 12 288² — 450 МБ на буфер, а процессов подготовки шесть)
        q = _blurred_background(img, qx, qy, side, exact_blur, sigma_frac, work_side)
        a0, a1, b0, b1 = max(qx, 0), min(qx + side, img.shape[1]), max(qy, 0), min(qy + side, img.shape[0])
        if a1 > a0 and b1 > b0:  # вне кадра маска пуста
            sub, msub = q[b0 - qy:b1 - qy, a0 - qx:a1 - qx], m[b0 - qy:b1 - qy, a0 - qx:a1 - qx]
            sub[msub] = img[b0:b1, a0:a1][msub]
    else:
        fill = (0, 0, 0) if b == "0" else MEAN_RGB
        q = _paste(img, qx, qy, qx + side, qy + side, fill)
        q[~m] = fill
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return q if side == size else cv2.resize(q, (size, size), interpolation=interp)
