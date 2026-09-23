"""Усреднение под маской $P$: мягкая привязка маски к сетке патчей, среднее, нормировка.

Один проход энкодера на изображение: карта $F$ обслуживает все маски и оба варианта — $P$ и, для DINOv3,
$P^\\perp$; карта не кешируется.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from src.encode import debias as D
from src.encode import model as M
from src.segment import sam2 as S

POOL_LONG_SIDE = S.LONG_SIDE  # вход $E$ — вход сегментатора


def patch_grid_mask(m_seg: np.ndarray, in_hw: tuple[int, int], patch: int, resize: bool = True) -> np.ndarray:
    """$\\tilde m_p\\in[0,1]$ — доля пикселей патча под маской.

    `m_seg` — маска в разрешении входа $S$; `in_hw` — размер входа $E$ до обрезки под шаг патча.
    Если вход $E$ меньше входа $S$ (`pool_long_side`), маска приводится к нему усреднением по площади; при
    `resize=False` (вход $E$ равен входу $S$) расхождение размеров — испорченная запись кеша, а не повод масштабировать.
    """
    m = m_seg.astype(np.float32)
    if tuple(m.shape) != tuple(in_hw):
        if not resize:
            raise ValueError(f"маска {tuple(m.shape)} не в разрешении входа E {tuple(in_hw)} при равных входах S и E")
        m = cv2.resize(m, (in_hw[1], in_hw[0]), interpolation=cv2.INTER_AREA)
    gh, gw = in_hw[0] // patch, in_hw[1] // patch
    return m[:gh * patch, :gw * patch].reshape(gh, patch, gw, patch).mean((1, 3))


def pool(f: torch.Tensor, m_grid: torch.Tensor, centers: torch.Tensor | None = None
         ) -> tuple[torch.Tensor, torch.Tensor]:
    """$z_{\\mathrm{pool}}$ для K масок: `f` (P, d) fp32, `m_grid` (K, P). Возвращает (K, d) и флаги пустых масок.

    Маске с $\\sum_p\\tilde m_p=0$ даётся патч `centers[k]` с весом 1.
    """
    s = m_grid.sum(1)
    empty = s == 0
    if empty.any():
        if centers is None:
            raise ValueError("маска не попала ни в один патч, а центры не заданы")
        m_grid = m_grid.clone()
        m_grid[empty, centers[empty]] = 1.0
        s = m_grid.sum(1)
    z = (m_grid @ f) / s[:, None]
    return torch.nn.functional.normalize(z, dim=-1), empty


def center_patch(box, hw: tuple[int, int], in_hw: tuple[int, int], grid: tuple[int, int], patch: int) -> int:
    """Патч, ближайший к центру маски — центру её описывающей рамки $c$, — номер в сетке.

    `box` — в пикселях исходного снимка `hw`; `in_hw` — вход $E$ до обрезки под шаг патча. Нужен только маске,
    не попавшей ни в один блок: это маска целиком в полосе ≤ 13 / 15 пикс., срезанной снизу или справа.
    """
    cx = (box[0] + box[2]) / 2 * in_hw[1] / hw[1]
    cy = (box[1] + box[3]) / 2 * in_hw[0] / hw[0]
    return min(max(int(cy // patch), 0), grid[0] - 1) * grid[1] + min(max(int(cx // patch), 0), grid[1] - 1)


def encoder_input(img: np.ndarray, patch: int, long_side: int = POOL_LONG_SIDE) -> tuple[np.ndarray, tuple[int, int]]:
    """Вход $E$ для $z_{\\mathrm{pool}}$: то же уменьшение, что у $S$, и обрезка снизу / справа до кратного шагу патча.

    Возвращает вход и его размер до обрезки — к нему приводится маска.
    """
    x = S.seg_input(img, long_side)
    return M.crop_to_patch_multiple(x, patch), x.shape[:2]


@torch.no_grad()
def pool_image(model, img: np.ndarray, entry, u_r: torch.Tensor | None = None, long_side: int = POOL_LONG_SIDE,
               outputs: tuple[str, ...] = ("p", "p_perp")) -> dict:
    """$z_{\\mathrm{pool}}$ всех масок снимка одним проходом энкодера; с `u_r` — ещё и $P^\\perp$ из той же карты $F$.

    `entry` — `segment.cache.MaskEntry`; маски декодируются по одной. `outputs` — что считать (для сверки
    «один проход против раздельного»); `p_perp` без `u_r` не считается.
    """
    patch = M.patch_size(model)
    x, in_hw = encoder_input(img, patch, long_side)
    f, grid = M.encode_full(model, x)
    n = len(entry)
    if n == 0:
        return {"grid": grid, "in_hw": in_hw, "empty": np.zeros(0, bool)}
    resize = long_side != S.LONG_SIDE  # маска масштабируется только при ином входе E
    m_grid = np.stack([patch_grid_mask(entry.decode(i), in_hw, patch, resize).reshape(-1) for i in range(n)])
    centers = torch.tensor([center_patch(r["box"], entry.hw, in_hw, grid, patch) for r in entry.records],
                           dtype=torch.long, device=f.device)
    m_t = torch.from_numpy(m_grid).to(f.device)
    out = {"grid": grid, "in_hw": in_hw, "patches_under_mask": m_grid.sum(1)}
    if "p" in outputs:
        z, empty = pool(f, m_t, centers)
        out["p"], out["empty"] = z.cpu().numpy(), empty.cpu().numpy()
    if "p_perp" in outputs and u_r is not None:
        z, empty = pool(D.project(f, u_r), m_t, centers)  # проектор — к каждому патч-признаку, до усреднения
        out["p_perp"], out["empty"] = z.cpu().numpy(), empty.cpu().numpy()
    for k in ("p", "p_perp"):
        if k in out and not np.isfinite(out[k]).all():
            raise FloatingPointError(f"не-конечный эмбеддинг {k}")
    return out
