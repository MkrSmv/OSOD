"""Позиционный дебиасинг $P^\\perp$ по [34]: только DINOv3, только $z_{\\mathrm{pool}}$.

Один $U_r$ на энкодер: шумовое изображение размера входа $E$ для сцен HR-InsDet, проход с весами в fp32,
SVD в fp32, первые $r$ правых сингулярных векторов. Центрирования нет — разложение самой карты признаков,
как записано в §2.2.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from src.encode import model as M

R = 500
NOISE_SEED = 20260918
# вход $E$ для сцены HR-InsDet до приведения к кратному шагу патча: 6144×8192 → 1536×2048
NOISE_HW = (1536, 2048)


def noise_input(patch: int, hw: tuple[int, int] = NOISE_HW, seed: int = NOISE_SEED) -> torch.Tensor:
    """Шум $\\mathcal N(0,1)$ по каналам, (1, 3, H, W) float32 — подаётся как `pixel_values` без нормировки.

    Генератор — на CPU: отсчёты не зависят от устройства.
    """
    h, w = hw[0] // patch * patch, hw[1] // patch * patch
    return torch.randn((1, 3, h, w), generator=torch.Generator().manual_seed(seed), dtype=torch.float32)


def estimate_u_r(model_fp32, r: int = R, hw: tuple[int, int] = NOISE_HW, seed: int = NOISE_SEED) -> tuple[np.ndarray, dict]:
    """$U_r\\in\\mathbb R^{d\\times r}$ (float32) и сведения для журнала окружения."""
    if next(model_fp32.parameters()).dtype != torch.float32:
        raise ValueError("U_r оценивается проходом в fp32")
    x = noise_input(M.patch_size(model_fp32), hw, seed)
    f, grid = M.patch_features(model_fp32, x)
    # SVD на CPU (LAPACK): секунды, зато результат не зависит от алгоритма cuSOLVER
    _, s, vh = torch.linalg.svd(f.cpu(), full_matrices=False)
    u_r = vh[:r].T.contiguous().numpy()
    energy = (s ** 2).cumsum(0) / (s ** 2).sum()
    info = {"r": r, "d": int(f.shape[1]), "noise_hw": list(x.shape[-2:]), "noise_seed": seed, "grid": list(grid),
            "weights_dtype": "float32", "svd": "torch.linalg.svd, CPU, float32, без центрирования",
            # спектр: есть ли разрыв у границы r — от этого зависит, насколько подпространство определено
            "singular_values": {str(k): float(s[k - 1]) for k in (1, 2, 3, 5, 10, 20, 50, 100, 200, r, r + 1, len(s))},
            "energy_fraction_in_u_r": float(energy[r - 1]),
            "orthonormality_max_abs_err": float(np.abs(u_r.T @ u_r - np.eye(r, dtype=np.float32)).max())}
    return u_r, info


def project(f: torch.Tensor, u_r: torch.Tensor) -> torch.Tensor:
    """$P_\\perp F_p=(I_d-U_rU_r^\\top)F_p$ для каждого патч-признака (fp32), до усреднения."""
    return f - (f @ u_r) @ u_r.T


def sha256(u_r: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(u_r).tobytes()).hexdigest()


def save(path: Path, u_r: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:  # `np.save` по имени дописал бы «.npy» к временному файлу
        np.save(fh, u_r)
    tmp.replace(path)


def load(path: Path, r: int = R) -> np.ndarray:
    u_r = np.load(path)
    if u_r.dtype != np.float32 or u_r.ndim != 2 or u_r.shape[1] != r:
        raise ValueError(f"{path}: ожидается float32 (d, {r}), получено {u_r.dtype} {u_r.shape}")
    return u_r
