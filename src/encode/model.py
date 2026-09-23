"""Энкодер $E$: загрузка в рабочей точности, нормировка вручную, CLS и карта патч-признаков."""

from __future__ import annotations

import numpy as np
import torch

from src import env

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
N_PREFIX = 5  # CLS + 4 регистра перед патч-токенами в `last_hidden_state`


def load(name: str, fp32: bool = False):
    """`name` — `encoder_dinov2` | `encoder_dinov3`; веса заморожены, внимание — SDPA.

    `fp32=True` — только для оценки $U_r$ и сверок точности; рабочая точность — `env.ENCODER_DTYPE`.
    """
    from transformers import AutoModel

    dtype = torch.float32 if fp32 else getattr(torch, env.ENCODER_DTYPE[name])
    model = AutoModel.from_pretrained(env.MODEL_IDS[name], dtype=dtype, attn_implementation="sdpa")
    assert model.config.num_register_tokens == N_PREFIX - 1
    return model.eval().requires_grad_(False).cuda()


def patch_size(model) -> int:
    return model.config.patch_size


def _pixel_values(x_uint8: np.ndarray, model) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB → нормированный тензор в точности весов. `AutoImageProcessor` не используется."""
    dtype = next(model.parameters()).dtype
    x = torch.from_numpy(x_uint8).cuda().permute(0, 3, 1, 2).float() / 255
    mean, std = (torch.tensor(v, device="cuda").view(1, 3, 1, 1) for v in (MEAN, STD))
    return ((x - mean) / std).to(dtype)


@torch.no_grad()
def encode_crops(model, crops: np.ndarray) -> torch.Tensor:
    """CLS-токен вырезок, fp32, нормированный на единицу."""
    g = model(pixel_values=_pixel_values(crops, model)).pooler_output.float()
    if not torch.isfinite(g).all():
        raise FloatingPointError("не-конечный эмбеддинг вырезки")
    return torch.nn.functional.normalize(g, dim=-1)


def crop_to_patch_multiple(img: np.ndarray, patch: int) -> np.ndarray:
    """Обрезка снизу и справа до размеров, кратных шагу патча."""
    h, w = img.shape[:2]
    return img[:h // patch * patch, :w // patch * patch]


@torch.no_grad()
def patch_features(model, pixel_values: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Карта патч-признаков $F$ (P, d) в fp32 и сетка патчей для готового входа (1, 3, H, W); H, W кратны шагу патча."""
    p = patch_size(model)
    h, w = pixel_values.shape[-2:]
    if h % p or w % p:  # не кратный размер свёртка patch-embedding обрезает молча
        raise ValueError(f"вход {h}×{w} не кратен шагу патча {p}")
    dtype = next(model.parameters()).dtype
    f = model(pixel_values=pixel_values.cuda().to(dtype)).last_hidden_state[0, N_PREFIX:].float()
    assert f.shape[0] == (h // p) * (w // p)
    if not torch.isfinite(f).all():
        raise FloatingPointError("не-конечные патч-признаки")
    return f, (h // p, w // p)


def encode_full(model, img: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    """То же для снимка (H, W, 3) uint8 RGB, уже кратного шагу патча; нормировка вручную."""
    return patch_features(model, _pixel_values(img[None], model))
