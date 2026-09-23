"""Вход $S$, авторежим и промпт-рамка SAM 2, перевод масок в исходные координаты.

Рамки везде исключающие, в пикселях исходного снимка; RLE — в разрешении входа $S$ вместе с масштабом.
Кеша здесь нет: функции возвращают записи масок; кеш на диске — `src.segment.cache`.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from pycocotools import mask as rle_api

from src import env

LONG_SIDE = 2048
# Не значение по умолчанию (64); одно значение для всех снимков.
POINTS_PER_BATCH = 16


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"снимок не читается: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def seg_input(img: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    """Уменьшение до `long_side` по длинной стороне; меньшие снимки не увеличиваются."""
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(h, w))
    if s == 1.0:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def build_generator(crop_n_layers: int, points_per_batch: int = POINTS_PER_BATCH):
    """Штатный генератор масок: все параметры, кроме двух названных, — значения по умолчанию установленной версии."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    return SAM2AutomaticMaskGenerator.from_pretrained(
        env.MODEL_IDS["segmenter"], device="cuda", crop_n_layers=crop_n_layers,
        points_per_batch=points_per_batch, output_mode="coco_rle")


def build_predictor():
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor.from_pretrained(env.MODEL_IDS["segmenter"], device="cuda")


def box_mode_params(pred=None) -> tuple[float, float]:
    """Порог маски и сдвиг stability score режима рамки — значения генератора установленной версии (в ключе кеша)."""
    d = env.mask_generator_defaults()
    thr, off = float(d["mask_threshold"]), float(d["stability_score_offset"])
    if pred is not None and float(pred.mask_threshold) != thr:
        raise ValueError(f"mask_threshold предиктора {pred.mask_threshold} расходится с генератором {thr}")
    return thr, off


def _record(rle: dict, box_incl_xyxy, area_seg: float, sx: float, sy: float, **scores) -> dict:
    x0, y0, x1, y1 = box_incl_xyxy
    # Включающий край генератора → исключающий (+1), затем в координаты исходного снимка.
    # Площадь — в пикселях исходного снимка: площадь seg-маски на масштаб, без декодирования в полный размер.
    return {"rle": rle, "box": [float(x0 * sx), float(y0 * sy), float((x1 + 1) * sx), float((y1 + 1) * sy)],
            "area": float(area_seg * sx * sy), **scores}


def generate(gen, img: np.ndarray, long_side: int = LONG_SIDE, autocast: bool = True) -> tuple[list[dict], dict]:
    """Авторежим: все маски $M(I)$ без отбора. Возвращает записи масок и сведения о проходе.

    `autocast=False` — проход в fp32, только для сверки точности; в кеш такие маски не идут.
    """
    import torch  # здесь, а не в заголовке: модуль читают и процессы подготовки вырезок, которым torch не нужен

    h, w = img.shape[:2]
    t0 = time.perf_counter()
    x = seg_input(img, long_side)
    t1 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for attempt in range(3):
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                out = gen.generate(x)
            torch.cuda.synchronize()
            break
        except RuntimeError as e:  # сбой драйвера CUDA под WSL при почти полной памяти — повтор того же снимка
            if "CUDA" not in str(e) or attempt == 2:
                raise
            print(f"повтор после ошибки CUDA: {e}", flush=True)
            torch.cuda.empty_cache()
            time.sleep(10)
    t2 = time.perf_counter()
    sx, sy = w / x.shape[1], h / x.shape[0]
    recs = [_record(m["segmentation"], (m["bbox"][0], m["bbox"][1], m["bbox"][0] + m["bbox"][2],
                                        m["bbox"][1] + m["bbox"][3]), m["area"], sx, sy,
                    predicted_iou=float(m["predicted_iou"]), stability_score=float(m["stability_score"]),
                    # окно слоя и точка решётки, породившие маску, — в координатах входа $S$, как у генератора
                    crop_box=[int(v) for v in m["crop_box"]], point_coords=[float(v) for v in m["point_coords"][0]])
            for m in out]
    # `attempts` > 1 — снимок пересчитан после сбоя драйвера CUDA; помечается, чтобы запись можно было найти
    info = {"seg_hw": list(x.shape[:2]), "scale_xy": [sx, sy], "sec_resize": t1 - t0, "sec_generate": t2 - t1,
            "vram_peak_mib": torch.cuda.max_memory_allocated() // 2**20, "attempts": attempt + 1}
    return recs, info


def predict_boxes(pred, img: np.ndarray, boxes: list[list[float]], long_side: int = LONG_SIDE,
                  one_by_one: bool = False) -> tuple[list[dict], dict]:
    """Промпт-рамка: одна рамка → одна маска; рамки — исключающие, в пикселях исходного снимка.

    `one_by_one` — снимок с несколькими эталонами (PCB): `set_image` один, а `predict` — отдельный на каждую рамку,
    то есть тем же вызовом с одной рамкой, что у эталона HR-InsDet; время каждого — в `info["sec_predict_each"]`.
    """
    import torch

    h, w = img.shape[:2]
    t0 = time.perf_counter()
    x = seg_input(img, long_side)
    sx, sy = w / x.shape[1], h / x.shape[0]
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred.set_image(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        b = np.array(boxes, np.float32) / [sx, sy, sx, sy]
        # логиты вместо готовой маски: по ним же считается stability score — той же формулой, что в авторежиме
        logits, ious, each = [], [], []
        for bb in (b[:, None] if one_by_one else [b]):
            t = time.perf_counter()
            lg, q, _ = pred.predict(box=bb, multimask_output=False, return_logits=True)
            torch.cuda.synchronize()
            logits.append(lg.reshape(-1, *lg.shape[-2:]))
            ious.append(np.asarray(q).reshape(-1))
            each.append(time.perf_counter() - t)
    t2 = time.perf_counter()
    logits, ious = np.concatenate(logits), np.concatenate(ious)
    thr, off = box_mode_params(pred)
    recs = []
    for lg, q, pb in zip(logits, ious, boxes):
        m = lg > thr  # то же, что делает `predict` при `return_logits=False`
        rle = rle_api.encode(np.asfortranarray(m.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        inc = (xs[0], ys[0], xs[-1], ys[-1]) if len(xs) else (0, 0, -1, -1)
        union = int((lg > thr - off).sum())
        stab = float((lg > thr + off).sum() / union) if union else 0.0
        recs.append(_record(rle, inc, float(m.sum()), sx, sy, predicted_iou=float(q), stability_score=stab,
                            prompt_box=[float(v) for v in pb]))
    info = {"seg_hw": list(x.shape[:2]), "scale_xy": [sx, sy], "sec_set_image": t1 - t0, "sec_predict": t2 - t1,
            "vram_peak_mib": torch.cuda.max_memory_allocated() // 2**20}
    if one_by_one:
        info["sec_predict_each"] = each
    return recs, info


def decode(rle: dict) -> np.ndarray:
    """Маска в разрешении входа $S$ (bool)."""
    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("ascii")
    return rle_api.decode(r).astype(bool)


def mask_window(m_seg: np.ndarray, scale_xy, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Окно маски $[y_0,y_1)\\times[x_0,x_1)$ в исходных координатах без полноразмерного массива.

    Значение в центре исходного пикселя — билинейная интерполяция seg-маски (то же, что `cv2.INTER_LINEAR`
    при увеличении всей маски), порог 0,5; окно может выходить за кадр — там маска пуста.
    """
    sx, sy = scale_xy
    hs, ws = m_seg.shape
    # участок seg-маски, покрывающий окно, с запасом в пиксель под интерполяцию
    u0, u1 = int(np.floor(x0 / sx)) - 2, int(np.ceil(x1 / sx)) + 2
    v0, v1 = int(np.floor(y0 / sy)) - 2, int(np.ceil(y1 / sy)) + 2
    # край кадра повторяется, как при увеличении всей маски (`BORDER_REPLICATE` у `cv2.resize`)
    sub = m_seg[np.ix_(np.clip(np.arange(v0, v1), 0, hs - 1), np.clip(np.arange(u0, u1), 0, ws - 1))].astype(np.float32)
    # координата seg-пикселя для центра исходного пикселя: (x + 0,5)/sx − 0,5 (соглашение cv2.resize);
    # аффинное отображение вместо сетки координат: окно 12 288² с сеткой заняло бы больше гигабайта
    A = np.array([[1 / sx, 0, (x0 + 0.5) / sx - 0.5 - u0], [0, 1 / sy, (y0 + 0.5) / sy - 0.5 - v0]], np.float64)
    out = cv2.warpAffine(sub, A, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_REPLICATE) > 0.5
    h, w = round(hs * sy), round(ws * sx)
    out[:max(0, -y0)] = False
    out[:, :max(0, -x0)] = False
    out[max(0, h - y0):] = False
    out[:, max(0, w - x0):] = False
    return out
