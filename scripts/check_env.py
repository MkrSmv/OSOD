"""Журнал окружения `experiments/env.json`.

Проверяет GPU и SDPA, скачивает снимки четырёх моделей и записывает их ревизии,
читает коммит SAM 2 и значения по умолчанию генератора масок из установленного пакета,
пробно загружает каждую модель. Опубликованный `experiments/env.json` — окружение, в котором получены числа
журнала: если он уже есть, отчёт этой машины пишется в `logs/env_check.json`, а `env.json` не меняется. С
`--overwrite` заменяются только свои разделы `env.json`; ключи, дописанные другими скриптами, сохраняются.

    python scripts/check_env.py              # всё; веса моделей скачиваются здесь
    python scripts/check_env.py --no-load    # без пробной загрузки моделей на GPU
    python scripts/check_env.py --overwrite  # заменить свои разделы experiments/env.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import env  # noqa: E402

OWN_SECTIONS = ["written", "platform", "gpu_check", "packages", "lock", "sam2", "models", "model_load", "encoder_precision"]


def platform_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "os": platform.freedesktop_os_release().get("PRETTY_NAME"),
        "kernel": platform.release(),
        "gpu": p.name,
        "gpu_total_mib": p.total_memory // 2**20,
        "gpu_arch": f"sm_{p.major}{p.minor}",
        "driver": driver,
        "torch_cuda": torch.version.cuda,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "env": {k: os.environ.get(k) for k in ("PYTORCH_CUDA_ALLOC_CONF", "HF_HUB_ENABLE_HF_TRANSFER", "SAM2_BUILD_CUDA")},
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
    }


def gpu_check() -> dict:
    """Проверка: пик SDPA на 16 тыс. токенов; гигабайты — математический fallback."""
    q = torch.randn(1, 16, 16384, 64, device="cuda", dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    peak = torch.cuda.max_memory_allocated() // 2**20
    q = q.to(torch.bfloat16)  # DINOv3 считается в bf16
    torch.cuda.reset_peak_memory_stats()
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    peak_bf16 = torch.cuda.max_memory_allocated() // 2**20
    del q
    torch.cuda.empty_cache()
    return {
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "sdpa_16k_tokens_peak_mib": int(peak),
        "sdpa_16k_tokens_peak_mib_bf16": int(peak_bf16),
    }


def download_models() -> dict:
    from huggingface_hub import snapshot_download

    out = {}
    for role, repo_id in env.MODEL_IDS.items():
        path = Path(snapshot_download(repo_id))
        files = {str(f.relative_to(path)): f.stat().st_size for f in sorted(path.rglob("*")) if f.is_file()}
        out[role] = {"id": repo_id, "revision": path.name, "files": files}
        print(role, repo_id, path.name, flush=True)
    return out


def _vram_mib() -> int:
    return int(torch.cuda.max_memory_allocated() // 2**20)


def load_models() -> dict:
    """Пробная загрузка в тех режимах точности, что заданы; сегментатор и энкодер — по очереди."""
    out = {}

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    torch.cuda.reset_peak_memory_stats()
    gen = SAM2AutomaticMaskGenerator.from_pretrained(env.MODEL_IDS["segmenter"])
    model = gen.predictor.model
    out["segmenter"] = {
        "class": type(model).__name__,
        "image_size": int(model.image_size),
        "param_dtype": str(next(model.parameters()).dtype),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "vram_peak_mib": _vram_mib(),
    }
    del gen, model
    torch.cuda.empty_cache()

    from transformers import AutoModel, Owlv2ForObjectDetection

    for role in ("encoder_dinov2", "encoder_dinov3"):
        torch.cuda.reset_peak_memory_stats()
        dt = getattr(torch, env.ENCODER_DTYPE[role])
        m = AutoModel.from_pretrained(env.MODEL_IDS[role], dtype=dt, attn_implementation="sdpa").to("cuda").eval()
        with torch.no_grad():
            h = m(pixel_values=torch.zeros(1, 3, 448, 448, device="cuda", dtype=dt)).last_hidden_state
        cfg = m.config
        out[role] = {
            "class": type(m).__name__,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "patch_size": cfg.patch_size,
            "num_register_tokens": cfg.num_register_tokens,
            "attn_implementation": cfg._attn_implementation,
            "param_dtype": str(next(m.parameters()).dtype),
            "tokens_at_448": int(h.shape[1]),
            "output_finite": bool(torch.isfinite(h).all()),
            "vram_peak_mib": _vram_mib(),
        }
        del m, h
        torch.cuda.empty_cache()

    # OWLv2: SDPA запрашивается, как у энкодеров; если класс установленной версии его не поддерживает
    # (transformers 4.57 — ValueError), берётся eager, и это записывается. Без SDPA матрица внимания
    # материализуется, поэтому пик зрительной башни на штатном входе замеряется здесь же.
    torch.cuda.reset_peak_memory_stats()
    try:
        m = Owlv2ForObjectDetection.from_pretrained(env.MODEL_IDS["owlv2"], dtype=torch.float16, attn_implementation="sdpa")
        sdpa_error = None
    except ValueError as e:
        sdpa_error = str(e).split(". ")[0]
        m = Owlv2ForObjectDetection.from_pretrained(env.MODEL_IDS["owlv2"], dtype=torch.float16, attn_implementation="eager")
    m = m.to("cuda").eval()
    size = m.config.vision_config.image_size
    with torch.no_grad():
        h = m.owlv2.vision_model(pixel_values=torch.zeros(1, 3, size, size, device="cuda", dtype=torch.float16)).last_hidden_state
    out["owlv2"] = {
        "class": type(m).__name__,
        "image_size": size,
        "patch_size": m.config.vision_config.patch_size,
        "attn_implementation": m.config._attn_implementation,
        "sdpa_error": sdpa_error,
        "param_dtype": str(next(m.parameters()).dtype),
        "vision_tokens": int(h.shape[1]),
        "output_finite": bool(torch.isfinite(h).all()),
        "vision_forward_vram_peak_mib": _vram_mib(),
    }
    del m, h
    torch.cuda.empty_cache()
    return out


def _pcb_crops(n: int = 6) -> torch.Tensor:
    """Фиксированные фрагменты первых по имени снимков PCB, 448×448, нормировка вручную."""
    import cv2
    import numpy as np

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    files = sorted((env.REPO_ROOT / "data" / "pku_market_pcb" / "images").glob("*/*.jpg"))[::120][:n]
    assert len(files) == n, "нет снимков PKU-Market-PCB в data/"
    xs = []
    for f in files:
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)[200:1096, 200:1096]
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        xs.append(torch.from_numpy((img - mean) / std).permute(2, 0, 1))
    return torch.stack(xs)


def encoder_precision() -> dict:
    """Сверка половинных точностей с fp32 для обоих энкодеров — основание выбора ENCODER_DTYPE."""
    from transformers import AutoModel

    nf = torch.nn.functional
    x = _pcb_crops()
    out = {"input": "6 фрагментов PKU-Market-PCB 448×448 (scripts/check_env.py, _pcb_crops)"}
    for role in ("encoder_dinov2", "encoder_dinov3"):
        ref, res = None, {}
        for name in ("float32", "float16", "bfloat16"):
            dt = getattr(torch, name)
            m = AutoModel.from_pretrained(env.MODEL_IDS[role], dtype=dt, attn_implementation="sdpa").to("cuda").eval()
            with torch.no_grad():
                o = m(pixel_values=x.to("cuda", dt), output_hidden_states=True)
            h = o.last_hidden_state.float()
            r = {
                "finite": bool(torch.isfinite(h).all()),
                "max_abs_hidden": round(max(float(hs.float().abs().nan_to_num(posinf=0, neginf=0).max()) for hs in o.hidden_states)),
            }
            if ref is None:
                ref = h
            elif r["finite"]:
                g, gr = nf.normalize(h[:, 0], dim=1), nf.normalize(ref[:, 0], dim=1)
                r["cls_cos_to_fp32_min"] = round(float((g * gr).sum(1).min()), 5)
                r["cls_pairwise_sim_max_abs_diff"] = round(float((g @ g.T - gr @ gr.T).abs().max()), 5)
                r["patch_cos_to_fp32_min"] = round(float(nf.cosine_similarity(h[:, 5:], ref[:, 5:], dim=2).min()), 5)
            res[name] = r
            del m, o, h
            torch.cuda.empty_cache()
        out[role] = {"chosen": env.ENCODER_DTYPE[role], **res}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-load", action="store_true", help="не загружать модели на GPU")
    ap.add_argument("--overwrite", action="store_true", help="заменить свои разделы опубликованного experiments/env.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA не видна"
    report = {
        "written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        "platform": platform_info(),
        "gpu_check": gpu_check(),
        "packages": env.package_versions(),
        "lock": {"file": env.LOCK_FILE.name, "sha256": env.lock_sha256()},
        "sam2": {**env.sam2_install_info(), "automatic_mask_generator_defaults": env.mask_generator_defaults()},
        "models": download_models(),
    }
    if not args.no_load:
        report["model_load"] = load_models()
        report["encoder_precision"] = encoder_precision()

    if env.ENV_JSON.exists() and not args.overwrite:
        local = Path("logs/env_check.json")
        local.parent.mkdir(exist_ok=True)
        local.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"{env.ENV_JSON} уже есть и не перезаписывается; отчёт этой машины — {local}; замена разделов — --overwrite",
              file=sys.stderr)
        return
    old = json.loads(env.ENV_JSON.read_text()) if env.ENV_JSON.exists() else {}
    kept = {k: v for k, v in old.items() if k not in OWN_SECTIONS}
    if args.no_load:
        kept.update({k: old[k] for k in ("model_load", "encoder_precision") if k in old})
    env.ENV_JSON.parent.mkdir(exist_ok=True)
    tmp = env.ENV_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({**report, **kept}, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(env.ENV_JSON)
    print("записано:", env.ENV_JSON, file=sys.stderr)


if __name__ == "__main__":
    main()
