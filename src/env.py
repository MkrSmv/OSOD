"""Сведения об установленном окружении для журнала `experiments/env.json`.

Всё читается из установленных пакетов, а не из памяти о них:
коммит SAM 2 — из `direct_url.json`, значения по умолчанию генератора масок — из сигнатуры
`SAM2AutomaticMaskGenerator.__init__`. Функции не требуют GPU.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # src/env.py → корень репозитория
ENV_JSON = REPO_ROOT / "experiments" / "env.json"
LOCK_FILE = REPO_ROOT / "requirements.lock.txt"

# Таблица моделей
MODEL_IDS = {
    "segmenter": "facebook/sam2.1-hiera-large",
    "encoder_dinov2": "facebook/dinov2-with-registers-large",
    "encoder_dinov3": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "owlv2": "google/owlv2-large-patch14-ensemble",
}

# Точность весов энкодеров: у DINOv3 ViT-L активации остаточного потока (~1,6e5) выходят
# за предел fp16 (65 504), поэтому он считается в bf16; сверка с fp32 — `encoder_precision` в env.json.
ENCODER_DTYPE = {"encoder_dinov2": "float16", "encoder_dinov3": "bfloat16"}

# Пакеты, от версий которых зависят результаты; полный список версий — `requirements.lock.txt`.
KEY_PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "huggingface-hub",
    "hf-transfer",
    "safetensors",
    "sam-2",
    "hydra-core",
    "iopath",
    "faiss-cpu",
    "numpy",
    "opencv-python-headless",
    "pillow",
    "pycocotools",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "pyyaml",
    "pytest",
]


def package_versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in KEY_PACKAGES}


def sam2_install_info() -> dict:
    """Коммит SAM 2 из `direct_url.json` установленного дистрибутива."""
    dist = metadata.distribution("sam-2")
    direct_url = json.loads(dist.read_text("direct_url.json"))
    return {
        "version": dist.version,
        "url": direct_url["url"],
        "commit": direct_url["vcs_info"]["commit_id"],
        "requested_revision": direct_url["vcs_info"].get("requested_revision"),
    }


def mask_generator_defaults() -> dict:
    """Фактические значения по умолчанию `SAM2AutomaticMaskGenerator` — часть определения M(I)."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sig = inspect.signature(SAM2AutomaticMaskGenerator.__init__)
    return {
        name: p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def lock_sha256() -> str | None:
    """Хеш файла зависимостей, если он есть рядом с пакетом."""
    return hashlib.sha256(LOCK_FILE.read_bytes()).hexdigest() if LOCK_FILE.is_file() else None


def code_stamp() -> dict:
    """Версия кода и время записи для журналов замеров: порядок «критерий → замер» должен читаться из самой записи."""
    import datetime
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    return {"code_commit": git("rev-parse", "HEAD"), "code_dirty": bool(git("status", "--porcelain", "--", "src", "scripts")),
            "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
