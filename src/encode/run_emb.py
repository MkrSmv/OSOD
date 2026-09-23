"""Рабочие файлы прогона: эмбеддинги масок по изображениям.

`cache/run_emb/<run_id>/<image_id>.npz`, fp16; это не кеш между прогонами (единственный такой кеш — маски):
ключ содержит `run_id`. Наличие файла означает «снимок закодирован»; файл пишется во временный и переименовывается.
Конфигурация прогона лежит в `key.json` каталога, её хеш повторён в каждом файле: прогон, возобновлённый после
смены любого параметра (например, рабочей стороны размытия), не смешает эмбеддинги двух конфигураций —
расхождение при записи или чтении есть ошибка, как у кеша масок.
Поиск всегда идёт по значениям из этих файлов (fp16 → fp32), а не по fp32 из памяти, — возобновлённый прогон
даёт те же $s^*$, что непрерывный.

Эмбеддинги эталонов при сборке галереи идут через такие же файлы (`<run_id>__gallery`, `src.gallery.build`), но
в float32: галерея хранится в одинарной точности (§2.5), и `emb.npy` всегда собирается из этих файлов без потерь.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("cache/run_emb")


class RunEmb:
    def __init__(self, run_id: str, config: dict, root: Path | None = None, dtype=np.float16):
        """`config` — полная конфигурация прогона (`RunConfig.to_dict()`) и всё прочее, от чего зависят эмбеддинги."""
        self.dir = (ROOT if root is None else Path(root)) / run_id
        self.dtype = np.dtype(dtype)
        self.config = json.loads(json.dumps(config, sort_keys=True, ensure_ascii=False))
        self.digest = hashlib.sha1(json.dumps(self.config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        kf = self.dir / "key.json"
        if not kf.exists():
            tmp = kf.with_name(f"{kf.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(self.config, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            tmp.replace(kf)
        elif json.loads(kf.read_text()) != self.config:
            raise ValueError(f"{kf}: рабочие файлы прогона записаны при другой конфигурации — удалить каталог {self.dir}")

    def path(self, image_id: str) -> Path:
        if "__" in image_id:
            raise ValueError(f"«__» в идентификаторе снимка: имя файла перестало бы быть однозначным ({image_id})")
        return self.dir / f"{image_id.replace('/', '__')}.npz"

    def has(self, image_id: str) -> bool:
        if not self.path(image_id).is_file():
            return False
        self._ensure_dir()  # посчитанное при другой конфигурации — ошибка, а не «уже сделано»
        return True

    def save(self, image_id: str, z: np.ndarray, **flags: np.ndarray) -> None:
        """`z` — (n, d), нормированные; не-конечное значение — ошибка прогона. `flags` — массивы длины n."""
        if not np.isfinite(z).all():
            raise FloatingPointError(f"{image_id}: не-конечный эмбеддинг")
        if any(len(v) != len(z) for v in flags.values()):
            raise ValueError("флаги — по одному значению на маску")
        self._ensure_dir()
        p = self.path(image_id)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, z=z.astype(self.dtype), config_sha1=np.array(self.digest), **flags)
        tmp.replace(p)

    def load(self, image_id: str) -> dict[str, np.ndarray]:
        """Эмбеддинги — float32 C-contiguous, как требует Faiss."""
        with np.load(self.path(image_id)) as f:
            out = {k: f[k] for k in f.files}
        if str(out.pop("config_sha1")) != self.digest:
            raise ValueError(f"{self.path(image_id)}: файл записан при другой конфигурации прогона")
        if out["z"].dtype != self.dtype:
            raise ValueError(f"{self.path(image_id)}: файл записан в {out['z'].dtype}, ожидается {self.dtype}")
        out["z"] = np.ascontiguousarray(out["z"], dtype=np.float32)
        return out


def scene_store(cfg, mask_key: dict, root: Path | None = None) -> RunEmb:
    """Рабочие файлы масок сцен прогона сетки `cfg` (`scripts/run_4_3.py`); их же читают контрольный прогон baseline
    и замер полноты HNSW на галереях сетки (`scripts/check_hnsw.py`, этап `grid`)."""
    return RunEmb(cfg.run_id, {"what": "run_4_3_scenes", "config": cfg.to_dict(), "mask_key": mask_key}, root=root)


PROGRESS = "progress.jsonl"


def log_progress(store: RunEmb, **fields) -> None:
    """Строка о закодированном снимке: время кодирования прогона, считанного урывками, складывается из этих строк. На эмбеддинги не влияет; в ключ не входит."""
    store._ensure_dir()
    with open(store.dir / PROGRESS, "a") as fh:
        fh.write(json.dumps(fields, ensure_ascii=False) + "\n")


def read_progress(store: RunEmb) -> list[dict]:
    p = store.dir / PROGRESS
    return [json.loads(line) for line in p.read_text().splitlines()] if p.is_file() else []
