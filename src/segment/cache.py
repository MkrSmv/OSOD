"""Кеш масок SAM 2 на диске — единственный кеш между прогонами.

Раскладка: `cache/masks/<dataset>/<каталог ключа>/<изображение>.json`. Ключ — всё, от чего зависит $M(I)$ помимо
самого снимка: режим (`auto` | `box`), модель и её ревизия, коммит SAM 2, разрешение входа $S$, точность счёта,
а в авторежиме — `crop_n_layers`, `points_per_batch` и остальные параметры генератора установленной версии.
Каталог назван по главным полям ключа и хешу ключа целиком: смена любого поля даёт другой каталог, а не подмену,
поэтому маски при `crop_n_layers=0` не могут оказаться на месте масок при `=1`. Ключ лежит рядом (`key.json`)
и повторён в каждом файле; расхождение при чтении — ошибка, а не предупреждение. В режиме `box` в имя файла
входит хеш рамок промпта: другая рамка — другой файл. Датасет в ключ не входит — ключ описывает счёт $S$, а не данные,
и у HR-InsDet и PCB он один; датасеты различает каталог, а запись несёт поле `dataset`, которое сверяется при чтении:
каталог кеша одного датасета, подставленный на место другого, даёт ошибку, а не маски.

Файл пишется во временный и переименовывается: недописанная запись на месте валидной не остаётся,
а наличие файла означает, что снимок посчитан. Маски хранятся как RLE в разрешении входа $S$ и декодируются
по одной, по требованию (`MaskEntry.decode`, `MaskEntry.window`): сотни масок сцены в памяти не держатся.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src import env
from src.segment import sam2 as S

ROOT = Path("cache/masks")
FORMAT = 1
AUTOCAST = "bfloat16"  # веса SAM 2 в fp32, счёт под bf16-autocast
# Только для сверки точности на калибровочных сценах: счёт без autocast, свой каталог.
AUTOCAST_OFF = "off"
# Записи, сделанные до появления поля `dataset`, — только HR-InsDet: другого кеша тогда не было.
DATASET_BEFORE_FIELD = "hr_insdet"
# Параметры генератора, не входящие в определение $M(I)$: формат выдачи.
_NOT_IN_KEY = {"output_mode"}


def _segmenter_identity() -> dict:
    rec = json.loads(env.ENV_JSON.read_text())["models"]["segmenter"]
    if rec["id"] != env.MODEL_IDS["segmenter"]:
        raise ValueError("env.json описывает другой сегментатор — выполнить scripts/check_env.py")
    return {"model": rec["id"], "model_revision": rec["revision"], "sam2_commit": env.sam2_install_info()["commit"]}


def auto_key(crop_n_layers: int, long_side: int = S.LONG_SIDE, points_per_batch: int = S.POINTS_PER_BATCH,
             fp32: bool = False) -> dict:
    """Ключ авторежима. Параметры генератора — значения по умолчанию установленной версии и два отступления от них."""
    gen = {k: v for k, v in env.mask_generator_defaults().items() if k not in _NOT_IN_KEY}
    gen.update(crop_n_layers=int(crop_n_layers), points_per_batch=int(points_per_batch))
    return {"format": FORMAT, "mode": "auto", **_segmenter_identity(), "long_side": int(long_side),
            "autocast": AUTOCAST_OFF if fp32 else AUTOCAST, "generator": gen}


def box_key(long_side: int = S.LONG_SIDE) -> dict:
    """Ключ режима промпта-рамки; сами рамки входят в имя файла (`MaskCache.path`)."""
    thr, off = S.box_mode_params()
    return {"format": FORMAT, "mode": "box", **_segmenter_identity(), "long_side": int(long_side),
            "autocast": AUTOCAST, "multimask_output": False, "mask_threshold": thr, "stability_score_offset": off}


def _digest(obj, n: int) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:n]


def key_dirname(key: dict) -> str:
    head = f"auto_cnl{key['generator']['crop_n_layers']}" if key["mode"] == "auto" else "box"
    ppb = f"_ppb{key['generator']['points_per_batch']}" if key["mode"] == "auto" else ""
    prec = "" if key["autocast"] == AUTOCAST else "_fp32"
    return f"{head}_ls{key['long_side']}{ppb}{prec}_sam2-{key['sam2_commit'][:7]}_{_digest(key, 8)}"


def check_generator(gen, key: dict) -> None:
    """Построенный генератор обязан иметь ровно те параметры, что записаны в ключе."""
    diff = {k: (v, getattr(gen, k)) for k, v in key["generator"].items()
            if hasattr(gen, k) and k != "point_grids" and getattr(gen, k) != v}
    if diff:
        raise ValueError(f"параметры генератора расходятся с ключом кеша (ключ, генератор): {diff}")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


class MaskEntry:
    """Маски одного снимка: записи лёгкие (RLE-строки, рамки, оценки), декодирование — по требованию."""

    def __init__(self, data: dict):
        self.image_id: str = data["image_id"]
        self.hw: tuple[int, int] = tuple(data["hw"])          # исходный снимок
        self.seg_hw: tuple[int, int] = tuple(data["seg_hw"])  # вход S
        self.scale_xy: tuple[float, float] = tuple(data["scale_xy"])
        self.info: dict = data["info"]
        self.records: list[dict] = data["masks"]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def boxes(self) -> np.ndarray:
        """Описывающие рамки $\\mathrm{box}(m)$: исключающие, в пикселях исходного снимка, форма (n, 4)."""
        return np.array([r["box"] for r in self.records], float).reshape(-1, 4)

    def decode(self, i: int) -> np.ndarray:
        """Маска `i` в разрешении входа $S$ (bool); результат нигде не запоминается."""
        return S.decode(self.records[i]["rle"])

    def window(self, i: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """Окно маски `i` в исходных координатах без полноразмерного массива."""
        return S.mask_window(self.decode(i), self.scale_xy, x0, y0, x1, y1)

    def windower(self, i: int):
        """Функция окна для `encode.crop.make_crop`; seg-маска декодируется один раз и живёт, пока жива функция."""
        m = self.decode(i)
        return lambda x0, y0, x1, y1: S.mask_window(m, self.scale_xy, x0, y0, x1, y1)


class MaskCache:
    def __init__(self, dataset: str, key: dict, root: Path | None = None):
        self.dataset, self.key = dataset, key
        self.dir = (ROOT if root is None else Path(root)) / dataset / key_dirname(key)
        self._checked = False

    def path(self, image_id: str, prompt_boxes: list | None = None) -> Path:
        if "__" in image_id:
            raise ValueError(f"«__» в идентификаторе снимка: имя файла перестало бы быть однозначным ({image_id})")
        name = image_id.replace("/", "__")
        if self.key["mode"] == "box":
            if not prompt_boxes:
                raise ValueError("в режиме box рамки промпта — часть ключа")
            name += "__" + _digest([[float(v) for v in b] for b in prompt_boxes], 10)
        elif prompt_boxes is not None:
            raise ValueError("в авторежиме рамок промпта нет")
        return self.dir / f"{name}.json"

    def has(self, image_id: str, prompt_boxes: list | None = None) -> bool:
        """Наличие файла — признак «посчитано». Первая найденная запись сверяется целиком (ключ, датасет, снимок):
        каталог чужого датасета или ключа, оказавшийся на этом месте, даёт ошибку уже здесь, а не пропуск счёта."""
        found = self.path(image_id, prompt_boxes).is_file()
        if found and not self._checked:
            self.load(image_id, prompt_boxes)
            self._checked = True
        return found

    def _ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        kf = self.dir / "key.json"
        if not kf.exists():
            _atomic_write(kf, json.dumps(self.key, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        elif json.loads(kf.read_text()) != self.key:
            raise ValueError(f"{kf}: ключ каталога не совпадает с запрошенным")

    def save(self, image_id: str, hw, recs: list[dict], info: dict, prompt_boxes: list | None = None) -> Path:
        p = self.path(image_id, prompt_boxes)
        self._ensure_dir()
        info = dict(info)
        data = {"key": self.key, "dataset": self.dataset, "image_id": image_id, "hw": [int(v) for v in hw], "seg_hw": info.pop("seg_hw"),
                "scale_xy": info.pop("scale_xy"), "info": info, "masks": recs}
        _atomic_write(p, json.dumps(data, ensure_ascii=False))
        return p

    def load(self, image_id: str, prompt_boxes: list | None = None) -> MaskEntry:
        p = self.path(image_id, prompt_boxes)
        data = json.loads(p.read_text())
        if data["key"] != self.key or data["image_id"] != image_id:
            raise ValueError(f"{p}: запись кеша сделана с другим ключом или для другого снимка")
        if data.get("dataset", DATASET_BEFORE_FIELD) != self.dataset:
            raise ValueError(f"{p}: запись кеша сделана для датасета {data.get('dataset', DATASET_BEFORE_FIELD)}, "
                             f"а читается как {self.dataset}")
        return MaskEntry(data)
