"""HR-InsDet (InsDet-FULL): эталоны, сцены, истина по правилам разметки ниже."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.voc import read_voc

ROOT = Path("data/InsDet-FULL")
N_VIEWS = 24
SCENE_WH = (8192, 6144)

# Правила разметки: опечатка префикса и белые кружки вне списка 100 объектов.
RENAME = {"076_mouse_thinkpad": "075_mouse_thinkpad"}
NOT_IN_GT = {"077_mug_white", "062_cup_white"}

# Бины площади рамки разметки, пикс.² исходного снимка: < 200², 200²–400², > 400².
AREA_EDGES = [200.0 ** 2, 400.0 ** 2, 1e10]

# Помещения калибровочных сцен; остальные — тест.
CAL_ROOMS = ["easy/meeting_room_001", "easy/sink"]


def objects(root: Path = ROOT) -> list[str]:
    """Метки в порядке `category_id` (индекс в отсортированном списке `Objects/`)."""
    return sorted(p.name for p in (root / "Objects").iterdir() if p.is_dir())


def mask_dir(obj: str, root: Path = ROOT) -> Path:
    d = root / "Objects" / obj
    return d / "masks" if (d / "masks").is_dir() else d / "mask"


def read_ref_mask(path: Path) -> np.ndarray:
    """Маска GrabCut: оттенки серого, передний план > 127 (часть масок RGBA, одна с промежуточными значениями)."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise OSError(f"маска не читается: {path}")
    return g > 127


def mask_box(m: np.ndarray) -> list[int]:
    ys = np.flatnonzero(m.any(1))
    xs = np.flatnonzero(m.any(0))
    if not len(xs):
        raise ValueError("пустая маска")
    return [int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1]


def references(root: Path = ROOT) -> list[dict]:
    """Все эталоны: 100 объектов × 24 ракурса; рамка — передний план маски GrabCut."""
    out = []
    for cid, obj in enumerate(objects(root)):
        mdir = mask_dir(obj, root)
        views = sorted(p.stem for p in (root / "Objects" / obj / "images").glob("*.jpg"))
        if views != sorted(p.stem for p in mdir.glob("*.png")) or len(views) != N_VIEWS:
            raise ValueError(f"{obj}: снимки и маски не образуют {N_VIEWS} пар")
        for v in views:
            m = read_ref_mask(mdir / f"{v}.png")
            out.append({"id": f"{obj}/{v}", "label": obj, "category_id": cid,
                        "image": str((root / "Objects" / obj / "images" / f"{v}.jpg").relative_to(root)),
                        "mask": str((mdir / f"{v}.png").relative_to(root)),
                        "hw": list(m.shape), "box": mask_box(m)})
    return out


def scenes(root: Path = ROOT) -> list[dict]:
    """Сцены: идентификатор `<уровень>/<помещение>/rgb_NNN`, помещение — папка."""
    out = []
    for x in sorted((root / "Scenes").glob("*/*/*.xml")):
        room = f"{x.parent.parent.name}/{x.parent.name}"
        out.append({"id": f"{room}/{x.stem}", "level": x.parent.parent.name, "room": room,
                    "image": str(x.with_suffix(".jpg").relative_to(root)), "xml": str(x.relative_to(root))})
    return out


def scene_gt(scene: dict, labels: list[str], root: Path = ROOT) -> list[dict]:
    """Истина сцены по VOC XML с правилами разметки `RENAME` и `NOT_IN_GT`; `iscrowd=0` у всех рамок."""
    v = read_voc(root / scene["xml"])
    if tuple(v["wh"]) != SCENE_WH:
        raise ValueError(f"{scene['id']}: размер {v['wh']}")
    cid = {n: i for i, n in enumerate(labels)}
    out = []
    for o in v["objects"]:
        name = RENAME.get(o["name"], o["name"])
        if name in NOT_IN_GT:
            continue
        if name not in cid:
            raise ValueError(f"{scene['id']}: имя вне Objects/: {o['name']}")
        out.append({"label": name, "category_id": cid[name], "box": o["box"], "iscrowd": 0})
    return out
