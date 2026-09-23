"""PKU-Market-PCB: снимки, рамки дефектов, эталоны и роли плат."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.voc import read_voc

ROOT = Path("data/pku_market_pcb")
CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]

# Разбиение по платам.
BOARDS = {"gallery": ["01", "04"], "cal": ["09", "10"], "test": ["05", "06", "07", "08", "11", "12"]}

# Бины площади рамки разметки, пикс.² исходного снимка: терцили по всем 2 953 рамкам.
AREA_EDGES = [3570.0, 5250.0, 1e10]
DATA_CHECK = Path("experiments/data_check_pcb.json")


def images(root: Path = ROOT) -> list[dict]:
    """Исходные снимки (без `rotation/`): идентификатор — имя файла без расширения, плата — префикс."""
    out = []
    for c in CLASSES:
        for x in sorted((root / "Annotations" / c).glob("*.xml")):
            out.append({"id": x.stem, "board": x.stem.split("_", 1)[0], "label": c,
                        "image": f"images/{c}/{x.stem}.jpg", "xml": str(x.relative_to(root))})
    return out


def image_gt(img: dict, root: Path = ROOT) -> list[dict]:
    """Рамки дефектов снимка; метка — класс папки, `object/name` обязан с ней совпадать."""
    v = read_voc(root / img["xml"])
    if v["filename"] != Path(img["image"]).name:
        raise ValueError(f"{img['id']}: filename {v['filename']}")
    out = []
    for o in v["objects"]:
        if o["name"] != img["label"].lower():
            raise ValueError(f"{img['id']}: класс {o['name']} в папке {img['label']}")
        out.append({"label": img["label"], "category_id": CLASSES.index(img["label"]), "box": o["box"],
                    "iscrowd": 0})
    return out


def references(root: Path = ROOT) -> list[dict]:
    """Эталоны: рамки дефектов на снимках эталонных плат; идентификатор — `<снимок>/<номер рамки>`."""
    out = []
    for img in images(root):
        if img["board"] not in BOARDS["gallery"]:
            continue
        for k, g in enumerate(image_gt(img, root)):
            out.append({"id": f"{img['id']}/{k}", "label": g["label"], "category_id": g["category_id"],
                        "image": img["image"], "board": img["board"], "box": g["box"]})
    return out


def clean_image(board: str, root: Path = ROOT) -> str:
    """Бездефектный снимок платы (`PCB_USED/`), совмещённый со снимками платы попиксельно."""
    p = root / "PCB_USED" / f"{board}.JPG"
    if not p.exists():
        raise FileNotFoundError(p)
    return str(p.relative_to(root))


def ignore_zones(path: Path = DATA_CHECK) -> dict[str, list[list[int]]]:
    """Зоны игнорирования тестовых плат: области неразмеченных дефектов, найденные разностью с бездефектной платой — снимок → рамки, исключающие, в пикселях снимка."""
    cand = json.loads(path.read_text())["unlabeled_by_diff"]["candidates"]
    return {i: z for i, z in cand.items() if i.split("_", 1)[0] in BOARDS["test"]}


def gt_with_zones(img: dict, zones: dict[str, list[list[int]]], root: Path = ROOT) -> list[dict]:
    """Истина снимка тестовой платы: рамки разметки и зоны игнорирования с `iscrowd=1`.

    Категория зоны — тип дефекта снимка (на снимке все дефекты одного типа); при оценке без категорий не используется.
    """
    if img["board"] not in BOARDS["test"]:
        raise ValueError(f"{img['id']}: плата {img['board']} не тестовая")
    cid = CLASSES.index(img["label"])
    return image_gt(img, root) + [{"label": img["label"], "category_id": cid, "box": list(z), "iscrowd": 1}
                                  for z in zones.get(img["id"], [])]


def image_wh(img: dict, root: Path = ROOT) -> list[int]:
    return read_voc(root / img["xml"])["wh"]
