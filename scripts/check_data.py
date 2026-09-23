"""Проверка раздачи по содержимому, до подготовки данных.

HR-InsDet: декодирует каждый снимок и маску, сверяет пары «снимок — маска» эталонов, свойства масок,
разбирает VOC XML сцен и сверяет имена объектов разметки с каталогами `Objects/`.
PKU-Market-PCB: декодирует исходные снимки `images/` и платы `PCB_USED/`, сверяет XML с именем
и размером снимка, считает рамки по платам и типам дефектов и терцили площадей рамок.
Итог — `experiments/data_check_<dataset>.json`; выводы и принятые по ним решения, формат раздачи. `data/` только читается.

    python scripts/check_data.py                  # HR-InsDet
    python scripts/check_data.py --dataset pcb
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/InsDet-FULL")
OUT = Path("experiments/data_check_hr_insdet.json")
PCB_ROOT = Path("data/pku_market_pcb")
PCB_OUT = Path("experiments/data_check_pcb.json")
PCB_CLASSES = ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
# Разность со снимком бездефектной платы `PCB_USED/` (совмещён попиксельно): порог на максимуме модуля
# разности по каналам после размытия 5×5 (шум JPEG — медиана 2), дилатация 9×9, компоненты от 60 пикс.
# При пороге 25 изменение видно у всех рамок разметки, кроме 8 малоконтрастных на плате 01.
DIFF_THRESH = 25
DIFF_MIN_AREA = 60
# Снимок, где областей вне рамок больше этого, считается несовмещённым с `PCB_USED/` — разность не применима.
DIFF_MAX_REGIONS = 20
N_VIEWS = 24
SCENE_HW = (6144, 8192)


def _file(path: str) -> dict:
    p = Path(path)
    raw = p.read_bytes()
    a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    r = {"path": str(p.relative_to(ROOT)), "ok": a is not None, "md5": hashlib.md5(raw).hexdigest()}
    if a is None:
        return r
    r["hw"] = list(a.shape[:2])
    r["channels"] = 1 if a.ndim == 2 else a.shape[2]
    if p.suffix == ".jpg":
        eoi = raw.rfind(b"\xff\xd9")
        r["eoi"] = eoi > 0
        r["trailer"] = raw[-8:-2].decode("latin1") if eoi > 0 and len(raw) - eoi > 2 else ""
    else:
        g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        r["n_values"] = len(np.unique(g))
        r["n_intermediate"] = int(((g > 0) & (g < 255)).sum())
        if a.ndim == 3 and a.shape[2] == 4:
            r["alpha_opaque"] = bool((a[..., 3] == 255).all())
        fg = g > 127
        r["fg_frac"] = float(fg.mean())
        if fg.any():
            ys, xs = np.nonzero(fg)
            r["box"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return r


def _xml(p: Path) -> dict:
    t = ET.parse(p).getroot()
    objs = []
    for o in t.findall("object"):
        bb = o.find("bndbox")
        objs.append({"name": o.findtext("name"), "difficult": o.findtext("difficult"),
                     "box": [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]})
    s = t.find("size")
    return {"path": str(p.relative_to(ROOT)), "filename": t.findtext("filename"),
            "wh": [int(s.findtext("width")), int(s.findtext("height"))], "objects": objs}


def main_hr_insdet() -> None:
    files = sorted(str(p) for p in ROOT.rglob("*") if p.suffix in (".jpg", ".png"))
    with ProcessPoolExecutor(8) as ex:
        info = {Path(r["path"]).as_posix(): r for r in ex.map(_file, files, chunksize=4)}
    xmls = [_xml(p) for p in sorted(ROOT.glob("Scenes/*/*/*.xml"))]

    objects = sorted(p.name for p in (ROOT / "Objects").iterdir() if p.is_dir())
    per_object, mask_dirs, extra_files = {}, {}, []
    for o in objects:
        d = ROOT / "Objects" / o
        mdir = next(n for n in ("masks", "mask") if (d / n).is_dir())
        mask_dirs[o] = mdir
        extra_files += [str(p.relative_to(ROOT)) for p in d.rglob("*") if p.is_file() and p.suffix not in (".jpg", ".png")]
        imgs = sorted(p.stem for p in (d / "images").glob("*.jpg"))
        masks = sorted(p.stem for p in (d / mdir).glob("*.png"))
        sizes = {tuple(info[f"Objects/{o}/images/{s}.jpg"]["hw"]) for s in imgs}
        size_ok = all(info[f"Objects/{o}/images/{s}.jpg"]["hw"] == info[f"Objects/{o}/{mdir}/{s}.png"]["hw"]
                      for s in set(imgs) & set(masks))
        per_object[o] = {"images": len(imgs), "masks": len(masks), "paired": imgs == masks,
                         "hw": sorted(sizes), "mask_hw_matches": size_ok}

    masks = [r for k, r in info.items() if k.endswith(".png")]
    jpgs = [r for k, r in info.items() if k.endswith(".jpg")]
    scenes = collections.Counter(str(Path(x["path"]).parent) for x in xmls)
    scene_jpgs = collections.Counter(str(Path(k).parent) for k in info if k.startswith("Scenes/"))
    names = collections.Counter(o["name"] for x in xmls for o in x["objects"])
    by_folder = collections.Counter()
    for x in xmls:
        by_folder[str(Path(x["path"]).parent)] += len(x["objects"])
    per_image = sorted(len(x["objects"]) for x in xmls)
    md5 = collections.defaultdict(list)
    for k, r in info.items():
        md5[r["md5"]].append(k)

    report = {
        "written": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "root": str(ROOT),
        "files": {"jpg": len(jpgs), "png": len(masks), "xml": len(xmls), "other": extra_files},
        "undecodable": [r["path"] for r in info.values() if not r["ok"]],
        "duplicates_md5": [v for v in md5.values() if len(v) > 1],
        "jpeg_without_eoi": [r["path"] for r in jpgs if not r["eoi"]],
        "jpeg_trailers": dict(collections.Counter(r["trailer"] for r in jpgs)),
        "objects": {
            "n": len(objects),
            "incomplete": {o: v for o, v in per_object.items()
                           if not (v["images"] == v["masks"] == N_VIEWS and v["paired"] and v["mask_hw_matches"])},
            "mask_dir_not_masks": {o: m for o, m in mask_dirs.items() if m != "masks"},
            "hw": {"x".join(map(str, hw)): sorted(o for o, v in per_object.items() if hw in v["hw"])
                   for hw in {tuple(h) for v in per_object.values() for h in v["hw"]}},
        },
        "masks": {
            "channels": dict(collections.Counter(r["channels"] for r in masks)),
            "rgba_not_opaque": [r["path"] for r in masks if r.get("alpha_opaque") is False],
            "nonbinary": {r["path"]: r["n_intermediate"] for r in masks if r["n_intermediate"]},
            "empty": [r["path"] for r in masks if r["fg_frac"] == 0],
            "fg_frac_min_max": [min(r["fg_frac"] for r in masks), max(r["fg_frac"] for r in masks)],
            "touch_frame": [r["path"] for r in masks if "box" in r and (
                r["box"][0] == 0 or r["box"][1] == 0 or r["box"][2] == r["hw"][1] or r["box"][3] == r["hw"][0])],
        },
        "background": {"n": sum(k.startswith("Background/") for k in info),
                       "hw": sorted({tuple(r["hw"]) for k, r in info.items() if k.startswith("Background/")})},
        "scenes": {
            "images": dict(scene_jpgs),
            "xml": dict(scenes),
            "hw": sorted({tuple(r["hw"]) for k, r in info.items() if k.startswith("Scenes/")}),
            "xml_size_matches": all(tuple(x["wh"][::-1]) == SCENE_HW for x in xmls),
            "xml_filename_matches": all(x["filename"] == Path(x["path"]).with_suffix(".jpg").name for x in xmls),
            "boxes": sum(names.values()),
            "boxes_by_folder": dict(by_folder),
            "difficult": dict(collections.Counter(o["difficult"] for x in xmls for o in x["objects"])),
            "boxes_outside_frame": [x["path"] for x in xmls for o in x["objects"]
                                    if not (0 <= o["box"][0] < o["box"][2] <= SCENE_HW[1]
                                            and 0 <= o["box"][1] < o["box"][3] <= SCENE_HW[0])],
            "per_image_min_median_max": [per_image[0], per_image[len(per_image) // 2], per_image[-1]],
            "names_not_in_objects": {n: [x["path"] for x in xmls for o in x["objects"] if o["name"] == n]
                                     for n in names if n not in objects},
            "objects_never_annotated": [o for o in objects if o not in names],
            "same_name_twice_in_image": sum(c > 1 for x in xmls
                                            for c in collections.Counter(o["name"] for o in x["objects"]).values()),
        },
    }
    _write(OUT, report)
    print(json.dumps({k: report[k] for k in ("files", "undecodable", "duplicates_md5")}, ensure_ascii=False))
    print("objects.incomplete:", report["objects"]["incomplete"])
    print("scenes.names_not_in_objects:", report["scenes"]["names_not_in_objects"])


def _write(out: Path, report: dict) -> None:
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(out)


def _pcb_file(path: str) -> dict:
    p = Path(path)
    raw = p.read_bytes()
    a = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    r = {"path": str(p.relative_to(PCB_ROOT)), "ok": a is not None, "md5": hashlib.md5(raw).hexdigest()}
    if a is not None:
        r["hw"] = list(a.shape[:2])
        r["channels"] = 1 if a.ndim == 2 else a.shape[2]
    return r


def _pcb_diff(item: tuple[str, str, list]) -> dict:
    """Области, где снимок отличается от бездефектной платы, вне всех рамок разметки."""
    cls, stem, boxes = item
    a = cv2.imread(str(PCB_ROOT / "images" / cls / f"{stem}.jpg")).astype(np.int16)
    t = cv2.imread(str(PCB_ROOT / "PCB_USED" / f"{stem[:2]}.JPG")).astype(np.int16)
    d = cv2.GaussianBlur(np.abs(a - t).max(2).astype(np.uint8), (5, 5), 0)
    m = cv2.dilate((d > DIFF_THRESH).astype(np.uint8), np.ones((9, 9), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    hit, outside = [False] * len(boxes), []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in st[i])
        if area < DIFF_MIN_AREA:
            continue
        inside = [j for j, (x0, y0, x1, y1) in enumerate(boxes) if x < x1 and x + w > x0 and y < y1 and y + h > y0]
        for j in inside:
            hit[j] = True
        if not inside:
            outside.append([x, y, x + w, y + h])
    return {"image": stem, "median_diff": float(np.median(d)), "outside": outside,
            "boxes_without_change": [j for j, v in enumerate(hit) if not v]}


def main_pcb() -> None:
    imgs = sorted(str(p) for p in (PCB_ROOT / "images").rglob("*") if p.is_file())
    used = sorted(str(p) for p in (PCB_ROOT / "PCB_USED").iterdir() if p.is_file())
    with ProcessPoolExecutor(8) as ex:
        info = {r["path"]: r for r in ex.map(_pcb_file, imgs + used, chunksize=4)}
    rot = sorted(p for p in (PCB_ROOT / "rotation").rglob("*.jpg"))

    problems, boxes = [], []
    per_board_images = collections.defaultdict(collections.Counter)
    for c in PCB_CLASSES:
        stems_img = sorted(p.stem for p in (PCB_ROOT / "images" / c).iterdir())
        stems_xml = sorted(p.stem for p in (PCB_ROOT / "Annotations" / c).glob("*.xml"))
        if stems_img != stems_xml:
            problems.append({"class": c, "images_without_xml": sorted(set(stems_img) - set(stems_xml)),
                             "xml_without_image": sorted(set(stems_xml) - set(stems_img))})
        for stem in stems_xml:
            board, rest = stem.split("_", 1)
            if rest.rsplit("_", 1)[0].lower() != c.lower():
                problems.append({"file": stem, "issue": "класс в имени файла не совпадает с папкой"})
            per_board_images[board][c] += 1
            t = ET.parse(PCB_ROOT / "Annotations" / c / f"{stem}.xml").getroot()
            img_key = f"images/{c}/{stem}.jpg"
            hw = info[img_key]["hw"] if img_key in info else None
            s = t.find("size")
            wh = [int(s.findtext("width")), int(s.findtext("height"))]
            if t.findtext("filename") != f"{stem}.jpg":
                problems.append({"file": stem, "issue": "filename в XML", "value": t.findtext("filename")})
            if hw is None or wh != hw[::-1]:
                problems.append({"file": stem, "issue": "size в XML не совпадает со снимком", "xml": wh, "img": hw})
            for o in t.findall("object"):
                bb = o.find("bndbox")
                x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                boxes.append({"image": stem, "board": board, "class": c, "name": o.findtext("name"),
                              "difficult": o.findtext("difficult"), "box": [x0, y0, x1, y1],
                              "in_frame": 0 <= x0 < x1 <= wh[0] and 0 <= y0 < y1 <= wh[1],
                              "area": (x1 - x0) * (y1 - y0)})

    names = collections.Counter((b["class"], b["name"]) for b in boxes)
    per_board_boxes = collections.defaultdict(collections.Counter)
    for b in boxes:
        per_board_boxes[b["board"]][b["class"]] += 1
    per_image = collections.Counter(b["image"] for b in boxes)
    hw_by_board = collections.defaultdict(set)
    for k, r in info.items():
        if k.startswith("images/"):
            hw_by_board[Path(k).stem.split("_", 1)[0]].add(tuple(r["hw"]))
    areas = np.array([b["area"] for b in boxes])
    by_image = collections.defaultdict(list)
    for b in boxes:
        by_image[(b["class"], b["image"])].append(b["box"])
    with ProcessPoolExecutor(8) as ex:
        diff = list(ex.map(_pcb_diff, [(c, s, bx) for (c, s), bx in sorted(by_image.items())], chunksize=4))
    misaligned = [r["image"] for r in diff if len(r["outside"]) > DIFF_MAX_REGIONS]
    # Область, повторяющаяся на одном месте платы в нескольких снимках, — отличие самого шаблона, а не дефект.
    rep = collections.Counter((r["image"][:2], tuple(v // 20 for v in o[:2]))
                              for r in diff if r["image"] not in misaligned for o in r["outside"])
    unlabeled = {r["image"]: [o for o in r["outside"] if rep[(r["image"][:2], tuple(v // 20 for v in o[:2]))] == 1]
                 for r in diff if r["image"] not in misaligned}
    unlabeled = {k: v for k, v in unlabeled.items() if v}
    md5 = collections.defaultdict(list)
    for k, r in info.items():
        md5[r["md5"]].append(k)
    rot_names = {p.name for p in rot}

    report = {
        "written": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "root": str(PCB_ROOT),
        "files": {"images": len(imgs), "xml": sum(1 for _ in (PCB_ROOT / "Annotations").rglob("*.xml")),
                  "pcb_used": len(used), "rotation_jpg": len(rot)},
        "undecodable": [r["path"] for r in info.values() if not r["ok"]],
        "duplicates_md5": [v for v in md5.values() if len(v) > 1],
        "problems": problems,
        "channels": dict(collections.Counter(r["channels"] for r in info.values() if r["ok"])),
        "rotation_names_equal_to_images": len(rot_names & {Path(p).name for p in imgs}),
        "boards": {b: {"images": sum(per_board_images[b].values()),
                       "images_by_class": dict(sorted(per_board_images[b].items())),
                       "boxes_by_class": dict(sorted(per_board_boxes[b].items())),
                       "hw": sorted(hw_by_board[b])}
                   for b in sorted(per_board_images)},
        "unlabeled_by_diff": {
            "method": {"thresh": DIFF_THRESH, "min_area": DIFF_MIN_AREA, "max_regions": DIFF_MAX_REGIONS},
            "misaligned_images": misaligned,
            "template_artifacts": [f"{b}@{xy[0] * 20},{xy[1] * 20}: {n} снимков" for (b, xy), n in rep.items() if n > 1],
            "gt_boxes_without_change": {r["image"]: r["boxes_without_change"] for r in diff if r["boxes_without_change"]},
            "per_board": {b: sum(len(v) for k, v in unlabeled.items() if k[:2] == b) for b in sorted(per_board_images)},
            "candidates": unlabeled,
        },
        "pcb_used_hw": {Path(k).name: r["hw"] for k, r in info.items() if k.startswith("PCB_USED/")},
        "boxes": {
            "n": len(boxes),
            "names": {f"{c}:{n}": k for (c, n), k in sorted(names.items())},
            "difficult": dict(collections.Counter(b["difficult"] for b in boxes)),
            "outside_frame": [b["image"] for b in boxes if not b["in_frame"]],
            "per_image_min_median_max": [min(per_image.values()), int(np.median(list(per_image.values()))),
                                         max(per_image.values())],
            "images_with_mixed_classes": sorted({b["image"] for b in boxes if b["name"].lower() != b["class"].lower()}),
            "area_px": {"min": float(areas.min()), "median": float(np.median(areas)), "max": float(areas.max()),
                        "tertiles": [float(x) for x in np.quantile(areas, [1 / 3, 2 / 3])],
                        "tertiles_sqrt": [float(x) for x in np.sqrt(np.quantile(areas, [1 / 3, 2 / 3]))]},
        },
    }
    _write(PCB_OUT, report)
    print(json.dumps({k: report[k] for k in ("files", "undecodable", "duplicates_md5", "problems")}, ensure_ascii=False))
    print("boxes:", json.dumps(report["boxes"], ensure_ascii=False))
    u = report["unlabeled_by_diff"]
    print("unlabeled_by_diff:", json.dumps({k: u[k] for k in ("misaligned_images", "template_artifacts", "per_board")},
                                           ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hr_insdet", "pcb"], default="hr_insdet")
    {"hr_insdet": main_hr_insdet, "pcb": main_pcb}[ap.parse_args().dataset]()
