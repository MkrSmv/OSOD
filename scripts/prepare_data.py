"""Эталоны с рамками и разбиения → `splits/<dataset>.json`.

Состав галереи и разбиения — по помещениям HR-InsDet и платам PKU-Market-PCB (§4.1).
Ничего случайного здесь нет: разбиения заданы помещениями и платами. `seed` записывается
для выборок следующих этапов (пилотный замер, подвыборки галереи 4.5)
и объявлен до первого прогона. `data/` только читается.

    python scripts/prepare_data.py --dataset hr_insdet
    python scripts/prepare_data.py --dataset pcb
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import hr_insdet, pcb

SEED = 20260918
OUT = Path("splits")


def _write(name: str, d: dict) -> None:
    OUT.mkdir(exist_ok=True)
    p = OUT / f"{name}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(p)
    print(p, {k: len(v) for k, v in d.items() if isinstance(v, list)})


def ref_geometry(refs: list[dict]) -> dict:
    """Сколько места вокруг объекта в сыром снимке: основание решения об эталоне."""
    side = np.array([max(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) / r["hw"][1] for r in refs])

    def fits(r: dict, a: float) -> bool:
        (x0, y0, x1, y1), (h, w) = r["box"], r["hw"]
        half = a * max(x1 - x0, y1 - y0) / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return cx - half >= 0 and cy - half >= 0 and cx + half <= w and cy + half <= h

    return {"side_frac_min_q10_median_q90_max": [round(float(q), 3) for q in
                                                 np.quantile(side, [0, 0.1, 0.5, 0.9, 1])],
            "square_fits_frame": {str(a): round(float(np.mean([fits(r, a) for r in refs])), 3) for a in (1.0, 1.5)},
            "side_below_448_if_frame_to_1024": round(float(np.mean(side * 1024 < 448)), 3),
            "box_touches_frame": [r["id"] for r in refs if r["box"][0] == 0 or r["box"][1] == 0
                                  or r["box"][2] == r["hw"][1] or r["box"][3] == r["hw"][0]]}


def prepare_hr_insdet() -> None:
    labels = hr_insdet.objects()
    refs = hr_insdet.references()
    sc = hr_insdet.scenes()
    rooms = sorted({s["room"] for s in sc})
    assert set(hr_insdet.CAL_ROOMS) <= set(rooms)
    cal = [s["id"] for s in sc if s["room"] in hr_insdet.CAL_ROOMS]
    test = [s["id"] for s in sc if s["room"] not in hr_insdet.CAL_ROOMS]
    gt = {s["id"]: hr_insdet.scene_gt(s, labels) for s in sc}
    _write("hr_insdet", {
        "dataset": "hr_insdet",
        "written": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "root": str(hr_insdet.ROOT),
        "seed": SEED,
        "labels": labels,
        "rooms": {"cal": hr_insdet.CAL_ROOMS, "test": [r for r in rooms if r not in hr_insdet.CAL_ROOMS]},
        "counts": {"gallery": len(refs), "cal_scenes": len(cal), "test_scenes": len(test),
                   "gt_boxes_cal": sum(len(gt[i]) for i in cal), "gt_boxes_test": sum(len(gt[i]) for i in test),
                   "iscrowd": sum(g["iscrowd"] for v in gt.values() for g in v)},
        "ref_geometry": ref_geometry(refs),
        "gallery": refs,
        "cal": cal,
        "test": test,
    })


def prepare_pcb() -> None:
    imgs = pcb.images()
    boards = {i["board"] for i in imgs}
    roles = pcb.BOARDS
    assert sorted(b for v in roles.values() for b in v) == sorted(boards), "платы по ролям не покрывают раздачу"
    refs = pcb.references()
    per_type = collections.Counter(r["label"] for r in refs)
    split = {role: [i["id"] for i in imgs if i["board"] in b] for role, b in roles.items()}
    _write("pcb", {
        "dataset": "pcb",
        "written": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "root": str(pcb.ROOT),
        "seed": SEED,
        "labels": pcb.CLASSES,
        "boards": roles,
        "clean_images": {b: pcb.clean_image(b) for r in ("cal", "test") for b in roles[r]},
        "counts": {"gallery_images": len(split["gallery"]), "gallery": len(refs),
                   "gallery_per_type": dict(sorted(per_type.items())),
                   "cal_images": len(split["cal"]), "test_images": len(split["test"])},
        "gallery": refs,
        "gallery_images": split["gallery"],
        "cal": split["cal"],
        "test": split["test"],
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["hr_insdet", "pcb"], required=True)
    {"hr_insdet": prepare_hr_insdet, "pcb": prepare_pcb}[ap.parse_args().dataset]()
