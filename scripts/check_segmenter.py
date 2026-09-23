"""SAM 2 под bf16-autocast против fp32 и повторяемость.

Пишет в `experiments/env.json` два раздела:

- `sam2_precision` — 3 калибровочные сцены HR-InsDet (по seed разбиения), `crop_n_layers=0`: число масок в каждой
  точности и доля масок fp32, у которых есть маска autocast с IoU ≥ 0,9 (и обратная доля);
- `sam2_repeatability` — одна сцена, `crop_n_layers` 0 и 1: два запуска с одними параметрами обязаны дать побитово
  те же маски. Отпечатки первого запуска скрипта сохраняются; повторный запуск скрипта сверяет с ними свои —
  это повторяемость между процессами, которая и нужна прерываемому счёту.

Маски сверки в кеш не идут. Сцены — калибровочные: маски тестовых сцен до построения кеша масок не строятся.

    python scripts/check_segmenter.py

Вторая часть — после построения кеша масок: полнота по размеченным рамкам на 40 калибровочных сценах
(`crop_n_layers=1`) по маскам из кеша, под autocast и в fp32 → `sam2_precision.recall_cal`. Величина — та же, что в
4.2: доля рамок, для которых среди всех масок $M(I)$ есть маска с IoU описывающей рамки ≥ 0,5.
Порог расхождения записан до счёта и по результату не пересматривается.

    python scripts/check_segmenter.py recall

Третья часть — сегментация PKU-Market-PCB, перед построением её кеша масок → `sam2_pcb`: на снимках,
выбранных по seed разбиения (пять снимков пяти тестовых плат, по три снимка каждой эталонной платы), — повторный
счёт авторежима против записи кеша (побитово), число масок и время на снимок; у эталонов —
маска отдельного вызова `predict` после общего `set_image` (так пишется кеш) против отдельного `set_image` на каждую
рамку (обязаны совпасть побитово) и против батча всех рамок снимка.
Записи кеша, которых ещё нет, скрипт не создаёт: сначала `scripts/segment.py --dataset pcb … --only <снимки>`.

    python scripts/check_segmenter.py pcb
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as rle_api

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import env
from src.data import hr_insdet, pcb
from src.segment import cache as MC
from src.eval.boxes import best_iou_per_gt
from src.segment import sam2 as S

# --- записано до счёта, по результату не пересматривается ---
RECALL_IOU = 0.5
RECALL_DIFF_MAX = 0.01  # 1 п. п.: больше — режим не меняется автоматически, нужен разбор
RECALL_CROP_N_LAYERS = 1

N_SCENES = 3
IOU_MATCH = 0.9  # маски совпадают при IoU не ниже


def _rles(recs: list[dict]) -> list[dict]:
    return [{"size": r["rle"]["size"], "counts": r["rle"]["counts"].encode("ascii")} for r in recs]


def _matched(a: list[dict], b: list[dict]) -> dict:
    """Доля масок `a`, у которых в `b` есть маска с IoU ≥ 0,9; квантили наибольшего IoU.

    Для масок без пары — их оценки: маска у порога фильтра генератора (`pred_iou_thresh`, `stability_score_thresh`)
    проходит его в одной точности и не проходит в другой, и это видно по оценкам.
    """
    if not a or not b:
        return {"frac": 0.0, "best_iou_min_q05_median": [], "unmatched": []}
    best = rle_api.iou(_rles(a), _rles(b), [0] * len(b)).max(1)
    unmatched = [{"best_iou": round(float(best[k]), 3), "predicted_iou": round(a[k]["predicted_iou"], 3),
                  "stability_score": round(a[k]["stability_score"], 3), "area_px": round(a[k]["area"])}
                 for k in np.flatnonzero(best < IOU_MATCH)]
    return {"frac": float((best >= IOU_MATCH).mean()),
            "best_iou_min_q05_median": [float(v) for v in np.quantile(best, [0, 0.05, 0.5])], "unmatched": unmatched}


def _digest(recs: list[dict]) -> str:
    rows = [(r["rle"]["counts"], r["box"], r["predicted_iou"], r["stability_score"]) for r in recs]
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()


def _write_env(update) -> None:
    new = update(json.loads(env.ENV_JSON.read_text()))
    tmp = env.ENV_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n")  # отступ — как у файла в репозитории
    tmp.replace(env.ENV_JSON)


def recall_cal(allow_partial: bool) -> None:
    """Полнота на калибровочных сценах по кешу масок: bf16-autocast против fp32."""
    sp = json.loads(Path("splits/hr_insdet.json").read_text())
    labels = hr_insdet.objects()
    scenes = {x["id"]: x for x in hr_insdet.scenes()}
    caches = {"autocast": MC.MaskCache("hr_insdet", MC.auto_key(RECALL_CROP_N_LAYERS)),
              "fp32": MC.MaskCache("hr_insdet", MC.auto_key(RECALL_CROP_N_LAYERS, fp32=True))}
    ids = [i for i in sp["cal"] if all(c.has(i) for c in caches.values())]
    if len(ids) < len(sp["cal"]) and not allow_partial:
        sys.exit(f"в кеше обе точности есть у {len(ids)} из {len(sp['cal'])} калибровочных сцен — сначала пачка")
    best = {k: [] for k in caches}
    n_masks = {k: 0 for k in caches}
    for i in ids:
        gt = np.array([g["box"] for g in hr_insdet.scene_gt(scenes[i], labels)], float).reshape(-1, 4)
        for k, c in caches.items():
            e = c.load(i)
            best[k].append(best_iou_per_gt(gt, e.boxes))
            n_masks[k] += len(e)
    out = {"split": "cal", "n_scenes": len(ids), "crop_n_layers": RECALL_CROP_N_LAYERS, "iou": RECALL_IOU,
           "diff_max_recorded_before": RECALL_DIFF_MAX, "n_gt_boxes": int(sum(len(b) for b in best["autocast"]))}
    for k in caches:
        b = np.concatenate(best[k])
        out[k] = {"n_masks": n_masks[k], "recall": float((b >= RECALL_IOU).mean()),
                  "recall_iou75": float((b >= 0.75).mean())}
    out["recall_diff_fp32_minus_autocast"] = out["fp32"]["recall"] - out["autocast"]["recall"]
    out["within_recorded_limit"] = bool(abs(out["recall_diff_fp32_minus_autocast"]) <= RECALL_DIFF_MAX)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if allow_partial:
        print("неполный состав: в env.json не записано")
        return
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    _write_env(lambda old: {**old, "sam2_precision": {**old["sam2_precision"], "recall_cal": {"written": today, **out}}})
    if not out["within_recorded_limit"]:
        sys.exit("РАСХОЖДЕНИЕ ПОЛНОТЫ БОЛЬШЕ ЗАПИСАННОГО ПОРОГА")


def pcb_sample(sp: dict) -> tuple[list[str], list[str]]:
    """Состав сверки PCB — только по seed разбиения: снимки пяти тестовых плат и по три снимка эталонных плат."""
    info = {i["id"]: i for i in pcb.images()}
    rng = np.random.default_rng(sp["seed"])
    boards = list(rng.permutation(pcb.BOARDS["test"]))[:5]
    types = list(rng.permutation(pcb.CLASSES))

    def pick(pool: list[str], board: str, label: str) -> str:
        return str(rng.choice(sorted(i for i in pool if info[i]["board"] == board and info[i]["label"] == label)))

    test = [pick(sp["test"], b, t) for b, t in zip(boards, types)]
    refs = [pick(sp["gallery_images"], b, t) for b in pcb.BOARDS["gallery"] for t in list(rng.permutation(pcb.CLASSES))[:3]]
    return test, refs


def pcb_check() -> None:
    sp = json.loads(Path("splits/pcb.json").read_text())
    info = {i["id"]: i for i in pcb.images()}
    test, ref_imgs = pcb_sample(sp)
    out = {"code": env.code_stamp(), "rule": pcb_sample.__doc__, "test_images": test, "ref_images": ref_imgs}

    # авторежим: число масок и время — из записей кеша; повторный счёт одного снимка — побитово та же запись
    out["auto"] = {}
    for cnl in (1, 0):
        mc = MC.MaskCache("pcb", MC.auto_key(cnl))
        ents = [mc.load(i) for i in test]
        out["auto"][f"crop_n_layers={cnl}"] = {
            "n_masks": {e.image_id: len(e) for e in ents},
            "sec_generate_mean": float(np.mean([e.info["sec_generate"] for e in ents])),
            "vram_peak_mib_max": max(e.info["vram_peak_mib"] for e in ents)}
    gen = S.build_generator(0)
    MC.check_generator(gen, MC.auto_key(0))
    again, _ = S.generate(gen, S.read_rgb(pcb.ROOT / info[test[0]]["image"]))
    out["auto"]["repeat"] = {"image": test[0], "crop_n_layers": 0,
                             "bitwise_identical_to_cache": again == MC.MaskCache("pcb", MC.auto_key(0)).load(test[0]).records}
    del gen
    torch.cuda.empty_cache()

    # эталоны: запись кеша против отдельного `set_image` на рамку и против батча рамок снимка
    box, pred = MC.MaskCache("pcb", MC.box_key()), S.build_predictor()
    keys = ("rle", "box", "area", "predicted_iou", "stability_score", "prompt_box")
    rows = []
    for iid in ref_imgs:
        img = S.read_rgb(pcb.ROOT / info[iid]["image"])
        refs = [r for r in sp["gallery"] if r["id"].rsplit("/", 1)[0] == iid]
        batch, _ = S.predict_boxes(pred, img, [r["box"] for r in refs])
        for r, rb in zip(refs, batch):
            e = box.load(r["id"], [r["box"]])
            c = e.records[0]
            alone = S.predict_boxes(pred, img, [r["box"]])[0][0]
            a, b = S.decode(c["rle"]), S.decode(rb["rle"])
            x0, y0, x1, y1 = r["box"]
            rows.append({"ref": r["id"], "type": r["label"], "own_set_image_identical": all(c[k] == alone[k] for k in keys),
                         "batch_identical": all(c[k] == rb[k] for k in keys),
                         "batch_mask_iou": float((a & b).sum() / max(1, (a | b).sum())),
                         "mask_area_to_box_area": c["area"] / ((x1 - x0) * (y1 - y0)), "predicted_iou": c["predicted_iou"],
                         "sec_set_image": e.info["sec_set_image"], "sec_predict": e.info["sec_predict"]})
    ratio = np.array([r["mask_area_to_box_area"] for r in rows])
    out["box"] = {
        "n_refs": len(rows), "own_set_image_identical": sum(r["own_set_image_identical"] for r in rows),
        "batch_identical": sum(r["batch_identical"] for r in rows),
        "batch_mask_iou_min": min(r["batch_mask_iou"] for r in rows), "n_empty": int((ratio == 0).sum()),
        "mask_area_to_box_area_min_median_max": [float(ratio.min()), float(np.median(ratio)), float(ratio.max())],
        "predicted_iou_median_by_type": {t: float(np.median([r["predicted_iou"] for r in rows if r["type"] == t]))
                                         for t in pcb.CLASSES if any(r["type"] == t for r in rows)},
        "sec_set_image_median": float(np.median([r["sec_set_image"] for r in rows])),
        "sec_predict_median": float(np.median([r["sec_predict"] for r in rows])),
        "rows": rows}
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    _write_env(lambda cur: {**cur, "sam2_pcb": {"written": today, **out}})
    print(json.dumps({k: ({n: v for n, v in out[k].items() if n != "rows"} if isinstance(out[k], dict) else out[k])
                      for k in out}, ensure_ascii=False, indent=1))
    if not out["auto"]["repeat"]["bitwise_identical_to_cache"] or out["box"]["own_set_image_identical"] != len(rows):
        sys.exit("МАСКИ PCB НЕ ПОВТОРЯЮТСЯ ПОБИТОВО")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", nargs="?", default="precision", choices=["precision", "recall", "pcb"])
    ap.add_argument("--allow-partial", action="store_true", help="recall по тем сценам, что есть; без записи в env.json")
    args = ap.parse_args()
    if args.part == "recall":
        return recall_cal(args.allow_partial)
    if args.part == "pcb":
        return pcb_check()

    sp = json.loads(Path("splits/hr_insdet.json").read_text())
    scenes = sorted(np.random.default_rng(sp["seed"]).choice(sorted(sp["cal"]), N_SCENES, replace=False).tolist())
    imgs = {i: S.read_rgb(hr_insdet.ROOT / "Scenes" / f"{i}.jpg") for i in scenes}
    old = json.loads(env.ENV_JSON.read_text())

    # «fp32» — веса и активации fp32 без autocast; свёртки и матричные произведения — как разрешено средой (TF32)
    precision = {"scenes": scenes, "split": "cal", "crop_n_layers": 0, "points_per_batch": S.POINTS_PER_BATCH,
                 "iou_match": IOU_MATCH, "fp32_tf32": {"cudnn": torch.backends.cudnn.allow_tf32,
                                                        "matmul": torch.backends.cuda.matmul.allow_tf32},
                 "by_scene": {}}
    digests, identical = {}, {}
    for cnl in (0, 1):
        gen = S.build_generator(cnl)
        MC.check_generator(gen, MC.auto_key(cnl))
        first = None
        for i in scenes if cnl == 0 else scenes[:1]:
            a, info_a = S.generate(gen, imgs[i])
            first = a if first is None else first
            if cnl == 0:
                f, info_f = S.generate(gen, imgs[i], autocast=False)
                precision["by_scene"][i] = {
                    "n_masks_autocast": len(a), "n_masks_fp32": len(f),
                    "fp32_matched_in_autocast": _matched(f, a), "autocast_matched_in_fp32": _matched(a, f),
                    "sec_autocast": info_a["sec_generate"], "sec_fp32": info_f["sec_generate"],
                    "vram_peak_mib_fp32": info_f["vram_peak_mib"]}
                print(i, precision["by_scene"][i], flush=True)
        again, _ = S.generate(gen, imgs[scenes[0]])
        digests[f"crop_n_layers={cnl}"] = _digest(first)
        identical[f"crop_n_layers={cnl}"] = _digest(again) == _digest(first)
        del gen
        torch.cuda.empty_cache()

    rows = list(precision["by_scene"].values())
    precision["summary"] = {
        "n_masks_autocast": sum(r["n_masks_autocast"] for r in rows), "n_masks_fp32": sum(r["n_masks_fp32"] for r in rows),
        "fp32_matched_in_autocast_min": min(r["fp32_matched_in_autocast"]["frac"] for r in rows),
        "autocast_matched_in_fp32_min": min(r["autocast_matched_in_fp32"]["frac"] for r in rows)}
    repeat = {"scene": scenes[0], "same_process_bitwise_identical": identical, "digests": digests}
    prev = old.get("sam2_repeatability", {}).get("digests")
    if prev and old["sam2_repeatability"].get("scene") == scenes[0]:
        repeat["digests"] = prev  # отпечатки первого процесса остаются опорными
        repeat["cross_process_bitwise_identical"] = {k: digests[k] == prev.get(k) for k in digests}
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    keep = {k: v for k, v in old.get("sam2_precision", {}).items() if k == "recall_cal"}
    _write_env(lambda cur: {**cur, "sam2_precision": {"written": today, **precision, **keep},
                            "sam2_repeatability": {"written": today, **repeat}})
    print(json.dumps({"summary": precision["summary"], "repeatability": repeat}, ensure_ascii=False, indent=1))
    flags = list(identical.values()) + list(repeat.get("cross_process_bitwise_identical", {}).values())
    if not all(flags):
        sys.exit("МАСКИ НЕ ПОВТОРЯЮТСЯ ПОБИТОВО")


if __name__ == "__main__":
    main()
