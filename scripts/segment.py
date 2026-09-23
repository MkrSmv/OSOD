"""Кеш масок SAM 2 → `cache/masks/<dataset>/<каталог ключа>/`.

Авторежим — сцены, где нужны запросы и дистракторы (по умолчанию тестовые и калибровочные); промпт-рамка —
эталоны галереи. Посчитанное пропускается: снимок считается, только если записи с тем же ключом ещё нет,
поэтому прерванный запуск продолжается той же командой. Если считать нечего, модель не загружается.

    python scripts/segment.py --dataset hr_insdet --crop-n-layers 1
    python scripts/segment.py --dataset hr_insdet --crop-n-layers 0 --split test
    python scripts/segment.py --dataset hr_insdet --crop-n-layers 1 --only hard/office_001/rgb_003
    python scripts/segment.py --dataset hr_insdet --mode box                 # маски эталонов по рамке
    python scripts/segment.py --dataset hr_insdet --crop-n-layers 1 --status # что уже в кеше; GPU не нужен
    python scripts/segment.py --dataset hr_insdet --crop-n-layers 1 --split cal --fp32   # сверка точности, свой каталог
    python scripts/segment.py --dataset pcb --crop-n-layers 1                # тестовые платы PCB (361 снимок)
    python scripts/segment.py --dataset pcb --mode box                       # 716 эталонов на 240 снимках плат 01 и 04

PCB: авторежим — только тестовые платы; калибровочные платы 09 и 10
в авторежиме не сегментируются вовсе. Эталоны — рамки разметки дефектов на снимках эталонных плат: запись кеша — на эталон, как у HR-InsDet,
`set_image` — один на снимок, `predict` — отдельный на каждую рамку.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import hr_insdet, pcb
from src.segment import cache as MC

# Авторежим: какие части разбиения сегментируются. У PCB — только тестовые платы.
AUTO_SPLITS = {"hr_insdet": ["test", "cal"], "pcb": ["test"]}


def _work(dataset: str, mode: str, splits: list[str]) -> list[tuple[str, Path, list[tuple[str, list | None]]]]:
    """Список работы по снимкам: (снимок, путь, записи кеша на нём) — только из `splits/*.json`.

    Запись кеша — (идентификатор, рамки промпта либо None). В авторежиме и у эталонов HR-InsDet запись на снимке
    одна; у эталонов PCB — по записи на рамку разметки, в порядке рамок в XML (`splits/pcb.json`, `gallery`).
    """
    sp = json.loads(Path(f"splits/{dataset}.json").read_text())
    if dataset == "hr_insdet":
        if mode == "box":
            return [(r["id"], hr_insdet.ROOT / r["image"], [(r["id"], [r["box"]])]) for r in sp["gallery"]]
        return [(i, hr_insdet.ROOT / "Scenes" / f"{i}.jpg", [(i, None)]) for s in splits for i in sp[s]]
    image = {i["id"]: i for i in pcb.images()}
    wanted = {r["id"].rsplit("/", 1)[0] for r in sp["gallery"]} | {i for s in splits for i in sp[s]}
    missing = sorted(wanted - set(image))
    if missing:
        raise ValueError(f"снимков из splits/pcb.json нет в раздаче: {missing[:5]} — выполнить scripts/prepare_data.py")
    if mode == "box":
        by_image: dict[str, list] = {}
        for r in sp["gallery"]:
            iid, k = r["id"].rsplit("/", 1)
            if image[iid]["board"] not in pcb.BOARDS["gallery"] or r["image"] != image[iid]["image"]:
                raise ValueError(f"{r['id']}: эталон не с эталонной платы либо путь снимка расходится с раздачей")
            if int(k) != len(by_image.setdefault(iid, [])):
                raise ValueError(f"{r['id']}: номера рамок снимка идут не подряд")
            by_image[iid].append((r["id"], [r["box"]]))
        if sorted(by_image) != sorted(sp["gallery_images"]):
            raise ValueError("снимки эталонов расходятся со списком gallery_images разбиения")
        return [(iid, pcb.ROOT / image[iid]["image"], ents) for iid, ents in by_image.items()]
    out = []
    for s in splits:
        for i in sp[s]:
            if image[i]["board"] not in pcb.BOARDS[s]:
                raise ValueError(f"{i}: плата {image[i]['board']} не входит в часть «{s}» разбиения")
            out.append((i, pcb.ROOT / image[i]["image"], [(i, None)]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hr_insdet", "pcb"])
    ap.add_argument("--mode", default="auto", choices=["auto", "box"])
    ap.add_argument("--crop-n-layers", type=int, choices=[0, 1, 2], help="обязателен в авторежиме")
    ap.add_argument("--split", nargs="+", choices=["test", "cal"],
                    help="по умолчанию: hr_insdet — test и cal, pcb — только test")
    ap.add_argument("--only", nargs="+", metavar="ID", help="только эти снимки (или записи кеша) из списка работы")
    ap.add_argument("--fp32", action="store_true",
                    help="счёт без autocast — только сверка точности на калибровочных сценах; каталог кеша отдельный")
    ap.add_argument("--status", action="store_true", help="показать, сколько снимков уже в кеше, и выйти")
    args = ap.parse_args()
    if args.mode == "auto" and args.crop_n_layers is None:
        ap.error("в авторежиме нужен --crop-n-layers")
    if args.split is None:
        args.split = AUTO_SPLITS[args.dataset]
    if args.mode == "auto" and set(args.split) - set(AUTO_SPLITS[args.dataset]):
        ap.error(f"{args.dataset}: в авторежиме сегментируются только части {AUTO_SPLITS[args.dataset]} — калибровочные "
                 "платы PCB в авторежиме не сегментируются")
    if args.fp32 and (args.dataset != "hr_insdet" or args.mode != "auto" or args.split != ["cal"]):
        ap.error("--fp32 — только HR-InsDet, авторежим и только --split cal")

    key = MC.auto_key(args.crop_n_layers, fp32=args.fp32) if args.mode == "auto" else MC.box_key()
    mc = MC.MaskCache(args.dataset, key)
    work = _work(args.dataset, args.mode, args.split)
    if args.only:
        only = set(args.only)
        unknown = only - {i for i, _, _ in work} - {e for _, _, ents in work for e, _ in ents}
        if unknown:
            ap.error(f"нет в списке работы: {sorted(unknown)}")
        work = [(i, p, [e for e in ents if i in only or e[0] in only]) for i, p, ents in work]
        work = [w for w in work if w[2]]
    n_all = sum(len(ents) for _, _, ents in work)
    on_image = {i: len(ents) for i, _, ents in _work(args.dataset, args.mode, args.split)}
    todo = [(i, p, [e for e in ents if not mc.has(*e)]) for i, p, ents in work]
    todo = [w for w in todo if w[2]]
    n_todo = sum(len(ents) for _, _, ents in todo)
    tag = f"{args.dataset} {mc.dir.name}"
    unit = "снимков" if n_all == len(work) else f"эталонов на {len(work)} снимках"
    print(f"[{tag}] всего {n_all} {unit}, в кеше {n_all - n_todo}, считать {n_todo}"
          + ("" if n_all == len(work) else f" (снимков — {len(todo)})"), flush=True)
    if args.status or not todo:
        if not todo:
            print(f"[{tag}] ГОТОВО: все {n_all} {unit} в кеше", flush=True)
        return

    import torch

    from src.segment import sam2 as S

    if args.mode == "auto":
        model = S.build_generator(args.crop_n_layers)
        MC.check_generator(model, key)
    else:
        model = S.build_predictor()
    t_start, n_masks, empty, retried = time.perf_counter(), [], 0, []
    for k, (iid, path, ents) in enumerate(todo, 1):
        t = time.perf_counter()
        img = S.read_rgb(path)
        if args.mode == "auto":
            recs, info = S.generate(model, img, long_side=key["long_side"], autocast=not args.fp32)
            info["sec_read"] = time.perf_counter() - t - sum(v for n, v in info.items() if n.startswith("sec_"))
            mc.save(iid, img.shape[:2], recs, info)
            n_masks.append(len(recs))
            if info["attempts"] > 1:
                retried.append(iid)
        else:
            # одна запись кеша — один эталон с одной маской; у PCB на снимке несколько эталонов с общим `set_image`
            # и отдельным `predict` на рамку — всегда, сколько бы эталонов снимка ни осталось после обрыва
            recs, info = S.predict_boxes(model, img, [b for _, bs in ents for b in bs], long_side=key["long_side"],
                                         one_by_one=args.dataset == "pcb")
            each = info.pop("sec_predict_each", None)
            info["sec_read"] = time.perf_counter() - t - sum(v for n, v in info.items() if n.startswith("sec_"))
            for j, ((eid, boxes), rec) in enumerate(zip(ents, recs, strict=True)):
                one = dict(info)
                if each is not None:  # `sec_set_image` и `sec_read` — общие на снимок, `sec_predict` — этой рамки
                    one.update(sec_predict=each[j], prompts_on_image=on_image[iid])
                mc.save(eid, img.shape[:2], [rec], one, boxes)
            empty += sum(r["area"] == 0 for r in recs)
            n_masks.append(len(recs))
        sec = time.perf_counter() - t
        left = (time.perf_counter() - t_start) / k * (len(todo) - k)
        print(f"[{k}/{len(todo)}] {iid}: {'масок' if args.mode == 'auto' else 'эталонов'} {len(recs)}, {sec:.1f} с, пик {info['vram_peak_mib']} МиБ, "
              f"осталось ~{left / 60:.0f} мин", flush=True)
    del model
    torch.cuda.empty_cache()
    total = time.perf_counter() - t_start
    print(f"[{tag}] ГОТОВО: посчитано {len(todo)} за {total / 60:.1f} мин ({total / len(todo):.1f} с на снимок), "
          f"{'масок' if args.mode == 'auto' else 'эталонов'} на снимок: среднее {np.mean(n_masks):.0f}, "
          f"мин {min(n_masks)}, макс {max(n_masks)}", flush=True)
    if retried:
        print(f"[{tag}] ВНИМАНИЕ: пересчитаны после ошибки CUDA (info.attempts > 1): {retried}", flush=True)
    if empty:
        print(f"[{tag}] ВНИМАНИЕ: пустых масок по рамке: {empty} — промпт не дал маски, разобрать до галереи", flush=True)


if __name__ == "__main__":
    main()
