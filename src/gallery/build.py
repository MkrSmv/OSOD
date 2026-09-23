"""Индексация эталонов; §2.1.

Эталон HR-InsDet — сырой снимок ракурса целиком; маска — $S$ по рамке переднего плана маски GrabCut, из кеша масок
(режим `box`): сегментатор на этапе кодирования не загружается. Эталон PKU-Market-PCB — рамка разметки дефекта на
снимке эталонной платы (на снимке их несколько), маска — $S$ по этой рамке, запись кеша — на эталон; `mode=category`,
метка — тип дефекта; контроля по GrabCut нет — в паспорт идёт справочная сводка масок эталонов по типам. Эмбеддинг — тем же $\\varphi$ и тем же кодом, что
у масок сцен: `encode.pipeline` для вырезок, `encode.pool` для $P$ и $P^\\perp$; эталон — снимок с одной маской,
поэтому по правилу состава батчей он кодируется батчем из одной вырезки. Отбор гранулярности не выполняется.

Кодирование прерываемо: эмбеддинг каждого эталона пишется в рабочий файл
`cache/run_emb/<run_id>__gallery/` в float32, `emb.npy` всегда собирается из этих файлов — галерея, собранная
с перерывом, побитово та же, что собранная за один раз. Маска GrabCut остаётся контролем: в метаданных — путь
к ней и IoU с маской $S$ в разрешении снимка, сводка IoU — в паспорте галереи и журнале.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.encode.run_emb import RunEmb
from src.gallery import store
from src.segment import cache as MC

WORK_SUFFIX = "__gallery"
IOU_WORK = Path("cache/checks")


def data_root(dataset: str) -> Path:
    from src.data import hr_insdet, pcb

    if dataset not in ("hr_insdet", "pcb"):
        raise ValueError(dataset)
    return hr_insdet.ROOT if dataset == "hr_insdet" else pcb.ROOT


def references(dataset: str) -> list[dict]:
    """Эталоны из `splits/<dataset>.json` в порядке вставки: по идентификатору, то есть по имени файла."""
    refs = json.loads(Path(f"splits/{dataset}.json").read_text())["gallery"]
    if dataset == "pcb":  # эталоны — только платы 01 и 04: тестовая либо калибровочная плата в галерее — утечка
        from src.data import pcb

        wrong = [r["id"] for r in refs if r["board"] not in pcb.BOARDS["gallery"] or r["id"].split("_", 1)[0] != r["board"]]
        if wrong:
            raise ValueError(f"эталоны не с эталонных плат {pcb.BOARDS['gallery']}: {wrong[:3]}")
    return sorted(refs, key=lambda r: r["id"])


def work_store(cfg, mask_key: dict, root: Path | None = None) -> RunEmb:
    key = {"what": "gallery", "dtype": "float32", "config": cfg.to_dict(), "mask_key": mask_key}
    return RunEmb(cfg.run_id + WORK_SUFFIX, key, root=root, dtype=np.float32)


def _entry(mc: MC.MaskCache, ref: dict) -> MC.MaskEntry:
    e = mc.load(ref["id"], [ref["box"]])
    if len(e) != 1 or e.records[0]["area"] <= 0:
        raise ValueError(f"{ref['id']}: в кеше масок по рамке не одна непустая маска")
    return e


def encode_crop(cfgs: dict, models: dict, stores: dict[str, RunEmb], feeder, refs: list[dict], mc: MC.MaskCache,
                progress: Callable[[int, int], None] | None = None) -> int:
    """$z_{\\mathrm{crop}}$ эталонов, ещё не лежащих в рабочих файлах; `cfgs`, `models`, `stores` — по имени прогона.

    Несколько прогонов сразу — парный проход: вариант и параметры вырезки у них обязаны быть одни.
    """
    from src.encode import pipeline as PL

    cfg = next(iter(cfgs.values()))
    same = ("dataset", "variant", "crop_size", "blur_sigma_frac", "blur_work_side", "encoder_batch")
    if any(getattr(c, f) != getattr(cfg, f) for c in cfgs.values() for f in same) or cfg.phi.kind != "crop":
        raise ValueError("общий проход — только для одного варианта вырезки с одними параметрами")
    todo = [r for r in refs if not all(s.has(r["id"]) for s in stores.values())]
    root = data_root(cfg.dataset)
    images = [(PL.image_ref(cfg.dataset, r["id"], root / r["image"], mc.key, [r["box"]], mc.dir.parents[1]), 1)
              for r in todo]
    stream = PL.encode_variant_multi(models, feeder, images, cfg.phi.b, cfg.phi.alpha, cfg.encoder_batch,
                                     size=cfg.crop_size, sigma_frac=cfg.blur_sigma_frac, work_side=cfg.blur_work_side)
    for n, (ref, z) in enumerate(stream, 1):
        for name, store_ in stores.items():
            store_.save(ref.image_id, z[name])
        if progress:
            progress(n, len(todo))
    return len(todo)


def encode_pool(cfg, model, stores: dict[str, RunEmb], refs: list[dict], mc: MC.MaskCache, u_r=None,
                progress: Callable[[int, int], None] | None = None) -> int:
    """$z_{\\mathrm{pool}}$ эталонов; `stores` — по выходу `p` | `p_perp`: оба — из одного прохода энкодера."""
    from src.encode import pool as PO
    from src.segment import sam2 as S

    if not set(stores) <= {"p", "p_perp"} or ("p_perp" in stores and u_r is None):
        raise ValueError("выходы — `p` и `p_perp`; для `p_perp` нужен U_r")
    # один проход энкодера на снимок: эталоны группируются по снимку — у PCB на снимке 1–6 эталонов,
    # у HR-InsDet один. Снимок, у которого не хватает хотя бы одного эталона, считается заново всем составом: состав
    # прохода постоянен, и галерея, собранная с перерывом, побитово та же.
    groups: dict[str, list[dict]] = {}
    for r in refs:
        groups.setdefault(r["image"], []).append(r)
    todo = [g for g in groups.values() if not all(s.has(r["id"]) for r in g for s in stores.values())]
    n_todo, n = sum(map(len, todo)), 0
    root = data_root(cfg.dataset)
    for g in todo:
        out = PO.pool_image(model, S.read_rgb(root / g[0]["image"]), _merged([_entry(mc, r) for r in g]), u_r,
                            cfg.pool_long_side, outputs=tuple(stores))
        for k, r in enumerate(g):
            for name, store_ in stores.items():
                store_.save(r["id"], out[name][k:k + 1], empty=out["empty"][k:k + 1])
            n += 1
            if progress:
                progress(n, n_todo)
    return n_todo


def _merged(entries: list[MC.MaskEntry]) -> MC.MaskEntry:
    """Записи кеша эталонов одного снимка как одна запись с несколькими масками — для одного прохода энкодера."""
    if len(entries) == 1:
        return entries[0]
    e0 = entries[0]
    if any((e.hw, e.seg_hw, e.scale_xy) != (e0.hw, e0.seg_hw, e0.scale_xy) for e in entries):
        raise ValueError(f"{e0.image_id}: записи эталонов одного снимка расходятся в размерах")
    return MC.MaskEntry({"image_id": e0.image_id.rsplit("/", 1)[0], "hw": e0.hw, "seg_hw": e0.seg_hw,
                         "scale_xy": e0.scale_xy, "info": {}, "masks": [e.records[0] for e in entries]})


# ---------------------------------------------------------------------- контроль: маска S против GrabCut


def control_iou(task: tuple[str, str, str, list, str, str | None]) -> tuple[str, float]:
    """IoU маски $S$ с маской GrabCut в разрешении снимка эталона. Функция верхнего уровня — для `Pool.map`."""
    from src.data import hr_insdet
    from src.segment import sam2 as S

    dataset, ref_id, mask_path, box, key_json, cache_root = task
    e = MC.MaskCache(dataset, json.loads(key_json), root=cache_root).load(ref_id, [box])
    h, w = e.hw
    m = S.mask_window(e.decode(0), e.scale_xy, 0, 0, w, h)
    g = hr_insdet.read_ref_mask(Path(mask_path))
    if g.shape != m.shape:
        raise ValueError(f"{ref_id}: размер маски GrabCut {g.shape} расходится со снимком {m.shape}")
    return ref_id, float((m & g).sum() / max((m | g).sum(), 1))


def control_ious(dataset: str, refs: list[dict], mc: MC.MaskCache, workers: int = 6,
                 work_dir: Path = IOU_WORK) -> dict[str, float]:
    """IoU по всем эталонам; от варианта $\\varphi$ не зависит, поэтому считается один раз на ключ кеша масок."""
    import multiprocessing as mp

    path = work_dir / f"ref_mask_iou_{dataset}_{mc.dir.name}.json"
    done = json.loads(path.read_text()) if path.is_file() else {}
    todo = [r for r in refs if r["id"] not in done]
    if todo:
        root = data_root(dataset)
        tasks = [(dataset, r["id"], str(root / r["mask"]), r["box"], json.dumps(mc.key, sort_keys=True),
                  str(mc.dir.parents[1])) for r in todo]
        with mp.get_context("spawn").Pool(workers) as pool:
            done.update(pool.imap_unordered(control_iou, tasks, chunksize=8))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(dict(sorted(done.items())), indent=0) + "\n")
        tmp.replace(path)
    return {r["id"]: done[r["id"]] for r in refs}


def ref_mask_summary(refs: list[dict], mc: MC.MaskCache) -> dict:
    """Справочная сводка масок эталонов PCB по типам дефекта: доля рамки разметки под маской и оценка сегментатора —
    те же величины, что в пилоте; контроля по GrabCut у PCB нет."""
    rows: dict[str, list[tuple[float, float]]] = {}
    for r in refs:
        rec = _entry(mc, r).records[0]
        x0, y0, x1, y1 = rec["prompt_box"]
        rows.setdefault(r["label"], []).append((rec["area"] / max((x1 - x0) * (y1 - y0), 1e-9), rec["predicted_iou"]))
    out = {}
    for t, v in sorted(rows.items()):
        fill, piou = np.array(v).T
        out[t] = {"n": len(v), "box_fill_median": round(float(np.median(fill)), 4), "box_fill_min": round(float(fill.min()), 4),
                  "predicted_iou_median": round(float(np.median(piou)), 4)}
    return {"what": "маска эталона по промпту-рамке: доля площади рамки разметки под маской, оценка сегментатора", "by_type": out}


def iou_summary(ious: dict[str, float]) -> dict:
    v = np.array(list(ious.values()))
    worst = sorted(ious, key=ious.get)[:5]
    return {"n": len(v), "min": float(v.min()), "q05": float(np.quantile(v, 0.05)), "median": float(np.median(v)),
            "mean": float(v.mean()), "n_below_0.9": int((v < 0.9).sum()), "n_below_0.5": int((v < 0.5).sum()),
            "worst": {i: round(ious[i], 4) for i in worst}}


# ---------------------------------------------------------------------- сборка


def reference_entry(dataset: str, ref: dict, entry: MC.MaskEntry, iou: float | None) -> dict:
    """Строка `meta.jsonl`. Сверх обязательных полей — то, что нужно для пересчёта эмбеддинга без
    повторного сбора эталонов (§2.5): размер снимка, рамка маски, рамка промпта; контроль — маска GrabCut."""
    rec = entry.records[0]
    if dataset == "pcb":  # категорийный режим: метка — тип дефекта; эталон вырезан из снимка платы, профиля нет
        return {"id": ref["id"], "label": ref["label"], "mode": "category", "label2": None,
                "image_id": ref["id"].rsplit("/", 1)[0], "mask_rle": rec["rle"], "origin": "scene", "image": ref["image"],
                "board": ref["board"], "hw": list(entry.hw), "mask_scale_xy": list(entry.scale_xy), "box": rec["box"],
                "prompt_box": rec["prompt_box"], "predicted_iou": rec["predicted_iou"], "control_mask": None,
                "control_iou": None}
    if dataset != "hr_insdet":
        raise ValueError(dataset)
    return {"id": ref["id"], "label": ref["label"], "mode": "instance", "label2": None, "image_id": ref["id"],
            "mask_rle": rec["rle"], "origin": "profile", "image": ref["image"], "hw": list(entry.hw),
            "mask_scale_xy": list(entry.scale_xy), "box": rec["box"], "prompt_box": rec["prompt_box"],
            "control_mask": ref["mask"], "control_iou": iou}


def assemble(cfg, refs: list[dict], mc: MC.MaskCache, work: RunEmb, ious: dict[str, float] | None,
             root: Path | None = None, build_info: dict | None = None) -> store.Gallery:
    """Галерея из рабочих файлов: все эталоны одной вставкой в порядке `refs`. Файлы галереи заменяются; запись
    калибровки прежней галереи (`calib.json`) удаляется — она относится к прежним эмбеддингам и прежнему $N_0$."""
    t = time.perf_counter()
    z = [work.load(r["id"]) for r in refs]
    emb = np.concatenate([x["z"] for x in z])
    entries = [reference_entry(cfg.dataset, r, _entry(mc, r), None if ious is None else ious[r["id"]]) for r in refs]
    g = store.Gallery.create(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset, root),
                             store.passport(cfg.to_dict(), mc.key, emb.shape[1]))
    g.add(emb, entries)
    flagged = [r["id"] for r, x in zip(refs, z) if "empty" in x and x["empty"].any()]
    g.passport["build"] = {**(build_info or {}), "n_pool_center_patch_fallback": len(flagged),
                           "control_iou": None if ious is None else iou_summary(ious),
                           "sec_assemble": round(time.perf_counter() - t, 1)}
    if cfg.dataset == "pcb":
        from src.eval import rules as RU

        g.passport["build"]["ref_masks"] = ref_mask_summary(refs, mc)
        if cfg.run_id == RU.PCB_GALLERY_RUN:  # критерий осмысленности галереи PCB — один раз, по этой галерее
            g.passport["build"]["pcb_gallery_criterion"] = RU.pcb_gallery_criterion(
                g.emb, [m["label"] for m in g.meta], [m["board"] for m in g.meta])
    g.save()
    (g.path / "calib.json").unlink(missing_ok=True)
    return g
