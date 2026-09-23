"""Эксперимент 4.3 — прогон сетки кодирования → `experiments/runs/<run_id>.json`.

    python scripts/run_4_3.py --config configs/dinov2_c_0_10_hr_insdet.yaml                      # прогон сетки
    python scripts/run_4_3.py --config configs/dinov2_c_0_10_hr_insdet.yaml --protocol baseline  # контрольный прогон
    python scripts/run_4_3.py --config configs/dinov2_c_blur_15_hr_insdet.yaml \\
                              --config configs/dinov3_c_blur_15_hr_insdet.yaml                   # парный проход
    python scripts/run_4_3.py --config configs/dinov3_p_hr_insdet.yaml \\
                              --config configs/dinov3_p_perp_hr_insdet.yaml                      # P и P⊥ одним проходом
    python scripts/run_4_3.py --baseline-best dinov2                                             # baseline лучшего φ
    python scripts/run_4_3.py --config … --status

Прогон сетки: галерея (если ещё не собрана — тем же кодом, что `build_gallery.py`) → эмбеддинги всех масок $M(I)$
всех 160 сцен в `cache/run_emb/<run_id>/` (прерываемо: закодированные сцены пропускаются) → точный поиск по
значениям из этих файлов → оракульный отбор и дистракторы → метрики по 120 тестовым сценам, `selection_cal` по 40
калибровочным, оба протокола галереи → запись журнала. Запись появляется только по завершении; существующая запись
не пересчитывается. Два `--config` — общий проход: один вариант вырезки на двух энкодерах (батч вырезок готовится
один раз) либо $P$ и $P^\\perp$ на DINOv3 (один проход энкодера); записей всё равно две, с `paired_with`.

Контрольный прогон по протоколу baseline — по тем же эмбеддингам, без GPU: все маски $M(I)$, AP и AR итоговых
детекций на 160 сценах и на 120 тестовых → `experiments/runs/<run_id>__baseline.json` (`protocol=baseline`).
Выполняется для первого прогона (контроль на входе, пороги — `src.eval.rules`) и для лучшего $\\varphi$ каждого
энкодера, выбранного правилом по `selection_cal`; для прочих прогонов — отказ: сверх сетки ничего не считается.

Вывод — ещё и в `logs/<run_id>.log`. Поиск — только точный перебор; порог $\\tau$ не применяется.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as CFG
from src import env
from src.encode import run_emb as RE
from src.encode.variants import VARIANTS
from src.eval import oracle as OR
from src.eval import protocol as PR
from src.eval import recall as R
from src.eval import rules as RU
from src.gallery import build as GB
from src.gallery import store
from src.segment import cache as MC

RUNS = Path("experiments/runs")
U_R_PATH = Path("gallery/dinov3/p_perp/U_r.npy")
BASELINE_SUFFIX = "__baseline"
_STAMP = env.code_stamp()  # версия кода — на момент запуска процесса


def _logger(names: list[str]):
    """Вывод — в stdout и в `logs/<run_id>.log` каждого прогона прохода."""
    Path("logs").mkdir(exist_ok=True)
    logs = [open(Path("logs") / f"{n}.log", "a") for n in names]
    tag = "+".join(names)

    def say(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{tag}] {msg}"
        print(line, flush=True)
        for log in logs:
            log.write(line + "\n")
            log.flush()
    return say


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def record_path(run_id: str, baseline: bool = False) -> Path:
    return RUNS / f"{run_id}{BASELINE_SUFFIX if baseline else ''}.json"


def _caches(cfg) -> tuple[MC.MaskCache, MC.MaskCache]:
    scenes = MC.MaskCache(cfg.dataset, MC.auto_key(cfg.crop_n_layers, cfg.seg_long_side, cfg.points_per_batch))
    refs = MC.MaskCache(cfg.dataset, MC.box_key(cfg.seg_long_side))
    return scenes, refs


_PCB_IMAGES: dict[str, str] = {}


def _scene_path(cfg, scene_id: str) -> Path:
    from src.data import hr_insdet, pcb

    if cfg.dataset == "hr_insdet":
        return hr_insdet.ROOT / "Scenes" / f"{scene_id}.jpg"
    if not _PCB_IMAGES:
        _PCB_IMAGES.update({i["id"]: i["image"] for i in pcb.images()})
    return pcb.ROOT / _PCB_IMAGES[scene_id]


def _scene_ids(cfg, sp: dict) -> list[str]:
    """Снимки, которые кодирует прогон: HR-InsDet — все 160 сцен; PCB — только тестовые платы (калибровочные платы
    в авторежиме не сегментируются)."""
    return sp["cal"] + sp["test"] if cfg.dataset == "hr_insdet" else list(sp["test"])


def _cache_script(cfg) -> str:
    return (f"python scripts/segment.py --dataset {cfg.dataset} --crop-n-layers 1 и "
            f"python scripts/segment.py --dataset {cfg.dataset} --mode box")


# ---------------------------------------------------------------------- порядок прогонов и пары


def check_pass(cfgs: list) -> str:
    """Вид прохода: `single` | `paired_crop` | `pool_pair`. Иные сочетания — ошибка."""
    if len({c.run_id for c in cfgs}) != len(cfgs) or not 1 <= len(cfgs) <= 2:
        raise SystemExit("--config: один прогон либо пара разных прогонов")
    for c in cfgs:
        if c.batch_rule != CFG.PL.BATCH_RULE:
            raise SystemExit(f"{c.run_id}: правило состава батчей в конфигурации — не то, что реализует код")
        if c.max_dets not in R.MAX_DETS:
            raise SystemExit(f"{c.run_id}: max_dets={c.max_dets} вне списка COCOeval {R.MAX_DETS}")
    if len(cfgs) == 1:
        return "single"
    a, b = cfgs
    # общий проход считает обе конфигурации одними параметрами: всё, кроме энкодера и варианта, обязано совпадать —
    # иначе в журнал второго прогона попало бы значение, которое не использовалось
    diff = [k for k in a.to_dict() if k not in ("encoder", "variant") and a.to_dict()[k] != b.to_dict()[k]]
    if diff:
        raise SystemExit(f"конфигурации пары расходятся в {diff}: общий проход невозможен")
    if a.dataset != "hr_insdet" or b.dataset != "hr_insdet":
        raise SystemExit("общий проход — только на HR-InsDet")
    if {a.variant, b.variant} == {"p", "p_perp"} and a.encoder == b.encoder == "dinov3":
        return "pool_pair"
    if a.variant == b.variant and a.phi.kind == "crop" and a.encoder != b.encoder:
        if a.variant == "c_0_10":
            raise SystemExit("C(0,1.0) считается отдельно на каждом энкодере")
        return "paired_crop"
    raise SystemExit("пара — один вариант вырезки на двух энкодерах либо P и P⊥ на DINOv3")


def check_order(cfgs: list, say) -> None:
    """Порядок прогонов: первым и отдельно — `RU.FIRST_RUN` с контрольным прогоном; остальные — после того, как
    контроль на входе пройден; прогоны DINOv3 — после записи критерия «слабого результата»."""
    if [c.run_id for c in cfgs] == [RU.FIRST_RUN]:
        return
    if cfgs[0].dataset == "pcb":
        check_order_pcb(cfgs[0], say)
        return
    p = record_path(RU.FIRST_RUN, baseline=True)
    if not p.is_file():
        raise SystemExit(f"первым выполняется {RU.FIRST_RUN} с контрольным прогоном baseline: нет {p}")
    chk = json.loads(p.read_text())["first_run_check"]
    if chk["ap_ge_min"] and not chk["diff_within_max"] and RU.FIRST_RUN_DIFF_RESOLUTION is not None:
        say(f"контроль на входе: расхождение {chk['diff']:+.2f} п. разобрано, продолжение открыто записанным решением — "
            f"{RU.FIRST_RUN_DIFF_RESOLUTION}")
    elif not chk["passed"]:
        raise SystemExit(f"контроль на входе не пройден: AP {chk['ap']:.2f} при пороге {chk['ap_min']:g} и допуске "
                         f"±{chk['diff_max']:g} к {chk['published_ap']} — прогоны остановлены до разбора")
    if any(c.encoder == "dinov3" for c in cfgs) and RU.WEAK_DINOV3_CRITERION is None:
        raise SystemExit("критерий «слабого результата» DINOv3 не записан (`src.eval.rules.WEAK_DINOV3_CRITERION`): "
                         "он записывается до первого прогона DINOv3")
    say("порядок: контроль на входе пройден" + ("; критерий «слабого результата» DINOv3 записан"
                                                 if any(c.encoder == "dinov3" for c in cfgs) else ""))


def check_order_pcb(cfg, say) -> None:
    """Порядок прогонов PCB: после всех прогонов HR-InsDet; первым и отдельно —
    `RU.PCB_FIRST_RUN` с контрольным прогоном; остальные — когда нижний признак первого прогона не сработал либо разобран
    и продолжение открыто записанным решением; $P$ — последним."""
    from src.encode.variants import grid

    absent = [f"{e}_{v}_{d}" for e, v, d in grid() if d == "hr_insdet" and not record_path(f"{e}_{v}_{d}").is_file()]
    if absent:
        raise SystemExit(f"прогоны PCB — после всех прогонов HR-InsDet; нет записей: {absent}")
    if cfg.run_id == RU.PCB_FIRST_RUN:
        return
    for p in (record_path(RU.PCB_FIRST_RUN), record_path(RU.PCB_FIRST_RUN, baseline=True)):
        if not p.is_file():
            raise SystemExit(f"первым на PCB выполняется {RU.PCB_FIRST_RUN} с контрольным прогоном baseline: нет {p}")
    chk = json.loads(record_path(RU.PCB_FIRST_RUN).read_text())["pcb_first_run_check"]
    if not chk["passed"]:
        if RU.PCB_SANITY_RESOLUTION is None:
            raise SystemExit(f"нижний признак первого прогона PCB сработал (top-1 {chk['type']} {chk['top1']} при пороге "
                             f"{chk['top1_min']:.4f}; AUROC {chk.get('auroc')} при пороге {chk.get('auroc_min')}) — прогоны "
                             "остановлены до разбора по `PCB_FIRST_RUN_REVIEW` и записи его итога в `PCB_SANITY_RESOLUTION`")
        say(f"первый прогон PCB: нижний признак разобран, продолжение открыто записанным решением — {RU.PCB_SANITY_RESOLUTION}")
    if cfg.variant == RU.PCB_RUN_ORDER[-1]:
        waits = [v for v in RU.PCB_RUN_ORDER[:-1] if not record_path(f"{cfg.encoder}_{v}_pcb").is_file()]
        if waits:
            raise SystemExit(f"прогон P на PCB — последним (`PCB_RUN_ORDER`); нет записей вариантов: {waits}")
    say("порядок PCB: прогоны HR-InsDet закончены, первый прогон PCB с контрольным — есть")


def load_u_r(cfg) -> np.ndarray:
    """$U_r$ — тот, что записан сверкой кодирования (`scripts/check_encoder.py u_r`): хеш и параметры сверяются с `experiments/env.json`, `debias`, а параметры —
    ещё и с конфигурацией прогона; подменённый либо пересчитанный иначе проектор молча не проходит."""
    from src.encode import debias as D

    rec = json.loads(env.ENV_JSON.read_text())["debias"]
    u_r = D.load(U_R_PATH, cfg.debias_r)
    if (rec["r"], rec["noise_seed"]) != (cfg.debias_r, cfg.debias_noise_seed) or rec["path"] != str(U_R_PATH):
        raise SystemExit(f"U_r получен при r, seed = {rec['r']}, {rec['noise_seed']}, в конфигурации — {cfg.debias_r}, {cfg.debias_noise_seed}")
    if D.sha256(u_r) != rec["sha256"]:
        raise SystemExit(f"{U_R_PATH}: хеш расходится с записанным в experiments/env.json — python scripts/check_encoder.py u_r")
    return u_r


# ---------------------------------------------------------------------- галереи


def open_gallery(cfg, refs_cache) -> store.Gallery:
    return store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset),
                              store.expected(cfg.to_dict(), refs_cache.key))


def ensure_galleries(cfgs: list, kind: str, models: dict, feeder, u_r, say) -> dict[str, float]:
    """Галереи прогонов прохода; несобранные кодируются общим проходом. Возвращает секунды кодирования эталонов."""
    _, refs_cache = _caches(cfgs[0])
    have = {c.run_id: (store.gallery_dir(c.encoder, c.variant, c.dataset) / "gallery.json").is_file() for c in cfgs}
    for c in cfgs:
        if have[c.run_id]:
            open_gallery(c, refs_cache)  # чужая либо устаревшая галерея — ошибка, а не повод пересобрать молча
    todo = [c for c in cfgs if not have[c.run_id]]
    sec = {c.run_id: 0.0 for c in cfgs}
    if not todo:
        say("галереи собраны ранее: " + ", ".join(c.run_id for c in cfgs))
        return sec
    refs = GB.references(todo[0].dataset)
    absent = [r["id"] for r in refs if not refs_cache.has(r["id"], [r["box"]])]
    if absent:
        raise SystemExit(f"в кеше масок нет {len(absent)} эталонов (первый — {absent[0]}): {_cache_script(cfgs[0])}")
    work = {c.run_id: GB.work_store(c, refs_cache.key) for c in todo}
    t0 = time.perf_counter()

    def progress(n: int, total: int) -> None:
        if n % 200 == 0 or n == total:
            s = time.perf_counter() - t0
            say(f"эталоны: {n}/{total}, {n / s:.1f} в секунду, осталось ~{(total - n) / n * s / 60:.1f} мин")

    say(f"сборка галерей: {', '.join(work)} — эталонов {len(refs)}")
    if kind == "pool_pair" or todo[0].phi.kind == "pool":
        model = models[todo[0].encoder]
        n = GB.encode_pool(todo[0], model, {c.variant: work[c.run_id] for c in todo}, refs, refs_cache, u_r, progress)
    else:
        n = GB.encode_crop({c.run_id: c for c in todo}, {c.run_id: models[c.encoder] for c in todo}, work, feeder,
                           refs, refs_cache, progress)
    dt = time.perf_counter() - t0
    ious = GB.control_ious(todo[0].dataset, refs, refs_cache) if todo[0].dataset == "hr_insdet" else None  # у PCB GrabCut нет
    for c in todo:
        g = GB.assemble(c, refs, refs_cache, work[c.run_id], ious, build_info={
            **_STAMP, "n_encoded_this_session": n, "sec_encode_this_session": round(dt, 1),
            "shared_pass_with": [x.run_id for x in todo if x is not c], "built_by": "scripts/run_4_3.py"})
        sec[c.run_id] = dt
        say(f"галерея {g.path}: N = {len(g)}, меток {len(g.labels)}")
    return sec


# ---------------------------------------------------------------------- кодирование сцен


def encode_scenes(cfgs: list, kind: str, scene_ids: list[str], models: dict, feeder, u_r, say) -> None:
    scenes_cache, _ = _caches(cfgs[0])
    stores = {c.run_id: RE.scene_store(c, scenes_cache.key) for c in cfgs}
    todo = [s for s in scene_ids if not all(st.has(s) for st in stores.values())]
    say(f"сцен {len(scene_ids)}, закодировано {len(scene_ids) - len(todo)}, осталось {len(todo)}")
    if not todo:
        return
    n_masks = {s: len(scenes_cache.load(s)) for s in todo}
    n_total, n_done, t0 = sum(n_masks.values()), 0, time.perf_counter()
    shared = [c.run_id for c in cfgs]
    session = _STAMP["written_utc"]
    lacking = {s: [rid for rid, st in stores.items() if not st.has(s)] for s in todo}  # у пары, докодируемой частично
    feeder_info = feeder.journal(cfgs[0].encoder_batch) if feeder is not None else {}

    def done(k: int, scene_id: str, n: int, t_prev: float) -> float:
        nonlocal n_done
        now = time.perf_counter()
        n_done += n
        for rid in lacking[scene_id]:  # сцена, уже лежавшая в файлах прогона, второй раз во время не засчитывается
            RE.log_progress(stores[rid], image_id=scene_id, n_masks=n, sec=round(now - t_prev, 3), session=session,
                            shared_pass=shared, code_commit=_STAMP["code_commit"],
                            crop_workers=feeder_info.get("crop_workers"), crop_max_in_flight=feeder_info.get("crop_max_in_flight"))
        sec = now - t0
        say(f"[{k}/{len(todo)}] {scene_id}: масок {n}; всего {n_done}/{n_total}, {n_done / sec:.1f} маски/с, "
            f"осталось ~{(n_total - n_done) / max(n_done, 1) * sec / 60:.0f} мин")
        return now

    c0 = cfgs[0]
    t_prev = t0
    if c0.phi.kind == "crop":
        from src.encode import pipeline as PL

        images = [(PL.image_ref(c0.dataset, s, _scene_path(c0, s), scenes_cache.key), n_masks[s]) for s in todo]
        stream = PL.encode_variant_multi({c.run_id: models[c.encoder] for c in cfgs}, feeder, images, c0.phi.b,
                                         c0.phi.alpha, c0.encoder_batch, size=c0.crop_size,
                                         sigma_frac=c0.blur_sigma_frac, work_side=c0.blur_work_side)
        for k, (ref, z) in enumerate(stream, 1):
            for rid in lacking[ref.image_id]:
                stores[rid].save(ref.image_id, z[rid])
            t_prev = done(k, ref.image_id, n_masks[ref.image_id], t_prev)
    else:
        from src.encode import pool as PO
        from src.segment import sam2 as S

        model = models[c0.encoder]
        outputs = tuple(c.variant for c in cfgs)
        for k, s in enumerate(todo, 1):
            out = PO.pool_image(model, S.read_rgb(_scene_path(c0, s)), scenes_cache.load(s), u_r, c0.pool_long_side,
                                outputs=outputs)
            for c in cfgs:
                if c.run_id in lacking[s]:
                    z = out[c.variant] if n_masks[s] else np.zeros((0, model.config.hidden_size), np.float32)
                    stores[c.run_id].save(s, z, empty=out["empty"])
            t_prev = done(k, s, n_masks[s], t_prev)


# ---------------------------------------------------------------------- оценка и запись


def _load_run(cfg, say):
    """Сцены, истина, галерея и эмбеддинги масок из рабочих файлов (fp16 → fp32) — общее для обоих протоколов."""
    scenes_cache, refs_cache = _caches(cfg)
    scenes, labels, edges, sp = PR.load_scenes(cfg.dataset, scenes_cache)
    st = RE.scene_store(cfg, scenes_cache.key)
    missing = [s.id for s in scenes if not st.has(s.id)]
    if missing:
        raise SystemExit(f"{cfg.run_id}: не закодировано сцен — {len(missing)}; сначала прогон сетки")
    emb, n_fallback = [], 0
    for s in scenes:
        f = st.load(s.id)
        if len(f["z"]) != len(s.boxes):
            raise SystemExit(f"{s.id}: эмбеддингов {len(f['z'])}, масок в кеше {len(s.boxes)}")
        n_fallback += int(f["empty"].sum()) if "empty" in f else 0
        emb.append(f["z"])
    g = open_gallery(cfg, refs_cache)
    gt = PR.coco_truth(scenes, labels)  # у PCB — с зонами игнорирования (`iscrowd=1` в каждой категории)
    say(f"сцен {len(scenes)}, масок {sum(map(len, emb))}, галерея N = {len(g)}")
    return scenes, labels, edges, sp, emb, g, gt, st, scenes_cache, refs_cache, n_fallback


def _common(cfg, kind: str, sp: dict, g: store.Gallery, st, scenes_cache, refs_cache, scenes) -> dict:
    models = json.loads(env.ENV_JSON.read_text())["models"]
    seg = {k: round(float(sum(scenes_cache.load(s.id).info[k] for s in scenes)), 1)
           for k in ("sec_read", "sec_resize", "sec_generate")}
    return {
        "dataset": cfg.dataset, "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"], "started_utc": _STAMP["written_utc"],
        "config": cfg.to_dict(), "seed": cfg.seed, "variant_label": VARIANTS[cfg.variant].label,
        "splits": {"file": f"splits/{cfg.dataset}.json", "written": sp["written"], "seed": sp["seed"],
                   "n_cal": len(sp["cal"]), "n_test": len(sp["test"]), "n_encoded": len(_scene_ids(cfg, sp))},
        "versions": {"packages": env.package_versions(), "sam2": env.sam2_install_info(), "lock_sha256": env.lock_sha256(),
                     "encoder": {"id": models[cfg.model_key]["id"], "revision": models[cfg.model_key]["revision"],
                                 "dtype": env.ENCODER_DTYPE[cfg.model_key]},
                     "segmenter": models["segmenter"]["revision"]},
        "mask_cache": {"scenes": scenes_cache.dir.name, "references": refs_cache.dir.name},
        "run_emb": {"dir": str(st.dir), "config_sha1": st.digest, "dtype": str(st.dtype)},
        "gallery": {"path": str(g.path), "N": len(g), "n_labels": len(g.labels), "emb_sha256": g.passport.get("emb_sha256"),
                    "config_sha1": g.passport["config_sha1"], "build": g.passport.get("build")},
        "rules": {"oracle_iou_min": OR.ORACLE_IOU, "distractor_iou_max": OR.DISTRACTOR_IOU_MAX,
                  "search": "IndexFlatIP, k = N; порог τ не применяется", "area_edges_px2": None},
        "_segmentation_sec": seg,
    }


def _search_all(cfg, emb, g, say) -> tuple[dict, dict, float]:
    found, info, t0 = {}, {}, time.perf_counter()
    for proto in cfg.gallery_protocols:
        y, s, info[proto] = PR.search(emb, g, proto)
        found[proto] = (y, s)
        say(f"поиск, галерея «{proto}»: N = {info[proto]['N']}")
    return found, info, time.perf_counter() - t0


def _pcb_blocks(cfg, protocol: str, scenes, labels, edges, g: store.Gallery, found: dict, metrics: dict, say) -> dict:
    """Блоки записи прогона PCB сверх метрик: справочный AP без зон и
    жёсткие признаки ошибки пайплайна — потолок AP по полноте 4.2 по типам дефекта и состав оценки. При сработавшем
    жёстком признаке записи журнала нет: прогон останавливается здесь."""
    evaluate = PR.evaluate_oracle if protocol == "oracle" else PR.evaluate_baseline
    gt_nz = PR.coco_truth(scenes, labels, zones=False)
    kw = {"zones": False} if protocol == "oracle" else {"with_ar": False}  # AR без зон не нужен: справочно — только AP
    no_zones, ap_by_type = {}, {}
    for proto, (y, s) in found.items():
        m = evaluate(scenes, labels, edges, gt_nz, g.labels, y, s, cfg.max_dets, cfg.bootstrap, cfg.seed, names=("test/all",),
                     levels_=PR.levels(cfg.dataset), by_type=True, **kw)["subsets"]["test/all"]
        a, b = metrics[proto]["subsets"]["test/all"]["by_area"]["all"], m["by_area"]["all"]
        diff = 100 * (b["ap"] - a["ap"])
        no_zones[proto] = {"ap": b["ap"], "ap50": b["ap50"], "ap75": b["ap75"], "ap_with_zones": a["ap"],
                           "diff_ap_points": round(diff, 4), "close": bool(abs(diff) <= RU.PCB_ZONES_AP_CLOSE),
                           "by_type": {t: {k: v[k] for k in ("ap", "ap50", "ap75")} for t, v in m["by_type"].items()}}
        for by_type in (metrics[proto]["subsets"]["test/all"]["by_type"], m["by_type"]):  # с зонами и без них
            for t, v in by_type.items():  # потолок действует в любом протоколе оценки и галереи — берётся наибольшее
                for k in ("ap", "ap50", "ap75"):
                    cur = ap_by_type.setdefault(t, {}).get(k)
                    if v[k] is not None and (cur is None or v[k] > cur):
                        ap_by_type[t][k] = v[k]
        say(f"галерея «{proto}»: AP без зон {100 * b['ap']:.2f} против {100 * a['ap']:.2f} с зонами "
            f"({diff:+.3f} п.; «близки» — до {RU.PCB_ZONES_AP_CLOSE:g})")

    r42 = json.loads(Path(RU.PCB_RECALL_RECORD).read_text())["results"][str(cfg.crop_n_layers)]
    recall = {t: r42[f"test/type_{t}"]["by_area"]["all"] for t in labels}
    rows = g.rows("full")
    n_oracle = {t: int(sum(((s.oracle >= 0) & (s.gt_labels == c)).sum() for s in scenes)) for c, t in enumerate(labels)}
    n_found_4_2 = {t: int(round(recall[t]["recall_50"] * recall[t]["n_gt"])) for t in labels}
    composition = {"n_images": len(scenes), "n_gt": int(sum(len(s.gt) for s in scenes)),
                   "n_ignore_zones": int(sum(len(s.zones) for s in scenes)), "n_masks": int(sum(len(s.boxes) for s in scenes)),
                   "n_oracle_by_type_within_4_2": bool(all(n_oracle[t] <= n_found_4_2[t] for t in labels)),
                   "gallery_N": len(g), "gallery_labels": g.labels, "gallery_n_max": g.n_max(rows), "gallery_k": g.k(rows),
                   "gallery_boards": sorted({m["board"] for m in g.meta}), "gallery_modes": sorted({m["mode"] for m in g.meta})}
    expected = {"n_images": RU.PCB_N_TEST_IMAGES, "n_gt": RU.PCB_N_GT, "n_ignore_zones": RU.PCB_N_ZONES,
                "n_masks": r42["test/all"]["n_masks"], "n_oracle_by_type_within_4_2": True, "gallery_N": RU.PCB_N_REFS,
                "gallery_labels": list(labels), "gallery_n_max": RU.PCB_N_MAX, "gallery_k": RU.PCB_K,
                "gallery_boards": list(RU.PCB_GALLERY_BOARDS), "gallery_modes": ["category"]}
    if protocol == "baseline":
        composition["n_detections"], expected["n_detections"] = metrics["full"]["n_detections"], r42["test/all"]["n_masks"]
    checks = RU.pcb_hard_checks(ap_by_type, RU.pcb_ap_ceiling(recall), composition, expected)
    checks.update(ap_max_over_protocols_by_type=ap_by_type, n_oracle_by_type=n_oracle, n_gt_with_mask_iou50_in_4_2=n_found_4_2,
                  recall_record=RU.PCB_RECALL_RECORD, composition=composition)
    if not checks["passed"]:
        # записи прогона нет, но след остаётся в журнале: заглушка без метрик — что сработало и при какой версии кода
        stub = RUNS / f"{cfg.run_id}{BASELINE_SUFFIX if protocol == 'baseline' else ''}{RU.PCB_REJECTED_SUFFIX}{_STAMP['code_commit'][:7]}.json"
        _write(stub, {"run_id": stub.stem, "kind": "rejected_run", "experiment": "4.3", "dataset": cfg.dataset,
                      "protocol": protocol, "rejected_run": cfg.run_id, "code_commit": _STAMP["code_commit"],
                      "code_dirty": _STAMP["code_dirty"], "started_utc": _STAMP["written_utc"], "failed": checks["failed"],
                      "what": "жёсткий признак ошибки пайплайна на PCB: записи прогона нет, метрики не пишутся; причина и "
                              "исправление — в коммите исправления"})
        raise SystemExit(f"ПРИЗНАК ОШИБКИ ПАЙПЛАЙНА НА PCB — записи прогона нет (заглушка {stub}), прогон и пачка "
                         "останавливаются: " + "; ".join(checks["failed"]))
    say("признаки ошибки пайплайна PCB: потолок AP по полноте 4.2 по типам и состав оценки — не сработали")
    return {"ap_without_zones": {"what": "справочно: истина без зон игнорирования, маски у зон возвращены в дистракторы "
                                         "по общему правилу; подмножество test/all", "close_if_abs_diff_le_points":
                                 RU.PCB_ZONES_AP_CLOSE, **no_zones},
            "pcb_pipeline_checks": checks}


def _write_pcb_grid(cfg, kind, sp, g, st, scenes_cache, refs_cache, scenes, edges, progress, sub, names, metrics,
                    search_info, sec_search, sec_eval, n_fallback, pcb_blocks, say) -> Path:
    """Запись прогона сетки PCB: без `selection_cal` (калибровочные платы не сегментируются), без интервалов."""
    rec = _common(cfg, kind, sp, g, st, scenes_cache, refs_cache, scenes)
    seg = rec.pop("_segmentation_sec")
    rec["rules"]["area_edges_px2"] = edges[:2]
    rec["rules"]["ignore_zones"] = "iscrowd=1 в каждой категории; маска, чья рамка пересекает зону, — не дистрактор"
    for proto, info in search_info.items():
        info["k_hnsw_by_gallery"] = g.k(g.rows(proto))  # k = 2·n_max, но не больше N; поиск прогона — точный, k = N
    rec = {"run_id": cfg.run_id, "experiment": "4.3", "kind": "grid_run", "protocol": "oracle", **rec,
           "paired_with": None, "encode_sec_shared": False, "pass": kind,
           "status": "стресс-тест на границе применимости: только тестовые платы, один энкодер, без интервалов",
           "encode": {"encoder_batch": cfg.encoder_batch if cfg.phi.kind == "crop" else None,
                      "batch_rule": cfg.batch_rule if cfg.phi.kind == "crop" else "pool: один проход энкодера на снимок, все маски снимка разом",
                      "crop_workers": sorted({p.get("crop_workers") for p in progress} - {None}),
                      "crop_max_in_flight": sorted({p.get("crop_max_in_flight") for p in progress} - {None}),
                      "n_pool_center_patch_fallback": n_fallback, "sessions": sorted({p["session"] for p in progress}),
                      "code_commits": sorted({p["code_commit"] for p in progress})},
           "timing": {"segmentation_from_cache_info_sec": seg,
                      "encode_scenes_sec": round(sum(p["sec"] for p in progress), 1),
                      "encode_scenes_n_masks": int(sum(p["n_masks"] for p in progress)),
                      "encode_gallery_sec": (g.passport.get("build") or {}).get("sec_encode_this_session"),
                      "search_sec": round(sec_search, 1), "evaluate_sec": round(sec_eval, 1),
                      "note": "кодирование — сумма по снимкам из progress.jsonl рабочих файлов (счёт урывками)"},
           "counts": {n: PR.counts(scenes, sub[n], edges) for n in names},
           "search": search_info,
           "metrics_note": "доли (0–1); 361 снимок тестовых плат; разбивка — по платам (подмножества снимков) и по типам "
                           "дефекта (`by_type`: AP категории на всех снимках подмножества, top-1 — на оракульных масках "
                           f"типа); интервалов нет (bootstrap = {cfg.bootstrap}); AP — при maxDets={cfg.max_dets}, справочно — при 100",
           "metrics": metrics,
           "selection_cal": {"enabled": False, "reason": "калибровочные платы PCB в авторежиме не сегментируются; вариант "
                             f"контрольного прогона перенесён с HR-InsDet — {RU.PCB_BASELINE_VARIANT}"},
           **pcb_blocks}
    boot = {proto: metrics[proto].pop("boot_ap") for proto in metrics if "boot_ap" in metrics[proto]}
    if boot:  # только если интервалы на PCB будут досчитаны: место то же, что у записей HR-InsDet
        rec["bootstrap_ap"] = boot
    if cfg.run_id == RU.PCB_FIRST_RUN:
        crit = (g.passport.get("build") or {}).get("pcb_gallery_criterion")
        if crit is None:
            raise SystemExit(f"{g.path}: в паспорте галереи нет критерия осмысленности галереи PCB")
        rec["pcb_gallery_criterion"] = crit
        head = metrics[RU.SELECT_GALLERY]["subsets"]["test/all"]
        t = head["by_type"][RU.PCB_SANITY_TYPE]
        rec["pcb_first_run_check"] = chk = RU.pcb_first_run_check(t["top1"], t["n_oracle"], head["by_area"]["all"]["auroc"])
        say(f"ПЕРВЫЙ ПРОГОН PCB: top-1 {chk['type']} на {chk['n_oracle']} оракульных масках — {chk['top1']}, порог "
            f"{chk['top1_min']:.4f}; AUROC {chk['auroc']}, порог {chk['auroc_min']:g} — "
            f"{'выдержаны' if chk['passed'] else 'НЕ ВЫДЕРЖАНЫ: разбор до остальных прогонов'}")
        say(f"критерий осмысленности галереи PCB: среднее top-1 {crit['mean_top1']:.4f}, типов не ниже "
            f"{crit['type_top1_min']} — {crit['n_types_ge_min']}; критерий {'выполнен' if crit['run_4_5_on_pcb'] else 'не выполнен'}")
    out = record_path(cfg.run_id)
    _write(out, rec)
    say(f"ГОТОВО: {out}")
    return out


def evaluate_grid(cfg, kind: str, partner: str | None, say) -> Path:
    scenes, labels, edges, sp, emb, g, gt, st, scenes_cache, refs_cache, n_fallback = _load_run(cfg, say)
    found, search_info, sec_search = _search_all(cfg, emb, g, say)
    t0 = time.perf_counter()
    metrics = {}
    is_pcb = cfg.dataset == "pcb"
    lv, names = PR.levels(cfg.dataset), PR.grid_subsets(cfg.dataset)
    for proto, (y, s) in found.items():
        metrics[proto] = PR.evaluate_oracle(scenes, labels, edges, gt, g.labels, y, s, cfg.max_dets, cfg.bootstrap, cfg.seed,
                                            names=names if is_pcb else (*names, PR.SELECT_SUBSET), levels_=lv, by_type=is_pcb)
        a = metrics[proto]["subsets"]["test/all"]["by_area"]["all"]
        say(f"галерея «{proto}», {len(sp['test'])} тестовых снимков, оракул: AP {100 * a['ap']:.2f} AP50 {100 * a['ap50']:.2f} "
            f"AP75 {100 * a['ap75']:.2f} top-1 {100 * a['top1']:.2f} AUROC {a['auroc']:.4f}")
    pcb_blocks = _pcb_blocks(cfg, "oracle", scenes, labels, edges, g, found, metrics, say) if is_pcb else {}
    sec_eval = time.perf_counter() - t0
    progress = RE.read_progress(st)
    sub = PR.subsets(scenes, lv)
    if is_pcb:
        return _write_pcb_grid(cfg, kind, sp, g, st, scenes_cache, refs_cache, scenes, edges, progress, sub, names, metrics,
                               search_info, sec_search, sec_eval, n_fallback, pcb_blocks, say)
    sel = metrics[RU.SELECT_GALLERY]["subsets"].pop(PR.SELECT_SUBSET)
    for proto in metrics:  # калибровочные сцены в метрики 4.3 не входят
        metrics[proto]["subsets"].pop(PR.SELECT_SUBSET, None)
    a = sel["by_area"]["all"]
    rec = _common(cfg, kind, sp, g, st, scenes_cache, refs_cache, scenes)
    seg = rec.pop("_segmentation_sec")
    rec["rules"]["area_edges_px2"] = edges[:2]
    rec = {"run_id": cfg.run_id, "experiment": "4.3", "kind": "grid_run", "protocol": "oracle", **rec,
           "paired_with": partner, "encode_sec_shared": partner is not None, "pass": kind,
           # из конфигурации; путь P батча вырезок не использует: все маски снимка — один проход
           "encode": {"encoder_batch": cfg.encoder_batch if cfg.phi.kind == "crop" else None,
                      "batch_rule": cfg.batch_rule if cfg.phi.kind == "crop" else "pool: один проход энкодера на снимок, все маски снимка разом",
                      "crop_workers": sorted({p.get("crop_workers") for p in progress} - {None}),   # фактические, по сессиям
                      "crop_max_in_flight": sorted({p.get("crop_max_in_flight") for p in progress} - {None}),
                      "n_pool_center_patch_fallback": n_fallback, "sessions": sorted({p["session"] for p in progress}),
                      "code_commits": sorted({p["code_commit"] for p in progress})},
           "timing": {"segmentation_from_cache_info_sec": seg,
                      "encode_scenes_sec": round(sum(p["sec"] for p in progress), 1),
                      "encode_scenes_n_masks": int(sum(p["n_masks"] for p in progress)),
                      "encode_gallery_sec": (g.passport.get("build") or {}).get("sec_encode_this_session"),
                      "search_sec": round(sec_search, 1), "evaluate_sec": round(sec_eval, 1),
                      "note": "кодирование — сумма по сценам из progress.jsonl рабочих файлов (счёт урывками); у общего "
                              "прохода время общее на оба прогона; задержка по этапам для 4.1 — из отдельных замеров"},
           "counts": {n: PR.counts(scenes, sub[n], edges) for n in ("cal/all", *PR.GRID_SUBSETS)},
           "search": search_info,
           "metrics_note": "доли (0–1); метрики — по тестовым сценам; интервалы 95 % — бутстрэп по сценам, "
                           f"{cfg.bootstrap} повторов, перцентильный; AP — при maxDets={cfg.max_dets}, справочно — при 100",
           "metrics": metrics,
           "bootstrap_ap": {proto: metrics[proto].pop("boot_ap") for proto in metrics},
           "selection_cal": {"split": RU.SELECT_SPLIT, "gallery": RU.SELECT_GALLERY, "max_dets": cfg.max_dets,
                             "n_scenes": sel["n_scenes"], "ap": a["ap"], "ap50": a["ap50"], "ap75": a["ap75"],
                             "ci95": {k: a["ci95"][k] for k in ("ap", "ap50", "ap75")}, "n_gt": a["n_gt"],
                             "rule": "лучший вариант энкодера — `src.eval.rules.best_variant`; выбирает make_tables.py"},
           "ap_without_part_mask_labels": {
               "labels": list(RU.PART_MASK_LABELS),
               **{proto: {n: v["ap_without_part_mask_labels"] for n, v in metrics[proto]["subsets"].items()}
                  for proto in metrics}}}
    for proto in metrics:
        for v in metrics[proto]["subsets"].values():
            v.pop("ap_without_part_mask_labels")
    out = record_path(cfg.run_id)
    _write(out, rec)
    say(f"selection_cal (40 калибровочных сцен, полная галерея): AP {100 * a['ap']:.2f}, AP50 {100 * a['ap50']:.2f}")
    say(f"ГОТОВО: {out}")
    return out


def evaluate_baseline(cfg, say) -> Path:
    grid = record_path(cfg.run_id)
    if not grid.is_file():
        raise SystemExit(f"контрольный прогон идёт по эмбеддингам прогона сетки: нет {grid}")
    scenes, labels, edges, sp, emb, g, gt, st, scenes_cache, refs_cache, _ = _load_run(cfg, say)
    found, search_info, sec_search = _search_all(cfg, emb, g, say)
    t0 = time.perf_counter()
    metrics = {}
    is_pcb = cfg.dataset == "pcb"
    head = "test/all" if is_pcb else "all/all"
    for proto, (y, s) in found.items():
        metrics[proto] = PR.evaluate_baseline(scenes, labels, edges, gt, g.labels, y, s, cfg.max_dets, cfg.bootstrap, cfg.seed,
                                              names=PR.baseline_subsets(cfg.dataset), levels_=PR.levels(cfg.dataset),
                                              by_type=is_pcb)
        a = metrics[proto]["subsets"][head]["by_area"]["all"]
        say(f"галерея «{proto}», {len(scenes)} снимков, baseline: AP {100 * a['ap']:.2f} AP50 {100 * a['ap50']:.2f} AP75 {100 * a['ap75']:.2f}")
    pcb_blocks = _pcb_blocks(cfg, "baseline", scenes, labels, edges, g, found, metrics, say) if is_pcb else {}
    rec = _common(cfg, "baseline", sp, g, st, scenes_cache, refs_cache, scenes)
    rec.pop("_segmentation_sec")
    rec["rules"]["area_edges_px2"] = edges[:2]
    rec = {"run_id": cfg.run_id + BASELINE_SUFFIX, "experiment": "4.3", "kind": "control_run", "protocol": "baseline",
           "grid_run": cfg.run_id, **rec,
           "what": "все маски M(I), независимые решения, штатный NMS SAM 2, без оракула и без порога; AP — по всем "
                   "детекциям с оценкой s*; AR итоговых детекций — в четырёх вариантах",
           "timing": {"search_sec": round(sec_search, 1), "evaluate_sec": round(time.perf_counter() - t0, 1)},
           "search": search_info,
           "metrics_note": "доли (0–1); all — 160 сцен (набор опубликованного числа), test — 120 тестовых; интервалы "
                           f"95 % — бутстрэп по сценам, {cfg.bootstrap} повторов; AP — при maxDets={cfg.max_dets}, справочно — при 100",
           "ar_for_4_6": RU.AR_FOR_4_6,
           "metrics": metrics}
    if is_pcb:
        rec["what"] = rec["what"].replace("без оракула и без порога", "без оракула и без порога; только тестовые платы, зоны "
                                          "игнорирования — в истине с iscrowd=1 в каждой категории")
        rec["metrics_note"] = ("доли (0–1); test — 361 снимок тестовых плат, разбивка — по платам и по типам дефекта (AP "
                               f"категории на всех снимках); интервалов нет (bootstrap = {cfg.bootstrap}); AP — при "
                               f"maxDets={cfg.max_dets}, справочно — при 100")
        rec.update(pcb_blocks)
    else:
        rec["ap_without_part_mask_labels"] = {
            "labels": list(RU.PART_MASK_LABELS),
            **{proto: {n: v.pop("ap_without_part_mask_labels") for n, v in metrics[proto]["subsets"].items()}
               for proto in metrics}}
    if cfg.run_id == RU.FIRST_RUN:
        ap = 100 * metrics[RU.FIRST_RUN_GALLERY]["subsets"][f"{RU.FIRST_RUN_SCENES}/all"]["by_area"]["all"]["ap"]
        rec["first_run_check"] = chk = RU.first_run_check(ap)
        chk["review_if_diff_exceeds_max"] = RU.FIRST_RUN_REVIEW
        chk["published_ap_readme_vit_l"] = RU.PUBLISHED_AP_README_VIT_L
        say(f"КОНТРОЛЬ НА ВХОДЕ: AP {ap:.2f} на 160 сценах; опубликовано {chk['published_ap']}; разница {chk['diff']:+.2f}; "
            f"порог {chk['ap_min']:g} — {'выдержан' if chk['ap_ge_min'] else 'НЕ ВЫДЕРЖАН: ошибка пайплайна, прогоны останавливаются'}; "
            f"допуск ±{chk['diff_max']:g} — {'выдержан' if chk['diff_within_max'] else 'НЕ ВЫДЕРЖАН: разбирается до продолжения'}")
    out = record_path(cfg.run_id, baseline=True)
    _write(out, rec)
    say(f"ГОТОВО: {out}")
    return out


def best_of_encoder(encoder: str, dataset: str = "hr_insdet") -> str:
    """Лучший вариант энкодера по `selection_cal` — только когда посчитаны все его прогоны сетки."""
    from src.encode.variants import grid

    names = [v for e, v, d in grid() if e == encoder and d == dataset]
    missing = [v for v in names if not record_path(f"{encoder}_{v}_{dataset}").is_file()]
    if missing:
        raise SystemExit(f"лучший φ {encoder} выбирается по всем его прогонам; нет записей: {missing}")
    cal = {v: json.loads(record_path(f"{encoder}_{v}_{dataset}").read_text())["selection_cal"] for v in names}
    return RU.best_variant({v: {"ap": c["ap"], "ap50": c["ap50"]} for v, c in cal.items()})


# ---------------------------------------------------------------------- точка входа


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", type=Path, default=[], help="один прогон либо пара (общий проход)")
    ap.add_argument("--protocol", choices=["oracle", "baseline"], default="oracle")
    ap.add_argument("--baseline-best", choices=["dinov2", "dinov3"], help="контрольный прогон лучшего φ энкодера")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.baseline_best:
        best = best_of_encoder(args.baseline_best)
        args.config, args.protocol = [CFG.CONFIG_DIR / f"{args.baseline_best}_{best}_hr_insdet.yaml"], "baseline"
    cfgs = [CFG.load(p) for p in args.config]
    if not cfgs:
        ap.error("нужен --config либо --baseline-best")
    kind = check_pass(cfgs)
    say = _logger([c.run_id + (BASELINE_SUFFIX if args.protocol == "baseline" else "") for c in cfgs])

    if args.status:
        scenes_cache, _ = _caches(cfgs[0])
        sp = json.loads(Path(f"splits/{cfgs[0].dataset}.json").read_text())
        for c in cfgs:
            st = RE.scene_store(c, scenes_cache.key)
            n = sum(st.has(s) for s in _scene_ids(c, sp))
            say(f"{c.run_id}: сцен закодировано {n}/{len(_scene_ids(c, sp))}; запись сетки — "
                f"{'есть' if record_path(c.run_id).is_file() else 'нет'}; baseline — "
                f"{'есть' if record_path(c.run_id, True).is_file() else 'нет'}")
        return

    if args.protocol == "baseline":
        if kind != "single":
            raise SystemExit("контрольный прогон — по одному прогону сетки")
        cfg = cfgs[0]
        if cfg.dataset == "pcb":  # выбора на PCB нет: вариант перенесён с HR-InsDet и сверяется с его журналом
            if cfg.variant != RU.PCB_BASELINE_VARIANT or best_of_encoder(cfg.encoder, "hr_insdet") != RU.PCB_BASELINE_VARIANT:
                raise SystemExit(f"контрольный прогон PCB — только {RU.PCB_BASELINE_VARIANT}, лучшим φ {cfg.encoder} на "
                                 "HR-InsDet")
        elif cfg.run_id != RU.FIRST_RUN and cfg.variant != best_of_encoder(cfg.encoder, cfg.dataset):
            raise SystemExit("контрольный прогон — только для первого прогона и лучшего φ энкодера")
        if record_path(cfg.run_id, True).is_file():
            say(f"запись уже есть: {record_path(cfg.run_id, True)} — не пересчитывается")
            return
        evaluate_baseline(cfg, say)
        return

    todo = [c for c in cfgs if not record_path(c.run_id).is_file()]
    if not todo:
        say("записи уже есть: " + ", ".join(str(record_path(c.run_id)) for c in cfgs) + " — не пересчитываются")
        return
    if len(todo) != len(cfgs):
        raise SystemExit("у пары посчитан один прогон из двух: общий проход не делится — разобрать вручную")
    check_order(cfgs, say)
    say(f"проход «{kind}», версия кода {_STAMP['code_commit'][:7]}{' (есть незакоммиченные правки)' if _STAMP['code_dirty'] else ''}")

    scenes_cache, _ = _caches(cfgs[0])
    sp = json.loads(Path(f"splits/{cfgs[0].dataset}.json").read_text())
    scene_ids = _scene_ids(cfgs[0], sp)
    absent = [s for s in scene_ids if not scenes_cache.has(s)]
    if absent:
        raise SystemExit(f"в кеше масок нет {len(absent)} сцен: {_cache_script(cfgs[0])}")
    stores = [RE.scene_store(c, scenes_cache.key) for c in cfgs]
    need_gpu = (not all(st.has(s) for st in stores for s in scene_ids)
                or not all((store.gallery_dir(c.encoder, c.variant, c.dataset) / "gallery.json").is_file() for c in cfgs))
    if need_gpu:
        import contextlib

        from src.encode import pipeline as PL

        crop = cfgs[0].phi.kind == "crop"
        with (PL.CropFeeder() if crop else contextlib.nullcontext()) as feeder:  # процессы — до инициализации CUDA
            import torch

            from src.encode import debias as D
            from src.encode import model as M

            models = {e: M.load(CFG.ENCODERS[e]) for e in sorted({c.encoder for c in cfgs})}
            u_r = None
            if any(c.variant == "p_perp" for c in cfgs):
                u_r = torch.from_numpy(load_u_r(next(c for c in cfgs if c.variant == "p_perp"))).cuda()
            ensure_galleries(cfgs, kind, models, feeder, u_r, say)
            encode_scenes(cfgs, kind, scene_ids, models, feeder, u_r, say)
            del models
            torch.cuda.empty_cache()
    for c in cfgs:
        partner = next((x.run_id for x in cfgs if x is not c), None)
        evaluate_grid(c, kind, partner, say)


if __name__ == "__main__":
    main()
