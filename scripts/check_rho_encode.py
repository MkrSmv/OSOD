"""Замер кодирования с $\\rho$.

    python scripts/check_rho_encode.py run       # GPU, порядка 0,5 ч на оба энкодера; прерываемо — посчитанное пропускается
    python scripts/check_rho_encode.py status
    python scripts/check_rho_encode.py measure   # запись experiments/rho_encode_check.json по рабочим файлам (без GPU)

Только 40 калибровочных сцен HR-InsDet; тестовые сцены и их кеш не читаются. Это не прогон сетки и не источник метрик
качества: в таблицы результатов не идёт. Оба лучших $\\varphi$ (`rules.best_variant` по журналу) кодируют кодом
пайплайна (`src.encode.pipeline`, батч 16, маски одного изображения в порядке записи кеша) сначала все маски $M(I)$,
затем только $M^*$ из `cache/rho/`. Эмбеддинги обоих проходов копятся в рабочих файлах
`cache/run_emb/rho_encode_check__<run_id>__<all|rho>/` в fp16 — как у прогона сетки, — время по сценам — в
`progress.jsonl` каталога (прерванный замер складывается из строк, как время кодирования прогона сетки).

В запись идут: время кодирования обоих путей и, для масок $M^*$, расхождение нового эмбеддинга (кодирование только
$M^*$ — иной состав батча) с сохранённым эмбеддингом той же маски из прогона сетки `cache/run_emb/<run_id>/`: косинус,
наибольшее $|\\Delta s^*|$ и число изменившихся $\\hat y$ при полной галерее, точный перебор. Порог записан до счёта: медиана косинуса в каждом бине площади не ниже `COS_MEDIAN_MIN` у каждого энкодера; $|\\Delta s^*|$ и
число изменившихся $\\hat y$ — описательно, порога у них нет. Бин площади маски — по площади её описывающей рамки
$\\mathrm{box}(m)$ в пикселях исходного снимка, границы — `hr_insdet.AREA_EDGES` (как бин дистрактора в AUROC 4.3):
у большинства масок $M^*$ размеченной рамки нет. При невыдержанном пороге 4.4 не начинается.

Порядок проходов: сначала $M(I)$, затем $M^*$; перед замером каждого энкодера — разогрев (первый батч
первой сцены, в замер не входит). Время — от выдачи сцены до выдачи следующей, с подготовкой вырезок на CPU параллельно
GPU, то есть время кодирования пайплайна целиком, а не одного прохода энкодера.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as CFG  # noqa: E402
from src import env  # noqa: E402
from src.data import hr_insdet  # noqa: E402
from src.encode import run_emb as RE  # noqa: E402
from src.encode.variants import grid  # noqa: E402
from src.eval import recall as R  # noqa: E402
from src.eval import rules as RU  # noqa: E402
from src.gallery import store  # noqa: E402
from src.segment import cache as MC  # noqa: E402
from src.select import rho as RHO  # noqa: E402

DATASET, SPLIT, N_SCENES = "hr_insdet", "cal", 40
ENCODERS = ("dinov2", "dinov3")
PASSES = ("all", "rho")            # порядок проходов
COS_MEDIAN_MIN = 0.999             # порог, записанный до счёта: медиана косинуса в каждом бине площади, у каждого энкодера
EXPECTED_MASKS = {"all": 4534, "rho": 2498}
RUNS = Path("experiments/runs")
OUT = Path("experiments/rho_encode_check.json")
_STAMP = env.code_stamp()


def _say(msg: str) -> None:
    Path("logs").mkdir(exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [rho_encode] {msg}"
    print(line, flush=True)
    with open("logs/check_rho_encode.log", "a") as fh:
        fh.write(line + "\n")


def best_configs() -> dict:
    out = {}
    for e in ENCODERS:
        names = [v for enc, v, d in grid() if enc == e and d == DATASET]
        cal = {v: json.loads((RUNS / f"{e}_{v}_{DATASET}.json").read_text())["selection_cal"] for v in names}
        best = RU.best_variant({v: {"ap": c["ap"], "ap50": c["ap50"]} for v, c in cal.items()})
        out[e] = CFG.load(CFG.CONFIG_DIR / f"{e}_{best}_{DATASET}.yaml")
    return out


def _setup():
    cfgs = best_configs()
    c0 = cfgs[ENCODERS[0]]
    if any(c.phi.kind != "crop" for c in cfgs.values()):
        raise SystemExit("замер определён для вариантов вырезки: у z_pool отбор на эмбеддинг маски не влияет")
    mc = MC.MaskCache(DATASET, MC.auto_key(c0.crop_n_layers, c0.seg_long_side, c0.points_per_batch))
    sp = json.loads(Path(f"splits/{DATASET}.json").read_text())
    scenes = list(sp[SPLIT])
    if len(scenes) != N_SCENES or set(scenes) & set(sp["test"]):
        raise SystemExit("замер — только 40 калибровочных сцен")
    rc = RHO.RhoCache(mc)
    n = {s: len(mc.load(s)) for s in scenes}
    sel = {s: rc.selected(s).tolist() for s in scenes}
    got = {"all": sum(n.values()), "rho": sum(map(len, sel.values()))}
    if got != EXPECTED_MASKS:
        raise SystemExit(f"состав замера {got}, ожидается {EXPECTED_MASKS}")
    return cfgs, mc, scenes, n, sel


def _store(cfg, mc, which: str) -> RE.RunEmb:
    return RE.RunEmb(f"rho_encode_check__{cfg.run_id}__{which}",
                     {"what": "rho_encode_check", "pass": which, "config": cfg.to_dict(), "mask_key": mc.key,
                      "rho": None if which == "all" else {"theta": str(RHO.THETA), "gamma": str(RHO.GAMMA), "rule": RHO.RHO_RULE}})


def stage_status() -> None:
    cfgs, mc, scenes, _, _ = _setup()
    for e, cfg in cfgs.items():
        for which in PASSES:
            st = _store(cfg, mc, which)
            _say(f"{cfg.run_id} / {which}: сцен {sum(st.path(s).is_file() for s in scenes)} из {len(scenes)}")
    _say(f"запись: {'есть' if OUT.exists() else 'нет'} ({OUT})")


def stage_run() -> None:
    from src.encode import pipeline as PL

    cfgs, mc, scenes, n, sel = _setup()
    _say(f"версия кода {_STAMP['code_commit'][:7]}{' (есть незакоммиченные правки)' if _STAMP['code_dirty'] else ''}")
    with PL.CropFeeder(workers=PL.WORKERS) as feeder:      # процессы подготовки — до инициализации CUDA
        from src.encode import model as M
        import torch

        for e, cfg in cfgs.items():
            stores = {w: _store(cfg, mc, w) for w in PASSES}
            if all(st.has(s) and s in {r["image_id"] for r in RE.read_progress(st)} for st in stores.values() for s in scenes):
                _say(f"{cfg.run_id}: оба прохода посчитаны")
                continue
            model = M.load(cfg.model_key)
            kw = dict(size=cfg.crop_size, sigma_frac=cfg.blur_sigma_frac, work_side=cfg.blur_work_side)
            ref = lambda s: PL.image_ref(DATASET, s, hr_insdet.ROOT / "Scenes" / f"{s}.jpg", mc.key)
            warm = [(ref(scenes[0]), n[scenes[0]], list(range(min(cfg.encoder_batch, n[scenes[0]]))))]
            list(PL.encode_variant(model, feeder, warm, cfg.phi.b, cfg.phi.alpha, cfg.encoder_batch, **kw))
            for which in PASSES:
                st = stores[which]
                logged = {r["image_id"] for r in RE.read_progress(st)}
                todo = [s for s in scenes if not st.has(s) or s not in logged]  # обрыв между файлом и строкой времени
                images = [(ref(s), n[s], None if which == "all" else sel[s]) for s in todo]
                n_todo = sum(n[s] if which == "all" else len(sel[s]) for s in todo)
                _say(f"{cfg.run_id} / {which}: сцен к счёту {len(todo)}, масок {n_todo}")
                done, t0 = 0, time.perf_counter()
                t_prev = t0
                for k, (r, z) in enumerate(PL.encode_variant(model, feeder, images, cfg.phi.b, cfg.phi.alpha,
                                                             cfg.encoder_batch, **kw), 1):
                    now = time.perf_counter()
                    st.save(r.image_id, z)
                    RE.log_progress(st, image_id=r.image_id, n_masks=len(z), sec=round(now - t_prev, 3),
                                    session=_STAMP["written_utc"], code_commit=_STAMP["code_commit"],
                                    code_dirty=_STAMP["code_dirty"], crop_workers=feeder.workers)
                    done += len(z)
                    _say(f"{cfg.run_id} / {which} [{k}/{len(todo)}] {r.image_id}: масок {len(z)}; всего {done}/{n_todo}, "
                         f"{done / (now - t0):.1f} маски/с, осталось ~{(n_todo - done) / max(done, 1) * (now - t0) / 60:.0f} мин")
                    t_prev = time.perf_counter()
            del model
            torch.cuda.empty_cache()
    _say("кодирование завершено; запись — этап measure")


def _unit(z: np.ndarray) -> np.ndarray:
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def stage_measure() -> int:
    from src.search import decide as DE
    from src.search import index as IX

    cfgs, mc, scenes, n, sel = _setup()
    res, passed = {}, True
    for e, cfg in cfgs.items():
        stores = {w: _store(cfg, mc, w) for w in PASSES}
        missing = [(w, s) for w, st in stores.items() for s in scenes if not st.path(s).is_file()]
        if missing:
            raise SystemExit(f"{cfg.run_id}: не посчитано {len(missing)} сцен — сначала этап run")
        saved = RE.scene_store(cfg, mc.key)                       # рабочие файлы прогона сетки: эмбеддинги всех M(I)
        g = store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset),
                               store.expected(cfg.to_dict(), MC.box_key(cfg.seg_long_side)))
        index = IX.ExactIndex(g.emb, g.rows("full"))
        decide = lambda z: DE.decide(*index.search(np.ascontiguousarray(z)), g.label_ids, deleted=g.deleted)
        timing = {}
        for w, st in stores.items():
            rows = [r for r in RE.read_progress(st)]
            by_scene = {}
            for r in rows:
                if r["image_id"] in by_scene:
                    raise SystemExit(f"{st.dir}: сцена {r['image_id']} закодирована дважды — время неоднозначно")
                by_scene[r["image_id"]] = r
            if set(by_scene) != set(scenes):
                raise SystemExit(f"{st.dir}: строки времени — не по 40 сценам")
            sec, nm = sum(r["sec"] for r in by_scene.values()), sum(r["n_masks"] for r in by_scene.values())
            timing[w] = {"n_masks": nm, "sec": round(sec, 1), "masks_per_sec": round(nm / sec, 2),
                         "sec_per_scene_median": float(np.median([r["sec"] for r in by_scene.values()])),
                         "sessions": sorted({r["session"] for r in by_scene.values()}),
                         "code_commits": sorted({r["code_commit"] for r in by_scene.values()}),
                         "code_dirty": any(r["code_dirty"] for r in by_scene.values())}
        cos, area, ds, dy = [], [], [], 0
        for s in scenes:
            z_saved = saved.load(s)["z"]
            if len(z_saved) != n[s]:
                raise SystemExit(f"{s}: сохранённых эмбеддингов {len(z_saved)}, масок {n[s]}")
            z_all, z_rho = stores["all"].load(s)["z"], stores["rho"].load(s)["z"]
            if len(z_all) != n[s] or len(z_rho) != len(sel[s]):
                raise SystemExit(f"{s}: число эмбеддингов замера расходится с составом")
            k = np.asarray(sel[s], np.int64)
            if not len(k):
                continue
            cos.append((_unit(z_rho) * _unit(z_saved[k])).sum(1))
            b = mc.load(s).boxes[k]
            area.append((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
            (y0, s0), (y1, s1) = decide(z_saved[k]), decide(z_rho)
            ds.append(np.abs(s1.astype(np.float64) - s0.astype(np.float64)))
            dy += int((y0 != y1).sum())
        cos, area, ds = map(np.concatenate, (cos, area, ds))
        bins = R.area_bin(area, hr_insdet.AREA_EDGES)
        by_bin = {lab: {"n": int((bins == a).sum()), "cos_median": float(np.median(cos[bins == a])),
                        "cos_min": float(cos[bins == a].min())} for a, lab in enumerate(R.AREA_LABELS[1:]) if (bins == a).any()}
        ok = all(v["cos_median"] >= COS_MEDIAN_MIN for v in by_bin.values())
        passed &= ok
        res[e] = {"run_id": cfg.run_id, "variant_label": cfg.variant, "timing": timing,
                  "time_ratio_rho_to_all": round(timing["rho"]["sec"] / timing["all"]["sec"], 4),
                  "masks_ratio_rho_to_all": round(timing["rho"]["n_masks"] / timing["all"]["n_masks"], 4),
                  "rho_vs_saved": {"n_masks": int(len(cos)), "by_area_bin": by_bin, "cos_median_all": float(np.median(cos)),
                                   "cos_min_all": float(cos.min()), "max_abs_delta_s_star": float(ds.max()),
                                   "n_y_hat_changed": dy, "gallery": "full", "search": "IndexFlatIP, k = N"},
                  "passed": bool(ok)}
        _say(f"{cfg.run_id}: M(I) {timing['all']['n_masks']} масок за {timing['all']['sec']} с, M* {timing['rho']['n_masks']} "
             f"за {timing['rho']['sec']} с; косинус M* с сохранённым — по бинам {[(k, round(v['cos_median'], 6)) for k, v in by_bin.items()]}, "
             f"минимум {cos.min():.6f}; max|Δs*| {ds.max():.2e}; изменилось ŷ — {dy}")
    out = {"what": "замер кодирования с ρ на 40 калибровочных сценах; "
                   "не прогон сетки и не источник метрик качества",
           "dataset": DATASET, "split": SPLIT, "n_scenes": N_SCENES, **_STAMP,
           "threshold": {"cos_median_per_area_bin_min": COS_MEDIAN_MIN, "set_before_run": "до замера",
                         "area_bin": "площадь описывающей рамки маски в пикселях исходного снимка, hr_insdet.AREA_EDGES",
                         "descriptive_only": ["max_abs_delta_s_star", "n_y_hat_changed"]},
           "protocol": {"passes_order": list(PASSES), "warmup": "первый батч первой сцены до замера каждого энкодера",
                        "timing": "от выдачи сцены до выдачи следующей; вырезки на CPU параллельно GPU",
                        "embeddings": "fp16 рабочих файлов → fp32, как в прогоне сетки", "batch": cfgs[ENCODERS[0]].encoder_batch},
           "by_encoder": res, "passed": bool(passed)}
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT)
    _say(f"{'ПОРОГ ВЫДЕРЖАН' if passed else 'ПОРОГ НЕ ВЫДЕРЖАН — 4.4 не начинать'}: {OUT}")
    return 0 if passed else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "status", "measure"))
    stage = ap.parse_args().stage
    if stage == "run":
        stage_run()
    elif stage == "status":
        stage_status()
    else:
        sys.exit(stage_measure())


if __name__ == "__main__":
    main()
