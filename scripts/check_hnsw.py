"""Полнота HNSW относительно точного перебора и задержка обоих индексов → `experiments/runs/hnsw_<run_id>.json`.

    python scripts/check_hnsw.py --config configs/dinov2_c_0_10_hr_insdet.yaml encode    # запросы: маски cal-сцен, GPU
    python scripts/check_hnsw.py --config configs/dinov2_c_0_10_hr_insdet.yaml measure   # замер по готовым файлам, CPU
    python scripts/check_hnsw.py --config configs/dinov2_c_0_10_hr_insdet.yaml diagnose  # разбор несовпадений, в ту же запись
    python scripts/check_hnsw.py --config configs/<encoder>_<variant>_hr_insdet.yaml grid  # полнота на галерее прогона сетки

Первый замер (этапы `encode`, `measure`, `diagnose`) — на галерее $C(0,1{,}0)$ DINOv2. Протокол, состав запросов и
критерий записаны до замера и стоят здесь константами. Запросы — все маски $M(I)$ 40 калибровочных сцен, закодированные
тем же $\\varphi$, что галерея; тестовые сцены не открываются. Порог $\\tau$ не применяется (калибровка —
`scripts/calibrate.py`), метрики качества не считаются (их дают прогоны сетки 4.3). Этап `diagnose` добавлен после результата (критерий не выдержан) и критерия не касается: параметры графа
в нём те же, что в замере, ни один не варьируется — он описывает несовпавшие запросы и разброс полноты между
повторными построениями графа (многопоточный `add` Faiss недетерминирован). Этап `encode` прерываем: эмбеддинги запросов копятся в `cache/checks/hnsw/<run_id>/` (fp16, как рабочие
файлы прогона) и читаются оттуда, как в прогоне. Вывод — ещё и в `logs/hnsw_<run_id>.log`.

Этап `grid` — замер полноты на остальных галереях HR-InsDet (протокол записан до любого прогона 4.3): запросы — маски калибровочных сцен из рабочих файлов самого прогона
`cache/run_emb/<run_id>/`, одно однопоточное построение графа (вердикт по критерию — по нему) и 30 многопоточных,
задержка не повторяется. Для галереи первого замера действует уже сделанная запись: существующая запись не
перезаписывается.
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
from src.encode.run_emb import RunEmb
from src.gallery import store
from src.segment import cache as MC

# --- записано до замера (коммит f62d6d3); по результату не пересматривается
RECALL_MIN = 0.999      # доля запросов, у которых ŷ и s* совпали с точным перебором
S_STAR_TOL = 1e-5       # допуск на s*: разные ядра скалярного произведения в Faiss, не ошибка поиска
QUERY_SPLIT = "cal"     # только калибровочные сцены
REPEATS = 5             # повторов замера задержки; у каждого вызова — медиана по повторам
THREADS = ("default", 1)
ON_FAIL = "параметры графа не подбираются"

GRID_BUILDS_MULTI = 30  # этап grid: многопоточных построений рядом с одним однопоточным
GRID_QUANTILES = (0.95, 0.99)

OVERWRITE = False  # ставится флагом --overwrite
WORK_ROOT = Path("cache/checks/hnsw")
RUNS = Path("experiments/runs")
_STAMP = env.code_stamp()  # версия кода — на момент запуска процесса


def _setup(cfg):
    from src.data import hr_insdet

    if cfg.dataset != "hr_insdet":
        raise SystemExit("замер HNSW — на HR-InsDet")
    sp = json.loads(Path(f"splits/{cfg.dataset}.json").read_text())
    scenes = sp[QUERY_SPLIT]
    if len(scenes) != 40 or set(scenes) & set(sp["test"]):  # не assert: проверка не должна сниматься флагом -O
        raise SystemExit("запросы — только 40 калибровочных сцен; тестовые сцены не открываются")
    mc = MC.MaskCache(cfg.dataset, MC.auto_key(cfg.crop_n_layers, cfg.seg_long_side, cfg.points_per_batch))
    work = RunEmb(cfg.run_id, {"what": "hnsw_check_queries", "config": cfg.to_dict(), "mask_key": mc.key},
                  root=WORK_ROOT)
    return sp, scenes, mc, work, hr_insdet


def _logger(cfg):
    Path("logs").mkdir(exist_ok=True)
    log = open(Path("logs") / f"hnsw_{cfg.run_id}.log", "a")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] [hnsw {cfg.run_id}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    return say


def stage_encode(cfg) -> None:
    from src.encode import pipeline as PL

    sp, scenes, mc, work, hr_insdet = _setup(cfg)
    say = _logger(cfg)
    if cfg.phi.kind != "crop":
        raise SystemExit("первый замер HNSW — на варианте вырезки")
    todo = [s for s in scenes if not work.has(s)]
    say(f"сцен {len(scenes)}, закодировано {len(scenes) - len(todo)}, осталось {len(todo)}")
    if not todo:
        return
    images = [(PL.image_ref(cfg.dataset, s, hr_insdet.ROOT / "Scenes" / f"{s}.jpg", mc.key), len(mc.load(s)))
              for s in todo]
    n_total, n_done, t0 = sum(n for _, n in images), 0, time.perf_counter()
    with PL.CropFeeder() as feeder:  # процессы — до инициализации CUDA
        from src.encode import model as M

        model = M.load(cfg.model_key)
        stream = PL.encode_variant(model, feeder, images, cfg.phi.b, cfg.phi.alpha, cfg.encoder_batch,
                                   size=cfg.crop_size, sigma_frac=cfg.blur_sigma_frac, work_side=cfg.blur_work_side)
        for k, (ref, z) in enumerate(stream, 1):
            work.save(ref.image_id, z)
            n_done += len(z)
            sec = time.perf_counter() - t0
            say(f"[{k}/{len(todo)}] {ref.image_id}: масок {len(z)}; всего {n_done}/{n_total}, {n_done / sec:.1f} маски/с, "
                f"осталось ~{(n_total - n_done) / max(n_done, 1) * sec / 60:.1f} мин")
    say("ГОТОВО: запросы закодированы")


def _latency(search, calls: list[np.ndarray]) -> dict:
    """Таймер — только вокруг `search`; прогрев — полный проход; у каждого вызова — медиана по `REPEATS` повторам."""
    for q in calls:
        search(q)
    t = np.empty((REPEATS, len(calls)))
    for r in range(REPEATS):
        for c, q in enumerate(calls):
            t0 = time.perf_counter()
            search(q)
            t[r, c] = time.perf_counter() - t0
    per_call = np.median(t, 0) * 1e3
    n_q = sum(len(q) for q in calls)
    return {"n_calls": len(calls), "n_queries": n_q, "ms_per_call_median": float(np.median(per_call)),
            "ms_per_call_mean": float(per_call.mean()), "ms_per_call_p95": float(np.quantile(per_call, 0.95)),
            "ms_per_query_mean": float(per_call.sum() / n_q)}


def stage_measure(cfg) -> None:
    import faiss

    from src.search import decide as DE
    from src.search import index as IX

    sp, scenes, mc, work, hr_insdet = _setup(cfg)
    say = _logger(cfg)
    if (RUNS / f"hnsw_{cfg.run_id}.json").exists() and not OVERWRITE:
        # построение графа недетерминировано: повторный замер заменил бы записанное число другим и стёр бы разбор
        raise SystemExit(f"запись замера уже есть: {RUNS / f'hnsw_{cfg.run_id}.json'}; перезапись — только --overwrite, "
                         "отдельным решением с записью")
    missing = [s for s in scenes if not work.has(s)]
    if missing:
        raise SystemExit(f"не закодировано сцен: {len(missing)} — сначала этап encode")
    g = store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset),
                           store.expected(cfg.to_dict(), MC.box_key(cfg.seg_long_side)))
    rows = g.rows("full")
    k, label_ids, deleted = g.k(rows), g.label_ids, g.deleted

    per_scene, is_oracle, _, _ = _queries(cfg, scenes, mc, work, sp, hr_insdet)
    q_all = np.ascontiguousarray(np.concatenate(per_scene))
    say(f"галерея N = {len(rows)}, k = {k}; запросов {len(q_all)} на {len(scenes)} сценах, из них оракульных {is_oracle.sum()}")

    default_threads = faiss.omp_get_max_threads()
    t0 = time.perf_counter()
    exact = IX.ExactIndex(g.emb, rows)
    sec_flat = time.perf_counter() - t0
    t0 = time.perf_counter()
    hnsw = IX.HnswIndex(g.emb, rows, cfg.hnsw["M"], cfg.hnsw["efConstruction"], cfg.hnsw["efSearch_min"])
    sec_hnsw = time.perf_counter() - t0

    # --- полнота: агрегация по меткам одна и та же для обоих индексов
    sim_e, idx_e = exact.search(q_all)            # k = N: s_y по всей галерее
    sim_h, idx_h = hnsw.search(q_all, k)
    y_e, s_e = DE.decide(sim_e, idx_e, label_ids, deleted=deleted)
    y_h, s_h = DE.decide(sim_h, idx_h, label_ids, deleted=deleted)
    same_y, same_s = y_e == y_h, np.abs(s_e - s_h) <= S_STAR_TOL
    top_e = idx_e[:, :k]
    shared = np.array([len(np.intersect1d(a, b)) for a, b in zip(top_e, idx_h)]) / k

    def summary(m: np.ndarray) -> dict:
        return {"n_queries": int(m.sum()), "recall_y_and_s_star": float((same_y & same_s)[m].mean()),
                "same_y": float(same_y[m].mean()), "same_s_star": float(same_s[m].mean()),
                "n_mismatch": int((~(same_y & same_s))[m].sum()), "neighbors_shared_mean": float(shared[m].mean()),
                "max_abs_s_star_diff": float(np.abs(s_e - s_h)[m].max())}

    recall = {"all": summary(np.ones(len(q_all), bool)), "oracle_masks": summary(is_oracle),
              "other_masks": summary(~is_oracle),
              "n_hnsw_short_lists": int((idx_h < 0).any(1).sum())}
    passed = recall["all"]["recall_y_and_s_star"] >= RECALL_MIN
    say(f"полнота (ŷ и s*): {recall['all']['recall_y_and_s_star']:.5f} при критерии {RECALL_MIN} — "
        f"{'ВЫДЕРЖАН' if passed else 'НЕ ВЫДЕРЖАН: ' + ON_FAIL}")

    # --- задержка: только search; агрегация по меткам вне таймеров
    hnsw.prepare(k)
    searches = {f"flat_k{len(rows)}": lambda q: exact.index.search(q, len(rows)),
                f"flat_k{k}": lambda q: exact.index.search(q, k),
                f"hnsw_k{k}": lambda q: hnsw.index.search(q, k)}
    single = [q_all[i:i + 1] for i in range(len(q_all))]
    latency = {}
    for th in THREADS:
        faiss.omp_set_num_threads(default_threads if th == "default" else th)
        n_th = faiss.omp_get_max_threads()
        latency[f"threads_{th}"] = {"faiss_omp_threads": n_th,
                                    **{name: {"single_query": _latency(fn, single), "batch_per_scene": _latency(fn, per_scene)}
                                       for name, fn in searches.items()}}
        say(f"задержка, потоков {n_th}: " + "; ".join(
            f"{name}: {v['single_query']['ms_per_call_median']:.3f} мс на запрос, "
            f"{v['batch_per_scene']['ms_per_call_median']:.2f} мс на сцену"
            for name, v in latency[f"threads_{th}"].items() if name != "faiss_omp_threads"))
    faiss.omp_set_num_threads(default_threads)

    rec = {"run_id": f"hnsw_{cfg.run_id}", "kind": "hnsw_check",
           "written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
           **{**env.code_stamp(), "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"]},
           "config": cfg.to_dict(),
           "protocol": {"criterion_recall_min": RECALL_MIN, "s_star_tol": S_STAR_TOL, "on_fail": ON_FAIL,
                        "criterion_commit": "f62d6d3", "queries": "все маски M(I) 40 калибровочных сцен, тот же φ, из рабочих "
                        "файлов fp16 → fp32; тестовые сцены не открывались", "tau": None,
                        "latency": f"только index.search; прогрев — полный проход; {REPEATS} повторов, у вызова — медиана "
                                   "по повторам; агрегация по меткам вне таймеров", "quality_metrics": "не считаются (их дают прогоны сетки 4.3)"},
           "versions": {"faiss": faiss.__version__, "numpy": np.__version__},
           "gallery": {"path": str(g.path), "N": len(rows), "n_labels": len(g.labels), "n_max": g.n_max(rows), "k": k,
                       "n_deleted": int(deleted.sum()), "build": g.passport["build"]},
           "index": {"hnsw": {**cfg.hnsw, "efSearch": max(k, cfg.hnsw["efSearch_min"]), "metric": "METRIC_INNER_PRODUCT",
                              "build_sec": round(sec_hnsw, 3), "build_threads": default_threads,
                              "add": "один add всех эталонов в порядке insert_no",
                              "deterministic": default_threads == 1,  # многопоточный add Faiss недетерминирован:
                              # повторный замер даёт иное число; разброс — этап diagnose
                              },
                     "flat": {"build_sec": round(sec_flat, 4)}},
           "queries": {"split": QUERY_SPLIT, "n_scenes": len(scenes), "n": int(len(q_all)),
                       "per_scene_min_median_max": [int(min(map(len, per_scene))),
                                                    float(np.median(list(map(len, per_scene)))),
                                                    int(max(map(len, per_scene)))],
                       "mask_cache": mc.dir.name},
           "recall": recall, "passed": bool(passed), "latency": latency}
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{rec['run_id']}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(out)
    say(f"ГОТОВО: {out}")


def _queries(cfg, scenes, mc, work, sp, hr_insdet):
    """Запросы из рабочих файлов (fp16 → fp32), как в прогоне, и признак оракульной маски (справочная разбивка)."""
    from src.eval.oracle import distractors, oracle_assign

    by_scene = {s["id"]: s for s in hr_insdet.scenes()}
    per_scene, is_oracle, is_distr, area = [], [], [], []
    for s in scenes:
        z = work.load(s)["z"]
        entry = mc.load(s)
        if len(z) != len(entry):
            raise SystemExit(f"{s}: эмбеддингов {len(z)}, масок в кеше {len(entry)}")
        gt = np.array([o["box"] for o in hr_insdet.scene_gt(by_scene[s], sp["labels"])], float).reshape(-1, 4)
        a = oracle_assign(gt, entry.boxes)
        flag = np.zeros(len(z), bool)
        flag[a[a >= 0]] = True
        per_scene.append(z)
        is_oracle.append(flag)
        is_distr.append(distractors(gt, entry.boxes))
        area.append(np.array([r["area"] for r in entry.records], float))
    return per_scene, np.concatenate(is_oracle), np.concatenate(is_distr), np.concatenate(area)


# повторных построений графа с теми же параметрами: многопоточное недетерминировано — 30; однопоточное
# детерминировано — 2, только чтобы это показать
DIAG_BUILDS = {"default": 30, 1: 2}
DIAG_QUANTILES = (0.5, 0.95, 0.99)  # s* дистракторов: порог при ε = 0,05 и 0,01 ставится по квантилям 0,95 и 0,99


def stage_diagnose(cfg) -> None:
    """После результата, без изменения параметров графа: кто не совпал и насколько число зависит от построения."""
    import faiss

    from src.search import decide as DE
    from src.search import index as IX

    sp, scenes, mc, work, hr_insdet = _setup(cfg)
    say = _logger(cfg)
    out = RUNS / f"hnsw_{cfg.run_id}.json"
    rec = json.loads(out.read_text())
    g = store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset),
                           store.expected(cfg.to_dict(), MC.box_key(cfg.seg_long_side)))
    rows = g.rows("full")
    k, label_ids, deleted = g.k(rows), g.label_ids, g.deleted
    per_scene, is_oracle, is_distr, area = _queries(cfg, scenes, mc, work, sp, hr_insdet)
    q_all = np.ascontiguousarray(np.concatenate(per_scene))
    y_e, s_e = DE.decide(*IX.ExactIndex(g.emb, rows).search(q_all), label_ids, deleted=deleted)  # как в замере

    default_threads = faiss.omp_get_max_threads()
    builds, last = {}, None
    overs = {}
    q_exact = [float(v) for v in np.quantile(s_e[is_distr], DIAG_QUANTILES)]
    for th_name, n_builds in DIAG_BUILDS.items():
        th = default_threads if th_name == "default" else th_name
        faiss.omp_set_num_threads(th)
        vals, q_diff = [], np.zeros(len(DIAG_QUANTILES))
        over = overs[f"build_threads_{th}"] = {"n_builds": 0, "n_mismatched": 0, "n_oracle_mismatched": 0,
                                                 "n_mismatched_per_build": [], "n_oracle_mismatched_per_build": [],
                                                 "builds_with_oracle_mismatch": 0, "s_star_exact_max": -1.0,
                                                 "s_star_exact_max_oracle": None, "s_star_lost_max": 0.0}
        for _ in range(n_builds):
            h = IX.HnswIndex(g.emb, rows, cfg.hnsw["M"], cfg.hnsw["efConstruction"], cfg.hnsw["efSearch_min"])
            faiss.omp_set_num_threads(default_threads)  # поиск — как в замере
            y_h, s_h = DE.decide(*h.search(q_all, k), label_ids, deleted=deleted)
            faiss.omp_set_num_threads(th)
            ok = (y_e == y_h) & (np.abs(s_e - s_h) <= S_STAR_TOL)
            vals.append(float(ok.mean()))
            bad_b = ~ok
            over["n_builds"] += 1
            over["n_mismatched"] += int(bad_b.sum())
            over["n_oracle_mismatched"] += int((bad_b & is_oracle).sum())
            over["n_mismatched_per_build"].append(int(bad_b.sum()))
            over["n_oracle_mismatched_per_build"].append(int((bad_b & is_oracle).sum()))
            q_diff = np.maximum(q_diff, np.abs(np.quantile(s_h[is_distr], DIAG_QUANTILES) - q_exact))
            if (bad_b & is_oracle).any():
                over["builds_with_oracle_mismatch"] += 1
                over["s_star_exact_max_oracle"] = max(over["s_star_exact_max_oracle"] or -1.0,
                                                      float(s_e[bad_b & is_oracle].max()))
            if bad_b.any():
                over["s_star_exact_max"] = max(over["s_star_exact_max"], float(s_e[bad_b].max()))
                over["s_star_lost_max"] = max(over["s_star_lost_max"], float((s_e - s_h)[bad_b].max()))
            last = (ok, s_h) if th == default_threads else last
        over["distractor_s_star_quantiles_max_abs_diff_vs_exact"] = [float(v) for v in q_diff]  # по `DIAG_QUANTILES`
        builds[f"build_threads_{th}"] = {"recall": vals, "min": min(vals), "median": float(np.median(vals)),
                                         "max": max(vals), "n_builds_passing": int(sum(v >= RECALL_MIN for v in vals))}
        say(f"построений {n_builds}, потоков построения {th}: полнота {min(vals):.5f}–{max(vals):.5f}, "
            f"медиана {np.median(vals):.5f}, критерий выдержан у {builds[f'build_threads_{th}']['n_builds_passing']}")
    faiss.omp_set_num_threads(default_threads)

    ok, s_h = last
    bad = ~ok
    qs = lambda v: [float(x) for x in np.quantile(v, [0, 0.25, 0.5, 0.75, 1])]  # noqa: E731
    rec["post_hoc_diagnostics"] = {
        "note": "добавлено после результата; критерия не касается; параметры графа те же, что в замере, не варьируются",
        "written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        **{**env.code_stamp(), "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"]},
        "repeat_builds": builds,
        "mismatched_over_all_builds": overs,
        "distractor_s_star_quantiles": {"q": list(DIAG_QUANTILES), "exact": q_exact,
                                        "note": "наибольшее по построениям отклонение квантилей HNSW от точных — "
                                                "`distractor_s_star_quantiles_max_abs_diff_vs_exact` в блоке выше; F калибровки "
                                                "требует сходств всех пар «дистрактор — эталон» и из k соседей HNSW не строится"},
        "gallery_emb_sha256": g.passport.get("emb_sha256"),
        "mismatched_in_last_default_build": {
            "n": int(bad.sum()), "n_oracle": int((bad & is_oracle).sum()), "n_distractor": int((bad & is_distr).sum()),
            "s_star_exact_min_q25_median_q75_max": qs(s_e[bad]) if bad.any() else None,
            "s_star_lost_min_q25_median_q75_max": qs((s_e - s_h)[bad]) if bad.any() else None,
            "n_with_s_star_exact_ge_0.4": int((s_e[bad] >= 0.4).sum()),
            "mask_area_px_min_median_max": [float(v) for v in np.quantile(area[bad], [0, 0.5, 1])] if bad.any() else None},
        "all_queries": {"s_star_exact_min_q25_median_q75_max": qs(s_e),
                        "s_star_exact_oracle_min_q25_median_q75_max": qs(s_e[is_oracle]),
                        "s_star_exact_distractor_min_q25_median_q75_max": qs(s_e[is_distr]),
                        "n_distractors": int(is_distr.sum())}}
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(out)
    say(f"несовпавшие (последнее построение): {json.dumps(rec['post_hoc_diagnostics']['mismatched_in_last_default_build'])}")
    say(f"ГОТОВО: блок post_hoc_diagnostics в {out}")


def stage_grid(cfg, out_dir: Path = RUNS) -> None:
    """Шаг 5: полнота HNSW на галерее прогона сетки по запросам из его рабочих файлов."""
    import faiss

    from src.encode.run_emb import scene_store
    from src.search import decide as DE
    from src.search import index as IX

    sp, scenes, mc, _, hr_insdet = _setup(cfg)
    say = _logger(cfg)
    out = Path(out_dir) / f"hnsw_{cfg.run_id}.json"
    if out.exists():
        raise SystemExit(f"запись замера уже есть и не перезаписывается: {out}")
    work = scene_store(cfg, mc.key)  # рабочие файлы прогона; читаются только калибровочные сцены
    missing = [s for s in scenes if not work.has(s)]
    if missing:
        raise SystemExit(f"в рабочих файлах прогона нет сцен: {len(missing)} — сначала scripts/run_4_3.py")
    g = store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset),
                           store.expected(cfg.to_dict(), MC.box_key(cfg.seg_long_side)))
    rows = g.rows("full")
    k, label_ids, deleted = g.k(rows), g.label_ids, g.deleted
    per_scene, is_oracle, is_distr, _ = _queries(cfg, scenes, mc, work, sp, hr_insdet)
    q_all = np.ascontiguousarray(np.concatenate(per_scene))
    y_e, s_e = DE.decide(*IX.ExactIndex(g.emb, rows).search(q_all), label_ids, deleted=deleted)
    q_exact = np.quantile(s_e[is_distr], GRID_QUANTILES)
    say(f"галерея N = {len(rows)}, k = {k}; запросов {len(q_all)} на {len(scenes)} сценах, оракульных {is_oracle.sum()}")

    default_threads = faiss.omp_get_max_threads()
    q_diff = np.zeros(len(GRID_QUANTILES))

    def build(threads: int) -> dict:
        nonlocal q_diff
        faiss.omp_set_num_threads(threads)
        h = IX.HnswIndex(g.emb, rows, cfg.hnsw["M"], cfg.hnsw["efConstruction"], cfg.hnsw["efSearch_min"])
        faiss.omp_set_num_threads(default_threads)  # поиск — как в первом замере
        y_h, s_h = DE.decide(*h.search(q_all, k), label_ids, deleted=deleted)
        ok = (y_e == y_h) & (np.abs(s_e - s_h) <= S_STAR_TOL)
        bad = ~ok
        q_diff = np.maximum(q_diff, np.abs(np.quantile(s_h[is_distr], GRID_QUANTILES) - q_exact))
        return {"recall": float(ok.mean()), "same_y": float((y_e == y_h).mean()), "n_mismatched": int(bad.sum()),
                "n_oracle_mismatched": int((bad & is_oracle).sum()),
                "s_star_exact_max": float(s_e[bad].max()) if bad.any() else None,
                "s_star_exact_max_oracle": float(s_e[bad & is_oracle].max()) if (bad & is_oracle).any() else None,
                "s_star_lost_max": float((s_e - s_h)[bad].max()) if bad.any() else None}

    single = build(1)
    multi = [build(default_threads) for _ in range(GRID_BUILDS_MULTI)]
    vals = [b["recall"] for b in multi]
    worst = lambda key: max((b[key] for b in multi if b[key] is not None), default=None)  # noqa: E731
    passed = single["recall"] >= RECALL_MIN
    rec = {"run_id": f"hnsw_{cfg.run_id}", "kind": "hnsw_check",
           "written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
           **{**env.code_stamp(), "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"]},
           "config": cfg.to_dict(),
           "protocol": {"criterion_recall_min": RECALL_MIN, "s_star_tol": S_STAR_TOL, "on_fail": ON_FAIL,
                        "verdict_by": "однопоточное (детерминированное) построение графа",
                        "queries": "все маски M(I) 40 калибровочных сцен из рабочих файлов прогона, fp16 → fp32; тестовые "
                                   "сцены в замер не читаются", "tau": None, "latency": "не повторяется (измерена первым замером, на галерее C(0,1.0) DINOv2)"},
           "versions": {"faiss": faiss.__version__, "numpy": np.__version__},
           "gallery": {"path": str(g.path), "N": len(rows), "n_labels": len(g.labels), "n_max": g.n_max(rows), "k": k,
                       "emb_sha256": g.passport.get("emb_sha256")},
           "index": {"hnsw": {**cfg.hnsw, "efSearch": max(k, cfg.hnsw["efSearch_min"]), "metric": "METRIC_INNER_PRODUCT",
                              "add": "один add всех эталонов в порядке insert_no"}},
           "queries": {"split": QUERY_SPLIT, "n_scenes": len(scenes), "n": int(len(q_all)), "n_oracle": int(is_oracle.sum()),
                       "n_distractors": int(is_distr.sum()), "source": str(work.dir), "mask_cache": mc.dir.name},
           "grid_check": {
               "single_thread": single,
               "multi_thread": {"n_builds": GRID_BUILDS_MULTI, "build_threads": default_threads, "recall": vals,
                                "min": min(vals), "median": float(np.median(vals)), "max": max(vals),
                                "n_builds_passing": int(sum(v >= RECALL_MIN for v in vals)),
                                "n_mismatched_per_build": [b["n_mismatched"] for b in multi],
                                "n_oracle_mismatched_per_build": [b["n_oracle_mismatched"] for b in multi],
                                "s_star_exact_max": worst("s_star_exact_max"), "s_star_lost_max": worst("s_star_lost_max")},
               "distractor_s_star_quantiles": {"q": list(GRID_QUANTILES), "exact": [float(v) for v in q_exact]},
               "distractor_s_star_quantiles_max_abs_diff_vs_exact": [float(v) for v in q_diff]},
           "passed": bool(passed)}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(out)
    say(f"полнота, однопоточное построение: {single['recall']:.5f} при критерии {RECALL_MIN} — "
        f"{'ВЫДЕРЖАН' if passed else 'НЕ ВЫДЕРЖАН: ' + ON_FAIL}; многопоточные: {min(vals):.5f}–{max(vals):.5f}")
    say(f"ГОТОВО: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("stage", choices=["encode", "measure", "diagnose", "grid"])
    ap.add_argument("--out-dir", type=Path, default=RUNS, help="grid: каталог записи (по умолчанию — журнал прогонов)")
    ap.add_argument("--overwrite", action="store_true", help="measure: заменить существующую запись замера")
    args = ap.parse_args()
    global OVERWRITE
    OVERWRITE = args.overwrite
    cfg = CFG.load(args.config)
    if args.stage == "grid":
        stage_grid(cfg, args.out_dir)
        return
    {"encode": stage_encode, "measure": stage_measure, "diagnose": stage_diagnose}[args.stage](cfg)


if __name__ == "__main__":
    main()
