"""Эксперимент 4.4 — отбор гранулярности.

    python scripts/run_4_4.py --config configs/dinov2_c_mean_10_hr_insdet.yaml      # → experiments/runs/4_4_<run_id>.json
    python scripts/run_4_4.py --config configs/dinov3_c_blur_15_hr_insdet.yaml
    python scripts/run_4_4.py --config … --status

Только HR-InsDet и только лучшие $\\varphi$ обоих энкодеров (`rules.RHO_RUNS`, сверяются с `rules.best_variant` по
журналу; иным прогонам — отказ). Без GPU и без нового кодирования: эмбеддинги — рабочие файлы прогона сетки
`cache/run_emb/<run_id>/` (все маски $M(I)$ 160 сцен), галерея — каталог прогона, $M^*$ — `cache/rho/`
(`scripts/select_rho.py`), состав `dedup` — `dedup.json` галереи (`scripts/dedup_gallery.py`). Поиск — точный
(`IndexFlatIP`), порог $\\tau$ не применяется, оценка детекции — $s^*$; оценка — тем же кодом, что контрольный прогон
(`protocol.evaluate_baseline` с подмножеством масок сцены). Не прогон сетки: в счёт 22 не входит.

Блоки записи (`rules.RHO_RECORD_BLOCKS`): `baseline_check` — контрольный прогон без отбора, пересчитанный по тем же
файлам с общими повторами бутстрэпа; точечные AP, AP50, AP75 обязаны совпасть с `<run_id>__baseline.json` (допуск
`rules.RHO_BASELINE_TOL`), иначе записи нет; `rho` — детекции только по $M^*$ и описательные величины; `paired_diff` —
главное сравнение (`rules.RHO_MAIN`) и справочные разности; `outcome` — исход по `rules.RHO_OUTCOMES`; `post_search` —
два ориентира отбора после поиска, без вердикта; `dedup` — полная галерея против дедуплицированной при $\\rho$, квантили
$s^*$ дистракторов $M^*$ только калибровочных сцен, задержка поиска и полнота HNSW на составе `dedup` по протоколам
первого замера HNSW и этапа `grid` (`scripts/check_hnsw.py`). Параметры $\\rho$ и $\\eta$ ни при каком исходе не меняются.

Счёт прерываемый: результат каждого блока копится в `cache/exp_4_4/<run_id>/` и при повторном запуске
пропускается, если посчитан той же версией кода; запись собирается из блоков одной версии кода при чистом дереве.
Существующая запись не перезаписывается. Вывод — ещё и в `logs/4_4_<run_id>.log`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as CFG  # noqa: E402
from src import env  # noqa: E402
from src.eval import granularity as GR  # noqa: E402
from src.eval import protocol as PR  # noqa: E402
from src.eval import rules as RU  # noqa: E402
from src.gallery import dedup as DD  # noqa: E402
from src.gallery import store  # noqa: E402
from src.select import post_search as PS  # noqa: E402
from src.select import rho as RHO  # noqa: E402

RUNS = Path("experiments/runs")
WORK = Path("cache/exp_4_4")
DESCRIBE_SUBSETS = RU.RHO_DESCRIBE_SUBSETS
QUANTILES = RU.RHO_DEDUP_QUANTILES
_STAMP = env.code_stamp()


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record_path(run_id: str) -> Path:
    return RUNS / f"4_4_{run_id}.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Blocks:
    """Рабочие файлы блоков: посчитанное той же версией кода при чистом дереве и по тем же входам (`inputs` — хеш
    файлов эмбеддингов, галереи, состава `dedup` и записей отбора) пропускается; иначе блок считается заново."""

    def __init__(self, run_id: str, inputs: str, say):
        self.dir, self.inputs, self.say = WORK / run_id, inputs, say
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, name: str, fn):
        p = self.dir / f"{name}.pkl"
        if p.is_file():
            rec = pickle.loads(p.read_bytes())
            if rec["code_commit"] == _STAMP["code_commit"] and not rec["code_dirty"] and not _STAMP["code_dirty"] \
                    and rec.get("inputs") == self.inputs:
                self.say(f"блок {name}: посчитан ранее этой версией кода ({rec['sec']} с) — пропуск")
                return rec["value"], rec["sec"]
        t0 = time.perf_counter()
        value = fn()
        sec = round(time.perf_counter() - t0, 1)
        tmp = p.with_suffix(".pkl.tmp")
        tmp.write_bytes(pickle.dumps({"code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"],
                                      "inputs": self.inputs, "sec": sec, "value": value}))
        tmp.replace(p)
        self.say(f"блок {name}: {sec} с")
        return value, sec


def _point(ev: dict, subset: str) -> dict:
    return ev["subsets"][subset]["by_area"]["all"]


def _strip(ev: dict) -> dict:
    """Оценка без массивов повторов — то, что идёт в запись."""
    return {k: v for k, v in ev.items() if k != "boot"}


def _diffs(a: dict, b: dict) -> dict:
    """Парные разности «a минус b»: с интервалом — на `RHO_DIFF_SUBSETS`, точечные — на easy / hard."""
    out = {}
    for n in RU.RHO_DIFF_SUBSETS:
        out[n] = {m: RU.paired_diff(a["boot"][n][m], b["boot"][n][m], _point(a, n)[m], _point(b, n)[m])
                  for m in RU.RHO_DIFF_METRICS}
    for n in ("test/easy", "test/hard"):
        out[n] = {m: {"diff": _point(a, n)[m] - _point(b, n)[m]} for m in RU.RHO_DIFF_METRICS}
    return out


def _brief(ev: dict) -> dict:
    return {n: {m: _point(ev, n)[m] for m in ("ap", "ap50", "ap75")} | {"ci95": _point(ev, n).get("ci95")}
            for n in ev["subsets"]}


def run(cfg, status: bool) -> None:
    R43 = _script("run_4_3")
    say = R43._logger([f"4_4_{cfg.run_id}"])
    out = record_path(cfg.run_id)
    if cfg.dataset != "hr_insdet" or cfg.run_id not in RU.RHO_RUNS or cfg.variant != R43.best_of_encoder(cfg.encoder):
        raise SystemExit(f"4.4 — только лучшие φ HR-InsDet {RU.RHO_RUNS} (сверка с журналом); запрошен {cfg.run_id}")
    if status:
        done = sorted(p.stem for p in (WORK / cfg.run_id).glob("*.pkl")) if (WORK / cfg.run_id).is_dir() else []
        say(f"запись: {'есть' if out.exists() else 'нет'} ({out}); блоки в рабочих файлах: {done}")
        return
    if out.exists():
        raise SystemExit(f"запись уже есть и не перезаписывается: {out}")
    # предусловия 4.4: сверка ρ с аналитикой пройдена поштучно, порог замера кодирования выдержан
    pre = {}
    for path in RU.RHO_PRECONDITIONS:
        if not Path(path).is_file() or json.loads(Path(path).read_text()).get("passed") is not True:
            raise SystemExit(f"4.4 не начинается: нет записи {path} либо в ней не `passed: true`")
        pre[path] = {"passed": True, "sha256": _sha256(path)}
    say(f"версия кода {_STAMP['code_commit'][:7]}{' (есть незакоммиченные правки)' if _STAMP['code_dirty'] else ''}")
    journal = json.loads(R43.record_path(cfg.run_id, baseline=True).read_text())

    scenes, labels, edges, sp, emb, g, gt, st, scenes_cache, refs_cache, _ = R43._load_run(cfg, say)
    if set(cfg.gallery_protocols) != set(RU.RHO_GALLERIES):
        raise SystemExit(f"протоколы галереи конфигурации {cfg.gallery_protocols}, ожидаются {RU.RHO_GALLERIES}")
    emb_files = {s.id: _sha256(st.path(s.id)) for s in scenes}
    gallery_sha = _sha256(g.path / "emb.npy")
    if gallery_sha != g.passport["emb_sha256"] or gallery_sha != journal["gallery"]["emb_sha256"]:
        raise SystemExit("галерея не та, что у контрольного прогона журнала")

    rc = RHO.RhoCache(scenes_cache)
    rho_recs = [rc.load(s.id) for s in scenes]
    if any(r["n_masks"] != len(s.boxes) for r, s in zip(rho_recs, scenes)):
        raise SystemExit("число масок в записях отбора расходится с кешем масок")
    m_star = [np.asarray(r["selected"], np.int64) for r in rho_recs]
    sub = PR.subsets(scenes)
    inputs = hashlib.sha256(json.dumps(
        {"emb": emb_files, "gallery": gallery_sha, "dedup": DD.file_sha256(g), "rho_key": rc.key,
         "rho": {r["image_id"]: [r["masks_sha256"], r["selected"]] for r in rho_recs},
         "bootstrap": [cfg.bootstrap, cfg.seed, cfg.max_dets]}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    blocks = Blocks(cfg.run_id, inputs, say)
    kw = dict(max_det=cfg.max_dets, n_boot=cfg.bootstrap, seed=cfg.seed, boot_subsets=RU.RHO_DIFF_SUBSETS)
    evaluate = lambda y, s, keep=None, with_ar=True: PR.evaluate_baseline(  # noqa: E731
        scenes, labels, edges, gt, g.labels, y, s, keep_masks=keep, with_ar=with_ar, **kw)

    found, search_info = {}, {}
    for proto in (*RU.RHO_GALLERIES, "dedup"):
        y, s, search_info[proto] = PR.search(emb, g, proto)
        found[proto] = (y, s)
        say(f"поиск, галерея «{proto}»: N = {search_info[proto]['N']}")

    # --- 1. контрольный прогон без отбора, пересчитанный с общими повторами; сверка с журналом
    base, sec_base = blocks.get("baseline_check", lambda: {p: evaluate(*found[p]) for p in RU.RHO_GALLERIES})
    check = {}
    for proto in RU.RHO_GALLERIES:
        for n in PR.BASELINE_SUBSETS:
            ours, theirs = _point(base[proto], n), journal["metrics"][proto]["subsets"][n]["by_area"]["all"]
            check[f"{proto}:{n}"] = max(abs(ours[m] - theirs[m]) for m in ("ap", "ap50", "ap75"))
    worst = max(check.values())
    say(f"контрольный прогон против журнала: наибольшее расхождение AP / AP50 / AP75 — {worst:.2e} (допуск {RU.RHO_BASELINE_TOL:g})")
    if worst > RU.RHO_BASELINE_TOL or base["full"]["n_detections"] != journal["metrics"]["full"]["n_detections"]:
        raise SystemExit("пересчитанный контрольный прогон расходится с записью журнала — запись 4.4 не создаётся")

    # --- 2. детекции только по M*
    rho_ev, sec_rho = blocks.get("rho", lambda: {p: evaluate(*found[p], keep=m_star) for p in RU.RHO_GALLERIES})
    y_full = found["full"][0]
    describe = {}
    for n in DESCRIBE_SUBSETS:
        sel = sub[n]
        describe[n] = {
            "n_scenes": int(len(sel)), "n_masks": int(sum(len(scenes[k].boxes) for k in sel)),
            "n_selected": int(sum(len(m_star[k]) for k in sel)),
            "rho_sec": round(float(sum(rho_recs[k]["sec"] for k in sel)), 3),
            "recall": {"M(I)": GR.recall_after_selection(scenes, None, sel), "M*": GR.recall_after_selection(scenes, m_star, sel)},
            "multiple_detections": {"M(I)": GR.multiplicity(scenes, None, y_full, sel),
                                    "M*": GR.multiplicity(scenes, m_star, y_full, sel)}}
        describe[n]["share_not_encoded"] = 1 - describe[n]["n_selected"] / describe[n]["n_masks"]

    # --- 3, 4. главное сравнение и исход
    paired = {p: _diffs(rho_ev[p], base[p]) for p in RU.RHO_GALLERIES}
    main = paired[RU.RHO_MAIN["gallery"]][RU.RHO_MAIN["subset"]][RU.RHO_MAIN["metric"]]
    outcome = {**RU.rho_outcome(main["ci95"]), "diff": main["diff"], "main": RU.RHO_MAIN,
               "share_not_encoded_test": describe["test/all"]["share_not_encoded"],
               "note": "параметры ρ и η по исходу не меняются"}
    say(f"ГЛАВНОЕ СРАВНЕНИЕ, test/all, полная галерея: AP с ρ {100 * _point(rho_ev['full'], 'test/all')['ap']:.2f}, без отбора "
        f"{100 * _point(base['full'], 'test/all')['ap']:.2f}; разность {100 * main['diff']:+.2f} п., 95 % интервал "
        f"[{100 * main['ci95'][0]:+.2f}; {100 * main['ci95'][1]:+.2f}] — исход ({outcome['code']}): {outcome['text']}")

    # --- 5. ориентиры отбора после поиска — по тем же эмбеддингам всех масок M(I), полная галерея
    s_full = found[RU.POST_SEARCH_GALLERY][1]

    def keep_post(rule: str) -> list[np.ndarray]:
        keep = []
        for k, s in enumerate(scenes):
            if rule == "chain_max":
                area, inter = RHO.geometry([r["rle"] for r in scenes_cache.load(s.id).records])
                keep.append(PS.chain_max(RHO.nesting(area, inter)[0], s_full[k]))
            else:
                keep.append(PS.box_nms(s.boxes, s_full[k], RU.POST_NMS_IOU))
        return keep

    def post_block(rule: str):
        keep = keep_post(rule)
        return keep, evaluate(*found[RU.POST_SEARCH_GALLERY], keep=keep, with_ar=False)

    post = {}
    for rule in RU.POST_SEARCH:
        (keep, ev), sec = blocks.get(f"post_search_{rule}", lambda rule=rule: post_block(rule))
        post[rule] = {"gallery": RU.POST_SEARCH_GALLERY, "n_detections": ev["n_detections"], "ap": _brief(ev),
                      "n_kept": {n: int(sum(len(keep[k]) for k in sub[n])) for n in DESCRIBE_SUBSETS},
                      "multiple_detections": {n: GR.multiplicity(scenes, keep, y_full, sub[n]) for n in DESCRIBE_SUBSETS},
                      "paired_diff_minus_rho": _diffs(ev, rho_ev[RU.POST_SEARCH_GALLERY]),
                      "paired_diff_note": "ориентир минус ρ; интервал — описательно, вердикта нет", "sec": sec}
        say(f"ориентир {rule}: AP test/all {100 * _point(ev, 'test/all')['ap']:.2f}, детекций {ev['n_detections']}")
    post["params"] = {"theta": str(RHO.THETA), "post_nms_iou": RU.POST_NMS_IOU,
                      "note": "константы без подбора; метки ŷ в правилах не участвуют; по маскам M(I) после штатного NMS SAM 2"}

    # --- 6. избыточность галереи при ρ: строки dedup против full
    dd_ev, sec_dd = blocks.get("dedup_eval", lambda: evaluate(*found["dedup"], keep=m_star, with_ar=False))
    rows = {p: g.rows(p) for p in ("full", "dedup")}
    cal = sub["cal/all"]
    d_cal = {p: np.concatenate([found[p][1][k][m_star[k]][scenes[k].distractor[m_star[k]]] for k in cal]) for p in rows}
    hnsw, sec_hnsw = blocks.get("dedup_hnsw", lambda: _hnsw(cfg, g, rows, [emb[k][m_star[k]] for k in cal], say))
    dedup = {"what": "те же детекции по M* при строках галереи dedup против full; вердикта нет",
             "dedup_json_sha256": DD.file_sha256(g), "eta": DD.eta_for(g.passport), "rule": DD.DEDUP_RULE,
             "gallery": {p: {"N": int(len(rows[p])), "n_max": g.n_max(rows[p]), "k": g.k(rows[p]),
                             "rows_sha256": store.rows_sha256(g, rows[p])} for p in rows},
             "ap": {"full": _brief(rho_ev["full"]), "dedup": _brief(dd_ev)},
             "n_detections": dd_ev["n_detections"],
             "paired_diff_dedup_minus_full": _diffs(dd_ev, rho_ev["full"]),
             "distractor_s_star_quantiles_cal": {
                 "what": "дистракторы D_cal из M* 40 калибровочных сцен; тестовые сцены в квантили не входят",
                 "n": int(len(d_cal["full"])), "q": list(QUANTILES),
                 **{p: [float(v) for v in np.quantile(d_cal[p].astype(np.float64), QUANTILES)] for p in rows}},  # как `tau_q`
             **hnsw}
    say(f"dedup: N {len(rows['full'])} → {len(rows['dedup'])}; AP test/all {100 * _point(rho_ev['full'], 'test/all')['ap']:.2f} → "
        f"{100 * _point(dd_ev, 'test/all')['ap']:.2f}; полнота HNSW на dedup (однопоточное построение) — "
        f"{hnsw['hnsw_recall_dedup']['single_thread']['recall']:.5f}")

    if _STAMP["code_dirty"] or env.code_stamp()["code_dirty"]:
        raise SystemExit("есть незакоммиченные правки src или scripts — запись 4.4 не создаётся (блоки сохранены)")
    rec = R43._common(cfg, "baseline", sp, g, st, scenes_cache, refs_cache, scenes)
    rec.pop("_segmentation_sec")
    rec["rules"]["area_edges_px2"] = edges[:2]
    rec = {"run_id": f"4_4_{cfg.run_id}", "experiment": "4.4", "kind": "exp_4_4", "grid_run": cfg.run_id,
           "baseline_record": str(R43.record_path(cfg.run_id, baseline=True)), **rec,
           "what": "отбор гранулярности ρ до поиска против контрольного прогона без отбора по сохранённым эмбеддингам всех "
                   "масок M(I) прогона сетки; без порога τ, оценка детекции — s*; не прогон сетки",
           "metrics_note": f"доли (0–1); all — 160 сцен, test — 120 тестовых; интервалы 95 % — бутстрэп по сценам, "
                           f"{cfg.bootstrap} повторов, общих у всех блоков; AP — при maxDets={cfg.max_dets}",
           "preconditions": pre,
           "inputs": {"blocks_inputs_sha256": inputs, "run_emb_files_sha256": emb_files, "gallery_emb_sha256": gallery_sha,
                      "rho_cache": str(rc.dir), "rho": {"theta": str(RHO.THETA), "gamma": str(RHO.GAMMA), "rule": RHO.RHO_RULE},
                      "embeddings_note": "эмбеддинг маски из M* — сохранённый эмбеддинг той же маски, посчитанный при "
                                         "кодировании всех M(I); отличие от кодирования одних M* — experiments/rho_encode_check.json"},
           "search": search_info, "timing_sec": {"baseline_check": sec_base, "rho": sec_rho, "dedup_eval": sec_dd,
                                                 "dedup_hnsw": sec_hnsw},
           "baseline_check": {"max_abs_diff_vs_journal": check, "tol": RU.RHO_BASELINE_TOL, "passed": True,
                              "metrics": {p: _strip(base[p]) for p in RU.RHO_GALLERIES}},
           "rho": {"metrics": {p: _strip(rho_ev[p]) for p in RU.RHO_GALLERIES}, "by_subset": describe,
                   "multiple_detections_definition": "основное — доля размеченных рамок с двумя и больше масками при IoU рамок "
                                                     "≥ 0,5 (§2.3, без меток); с учётом ŷ — справочно"},
           "paired_diff": {"what": "ρ минус без отбора, общие повторы бутстрэпа", "main": RU.RHO_MAIN, **paired},
           "outcome": outcome, "post_search": post, "dedup": dedup}
    missing = [b for b in RU.RHO_RECORD_BLOCKS if b not in rec]
    if missing:
        raise AssertionError(f"в записи нет блоков {missing}")
    R43._write(out, rec)
    say(f"ГОТОВО: {out}")


def _hnsw(cfg, g, rows: dict, per_scene: list[np.ndarray], say) -> dict:
    """Задержка поиска при полной и дедуплицированной галерее (протокол первого замера HNSW) и полнота HNSW на составе
    `dedup` (протокол этапа `grid`): запросы — маски $M^*$ 40 калибровочных сцен."""
    import faiss

    from src.search import decide as DE
    from src.search import index as IX

    H = _script("check_hnsw")
    per_scene = [np.ascontiguousarray(z) for z in per_scene if len(z)]
    q_all = np.ascontiguousarray(np.concatenate(per_scene))
    single = [q_all[i:i + 1] for i in range(len(q_all))]
    default_threads = faiss.omp_get_max_threads()
    label_ids, deleted = g.label_ids, g.deleted
    latency = {}
    for proto, r in rows.items():
        k = g.k(r)
        exact = IX.ExactIndex(g.emb, r)
        faiss.omp_set_num_threads(default_threads)  # граф для замера задержки — как в первом замере HNSW: потоков по умолчанию
        hn = IX.HnswIndex(g.emb, r, cfg.hnsw["M"], cfg.hnsw["efConstruction"], cfg.hnsw["efSearch_min"])
        hn.prepare(k)
        searches = {f"flat_k{len(r)}": lambda q, e=exact, n=len(r): e.index.search(q, n),
                    f"hnsw_k{k}": lambda q, h=hn, k=k: h.index.search(q, k)}
        latency[proto] = {"N": int(len(r)), "k": k, "hnsw_build_threads": default_threads}
        for th in H.THREADS:
            faiss.omp_set_num_threads(default_threads if th == "default" else th)
            latency[proto][f"threads_{th}"] = {
                "faiss_omp_threads": faiss.omp_get_max_threads(),
                **{name: {"single_query": H._latency(fn, single), "batch_per_scene": H._latency(fn, per_scene)}
                   for name, fn in searches.items()}}
        faiss.omp_set_num_threads(default_threads)
        say(f"задержка, галерея «{proto}» (N = {len(r)}, k = {k}): " + "; ".join(
            f"{n}: {v['single_query']['ms_per_call_median']:.3f} мс на запрос"
            for n, v in latency[proto]["threads_default"].items() if isinstance(v, dict)))

    r = rows["dedup"]
    k = g.k(r)
    y_e, s_e = DE.decide(*IX.ExactIndex(g.emb, r).search(q_all), label_ids, deleted=deleted)

    def build(threads: int) -> dict:
        faiss.omp_set_num_threads(threads)
        h = IX.HnswIndex(g.emb, r, cfg.hnsw["M"], cfg.hnsw["efConstruction"], cfg.hnsw["efSearch_min"])
        faiss.omp_set_num_threads(default_threads)
        y_h, s_h = DE.decide(*h.search(q_all, k), label_ids, deleted=deleted)
        ok = (y_e == y_h) & (np.abs(s_e - s_h) <= H.S_STAR_TOL)
        return {"recall": float(ok.mean()), "same_y": float((y_e == y_h).mean()), "n_mismatched": int((~ok).sum())}

    one = build(1)
    multi = [build(default_threads) for _ in range(H.GRID_BUILDS_MULTI)]
    vals = [b["recall"] for b in multi]
    return {"search_latency": {"protocol": "как в первом замере HNSW (scripts/check_hnsw.py): таймер только вокруг search, прогрев — "
                                           f"полный проход, медиана по {H.REPEATS} повторам; полная галерея — для сравнения «полная "
                                           "против дедуплицированной»", "queries": "маски M* 40 калибровочных сцен",
                               "n_queries": int(len(q_all)), **latency},
            "hnsw_recall_dedup": {"protocol": "как на галереях сетки (scripts/check_hnsw.py, этап grid); описательно", "k": k,
                                  "criterion_recall_min": H.RECALL_MIN, "s_star_tol": H.S_STAR_TOL,
                                  "n_queries": int(len(q_all)), "single_thread": one,
                                  "single_thread_meets_criterion": bool(one["recall"] >= H.RECALL_MIN),
                                  "multi_thread": {"n_builds": H.GRID_BUILDS_MULTI, "build_threads": default_threads,
                                                   "min": min(vals), "median": float(np.median(vals)), "max": max(vals),
                                                   "n_builds_passing": int(sum(v >= H.RECALL_MIN for v in vals)),
                                                   "n_mismatched_per_build": [b["n_mismatched"] for b in multi]}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    run(CFG.load(args.config), args.status)


if __name__ == "__main__":
    main()
