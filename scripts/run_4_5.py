"""Эксперимент 4.5 — отклонение и калибровка порога при росте галереи, HR-InsDet.

    python scripts/run_4_5.py --config configs/dinov2_c_mean_10_hr_insdet.yaml      # → experiments/runs/4_5_<run_id>.json
    python scripts/run_4_5.py --config configs/dinov3_c_blur_15_hr_insdet.yaml
    python scripts/run_4_5.py --config … --status

Только прогоны `rules.REJ_RUNS` (лучшие $\\varphi$ обоих энкодеров, сверяются с журналом); иным — отказ. Без GPU и без
нового кодирования: эмбеддинги масок — рабочие файлы прогонов сетки, галерея — полная, подвыборки — подмножества её
строк по закоммиченному списку `splits/gallery_subsets_hr_insdet.json` (сверяется с пересчётом по seed и с индексом
git); маски — $M^*$ (`cache/rho/`), порог — по $\\mathcal D_{\\mathrm{cal}}$ из $M^*$ 40 калибровочных сцен. Поиск —
точный (`IndexFlatIP`). Протокол заморожен: константы — `src.eval.rules`, счёт — `src.eval.rejection`.

Предусловия: `calib.json` галереи — с `D_source` = `M_star` при действующих $\\theta$ и $\\gamma$, по составу строк
`full`, сводка калибровки — с пройденным `verify` (`scripts/calibrate.py`); состав `dedup` — `dedup.json` галереи
(`scripts/dedup_gallery.py`). Калибровка при 100 экземплярах обязана совпасть с `calib.json` точно — иначе записи нет.
Блоки записи — `rules.REJ_RECORD_BLOCKS`. Справочные строки AP полного метода (тот же вызов оценки, что в 4.4, с
интервалами) — минуты на строку; их результат копится в `cache/exp_4_5/<run_id>/` и при повторном запуске той же
версией кода не пересчитывается. Запись создаётся только при чистом дереве и не перезаписывается. Не прогон сетки.
Вывод — ещё и в `logs/4_5_<run_id>.log`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as CFG  # noqa: E402
from src import env  # noqa: E402
from src.calib import threshold as TH  # noqa: E402
from src.eval import oracle as OR  # noqa: E402
from src.eval import protocol as PR  # noqa: E402
from src.eval import rejection as RJ  # noqa: E402
from src.eval import rules as RU  # noqa: E402
from src.gallery import dedup as DD  # noqa: E402
from src.gallery import store  # noqa: E402
from src.gallery import subsets as SB  # noqa: E402
from src.search import index as IX  # noqa: E402
from src.select import rho as RHO  # noqa: E402

RUNS = Path("experiments/runs")
WORK = Path("cache/exp_4_5")
_STAMP = env.code_stamp()
EK = store.eps_key


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record_path(run_id: str) -> Path:
    return RUNS / f"4_5_{run_id}.json"


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _subsets_committed(dataset: str) -> dict:
    """Списки подвыборок — закоммиченный файл без правок (коммит до первого запуска) и ровно пересчёт по seed."""
    p = SB.path(dataset)
    rec = SB.load(dataset)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(p)], capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        raise SystemExit(f"{p}: списки подвыборок обязаны быть закоммичены до прогона и не иметь правок")
    commit = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", str(p)], capture_output=True, text=True).stdout.strip()
    return {"rec": rec, "info": {"file": str(p), "sha256": _sha256(p), "last_commit": commit}}


class Blocks:
    """Рабочие файлы долгих блоков (как в `run_4_4.py`): посчитанное той же версией кода при чистом дереве и по тем же
    входам пропускается."""

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


# ---------------------------------------------------------------------- общее: ячейки по цепочкам


def _cells(imgs, S_cal, g, sub, in_subset_of, eps_levels, delta, W, say, groups: dict | None = None) -> dict:
    """Калибровки подвыборок, ячейки «цепочка × (n0 → n) × ε × правило», проверка допущений и условие перекалибровки.

    `in_subset_of(chain, n)` — (|Y|,) bool; `W` — веса бутстрэпа по снимкам `imgs` либо `None` (без интервалов);
    `groups` — имя → номера снимков: величины приводятся ещё и по этим группам."""
    rec, label_ids = sub["rec"], g.label_ids
    sizes = tuple(rec["sizes"])
    if sizes != RU.GALLERY_SUBSET_SIZES or rec["n_chains"] != RU.GALLERY_CHAINS:
        raise SystemExit("списки подвыборок расходятся с константами протокола")
    pairs = [(n0, n) for n0 in sizes for n in sizes if n >= n0]
    if [p for p in pairs if p[0] != p[1]] != list(RU.GROWTH_PAIRS):
        raise AssertionError("пары роста расходятся с rules.GROWTH_PAIRS")
    cache: dict = {}

    def at(chain, n):
        rows = SB.rows(g, rec, chain, n)
        key = hashlib.sha1(rows.tobytes()).hexdigest()
        if key not in cache:  # 100 экземпляров — вся галерея: одна у всех цепочек
            cache[key] = {"rows": rows, "rows_sha256": store.rows_sha256(g, rows), "cal": RJ.calibrate_rows(S_cal, rows, eps_levels, delta),
                          "ans": [RJ.answers(im.S, rows, label_ids) for im in imgs], "memo": {}}
        return cache[key]

    def meas(node, in_sub, tau=None, top_k=None):
        k = (in_sub.tobytes(), tau, top_k)
        if k not in node["memo"]:
            m = RJ.measure(imgs, node["ans"], in_sub, tau, top_k, W)
            if groups:
                m["by_group"] = {name: {kk: v for kk, v in RJ.measure(imgs, node["ans"], in_sub, tau, top_k, None, sel).items()}
                                 for name, sel in groups.items()}
            node["memo"][k] = m
        return dict(node["memo"][k])

    cells, calibs, assumptions, recal, auroc = {}, {}, {}, {}, {}
    holds = {EK(e): {r: {} for r in RU.REJ_RULES} for e in eps_levels}
    for chain in rec["chains"]:
        r = chain["r"]
        nodes = {n: at(chain, n) for n in sizes}
        in_sub = {n: in_subset_of(chain, n) for n in sizes}
        calibs[r] = {n: {"gallery_rows": SB.name(r, n), "gallery_rows_sha256": nodes[n]["rows_sha256"], "N0": nodes[n]["cal"]["N0"],
                         "F": RJ.tail_summary(nodes[n]["cal"]["tail"]),
                         "by_eps": nodes[n]["cal"]["by_eps"]} for n in sizes}
        auroc[r] = {n: {"all": RJ.auroc_known_unknown(imgs, nodes[n]["ans"], in_sub[n]),
                        **({name: RJ.auroc_known_unknown(imgs, nodes[n]["ans"], in_sub[n], sel) for name, sel in groups.items()}
                           if groups else {})} for n in sizes}
        cells[r], assumptions[r], recal[r] = {}, {}, {}
        for n0, n in pairs:
            pk = RJ.pair_key(n0, n)
            c0, cn = nodes[n0]["cal"], nodes[n]["cal"]
            N0, N = c0["N0"], cn["N0"]
            cells[r][pk] = {}
            for eps in eps_levels:
                e = EK(eps)
                taus = {"quantile_recal": cn["by_eps"][e]["tau_q"], "model": RJ.model_tau(c0, eps, N),
                        "quantile_frozen": c0["by_eps"][e]["tau_q"], "fixed": RU.FIXED_TAU}
                out = {}
                for rule in RU.REJ_RULES:
                    if rule == "top_k":
                        m = meas(nodes[n], in_sub[n], top_k=RU.TOP_K)
                        m0 = meas(nodes[n0], in_sub[n0], top_k=RU.TOP_K)
                    elif taus[rule] is None:  # κ вне отрезка: правило по модели в ячейке считается не удержавшим уровень
                        out[rule] = {"tau": None, "kappa_out_of_range": True, "eps_holds": False if W is not None else None}
                        if n != n0 and W is not None:
                            holds[e][rule].setdefault(r, {})[(n0, n)] = False
                        continue
                    else:
                        m = meas(nodes[n], in_sub[n], tau=float(taus[rule]))
                        # FPR при N0 — того порога, который правило даёт при самой галерее G_n0
                        t0 = {"quantile_recal": c0["by_eps"][e]["tau_q"], "model": RJ.model_tau(c0, eps, N0),
                              "quantile_frozen": c0["by_eps"][e]["tau_q"], "fixed": RU.FIXED_TAU}[rule]
                        m0 = meas(nodes[n0], in_sub[n0], tau=float(t0))
                        m["tau"] = float(taus[rule])
                    m["fpr_at_n0"] = m0["fpr"]
                    m["fpr_ratio_to_n0"] = (m["fpr"] / m0["fpr"]) if m0["fpr"] else None
                    m["predicted_ratio"] = RJ.predicted_ratio_frozen(eps, N, N0) if rule == "quantile_frozen" else \
                        (1.0 if rule in ("quantile_recal", "model") else None)
                    if W is not None:
                        m["eps_holds"] = RU.eps_holds(m["fpr_ci95"][0], eps, delta)
                        if n != n0:
                            holds[e][rule].setdefault(r, {})[(n0, n)] = m["eps_holds"]
                    out[rule] = m
                cells[r][pk][e] = out
            if n != n0:
                levels = {f"tau_m_n_eps_{EK(eps)}": RJ.model_tau(c0, eps, N) for eps in eps_levels}
                levels.update({f"F_n0_q{q:g}": float(np.interp(q, c0["tail"].levels, c0["tail"].q)) for q in RU.REJ_TAIL_QUANTILES})
                assumptions[r][pk] = RJ.new_exemplar_tail(S_cal, nodes[n0]["rows"], nodes[n]["rows"], levels)
                recal[r][pk] = {"N0": N0, "N": N, "growth_requires_recalibration": TH.growth_requires_recalibration(N, N0, delta),
                                **{EK(eps): TH.control_check(cn["s_star_cal"], c0["by_eps"][EK(eps)]["tau_q"], eps, delta)
                                   for eps in eps_levels}}
        say(f"цепочка {r}: κN0 при ε = {EK(eps_levels[0])} — " + ", ".join(
            f"{n}: {calibs[r][n]['by_eps'][EK(eps_levels[0])]['kappa_N0']}" for n in sizes))
    kappa = {r: {n: {e: {"kappa": v["kappa"], "kappa_N0": v["kappa_N0"], "kappa_out_of_range": v["kappa_out_of_range"]}
                     for e, v in calibs[r][n]["by_eps"].items()} for n in sizes} for r in calibs}
    return {"cells": cells, "calibrations": calibs, "auroc": auroc, "holds": holds, "nodes_full": at(rec["chains"][0], sizes[-1]),
            "assumptions": {"what": "описательно: κ по объёмам и ε; хвост пар «дистрактор — новый эталон» против F_n0",
                            "kappa": kappa, "new_exemplar_tail": assumptions},
            "recalibration": {"what": "описательно: контрольная проверка порога τ_q(n0) по всей D_cal при галерее G_n и условие "
                                      "N/N0 > 1 + δ", "by_chain": recal}}


def _common_record(cfg, R43, sp, g, st, scenes_cache, refs_cache, scenes, edges) -> dict:
    rec = R43._common(cfg, "baseline", sp, g, st, scenes_cache, refs_cache, scenes)
    rec.pop("_segmentation_sec")
    rec["rules"]["area_edges_px2"] = edges[:2]
    rec["rules"]["search"] = "IndexFlatIP, k = N по полной галерее; подвыборка — подмножество строк"
    return rec


def _protocol_block() -> dict:
    return {"gallery_chains": RU.GALLERY_CHAINS, "chain_majority": RU.CHAIN_MAJORITY, "subset_sizes": list(RU.GALLERY_SUBSET_SIZES),
            "growth_pairs": [list(p) for p in RU.GROWTH_PAIRS], "rules": list(RU.REJ_RULES), "verdict_rules": list(RU.REJ_VERDICT_RULES),
            "top_k": RU.TOP_K, "fixed_tau": RU.FIXED_TAU, "eps_holds_rule": RU.EPS_HOLDS_RULE,
            "distractor_iou_max": OR.DISTRACTOR_IOU_MAX, "oracle_iou_min": OR.ORACLE_IOU,
            "frozen": "протокол заморожен с первого запуска run_4_4.py"}


# ---------------------------------------------------------------------- HR-InsDet


def run_hr(cfg, R43, say, out: Path) -> None:
    if cfg.variant != R43.best_of_encoder(cfg.encoder):
        raise SystemExit(f"4.5 на HR-InsDet — только лучшие φ {RU.REJ_RUNS['hr_insdet']} (сверка с журналом)")
    if cfg.bootstrap != RU.REJ_BOOTSTRAP:
        raise SystemExit(f"повторов бутстрэпа в конфигурации {cfg.bootstrap}, по протоколу {RU.REJ_BOOTSTRAP}")
    sub = _subsets_committed(cfg.dataset)
    CAL = _script("calibrate")
    scenes, labels, edges, sp, emb, g, gt, st, scenes_cache, refs_cache, _ = R43._load_run(cfg, say)
    if labels != g.labels:
        raise SystemExit("номера меток галереи расходятся с категориями истины")

    # --- предусловия: пересчитанная калибровка с ρ
    eps_levels = tuple(cfg.eps)
    stored = store.read_calib(g.path, eps_levels, g, "full")  # состав строк сверяется здесь
    src = stored.get("D_source", {})
    src_rho = src.get("rho") or {}
    if src.get("masks") != "M_star" or src_rho.get("theta") != str(RHO.THETA) or src_rho.get("gamma") != str(RHO.GAMMA):
        raise SystemExit("calib.json поставлен не по M* с действующими θ и γ — сначала пересчёт калибровки "
                         "(scripts/calibrate.py … run --supersede)")
    summ = json.loads(CAL.summary_path(cfg).read_text())
    if summ.get("verify", {}).get("passed") is not True or summ["calib_json_sha256"] != _sha256(g.path / "calib.json"):
        raise SystemExit("сводка калибровки без пройденного verify либо не от этого calib.json")

    # --- D_cal: тот же отбор и тот же поиск одним батчем, что в calibrate.py
    cal_scenes, work, _, rows_full, source, m_star_cal = CAL._setup(cfg)
    z_cal, ids_cal = CAL._d_cal(cal_scenes, work, m_star_cal)
    exact = IX.ExactIndex(g.emb, rows_full)
    S_cal = RJ.dense(*exact.search(z_cal), len(g))
    if ids_cal != stored["check_set_ids"]:
        raise SystemExit("D_cal расходится с check_set_ids записи калибровки")

    # --- снимки оценки: 120 тестовых сцен, маски M*
    rc = RHO.RhoCache(scenes_cache)
    rho_recs = [rc.load(s.id) for s in scenes]
    if any(r["n_masks"] != len(s.boxes) for r, s in zip(rho_recs, scenes)):
        raise SystemExit("число масок в записях отбора расходится с кешем масок")
    m_star = [np.asarray(r["selected"], np.int64) for r in rho_recs]
    test_idx = np.array([k for k, s in enumerate(scenes) if s.split == "test"])
    if [scenes[k].id for k in test_idx] != list(sp["test"]) or len(scenes) != len(sp["cal"]) + len(sp["test"]) or any(scenes[k].split != "cal" for k in range(len(scenes)) if k not in set(test_idx)):
        raise SystemExit("состав сцен расходится с splits: калибровочные, затем тестовые")
    imgs = []
    for k in test_idx:
        s, sel = scenes[k], m_star[k]
        S = RJ.dense(*exact.search(emb[k]), len(g))[sel]  # поиск — по всем маскам сцены, как в прогонах; затем M*
        gl = np.full(len(sel), -1, np.int64)
        o = OR.oracle_assign(s.gt_boxes, s.boxes[sel])  # оракульное назначение среди M*, один раз по всем рамкам сцены
        gl[o[o >= 0]] = s.gt_labels[o >= 0]
        if (s.distractor[sel] & (gl >= 0)).any():
            raise AssertionError(f"{s.id}: маска одновременно оракульная и дистрактор")
        imgs.append(RJ.Img(s.id, s.level, S, sel, s.distractor[sel], gl))
    W = PR._weights(PR.subsets(scenes), ("test/all",), len(scenes), cfg.bootstrap, cfg.seed)["test/all"]
    if W[:, [k for k in range(len(scenes)) if k not in set(test_idx)]].any():
        raise AssertionError("повторы бутстрэпа test/all задевают калибровочные сцены")
    W = W[:, test_idx]
    n_mstar_test = [len(m_star[k]) for k in test_idx]
    say(f"D_cal {len(ids_cal)}; тестовых сцен {len(imgs)}, масок M* {sum(n_mstar_test)}, дистракторов "
        f"{sum(int(i.distractor.sum()) for i in imgs)}, оракульных {sum(int((i.gt_label >= 0).sum()) for i in imgs)}")

    lab_id = {y: i for i, y in enumerate(g.labels)}

    def in_subset_of(chain, n):
        m = np.zeros(len(g.labels), bool)
        m[[lab_id[y] for y in SB.chain_ids(sub["rec"], chain, n)]] = True
        return m

    res = _cells(imgs, S_cal, g, sub, in_subset_of, eps_levels, cfg.delta, W, say)

    # --- сверка: калибровка при 100 экземплярах против calib.json — точно
    full = res["nodes_full"]
    if not np.array_equal(full["rows"], rows_full):
        raise AssertionError("100 экземпляров — не вся галерея")
    check = {}
    for e, r in stored["by_eps"].items():
        mine = full["cal"]["by_eps"][e]
        check[e] = {k: {"calib_json": r[k], "exp_4_5": mine[k], "equal": bool(r[k] == mine[k])} for k in ("tau_q", "tau_m", "kappa")}
    same_F = full["cal"]["tail"].to_dict() == stored["F"]
    ok = bool(same_F and all(v["equal"] for c in check.values() for v in c.values()))
    say(f"калибровка при 100 экземплярах против calib.json: {'совпала точно' if ok else 'РАСХОДИТСЯ'}")
    if not ok:
        raise SystemExit(f"калибровка при 100 экземплярах не совпала с calib.json — записи нет: {check}, F: {same_F}")

    verdict = RJ.collect_verdict(res["holds"])
    outcome = {}
    for eps in eps_levels:
        e = EK(eps)
        v = verdict[e]
        recal_n0 = {n: [res["cells"][c["r"]][RJ.pair_key(n, n)][e]["quantile_recal"]["eps_holds"] for c in sub["rec"]["chains"]]
                    for n in RU.GALLERY_SUBSET_SIZES}
        outcome[e] = {**RU.rejection_outcome(v["model"]["holds"], v["quantile_recal"]["holds"], v["fixed"]["holds"]),
                      "n_chains_hold": {r: v[r]["n_chains_hold"] for r in RU.REJ_RULES},
                      "fails_at_n0": {**RU.transfer_failure(recal_n0),
                                      "reading": "провал при N = N0 — провал переноса калибровочного набора (40 сцен уровня easy) на "
                                                 "тестовые сцены, а не правила роста"}}
        say(f"ИСХОД, ε = {e}: ({outcome[e]['refined_21sep']['code']}) — {outcome[e]['refined_21sep']['text']}; согласных цепочек: "
            f"model {v['model']['n_chains_hold']}/9, quantile_recal {v['quantile_recal']['n_chains_hold']}/9, fixed "
            f"{v['fixed']['n_chains_hold']}/9; по формулировке 12 сентября — ({outcome[e]['as_written_12sep']['code']})")

    # --- точка на дедуплицированной галерее и справочные строки AP полного метода
    found = {p: PR.search(emb, g, p) for p in ("full", "dedup")}
    qp = CAL.quantile_point(cfg, "dedup")
    all_known = np.ones(len(g.labels), bool)
    ans = {p: [(found[p][0][k][m_star[k]], found[p][1][k][m_star[k]]) for k in test_idx] for p in found}
    for a, b in zip(ans["full"], full["ans"]):
        if not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])):
            raise AssertionError("ответы полной галереи из матрицы сходств расходятся с поиском прогона")
    dedup_point = {"what": "одна точка с порогом по дедуплицированной галерее: только правило по квантили; без подвыборок, без "
                           "правила по модели, без вердикта; в исходы (а)–(в) не входит",
                   "dedup_json_sha256": DD.file_sha256(g), "eta": DD.eta_for(g.passport),
                   "calibration": {k: v for k, v in qp.items() if k != "D_source"}, "by_eps": {}}
    for eps in eps_levels:
        e = EK(eps)
        md = RJ.measure(imgs, ans["dedup"], all_known, float(qp["by_eps"][e]["tau_q"]), None, W)
        mf = RJ.measure(imgs, ans["full"], all_known, float(stored["by_eps"][e]["tau_q"]), None, W)
        for m, t in ((md, qp["by_eps"][e]["tau_q"]), (mf, stored["by_eps"][e]["tau_q"])):
            m["tau"] = float(t)
        dedup_point["by_eps"][e] = {"dedup": md, "full": mf}
        say(f"точка dedup, ε = {e}: FPR {md['fpr']} (полная {mf['fpr']}), пропусков {md['miss_rate']} ({mf['miss_rate']})")

    inputs = hashlib.sha256(json.dumps({"emb": {s.id: _sha256(st.path(s.id)) for s in scenes}, "gallery": g.passport["emb_sha256"],
                                        "dedup": DD.file_sha256(g), "calib": _sha256(g.path / "calib.json"),
                                        "tau_dedup": qp["by_eps"][EK(RU.REJ_FULL_AP_EPS)]["tau_q"],
                                        "rho": {r["image_id"]: [r["masks_sha256"], r["selected"]] for r in rho_recs},
                                        "bootstrap": [cfg.bootstrap, cfg.seed, cfg.max_dets]}, sort_keys=True).encode()).hexdigest()
    blocks = Blocks(cfg.run_id, inputs, say)
    e_ap = EK(RU.REJ_FULL_AP_EPS)
    tau_ap = {"full": float(stored["by_eps"][e_ap]["tau_q"]), "dedup": float(qp["by_eps"][e_ap]["tau_q"])}
    full_ap = {"what": "справочно, без вердикта: AP по D(I) — маски M* с s* ≥ τ_q при ε = 0,05, оценка детекции — s*; тот же вызов "
                       "оценки и те же подмножества сцен, что в 4.4; в таблицу 4.6 — с пометкой «с порогом»",
               "eps": RU.REJ_FULL_AP_EPS, "galleries": {}}
    for p in ("full", "dedup"):
        y, s = found[p][0], found[p][1]
        keep = [m_star[k][s[k][m_star[k]] >= tau_ap[p]] for k in range(len(scenes))]
        ev, sec = blocks.get(f"full_method_ap_{p}", lambda y=y, s=s, keep=keep: PR.evaluate_baseline(
            scenes, labels, edges, gt, g.labels, y, s, max_det=cfg.max_dets, n_boot=cfg.bootstrap, seed=cfg.seed,
            keep_masks=keep, with_ar=False))
        full_ap["galleries"][p] = {"tau_q": tau_ap[p], "N": found[p][2]["N"], "n_detections": ev["n_detections"], "sec": sec,
                                   "ap": {n: {m: ev["subsets"][n]["by_area"]["all"][m] for m in ("ap", "ap50", "ap75")}
                                          | {"ci95": ev["subsets"][n]["by_area"]["all"].get("ci95")} for n in ev["subsets"]}}
        say(f"AP полного метода, галерея «{p}», τ = {tau_ap[p]:.4f}: test/all AP {100 * full_ap['galleries'][p]['ap']['test/all']['ap']:.2f}")

    if _STAMP["code_dirty"] or env.code_stamp()["code_dirty"]:
        raise SystemExit("есть незакоммиченные правки src или scripts — запись 4.5 не создаётся")
    rec = _common_record(cfg, R43, sp, g, st, scenes_cache, refs_cache, scenes, edges)
    rec = {"run_id": f"4_5_{cfg.run_id}", "experiment": "4.5", "kind": "exp_4_5", "grid_run": cfg.run_id, **rec,
           "what": "отклонение и калибровка порога при росте галереи: подвыборки 25 / 50 / 100 экземпляров полной галереи, 9 "
                   "вложенных цепочек; маски M*, дистракторы — IoU рамки < 0,1 ко всем размеченным рамкам сцены; порог — по D_cal "
                   "(дистракторы M* 40 калибровочных сцен), измерение — на 120 тестовых сценах; не прогон сетки",
           "metrics_note": f"доли (0–1); интервалы 95 % частоты ложных срабатываний — перцентильный бутстрэп по сценам, {cfg.bootstrap} "
                           "повторов, seed конфигурации, отношение сумм по сценам; correct_rate — доля известных масок с ŷ ≠ ∅ и "
                           "верной меткой; ячейки N = N0 в вердикт не входят",
           "protocol": _protocol_block(), "subsets": sub["info"],
           "inputs": {"run_emb_files_sha256": {s.id: _sha256(st.path(s.id)) for s in scenes},
                      "gallery_emb_sha256": g.passport["emb_sha256"], "calib_json_sha256": _sha256(g.path / "calib.json"),
                      "calib_gallery_rows": stored["gallery_rows"], "calib_gallery_rows_sha256": stored["gallery_rows_sha256"],
                      "calib_N0": stored["N0"], "calib_summary": str(CAL.summary_path(cfg)), "D_source": source, "rho_cache": str(rc.dir),
                      "embeddings_note": "эмбеддинг маски из M* — сохранённый эмбеддинг той же маски из кодирования всех M(I) "
                                         "(experiments/rho_encode_check.json)"},
           "composition": {"n_D_cal": len(ids_cal), "n_test_scenes": len(imgs), "n_masks_M_star_test": int(sum(n_mstar_test)),
                           "n_masks_M_star_per_scene_min_median_max": [int(min(n_mstar_test)), float(np.median(n_mstar_test)),
                                                                       int(max(n_mstar_test))],
                           "n_scenes_with_M_star_le_top_k": int(sum(n <= RU.TOP_K for n in n_mstar_test)),
                           "n_distractors_test": int(sum(int(i.distractor.sum()) for i in imgs)),
                           "n_oracle_masks_test": int(sum(int((i.gt_label >= 0).sum()) for i in imgs))},
           "calibration_check": {"what": "калибровка при 100 экземплярах против calib.json галереи — точное равенство",
                                 "by_eps": check, "F_identical": same_F, "passed": True},
           "calibrations": res["calibrations"], "cells": res["cells"], "auroc": res["auroc"], "verdict": verdict,
           "outcome": outcome, "assumptions": res["assumptions"], "recalibration": res["recalibration"],
           "dedup_point": dedup_point, "full_method_ap": full_ap}
    _finish(rec, cfg, R43, out, say)


def _finish(rec: dict, cfg, R43, out: Path, say) -> None:
    missing = [b for b in RU.REJ_RECORD_BLOCKS[cfg.dataset] if b not in rec]
    if missing:
        raise AssertionError(f"в записи нет блоков {missing}")
    R43._write(out, _jsonable(rec))
    say(f"ГОТОВО: {out}")


def _jsonable(o):
    """Ключи-числа цепочек и объёмов — строками; numpy — в типы Python."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    cfg = CFG.load(args.config)
    R43 = _script("run_4_3")
    say = R43._logger([f"4_5_{cfg.run_id}"])
    out = record_path(cfg.run_id)
    if cfg.run_id not in RU.REJ_RUNS.get(cfg.dataset, ()):
        raise SystemExit(f"4.5 — только прогоны {RU.REJ_RUNS}; запрошен {cfg.run_id}")
    if args.status:
        say(f"запись: {'есть' if out.exists() else 'нет'} ({out})")
        return
    if out.exists():
        raise SystemExit(f"запись уже есть и не перезаписывается: {out}")
    say(f"версия кода {_STAMP['code_commit'][:7]}{' (есть незакоммиченные правки)' if _STAMP['code_dirty'] else ''}")
    run_hr(cfg, R43, say, out)


if __name__ == "__main__":
    main()
