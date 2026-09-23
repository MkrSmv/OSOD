"""Калибровка порога на лучшем варианте $\\varphi$.

    python scripts/calibrate.py --config configs/dinov2_c_mean_10_hr_insdet.yaml run      # calib.json и сводка
    python scripts/calibrate.py --config configs/dinov2_c_mean_10_hr_insdet.yaml verify   # воспроизводимость
    python scripts/calibrate.py --config configs/dinov2_c_mean_10_hr_insdet.yaml status

`run`: $\\mathcal D_{\\mathrm{cal}}$ — дистракторы $M^*$ 40 калибровочных сцен (маски после отбора $\\rho$ из
`cache/rho/`; первые калибровки, до отбора $\\rho$, шли по $M(I)$, их сводки переименованы `__superseded_`); ожидаемое число —
`D_CAL_EXPECTED`, иное — отказ; эмбеддинги — из рабочих файлов прогона сетки `cache/run_emb/<run_id>/` (fp16 → fp32:
эмбеддинг маски из $M^*$ — сохранённый эмбеддинг той же маски, `experiments/rho_encode_check.json`); точный перебор по всей галерее ($N_0$) даёт все пары для $F$ и $s^*$;
$\\tau^{\\mathrm q}$, $F$, $\\kappa$ и $\\tau^{\\mathrm m}$ при обоих $\\varepsilon$; контрольная проверка при $N_0$ по всей
$\\mathcal D_{\\mathrm{cal}}$ (доля против $\\varepsilon(1+\\delta)$);
запись — `gallery/<encoder>/<variant>/hr_insdet/calib.json` и сводка `experiments/calib_<run_id>.json`. Существующая
запись не перезаписывается; `run --supersede` — замена: прежняя сводка переименовывается в
`calib_<run_id>__superseded_<её коммит>.json`, `calib.json` заменяется. Тестовые сцены не читаются. Только CPU, минуты.

`verify` (новый процесс): калибровка с нуля — `calib.json` обязан совпасть побайтно; контрольная проверка — только по
`check_set_ids` из записанного `calib.json` и галерее при `upto_insert_no` = `insert_no_max` — те же числа, что в
сводке. Итог — блок `verify` сводки. Вывод — ещё и в `logs/calib_<run_id>.log`.

`run --rows dedup` — точка 4.5 на дедуплицированной галерее: только правило по квантили по строкам `dedup`, вывод на
экран; в `calib.json` галереи и в сводку не пишется — запись ведёт блок `dedup_point` записи эксперимента 4.5
(`scripts/run_4_5.py` вызывает тот же `quantile_point`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as CFG
from src import env
from src.calib import threshold as TH
from src.encode import run_emb as RE
from src.encode.variants import grid
from src.eval import oracle as OR
from src.eval import protocol as PR
from src.eval import rules as RU
from src.gallery import store
from src.segment import cache as MC
from src.select import rho as RHO

RUNS = Path("experiments/runs")
OUT = Path("experiments")
SPLIT = "cal"
N_CAL_SCENES = 40
GALLERY_PROTOCOL = "full"
# ожидаемый объём D_cal из M* — `experiments/analysis/rho_rules_part3.json`, θ = 0,9, `neardup_0.8`, роль
# `distractor`; иное число — ошибка отбора дистракторов либо ρ: запись не создаётся
D_CAL_EXPECTED = 1552
HNSW_Q = {0.05: 0.95, 0.01: 0.99}  # справочно: квантили s* дистракторов M(I) записи замера HNSW рядом с τ_q (до ρ — сверка)
_STAMP = env.code_stamp()  # версия кода — на момент запуска процесса


def _logger(run_id: str):
    Path("logs").mkdir(exist_ok=True)
    log = open(Path("logs") / f"calib_{run_id}.log", "a")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [calib {run_id}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    return say


def summary_path(cfg) -> Path:
    return OUT / f"calib_{cfg.run_id}.json"


def _write_json(path: Path, rec: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(path)


def best_variant(encoder: str, dataset: str) -> str:
    """Лучший вариант энкодера по `selection_cal` журнала (`rules.best_variant`); без всех записей — отказ."""
    names = [v for e, v, d in grid() if e == encoder and d == dataset]
    recs = {v: RUNS / f"{encoder}_{v}_{dataset}.json" for v in names}
    missing = [v for v, p in recs.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"лучший φ {encoder} выбирается по всем его прогонам; нет записей: {missing}")
    cal = {v: json.loads(p.read_text())["selection_cal"] for v, p in recs.items()}
    return RU.best_variant({v: {"ap": c["ap"], "ap50": c["ap50"]} for v, c in cal.items()})


def _setup(cfg):
    """Сверки состава и входные данные: галерея $G_{N_0}$, $\\mathcal D_{\\mathrm{cal}}$ и его источник."""
    if cfg.dataset != "hr_insdet":
        raise SystemExit("калибровка — только HR-InsDet: D_cal PCB не сегментирован")
    best = best_variant(cfg.encoder, cfg.dataset)
    if cfg.variant != best:
        raise SystemExit(f"калибровка — на лучшем φ энкодера {cfg.encoder}: {best}, запрошен {cfg.variant}")
    mc = MC.MaskCache(cfg.dataset, MC.auto_key(cfg.crop_n_layers, cfg.seg_long_side, cfg.points_per_batch))
    scenes, *_, sp = PR.load_scenes(cfg.dataset, mc, splits=(SPLIT,))  # тестовые сцены и их кеш масок не читаются
    if len(scenes) != N_CAL_SCENES or any(s.split != SPLIT for s in scenes) or set(sp[SPLIT]) & set(sp["test"]):
        raise SystemExit("D_cal — только 40 калибровочных сцен")
    if [s.id for s in scenes] != list(sp[SPLIT]):
        raise SystemExit("порядок сцен расходится с splits — состав D_chk зависел бы от порядка чтения")
    work = RE.scene_store(cfg, mc.key)
    g = store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset),
                           store.expected(cfg.to_dict(), MC.box_key(cfg.seg_long_side)))
    rows = g.rows(GALLERY_PROTOCOL)
    if not np.array_equal(rows, np.arange(len(g))) or g.deleted.any():
        raise SystemExit("G_N0 — вся галерея без помеченных; строки расходятся с номерами вставки")
    rc = RHO.RhoCache(mc)
    m_star = {s.id: rc.selected(s.id) for s in scenes}  # запись отбора сверяется с файлом масок при чтении
    h = hashlib.sha256((rc.dir / "key.json").read_bytes())
    for s in scenes:
        h.update(rc.path(s.id).name.encode() + b"\0" + rc.path(s.id).read_bytes())
    source = {"masks": "M_star", "rho": {"theta": str(RHO.THETA), "gamma": str(RHO.GAMMA), "rule": RHO.RHO_RULE},
              "rho_cache": str(rc.dir), "rho_cache_sha256": h.hexdigest(),
              "rho_cache_sha256_of": "key.json и записи отбора 40 калибровочных сцен (имя файла, байты) в порядке сцен",
              "dataset": cfg.dataset, "split": SPLIT, "n_scenes": len(scenes),
              "crop_n_layers": cfg.crop_n_layers, "distractor_rule": f"IoU описывающей рамки < {OR.DISTRACTOR_IOU_MAX:g} "
              "с каждой размеченной рамкой (src.eval.oracle.distractors; истина — src.data.hr_insdet.scene_gt)",
              "mask_cache": mc.dir.name, "embeddings": str(work.dir), "embeddings_dtype": "float16 → float32"}
    return scenes, work, g, rows, source, m_star


def _d_cal(scenes, work, m_star) -> tuple[np.ndarray, list[str]]:
    """Эмбеддинги и идентификаторы $\\mathcal D_{\\mathrm{cal}}$ в порядке сцен и номеров масок: маски $M^*$, являющиеся
    дистракторами; номер маски — номер в записи кеша масок, как до отбора."""
    z, ids = [], []
    for s in scenes:
        e = work.load(s.id)["z"]
        if len(e) != len(s.boxes):
            raise SystemExit(f"{s.id}: эмбеддингов {len(e)}, масок в кеше {len(s.boxes)}")
        k = m_star[s.id][s.distractor[m_star[s.id]]]
        z.append(e[k])
        ids += [TH.mask_id(s.id, i) for i in k]
    if len(ids) != D_CAL_EXPECTED:
        raise SystemExit(f"|D_cal| = {len(ids)}, ожидается {D_CAL_EXPECTED} — ошибка отбора дистракторов либо ρ; "
                         "запись не создаётся")
    return np.ascontiguousarray(np.concatenate(z)), ids


def _s_star(g, rows, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Точный перебор при $k=N_0$: матрица всех сходств (строки — запросы) и $s^*$ по правилу решения."""
    from src.search import decide as DE
    from src.search import index as IX

    sim, idx = IX.ExactIndex(g.emb, rows).search(z)
    _, s_star = DE.decide(sim, idx, g.label_ids, deleted=g.deleted)
    if (idx < 0).any() or not np.array_equal(np.sort(idx, 1), np.tile(rows, (len(idx), 1))):
        raise AssertionError("точный перебор вернул не все строки галереи")
    return sim, s_star


def compute(cfg) -> tuple[dict, dict, dict, dict[str, float]]:
    """Запись `calib.json` (без `format`), сводка калибровки, справочные сведения о входе и $s^*$ по маскам
    $\\mathcal D_{\\mathrm{cal}}$."""
    scenes, work, g, rows, source, m_star = _setup(cfg)
    z, ids = _d_cal(scenes, work, m_star)
    sim, s_star = _s_star(g, rows, z)
    rec, summ = TH.calibrate(sim, s_star, ids, tuple(cfg.eps), cfg.delta)
    rec = {**rec, "insert_no_max": int(rows.max()), "D_source": source,
           "gallery_emb_sha256": g.passport.get("emb_sha256"),
           "gallery_rows": GALLERY_PROTOCOL, "gallery_rows_sha256": store.rows_sha256(g, rows)}
    info = {"n_D_cal": len(ids), "n_pairs": int(sim.size), "gallery": str(g.path), "n_labels": len(g.labels),
            "s_star_D_cal_min_median_max": [float(v) for v in np.quantile(s_star, [0, 0.5, 1])]}
    return rec, summ, info, dict(zip(ids, map(float, s_star)))


def quantile_point(cfg, protocol: str = "dedup") -> dict:
    """Точка 4.5 на ином составе строк той же галереи: только правило по квантили — $\\tau^{\\mathrm q}$ по той же
    $\\mathcal D_{\\mathrm{cal}}$ при обоих $\\varepsilon$; без правила по модели. Ничего не пишет."""
    if protocol == GALLERY_PROTOCOL:
        raise SystemExit("порог по полной галерее — запись calib.json (`run`), а не точка")
    scenes, work, g, _, source, m_star = _setup(cfg)
    z, ids = _d_cal(scenes, work, m_star)
    rows = g.rows(protocol)
    _, s_star = _s_star(g, rows, z)
    by_eps = {}
    for eps in cfg.eps:
        tq = TH.tau_q(s_star, eps)
        by_eps[store.eps_key(eps)] = {"tau_q": tq, "check_at_N0": TH.control_check(s_star, tq, eps, cfg.delta)}
    return {"gallery_rows": protocol, "gallery_rows_sha256": store.rows_sha256(g, rows), "N0": int(len(rows)),
            "n_D_cal": len(ids), "D_source": source, "gallery_emb_sha256": g.passport.get("emb_sha256"), "by_eps": by_eps}


def _calib_bytes(rec: dict, eps) -> bytes:
    """Байты `calib.json` так, как их пишет `store.write_calib`."""
    full = {"format": store.FORMAT, **rec}
    store.check_calib(full, eps)
    return (json.dumps(full, ensure_ascii=False) + "\n").encode()


def _hnsw_crosscheck(cfg, summ: dict) -> dict:
    p = RUNS / f"hnsw_{cfg.run_id}.json"
    if not p.is_file():
        return {"available": False}
    q = json.loads(p.read_text())["grid_check"]["distractor_s_star_quantiles"]
    by_q = dict(zip(q["q"], q["exact"]))
    out = {"available": True, "record": str(p),
           "note": "запись замера HNSW — по дистракторам M(I); при D_cal из M* квантили обязаны различаться, сверкой не служит"}
    for eps, level in HNSW_Q.items():
        tq = summ[store.eps_key(eps)]["tau_q"]
        out[store.eps_key(eps)] = {"hnsw_q": level, "hnsw_exact": by_q[level], "tau_q": tq, "abs_diff": abs(tq - by_q[level])}
    return out


SUPERSEDED = "__superseded_"


def _supersede(cfg, gpath: Path, say) -> None:
    """Замена записи: прежняя сводка переименовывается с коммитом прежней записи, `calib.json` удаляется — его sha256
    остаётся в переименованной сводке."""
    old = summary_path(cfg)
    if not old.exists():
        raise SystemExit(f"--supersede: прежней сводки нет ({old})")
    commit = json.loads(old.read_text())["code_commit"][:7]
    dst = old.with_name(f"calib_{cfg.run_id}{SUPERSEDED}{commit}.json")
    if dst.exists():
        raise SystemExit(f"--supersede: {dst} уже есть")
    old.rename(dst)
    (gpath / "calib.json").unlink(missing_ok=True)
    say(f"прежняя сводка → {dst}; прежний calib.json удалён")


def stage_run(cfg, supersede: bool = False) -> None:
    say = _logger(cfg.run_id)
    gpath = store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset)
    if supersede:
        _supersede(cfg, gpath, say)
    if (gpath / "calib.json").exists() or summary_path(cfg).exists():
        raise SystemExit(f"калибровка уже записана ({gpath / 'calib.json'}, {summary_path(cfg)}) и не перезаписывается; "
                         "замена — --supersede")
    say(f"версия кода {_STAMP['code_commit'][:7]}{' (есть незакоммиченные правки)' if _STAMP['code_dirty'] else ''}")
    t0 = time.time()
    rec, summ, info, _ = compute(cfg)
    store.write_calib(gpath, rec, tuple(cfg.eps))
    data = (gpath / "calib.json").read_bytes()
    if data != _calib_bytes(rec, tuple(cfg.eps)):
        raise AssertionError("calib.json записан не теми байтами, что проверяет verify")
    out = {"run_id": cfg.run_id, "kind": "calibration", "step": "калибровка порога; пересчёт с ρ после отбора",
           **{**env.code_stamp(), "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"]},
           "config": cfg.to_dict(), "calib_json": str(gpath / "calib.json"),
           "calib_json_sha256": hashlib.sha256(data).hexdigest(), "gallery_emb_sha256": rec["gallery_emb_sha256"],
           "N0": rec["N0"], "insert_no_max": rec["insert_no_max"], "D_source": rec["D_source"], **info,
           "check_set": {"size": len(rec["check_set_ids"]), "rule": rec["check_rule"]},
           "F": {"n_pairs": rec["F"]["n_pairs"], "n_top": len(rec["F"]["top"]), "max": rec["F"]["top"][0],
                 "top_min": rec["F"]["top"][-1],
                 "quantiles_0.5_0.9_0.99_0.999": [float(np.interp(q, np.linspace(0, 1, store.F_QUANTILES),
                                                                  rec["F"]["quantiles"])) for q in (0.5, 0.9, 0.99, 0.999)]},
           "delta": rec["delta"], "by_eps": summ, "hnsw_crosscheck": _hnsw_crosscheck(cfg, summ),
           "sec": round(time.time() - t0, 1)}
    _write_json(summary_path(cfg), out)
    for e, s in summ.items():
        say(f"ε = {e}: τ_q = {s['tau_q']:.6f}, κ = {s['kappa']:.6g}, τ_m = {s['tau_m']:.6f} (невязка {s['residual']:.1e}, "
            f"F^-1 — {s['f_inv_path']}); контроль при N0: τ_q — {s['check_at_N0']['tau_q']['n_ge_tau']}, "
            f"τ_m — {s['check_at_N0']['tau_m']['n_ge_tau']} из {s['check_at_N0']['tau_q']['n']} "
            f"(тревога с {s['check_at_N0']['tau_q']['k_trigger']}; перекалибровка — "
            f"{s['check_at_N0']['tau_q']['recalibrate'] or s['check_at_N0']['tau_m']['recalibrate']})")
    say(f"|D_cal| = {info['n_D_cal']}, пар {info['n_pairs']}, N0 = {rec['N0']}; {out['sec']} с")
    say(f"ГОТОВО: {gpath / 'calib.json'}, {summary_path(cfg)}")


def stage_verify(cfg) -> None:
    say = _logger(cfg.run_id)
    gpath = store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset)
    summ_rec = json.loads(summary_path(cfg).read_text())
    written = (gpath / "calib.json").read_bytes()
    g0 = store.Gallery.open(gpath, store.expected(cfg.to_dict(), MC.box_key(cfg.seg_long_side)))
    stored = store.read_calib(gpath, tuple(cfg.eps), g0, GALLERY_PROTOCOL)

    # (1) калибровка с нуля — те же байты
    rec, summ, _, s_full = compute(cfg)
    if "gallery_rows_sha256" not in json.loads(written):  # запись старая: состава строк в ней нет
        rec = {k: v for k, v in rec.items() if k not in store.ROWS_FIELDS}
    same_bytes = _calib_bytes(rec, tuple(cfg.eps)) == written
    same_sha = hashlib.sha256(written).hexdigest() == summ_rec["calib_json_sha256"]

    # (2) контрольная проверка только по check_set_ids записанного calib.json
    scenes, work, g, _, _, m_star = _setup(cfg)
    if g.passport.get("emb_sha256") != stored["gallery_emb_sha256"]:
        raise SystemExit("галерея не та, на которой записана калибровка")
    rows = g.rows(GALLERY_PROTOCOL, upto_insert_no=stored["insert_no_max"])
    cal_ids = {s.id for s in scenes}
    z = []
    for mid in stored["check_set_ids"]:
        scene, k = TH.parse_mask_id(mid)
        if scene not in cal_ids:
            raise SystemExit(f"{mid}: сцена не из калибровочной части")
        if k not in set(m_star[scene].tolist()):
            raise SystemExit(f"{mid}: маска не входит в M* — D_chk обязана состоять из масок после отбора ρ")
        z.append(work.load(scene)["z"][k])
    _, s_chk = _s_star(g, rows, np.ascontiguousarray(np.stack(z)))
    checks, same_checks = {}, True
    for e, r in stored["by_eps"].items():
        for rule in ("tau_q", "tau_m"):
            if stored["check_rule"] != TH.CHECK_RULE:
                raise SystemExit("правило контрольной проверки в calib.json — не то, что реализует код")
            c = TH.control_check(s_chk, r[rule], float(e), stored["delta"])
            checks[f"{e}/{rule}"] = c
            same_checks &= c == summ_rec["by_eps"][e]["check_at_N0"][rule]
    s_diff = max(abs(float(a) - s_full[m]) for a, m in zip(s_chk, stored["check_set_ids"]))
    ok = bool(same_bytes and same_sha and same_checks)
    summ_rec["verify"] = {**{**env.code_stamp(), "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"]},
                          "calib_json_bytes_identical": bool(same_bytes), "calib_json_sha256_matches_summary": bool(same_sha),
                          "control_check_from_check_set_ids_identical": bool(same_checks),
                          "control_check_recomputed": checks,
                          "s_star_chk_max_abs_diff_vs_full_search": s_diff, "passed": ok}
    _write_json(summary_path(cfg), summ_rec)
    say(f"verify: calib.json побайтно {'тот же' if same_bytes else 'ДРУГОЙ'}; sha256 {'совпал' if same_sha else 'НЕ совпал'}; "
        f"контрольная проверка по check_set_ids — {'те же числа' if same_checks else 'ДРУГИЕ числа'}")
    if not ok:
        raise SystemExit("калибровка не воспроизводится")
    say(f"ГОТОВО: блок verify в {summary_path(cfg)}")


def stage_status(cfg) -> None:
    gpath = store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset)
    s = summary_path(cfg)
    v = json.loads(s.read_text()).get("verify", {}).get("passed") if s.exists() else None
    print(f"{cfg.run_id}: calib.json — {'есть' if (gpath / 'calib.json').exists() else 'нет'}; сводка — "
          f"{'есть' if s.exists() else 'нет'}; verify — {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("stage", choices=["run", "verify", "status"])
    ap.add_argument("--supersede", action="store_true", help="run: заменить запись, прежнюю сводку переименовать")
    ap.add_argument("--rows", choices=["full", "dedup"], default=GALLERY_PROTOCOL,
                    help="run: dedup — точка 4.5 на дедуплицированной галерее, только τ_q, без записи")
    args = ap.parse_args()
    cfg = CFG.load(args.config)
    if args.rows != GALLERY_PROTOCOL:
        if args.stage != "run" or args.supersede:
            ap.error("--rows dedup — только `run` без --supersede: точка считается и печатается, записи нет")
        print(json.dumps(quantile_point(cfg, args.rows), ensure_ascii=False, indent=1))
        return
    if args.stage == "run":
        stage_run(cfg, args.supersede)
        return
    {"verify": stage_verify, "status": stage_status}[args.stage](cfg)


if __name__ == "__main__":
    main()
