"""Отбор гранулярности $\\rho$: $M^*$ сцен HR-InsDet в `cache/rho/`.

    python scripts/select_rho.py --dataset hr_insdet               # M* 160 сцен (CPU, минуты; посчитанное пропускается)
    python scripts/select_rho.py --dataset hr_insdet --status
    python scripts/select_rho.py --dataset hr_insdet --verify-cal  # сверка с аналитикой §2.3 → experiments/rho_verify_cal.json

Отбор геометрический: читаются только маски кеша (RLE входа $S$), разметка и эмбеддинги не нужны. Для PCB скрипт
отказывает — там $M^*=M(I)$.

`--verify-cal` — только split `cal`: состав $M^*$ каждой из 40 калибровочных сцен, прочитанный из `cache/rho/`
(целочисленная арифметика `src.select.rho`), обязан поштучно совпасть с `rule_neardup(…, 0.80)` скрипта аналитики при
$\\theta=0{,}90$ (арифметика с плавающей точкой), а сводные числа — с `experiments/analysis/rho_rules_part3.json`
(`neardup_0.8`). Расхождение хотя бы в одной маске — код возврата 1: 4.4 не начинается; арифметика
и пороги по расхождению не меняются.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as CFG  # noqa: E402
from src import env  # noqa: E402
from src.segment import cache as MC  # noqa: E402
from src.select import rho as RHO  # noqa: E402

CONFIG = "dinov2_c_mean_10_hr_insdet.yaml"   # ключ кеша масок у всех прогонов HR-InsDet один; берётся из конфигурации
OUT = Path("experiments/rho_verify_cal.json")
ANALYSIS = Path("experiments/analysis/rho_rules_part3.json")
ANALYSIS_RULE, ANALYSIS_THETA, ANALYSIS_A = "neardup_0.8", 0.90, 0.80
_STAMP = env.code_stamp()


def _say(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [select_rho] {msg}", flush=True)


def _caches(dataset: str):
    if dataset not in RHO.DATASETS:
        raise SystemExit(f"ρ применяется только к {RHO.DATASETS}: на {dataset} M* = M(I)")
    cfg = CFG.load(CFG.CONFIG_DIR / CONFIG)
    mc = MC.MaskCache(dataset, MC.auto_key(cfg.crop_n_layers, cfg.seg_long_side, cfg.points_per_batch))
    sp = json.loads(Path(f"splits/{dataset}.json").read_text())
    if set(sp["cal"]) & set(sp["test"]):
        raise SystemExit("калибровочные и тестовые сцены пересекаются")
    return mc, RHO.RhoCache(mc), sp


def stage_select(dataset: str, status: bool) -> None:
    mc, rc, sp = _caches(dataset)
    ids = [*sp["cal"], *sp["test"]]
    todo = [i for i in ids if not rc.has(i)]
    _say(f"{rc.dir}: посчитано {len(ids) - len(todo)} из {len(ids)}")
    if status:
        return
    t0 = time.time()
    for n, i in enumerate(todo, 1):
        rec = rc.compute(i)
        _say(f"{n}/{len(todo)} {i}: {rec['n_selected']} из {rec['n_masks']} масок, {rec['sec']:.2f} с")
    tot = [rc.load(i) for i in ids]
    for name, part in (("cal", sp["cal"]), ("test", sp["test"])):
        rs = [r for r in tot if r["image_id"] in set(part)]
        _say(f"{name}: сцен {len(rs)}, масок {sum(r['n_masks'] for r in rs)}, отобрано {sum(r['n_selected'] for r in rs)}")
    _say(f"ГОТОВО за {time.time() - t0:.0f} с")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_verify_cal(dataset: str) -> int:
    from src.eval import protocol as PR

    mc, rc, sp = _caches(dataset)
    R2 = _load_script("analyze_rho_cal")          # сам загружает `analyze_cal_masks` как `R2.A`
    A = R2.A
    scenes, *_ = PR.load_scenes(dataset, mc, splits=(A.SPLIT,))   # кеш масок тестовых сцен не читается
    if len(scenes) != A.N_SCENES or [s.id for s in scenes] != list(sp["cal"]) or any(s.split != "cal" for s in scenes):
        raise SystemExit("ожидаются ровно 40 калибровочных сцен в порядке splits")
    sms = [A.SceneMasks(s, mc.load(s.id)) for s in scenes]
    nest = {sm.id: {ANALYSIS_THETA: A.Nesting(sm, ANALYSIS_THETA)} for sm in sms}
    per_scene, mismatched, violations, n_equal = [], [], 0, 0
    ours: dict[str, list[int]] = {}
    for sm in sms:
        rec = rc.load(sm.id)
        if rec["n_masks"] != sm.n:
            raise SystemExit(f"{sm.id}: в записи отбора {rec['n_masks']} масок, в кеше {sm.n}")
        want = R2.rule_neardup(nest[sm.id][ANALYSIS_THETA], ANALYSIS_A)
        got = ours[sm.id] = rec["selected"]
        only_ours, only_theirs = sorted(set(got) - set(want)), sorted(set(want) - set(got))
        ne = nest[sm.id][ANALYSIS_THETA]
        violations += sum(len(ne.inside[i] & set(got)) for i in got)
        n_equal += rec["n_equal_area_duplicates"]
        per_scene.append({"scene": sm.id, "n_masks": sm.n, "n_selected": len(got), "equal": got == want,
                          "only_in_rho": only_ours, "only_in_analysis": only_theirs})
        if got != want:
            mismatched.append(sm.id)
    # сводные числа — кодом аналитики по составу из cache/rho/
    m = R2.rule_metrics(sms, nest, ANALYSIS_THETA, lambda ne: {"rho": (ours[ne.sm.id], 0)})["rho"]
    ref = json.loads(ANALYSIS.read_text())["rho_rules_part3"]["by_theta"][str(ANALYSIS_THETA)][ANALYSIS_RULE]
    keys = ("n_kept", "by_role", "recall_50", "recall_75", "recall_50_95", "n_gt", "gt_with_2plus_masks_at_50_share")
    totals = {k: {"rho": m[k], "analysis": ref[k], "equal": m[k] == ref[k]} for k in keys}
    n_found = round(m["recall_50"] * m["n_gt"])
    expected = {"n_kept": 2498, "n_masks": 4534, "by_role": {"object": 699, "near": 247, "distractor": 1552},
                "n_gt_found_at_50": 733, "n_gt": 831, "recall_percent": [88.21, 78.70, 77.22]}
    got = {"n_kept": m["n_kept"], "n_masks": sum(sm.n for sm in sms), "by_role": m["by_role"], "n_gt_found_at_50": n_found,
           "n_gt": m["n_gt"], "recall_percent": [round(100 * m[k], 2) for k in ("recall_50", "recall_75", "recall_50_95")]}
    passed = not mismatched and violations == 0 and all(t["equal"] for t in totals.values()) and got == expected
    out = {"what": "сверка ρ (целочисленная арифметика, cache/rho/) с аналитикой §2.3 (rule_neardup, float)",
           "dataset": dataset, "split": A.SPLIT, "n_scenes": len(sms), **_STAMP,
           "rho": {"theta": str(RHO.THETA), "gamma": str(RHO.GAMMA), "rule": RHO.RHO_RULE, "cache": str(rc.dir)},
           "analysis": {"script": "scripts/analyze_rho_cal.py", "rule": ANALYSIS_RULE, "theta": ANALYSIS_THETA,
                        "a": ANALYSIS_A, "record": str(ANALYSIS)},
           "n_scenes_mismatched": len(mismatched), "scenes_mismatched": mismatched,
           "antichain_violations": violations, "n_equal_area_duplicates": n_equal,
           "expected_by_tz": expected, "got": got, "totals_vs_analysis_record": totals, "passed": passed,
           "per_scene": per_scene}
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT)
    _say(f"сцен с расхождением: {len(mismatched)} из {len(sms)}; нарушений антицепи: {violations}; "
         f"масок {got['n_kept']} из {got['n_masks']}, по ролям {got['by_role']}, рамок {n_found} из {got['n_gt']}, "
         f"полнота {got['recall_percent']}")
    _say(f"{'СВЕРКА ПРОЙДЕНА' if passed else 'СВЕРКА НЕ ПРОЙДЕНА — 4.4 не начинать'}: {OUT}")
    return 0 if passed else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--verify-cal", action="store_true", help="сверка с аналитикой §2.3, только split cal")
    args = ap.parse_args()
    if args.verify_cal:
        sys.exit(stage_verify_cal(args.dataset))
    stage_select(args.dataset, args.status)


if __name__ == "__main__":
    main()
