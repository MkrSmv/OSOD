"""Разбор результатов 4.3 на HR-InsDet по записям журнала — после результата, без новых прогонов и без GPU.

    python scripts/analyze_4_3.py            # → experiments/post_hoc_4_3_hr_insdet.json

Не таблица главы 4 (те — только `make_tables.py`) и не заранее записанная проверка: сравнения выбраны, зная результат,
и в тексте подаются как разбор после результата. Два блока:

1. Парные бутстрэп-разности AP между прогонами сетки по осям «заполнитель», «контекст», «энкодер» — по повторам
   `bootstrap_ap` записей (120 тестовых сцен, оракул; повторы общие у всех прогонов). Кеш масок и эмбеддинги сцен не читаются.
2. Сторона галерей (только эталоны): медиана сходства эталонов одной метки и разных меток в каждой из 15 галерей —
   шкала сходств варианта $\\varphi$; к тестовым сценам отношения не имеет.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import env  # noqa: E402
from src.eval import rules as RU  # noqa: E402

OUT = Path("experiments/post_hoc_4_3_hr_insdet.json")
RUNS = Path("experiments/runs")
DATASET = "hr_insdet"
ENCODERS = ("dinov2", "dinov3")
CROPS = ("c_0_10", "c_0_15", "c_mean_10", "c_mean_15", "c_blur_10", "c_blur_15")
GALLERIES = ("full", "one_per_class")


def variants(encoder: str) -> tuple[str, ...]:
    return (*CROPS, "p") + (("p_perp",) if encoder == "dinov3" else ())


def load(encoder: str, variant: str) -> dict:
    return json.loads((RUNS / f"{encoder}_{variant}_{DATASET}.json").read_text())


def main() -> None:
    rec = {(e, v): load(e, v) for e in ENCODERS for v in variants(e)}
    out: dict = {"what": "разбор 4.3 на HR-InsDet после результата: парные бутстрэп-разности AP и шкала сходств галерей; "
                         "не таблица главы 4 и не заранее записанная проверка", "post_hoc": True, **env.code_stamp(),
                 "subset": RU.BOOT_AP_SUBSET, "unit": "пункты AP; интервал — 95 % перцентильный по общим повторам бутстрэпа",
                 "paired_diff": {}, "gallery_similarity": {}}

    def diff(gal: str, a: tuple[str, str], b: tuple[str, str]) -> dict:
        ba, bb = (np.asarray(rec[k]["bootstrap_ap"][gal]["values"], float) for k in (a, b))
        pa, pb = (rec[k]["metrics"][gal]["subsets"][RU.BOOT_AP_SUBSET]["by_area"]["all"]["ap"] for k in (a, b))
        lo, hi = np.quantile(100 * (ba - bb), [0.025, 0.975])
        return {"a": "_".join(a), "b": "_".join(b), "diff": round(100 * (pa - pb), 2), "ci95": [round(lo, 2), round(hi, 2)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    for gal in GALLERIES:
        d = out["paired_diff"][gal] = {"filler": [], "context": [], "encoder": [], "debias": []}
        for e in ENCODERS:
            for al in ("10", "15"):
                d["filler"] += [diff(gal, (e, f"c_mean_{al}"), (e, f"c_0_{al}")), diff(gal, (e, f"c_blur_{al}"), (e, f"c_0_{al}")),
                                diff(gal, (e, f"c_blur_{al}"), (e, f"c_mean_{al}"))]
            d["context"] += [diff(gal, (e, f"c_{b}_15"), (e, f"c_{b}_10")) for b in ("0", "mean", "blur")]
        d["encoder"] = [diff(gal, ("dinov3", v), ("dinov2", v)) for v in (*CROPS, "p")]
        d["debias"] = [diff(gal, ("dinov3", "p_perp"), ("dinov3", "p"))]

    for e in ENCODERS:
        for v in variants(e):
            g = Path("gallery") / e / v / DATASET
            z = np.load(g / "emb.npy")
            lab = np.array([json.loads(x)["label"] for x in (g / "meta.jsonl").read_text().splitlines()])
            s = z @ z.T
            same = lab[:, None] == lab[None, :]
            off = ~np.eye(len(z), dtype=bool)
            out["gallery_similarity"][f"{e}_{v}"] = {
                "n": int(len(z)), "same_label_median": round(float(np.median(s[same & off])), 4),
                "other_label_median": round(float(np.median(s[~same])), 4),
                "other_label_q99": round(float(np.quantile(s[~same], 0.99)), 4)}

    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(OUT)
    for gal in GALLERIES:
        for axis, rows in out["paired_diff"][gal].items():
            for r in rows:
                print(f"{gal:14s} {axis:8s} {r['a']:18s} − {r['b']:18s} {r['diff']:+6.2f} [{r['ci95'][0]:+.2f}; {r['ci95'][1]:+.2f}]"
                      f"{' *' if r['excludes_zero'] else ''}")
    for k, v in out["gallery_similarity"].items():
        print(f"{k:20s} одна метка {v['same_label_median']:.3f} разные {v['other_label_median']:.3f} q99 разных {v['other_label_q99']:.3f}")


if __name__ == "__main__":
    main()
