"""Таблицы главы 4 — только из журнала прогонов `experiments/runs/*.json` → `experiments/tables/`.

Вручную таблицы не заполняются: правка числа — это правка прогона. 4.2 — предел полноты сегментатора; 4.3 — сетка
кодирования с контрольными прогонами по протоколу baseline (лучший $\\varphi$ энкодера выбирается здесь же, правилом
`src.eval.rules.best_variant` по `selection_cal` записей, а не вручную); 4.4 — отбор гранулярности; 4.5 — отклонение
при пополнении галереи; 4.6 — сравнение с OWLv2 и опубликованными числами (`src.eval.published`) и задержка по этапам.
`print.md` собирает таблицы в том виде, в каком они напечатаны в тексте работы.

    python scripts/make_tables.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encode.variants import VARIANTS, grid
from src.eval import rules as RU

RUNS = Path("experiments/runs")
TABLES = Path("experiments/tables")

# Опубликованные AR@100 на HR-InsDet, %; адреса — `AR100_ADDRESS`.
# Это AR итоговых детекций (после сопоставления, ранжирование по сходству), а не всех предложений.
PUBLISHED_AR100 = {
    "SAM + DINOv2 [5]": {"all": 63.06, "easy": 71.96, "hard": 43.47, "small": 31.11, "medium": 70.40, "large": 90.36},
    "Grounding DINO + дообученный DINOv2 (IDOW) [8]": {"all": 77.09, "easy": None, "hard": None,
                                                       "small": 53.53, "medium": 83.73, "large": 94.06},
}
AR100_ADDRESS = "[5] — разд. 3.1, табл. 2, 3 и доп. материалы, табл. 7; [8] — доп. материалы, табл. 6"

# Опубликованные AP на HR-InsDet, %: 41,61 — SAM + DINOv2 [5]; разбивка по размеру — строка
# «OTS-FM SAM» табл. 1 [8] (тот же метод).
PUBLISHED_AP = {"SAM + DINOv2 [5]; по размеру — [8], табл. 1": {"all": 41.61, "small": 14.58, "medium": 45.83, "large": 69.14},
                "SAM + DINOv2 ViT-L/14 — README кода [5], не статья": {"all": RU.PUBLISHED_AP_README_VIT_L, "small": None,
                                                                       "medium": None, "large": None}}

SPLITS = [("cal", "калибровочные"), ("test", "тестовые"), ("all", "все")]
LEVELS = [("all", "все"), ("easy", "easy"), ("hard", "hard")]
AREAS = [("small", "< 200²"), ("medium", "200²–400²"), ("large", "> 400²")]
EFFECTIVE = {"0": "1024", "1": "2048"}


def pct(x, nd: int = 2) -> str:
    return "—" if x is None else f"{100 * x:.{nd}f}".replace(".", ",")


def num(x, nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def with_ci(m: dict, name: str) -> str:
    if m[name] is None:
        return "—"
    lo, hi = m["ci95"][name]
    return f"{pct(m[name])} [{pct(lo, 1)}; {pct(hi, 1)}]"


def table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(r) + " |" for r in rows])


def tables_4_2(rec: dict) -> str:
    res, cfg, sel = rec["results"], rec["config"], rec["selection"]
    out = [f"# 4.2. Предел полноты сегментатора — {rec['dataset']}", "",
           f"Источник чисел — `experiments/runs/{rec['run_id']}.json` (запись от {rec['written']}). "
           "Все величины — в процентах, по описывающим рамкам масок; в квадратных скобках — "
           f"интервал 95 %, бутстрэп по сценам ({cfg['bootstrap']['n']} повторов). Recall — по всем маскам $M(I)$, без предела "
           "их числа; AR@100 и AR@1000 — COCOeval без категорий, предложения ранжированы оценкой predicted IoU SAM 2, "
           "пороги IoU 0,5:0,95.", ""]

    out += ["## Таблица 4.2-1. Полнота по разрешению входа и набору сцен", ""]
    rows = []
    for cnl in map(str, cfg["crop_n_layers"]):
        for s, s_name in SPLITS:
            for lv, lv_name in LEVELS:
                r = res[cnl].get(f"{s}/{lv}")
                if r is None or (s == "cal" and lv != "all"):  # калибровочные сцены — все easy
                    continue
                m = r["by_area"]["all"]
                rows.append([f"{cnl} ({EFFECTIVE[cnl]})", s_name, lv_name, str(r["n_scenes"]), str(m["n_gt"]),
                             num(r["masks_per_scene_mean_median_max"][0], 1), with_ci(m, "recall_50"),
                             with_ci(m, "recall_75"), with_ci(m, "recall_50_95"), with_ci(m, "ar_100"), with_ci(m, "ar_1000")])
    out += [table(["`crop_n_layers` (эфф. разрешение)", "сцены", "уровень", "сцен", "рамок", "масок на сцену",
                   "recall, IoU ≥ 0,5", "recall, IoU ≥ 0,75", "recall, 0,5:0,95", "AR@100", "AR@1000"], rows), ""]

    out += ["## Таблица 4.2-2. Полнота по бинам площади рамки (пикс.² исходного снимка)", ""]
    rows = []
    for cnl in map(str, cfg["crop_n_layers"]):
        for s, s_name in SPLITS:
            for a, a_name in AREAS:
                m = res[cnl][f"{s}/all"]["by_area"][a]
                rows.append([f"{cnl} ({EFFECTIVE[cnl]})", s_name, a_name, str(m["n_gt"]), with_ci(m, "recall_50"),
                             with_ci(m, "recall_75"), with_ci(m, "recall_50_95"), with_ci(m, "ar_100"), with_ci(m, "ar_1000")])
    out += [table(["`crop_n_layers` (эфф. разрешение)", "сцены", "площадь", "рамок", "recall, IoU ≥ 0,5",
                   "recall, IoU ≥ 0,75", "recall, 0,5:0,95", "AR@100", "AR@1000"], rows), ""]

    out += ["## Таблица 4.2-3. Сопоставление с опубликованными AR@100 (все 160 сцен)", "",
            "Опубликованные числа — AR итоговых детекций (после порога сходства и сопоставления, ранжирование по сходству); "
            "наши — полнота предложений до этапа поиска: AR@100 — при ранжировании оценкой сегментатора, AR@1000 — без "
            f"усечения (масок на сцену больше 100). Величины разного рода; адреса чисел: {AR100_ADDRESS}.", ""]
    keys = [("all", "все", "all/all", "all"), ("easy", "easy", "all/easy", "all"), ("hard", "hard", "all/hard", "all"),
            ("small", "< 200²", "all/all", "small"), ("medium", "200²–400²", "all/all", "medium"),
            ("large", "> 400²", "all/all", "large")]
    rows = [[name] + ["—" if v[k] is None else num(v[k]) for k, *_ in keys] for name, v in PUBLISHED_AR100.items()]
    for cnl in map(str, cfg["crop_n_layers"]):
        for metric, label in (("ar_100", "AR@100"), ("ar_1000", "AR@1000")):
            rows.append([f"SAM 2, `crop_n_layers={cnl}`, {label} предложений"] +
                        [pct(res[cnl][subset]["by_area"][area][metric]) for _, _, subset, area in keys])
    out += [table(["метод"] + [k[1] for k in keys], rows), ""]

    lo, hi = sel["diff_ci95_bootstrap_scenes_reference_only"]
    out += ["## Выбор разрешения входа", "",
            f"Правило: {sel['rule']}. Recall при IoU ≥ {num(sel['iou'], 1)} на калибровочных сценах: "
            + "; ".join(f"`crop_n_layers={c}` — {pct(v)}" for c, v in sel["recall"].items())
            + f"; разность {pct(sel['diff_1_minus_0'])} п. п. (справочно, интервал 95 %: [{pct(lo, 1)}; {pct(hi, 1)}]). "
            f"**Выбрано `crop_n_layers={sel['selected_crop_n_layers']}`.** Тестовые сцены в выборе не участвуют.", ""]
    return "\n".join(out)


PCB_AREAS = [("small", "< 3 570"), ("medium", "3 570–5 250"), ("large", "≥ 5 250")]


def tables_4_2_pcb(rec: dict) -> str:
    """4.2 на PKU-Market-PCB: тестовые платы, без блока выбора; интервалов нет — снимки одной платы есть один шаблон
    с нанесёнными дефектами, бутстрэп по снимкам занизил бы разброс, а по платам интервалы в статусе «стресс-тест»
    не приводятся; разброс показан разбивкой по платам."""
    res, cfg, sel, pil, ref = (rec[k] for k in ("results", "config", "selection", "pilot_cal_reference",
                                                "reference_not_criterion"))
    cnls = list(map(str, cfg["crop_n_layers"]))
    cols = ["recall_50", "recall_75", "recall_50_95", "ar_100", "ar_1000"]
    head = ["recall, IoU ≥ 0,5", "recall, IoU ≥ 0,75", "recall, 0,5:0,95", "AR@100", "AR@1000"]
    out = [f"# 4.2. Предел полноты сегментатора — {rec['dataset']}", "",
           f"Источник чисел — `experiments/runs/{rec['run_id']}.json` (запись от {rec['written']}). "
           "Все величины — в процентах, по описывающим рамкам масок. Recall — по всем маскам $M(I)$, "
           "без предела их числа; AR@100 и AR@1000 — COCOeval без категорий, предложения ранжированы оценкой predicted "
           "IoU SAM 2, пороги IoU 0,5:0,95. Только тестовые платы; зоны игнорирования (неразмеченные дефекты, "
           f"{rec['n_ignore_zones']}) в полноту не входят. Интервалы не приводятся: снимки одной платы — один шаблон "
           "с нанесёнными дефектами, разброс показан разбивкой по платам.", ""]

    def rows_for(prefix: str, names: list[tuple[str, str]]) -> list[list[str]]:
        rows = []
        for cnl in cnls:
            for key, title in names:
                r = res[cnl][f"test/{prefix}{key}"]
                m = r["by_area"]["all"]
                rows.append([f"{cnl} ({EFFECTIVE[cnl]})", title, str(r["n_scenes"]), str(m["n_gt"]),
                             num(r["masks_per_scene_mean_median_max"][0], 1)] + [pct(m[c]) for c in cols])
        return rows

    boards = sorted(k.split("board_")[1] for k in res[cnls[0]] if k.startswith("test/board_"))
    types = [k.split("type_")[1] for k in res[cnls[0]] if k.startswith("test/type_")]
    first = ["`crop_n_layers` (эфф. разрешение)", None, "снимков", "рамок", "масок на снимок"] + head
    out += ["## Таблица 4.2-1. Полнота по разрешению входа: тестовые платы, по платам", "",
            table([c or "плата" for c in first],
                  rows_for("", [("all", "все тестовые")]) + rows_for("board_", [(b, b) for b in boards])), ""]
    out += ["## Таблица 4.2-2. Полнота по типу дефекта", "",
            table([c or "тип дефекта" for c in first], rows_for("type_", [(t, t) for t in types])), ""]

    out += ["## Таблица 4.2-3. Полнота по бинам площади рамки (пикс.² исходного снимка; справочно)", ""]
    rows = []
    for cnl in cnls:
        for a, a_name in PCB_AREAS:
            m = res[cnl]["test/all"]["by_area"][a]
            rows.append([f"{cnl} ({EFFECTIVE[cnl]})", a_name, str(m["n_gt"])] + [pct(m[c]) for c in cols])
    out += [table(["`crop_n_layers` (эфф. разрешение)", "площадь", "рамок"] + head, rows), ""]

    out += ["## Таблица 4.2-4. Рядом с пилотным замером на калибровочных платах", "",
            f"Пилот — стоп-критерий области ({pil['source']}): платы {', '.join(pil['boards'])}, {pil['n_boxes']} рамка, "
            f"`crop_n_layers={pil['crop_n_layers']}`, порог критерия {num(pil['stop_criterion_recall_min'], 1)}. "
            "Справочные столбцы — не критерий: recall при мягких порогах IoU и доля рамок, внутри которых есть маска, "
            "потому что рамка разметки свободнее дефекта.", ""]
    rows = [["пилот, калибровочные платы, `crop_n_layers=1`", str(pil["n_boxes"]), pct(pil["recall_50"]),
             pct(pil["recall_75"]), "—", "—", "—"]]
    for cnl in cnls:
        m, rf = res[cnl]["test/all"]["by_area"]["all"], ref[cnl]
        rows.append([f"4.2, тестовые платы, `crop_n_layers={cnl}`", str(m["n_gt"]), pct(m["recall_50"]), pct(m["recall_75"]),
                     pct(rf["recall_by_iou"]["0.1"]), pct(rf["recall_by_iou"]["0.25"]), pct(rf["share_with_mask_inside_box"])])
    out += [table(["замер", "рамок", "recall, IoU ≥ 0,5", "recall, IoU ≥ 0,75", "recall, IoU ≥ 0,1 (справочно)",
                   "recall, IoU ≥ 0,25 (справочно)", "маска внутри рамки (справочно)"], rows), ""]
    rows = [[t, str(pil["by_type"][t]["n"]), pct(pil["by_type"][t]["recall_050"])] +
            [x for cnl in cnls for x in (str(res[cnl][f"test/type_{t}"]["by_area"]["all"]["n_gt"]),
                                         pct(res[cnl][f"test/type_{t}"]["by_area"]["all"]["recall_50"]),
                                         pct(ref[cnl]["share_with_mask_inside_box_by_type"][t]))] for t in types]
    out += [table(["тип дефекта", "пилот: рамок", "пилот: recall, IoU ≥ 0,5"] +
                  [x for cnl in cnls for x in (f"4.2, `{cnl}`: рамок", f"4.2, `{cnl}`: recall, IoU ≥ 0,5",
                                               f"4.2, `{cnl}`: маска внутри рамки")], rows), ""]
    out += ["## Разрешение входа", "",
            f"Блок выбора отключён: {sel['reason'].replace('; ', ', ')}. **Действует `crop_n_layers={sel['carried_over_crop_n_layers']}`** "
            f"(`{sel['carried_over_from']}`). {pil['note'][0].upper() + pil['note'][1:]}.", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------- 4.3


def ci(m: dict, name: str, nd: int = 2) -> str:
    if m.get(name) is None:
        return "—"
    c = (m.get("ci95") or {}).get(name)
    return pct(m[name], nd) + ("" if not c else f" [{pct(c[0], 1)}; {pct(c[1], 1)}]")


def auc(m: dict) -> str:
    if m.get("auroc") is None:
        return "—"
    c = (m.get("ci95") or {}).get("auroc")
    return num(m["auroc"], 4) + ("" if not c else f" [{num(c[0], 3)}; {num(c[1], 3)}]")


def selection(recs: dict[str, dict], dataset: str) -> dict[str, dict]:
    """Лучший вариант каждого энкодера по `selection_cal`; пока посчитаны не все прогоны энкодера — выбора нет."""
    out = {}
    for enc in dict.fromkeys(e for e, _, d in grid() if d == dataset):
        names = [v for e, v, d in grid() if e == enc and d == dataset]
        have = {v: recs[f"{enc}_{v}_{dataset}"]["selection_cal"] for v in names if f"{enc}_{v}_{dataset}" in recs}
        out[enc] = {"variants": names, "have": have,
                    "best": RU.best_variant({v: {"ap": c["ap"], "ap50": c["ap50"]} for v, c in have.items()})
                    if len(have) == len(names) else None}
    return out


def categories_by_subset(dataset: str, rec: dict, base: dict[str, dict]) -> dict:
    """Число категорий с рамками по наборам сцен — по разметке; для тестовых подмножеств сверяется с записью журнала
    (в записи прогона сетки `n_categories` есть только у тестовых подмножеств)."""
    if dataset != "hr_insdet":
        raise NotImplementedError(dataset)
    from src.data import hr_insdet as H

    sp = json.loads(Path(f"splits/{dataset}.json").read_text())
    by_id = {s["id"]: s for s in H.scenes()}
    found = {role: {i: {g["category_id"] for g in H.scene_gt(by_id[i], sp["labels"])} for i in sp[role]} for role in ("cal", "test")}

    def union(role: str, level: str | None = None) -> set:
        return set().union(*(c for i, c in found[role].items() if level is None or by_id[i]["level"] == level))

    subsets = {"cal/all": len(union("cal")), "test/all": len(union("test")), "test/easy": len(union("test", "easy")),
               "test/hard": len(union("test", "hard"))}
    for n, m in rec["metrics"]["full"]["subsets"].items():
        if subsets[n] != m["by_area"]["all"]["n_categories"]:
            raise SystemExit(f"{n}: категорий по разметке {subsets[n]}, в записи {rec['run_id']} — {m['by_area']['all']['n_categories']}")
    cal, test = union("cal"), union("test")
    for r, brec in base.items():  # контрольные прогоны — все сцены: число категорий записи обязано совпасть с разметкой
        n_all = brec["metrics"]["full"]["subsets"]["all/all"]["by_area"]["all"]["n_categories"]
        if n_all != len(cal | test):
            raise SystemExit(f"all/all: категорий по разметке {len(cal | test)}, в записи {r} — {n_all}")
    return {"subsets": subsets, "all": len(cal | test), "only_cal": len(cal - test), "only_test": len(test - cal)}


def tables_4_3(dataset: str, recs: dict[str, dict], base: dict[str, dict]) -> str:
    order = [f"{e}_{v}_{d}" for e, v, d in grid() if d == dataset and f"{e}_{v}_{d}" in recs]
    first = recs[order[0]]
    n_grid = sum(d == dataset for _, _, d in grid())
    out = [f"# 4.3. Способ формирования эмбеддинга — {dataset}", "",
           f"Источник чисел — `experiments/runs/` (прогонов сетки посчитано {len(order)} из {n_grid}; "
           "контрольных прогонов — " + str(len(base)) + "). Все величины, кроме AUROC, — в процентах; в "
           f"квадратных скобках — интервал 95 %, бутстрэп по сценам ({first['config']['bootstrap']} повторов). Прогон сетки: "
           "оракульный отбор масок (взаимно-однозначный, IoU ≥ 0,5) и все дистракторы (IoU рамки < 0,1), точный перебор, "
           f"порог не применяется, AP — по рамкам масок с оценкой $s^*$ при `maxDets` = {first['config']['max_dets']}; "
           "метрики — по тестовым сценам.", ""]

    c = first["counts"]
    cats = categories_by_subset(dataset, first, base)
    rows = [[n, str(v["n_scenes"]), str(v["n_gt"]), str(cats["subsets"][n]), str(v["n_oracle"]),
             f"{v['n_segmenter_miss']} ({pct(v['n_segmenter_miss'] / v['n_gt'], 1)})",
             " / ".join(str(v["n_segmenter_miss_by_area"][a]) for a, _ in AREAS), str(v["n_distractors"]), str(v["n_masks"])]
            for n, v in c.items()]
    out += ["## Таблица 4.3-0. Состав оценки (один для всех прогонов сетки: маски и разметка от $\\varphi$ не зависят)", "",
            table(["сцены", "сцен", "рамок", "категорий с рамками", "оракульных масок", "пропусков сегментатора (доля рамок)",
                   "пропусков: < 200² / 200²–400² / > 400²", "дистракторов", "масок $M(I)$"], rows), "",
            "AP — среднее по категориям, имеющим рамки в наборе сцен, поэтому наборы категорий у трёх чисел разные: выбор "
            f"лучшего $\\varphi$ (`cal/all`) — {cats['subsets']['cal/all']}, метрики 4.3 (`test/all`) — {cats['subsets']['test/all']}, "
            f"контрольный прогон на всех сценах — {cats['all']}; только в калибровочных сценах встречаются {cats['only_cal']} "
            f"объектов, только в тестовых — {cats['only_test']}. Число категорий — по разметке сцен набора (истина — `src.data.hr_insdet.scene_gt`); для тестовых подмножеств и для всех сцен (контрольные прогоны) сверено с `n_categories` записей журнала; у калибровочных сцен этого поля в записях сетки нет.", ""]

    for proto, title in (("full", "полная галерея"), ("one_per_class", "один эталон на класс")):
        rows = []
        for r in order:
            rec = recs[r]
            m = rec["metrics"][proto]["subsets"]["test/all"]["by_area"]["all"]
            rows.append([rec["config"]["encoder"], rec["variant_label"], ci(m, "ap"), ci(m, "ap50"), ci(m, "ap75"),
                         pct(m["ap_maxdets_100"]), ci(m, "top1"), auc(m), rec.get("paired_with") or "—"])
        out += [f"## Таблица 4.3-1 ({proto}). Сетка кодирования, тестовые сцены, {title}", "",
                table(["энкодер", "$\\varphi$", "AP", "AP50", "AP75", "AP при `maxDets`=100", "top-1 на оракульных масках",
                       "AUROC по $s^*$", "общий проход с"], rows), ""]

    rows = []
    for r in order:
        rec = recs[r]
        for a, a_name in AREAS:
            m = rec["metrics"]["full"]["subsets"]["test/all"]["by_area"][a]
            rows.append([rec["config"]["encoder"], rec["variant_label"], a_name, str(m["n_gt"]), str(m["n_oracle"]),
                         ci(m, "ap"), ci(m, "top1"), auc(m)])
    out += ["## Таблица 4.3-2. По бинам площади рамки (пикс.² исходного снимка), тестовые сцены, полная галерея", "",
            "Бин AUROC: у оракульных масок — по площади размеченной рамки, у дистракторов — по площади собственной рамки.", "",
            table(["энкодер", "$\\varphi$", "площадь", "рамок", "оракульных масок", "AP", "top-1", "AUROC"], rows), ""]

    rows = []
    for r in order:
        rec = recs[r]
        for lv in ("easy", "hard"):
            m = rec["metrics"]["full"]["subsets"][f"test/{lv}"]["by_area"]["all"]
            rows.append([rec["config"]["encoder"], rec["variant_label"], lv, str(rec["metrics"]["full"]["subsets"][f"test/{lv}"]["n_scenes"]),
                         str(m["n_gt"]), ci(m, "ap"), ci(m, "ap50"), ci(m, "top1"), auc(m)])
    out += ["## Таблица 4.3-3. По уровню трудности, тестовые сцены, полная галерея", "",
            table(["энкодер", "$\\varphi$", "уровень", "сцен", "рамок", "AP", "AP50", "top-1", "AUROC"], rows), ""]

    sel = selection(recs, dataset)
    rows = []
    for enc, s in sel.items():
        for v, c_ in s["have"].items():
            rows.append([enc, VARIANTS[v].label, ci(c_, "ap"), ci(c_, "ap50"), "**лучший**" if v == s["best"] else ""])
    out += ["## Таблица 4.3-4. Выбор лучшего $\\varphi$ энкодера — только по калибровочным сценам", "",
            f"Правило выбора: наибольший AP с оракулом при полной галерее на "
            f"{first['selection_cal']['n_scenes']} калибровочных сценах, а в пределах {num(RU.SELECT_TIE_AP, 1)} п. — больший AP50, "
            "затем вариант, стоящий выше в таблице 2.1. Выбор объявляется, когда посчитаны все прогоны энкодера: "
            + ", ".join(f"{enc} — {len(s['have'])} из {len(s['variants'])}" + (f", лучший — {VARIANTS[s['best']].label}" if s["best"] else "")
                        for enc, s in sel.items()) + ".", "",
            table(["энкодер", "$\\varphi$", "AP, калибровочные сцены", "AP50", ""], rows), ""]

    if base:
        out += ["## Таблица 4.3-5. Контрольные прогоны по протоколу baseline (без оракула: все маски $M(I)$, независимые решения)", "",
                "AP — по всем детекциям с оценкой $s^*$, без порога. 160 сцен — набор опубликованного числа, 120 — тестовые "
                "сцены метрик 4.3. Перечень отличий от [5] — в `4_6_hr_insdet.md`: число рядом с 41,61 — не воспроизведение. "
                "Бины площади у нас — по площади размеченной рамки в пикселях исходного снимка; система координат бинов в [5] и [8] не названа.", ""]
        rows = [[name, "160", "полная", "—"] + [num(v["all"])] + ["—"] * 4 + [num(v[a]) for a, _ in AREAS] for name, v in PUBLISHED_AP.items()]
        role = lambda rec: ("ближайшая к [5] конфигурация" if rec["grid_run"] == RU.FIRST_RUN else "") + (  # noqa: E731
            "; лучший φ энкодера — число метода" if sel[rec["config"]["encoder"]]["best"] == rec["config"]["variant"] else "")
        for r, rec in base.items():
            for proto in rec["metrics"]:
                for n_sc, key in (("160", "all"), ("120", "test")):
                    sub = rec["metrics"][proto]["subsets"]
                    m = sub[f"{key}/all"]["by_area"]
                    rows.append([f"{rec['config']['encoder']}, {rec['variant_label']}" + (f" ({role(rec).strip('; ')})" if role(rec) else ""), n_sc, proto,
                                 pct(rec["ap_without_part_mask_labels"][proto][f"{key}/all"]["ap"]), ci(m["all"], "ap"),
                                 ci(m["all"], "ap50"), ci(m["all"], "ap75"), pct(sub[f"{key}/easy"]["by_area"]["all"]["ap"]),
                                 pct(sub[f"{key}/hard"]["by_area"]["all"]["ap"])] + [pct(m[a]["ap"]) for a, _ in AREAS])
        out += [table(["метод", "сцен", "галерея", f"AP без меток {', '.join(x[:3] for x in RU.PART_MASK_LABELS)} (справочно)", "AP",
                       "AP50", "AP75", "AP, easy", "AP, hard"] + [f"AP, {n}" for _, n in AREAS], rows), ""]
        for r, rec in base.items():
            chk = rec.get("first_run_check")
            if chk:
                out += [f"Контроль на входе — {rec['config']['encoder']}, {rec['variant_label']}, "
                        f"160 сцен, полная галерея: AP {num(chk['ap'])} при опубликованных {num(chk['published_ap'])}, разница "
                        + f"{chk['diff']:+.2f}".replace(".", ",") + f" п.; порог {num(chk['ap_min'], 0)} AP — "
                        f"{'выдержан' if chk['ap_ge_min'] else 'НЕ выдержан'}, допуск ±{num(chk['diff_max'], 0)} п. — "
                        f"{'выдержан' if chk['diff_within_max'] else 'НЕ выдержан'}."
                        + ("" if chk["diff_within_max"] or not RU.FIRST_RUN_DIFF_RESOLUTION or "post_hoc_review" not in rec else
                           f" Разбор по перечню, записанному до прогона (блок `post_hoc_review` записи): ошибок не найдено; "
                           f"итог разбора — {RU.FIRST_RUN_DIFF_RESOLUTION}."), ""]

        if all(sel[e]["best"] for e in ("dinov2", "dinov3")) and f"dinov3_p_perp_{dataset}" in recs:
            boot = lambda rid: recs[rid]["bootstrap_ap"]["full"]["values"]  # noqa: E731
            b2, b3 = f"dinov2_{sel['dinov2']['best']}_{dataset}", f"dinov3_{sel['dinov3']['best']}_{dataset}"
            w = RU.weak_dinov3(boot(b2), boot(b3), boot(f"dinov3_p_perp_{dataset}"))
            fmt = lambda d: f"[{_signed(100 * d['ci95'][0])}; {_signed(100 * d['ci95'][1])}]"  # noqa: E731
            out += ["Критерий «слабого результата» DINOv3 (записан до первого прогона; `rules.weak_dinov3`; 120 тестовых сцен, оракул, "
                    "полная галерея, парные разности по общим повторам бутстрэпа): интервал 95 % разности AP(лучший $\\varphi$ DINOv2, "
                    f"{recs[b2]['variant_label']}) − AP(лучший $\\varphi$ DINOv3, {recs[b3]['variant_label']}) — "
                    f"{fmt(w['best_dinov2_minus_best_dinov3'])} п. ({'целиком выше нуля' if w['best_dinov2_minus_best_dinov3']['above_zero'] else 'не выше нуля целиком'}); "
                    f"разности с $P^\\perp$ на DINOv3 — {fmt(w['best_dinov2_minus_p_perp_dinov3'])} п.; критерий "
                    f"{'СРАБОТАЛ' if w['weak'] else 'не сработал'}.", ""]

        keys = [("all", "все", "all/all", "all"), ("easy", "easy", "all/easy", "all"), ("hard", "hard", "all/hard", "all"),
                ("small", "< 200²", "all/all", "small"), ("medium", "200²–400²", "all/all", "medium"), ("large", "> 400²", "all/all", "large")]
        rows = [[name, "—", "—"] + ["—" if v[k] is None else num(v[k]) for k, *_ in keys] for name, v in PUBLISHED_AR100.items()]
        for r, rec in base.items():
            for vname, v in rec["metrics"]["full"]["subsets"]["all/all"]["ar_final_detections"].items():
                star = " **(для 4.6)**" if {"use_cats": v["use_cats"], "score_threshold": v["score_threshold"]} == RU.AR_FOR_4_6 else ""
                for md in RU.AR_MAX_DETS:
                    rows.append([f"{rec['config']['encoder']}, {rec['variant_label']}: {'с учётом' if v['use_cats'] else 'без учёта'} категорий, "
                                 f"{'без порога' if v['score_threshold'] is None else 'порог ' + num(v['score_threshold'], 1)}{star}", f"AR@{md}",
                                 "160"] + [pct(rec["metrics"]["full"]["subsets"][s_]["ar_final_detections"][vname]["by_area"][a_][f"ar_{md}"])
                                           for _, _, s_, a_ in keys])
        out += ["## Таблица 4.3-6. AR итоговых детекций контрольного прогона (полная галерея, 160 сцен) и опубликованные AR@100", "",
                "Детекции ранжированы по $s^*$; пороги IoU 0,5:0,95 (в [5] названо «от 0,5 до 1,0»). Опубликованные числа — AR@100 "
                f"итоговых детекций (после порога сходства и сопоставления); адреса чисел: {AR100_ADDRESS}. Порог 0,4 взят по [8] на "
                "шкале сходств CLS вырезки DINOv2; у усреднения под маской и у DINOv3 шкала другая, и строки с порогом для них несодержательны.", "",
                table(["метод и вариант", "величина", "сцен"] + [k[1] for k in keys], rows), ""]

    best_rows = []
    for enc, s in sel.items():
        if s["best"]:
            rec = recs[f"{enc}_{s['best']}_{dataset}"]
            for proto in rec["metrics"]:
                w = rec["ap_without_part_mask_labels"][proto]["test/all"]
                m = rec["metrics"][proto]["subsets"]["test/all"]["by_area"]["all"]
                best_rows.append([enc, rec["variant_label"], proto, ci(m, "ap"), ci(w, "ap"), pct(w["ap50"]), pct(w["ap75"])])
    if best_rows:
        out += [f"## Таблица 4.3-7. Лучший $\\varphi$ энкодера: AP без меток {', '.join(RU.PART_MASK_LABELS)} (справочно; тестовые сцены, оракул)", "",
                table(["энкодер", "$\\varphi$", "галерея", "AP, 100 меток", "AP, 98 меток", "AP50, 98 меток", "AP75, 98 меток"], best_rows), ""]
    return "\n".join(out)


# ---------------------------------------------------------------------- 4.4 — отбор гранулярности

RHO_ENCODE_CHECK = Path("experiments/rho_encode_check.json")
SUBSETS_4_4 = [("test/all", "120 тестовых"), ("test/easy", "тестовые easy"), ("test/hard", "тестовые hard"),
               ("all/all", "160 (набор [5])"), ("cal/all", "40 калибровочных")]


def _diff(d: dict) -> str:
    sign = lambda v: f"{100 * v:+.2f}".replace(".", ",")  # noqa: E731
    return sign(d["diff"]) + (f" [{sign(d['ci95'][0])}; {sign(d['ci95'][1])}]" if "ci95" in d else "")


def tables_4_4(recs: dict[str, dict]) -> str:
    """Таблицы 4.4 по записям `4_4_<run_id>.json`; сводный исход — `rules.rho_outcome_joint`."""
    from src.eval import rules as RU

    order = [r for r in RU.RHO_RUNS if f"4_4_{r}" in recs]
    rs = [recs[f"4_4_{r}"] for r in order]
    name = lambda r: f"{r['config']['encoder']}, {r['variant_label']}"  # noqa: E731
    complete = len(rs) == len(RU.RHO_RUNS)  # сводный исход — только по обоим энкодерам
    joint = RU.rho_outcome_joint({r["config"]["encoder"]: r["outcome"]["code"] for r in rs}) if complete else None
    L = ["# Эксперимент 4.4 — отбор гранулярности ρ, HR-InsDet", "",
         "Источник чисел — `experiments/runs/4_4_*.json`. AP — без порога τ, по "
         "сохранённым эмбеддингам всех масок M(I) прогонов лучших φ; разности — парные, по общим повторам бутстрэпа по сценам; "
         "в скобках — 95 % перцентильный интервал. Параметры ρ (θ = 0,90, γ = 0,80) и η заморожены и по исходу не меняются.", "",
         "## Таблица 4.4-1 — главное сравнение: ρ против контрольного прогона без отбора", "",
         table(["энкодер, φ", "галерея", "сцены", "AP без отбора", "AP с ρ", "ΔAP, п.", "ΔAP50, п.", "ΔAP75, п."],
               [[name(r), g, lab, ci(r["baseline_check"]["metrics"][g]["subsets"][n]["by_area"]["all"], "ap"),
                 ci(r["rho"]["metrics"][g]["subsets"][n]["by_area"]["all"], "ap"),
                 *(_diff(r["paired_diff"][g][n][m]) for m in ("ap", "ap50", "ap75"))]
                for r in rs for g in RU.RHO_GALLERIES for n, lab in SUBSETS_4_4[:4]]), "",
         "Исход выносится по первой строке каждого энкодера (120 тестовых сцен, полная галерея, AP). Уровни easy / hard — "
         "точечные оценки, без интервалов.", ""]
    for r in rs:
        o = r["outcome"]
        L.append(f"- **{name(r)}: исход ({o['code']})** — {o['text']}. ΔAP = {_diff(o).replace(' [', ', интервал от ').replace('; ', ' до ').rstrip(']')} п., ширина интервала "
                 f"{num(100 * o['ci95_width'])} п., доля некодируемых масок на тестовых сценах — {pct(o['share_not_encoded_test'], 1)} %.")
    L += ["", "**Сводный исход:** " + ("не выводится — есть запись только одного энкодера" if joint is None else
                                   f"общий — ({joint['code']}) {joint['text']}" if joint["same"] else joint["text"]) + ".", "",
          "## Таблица 4.4-2 — число масок, полнота после отбора и кратные детекции", ""]
    r0 = rs[0]  # от энкодера зависят только кратные детекции с учётом ŷ
    L += [table(["сцены", "масок M(I)", "масок M*", "не кодируется, %", "recall@0,5 M(I) → M*", "recall@0,75", "recall 0,5:0,95",
                 "рамок с ≥ 2 масками, % M(I) → M*", "время ρ, с"],
                [[lab, str(d["n_masks"]), str(d["n_selected"]), pct(d["share_not_encoded"], 1),
                  *(f"{pct(d['recall']['M(I)'][k])} → {pct(d['recall']['M*'][k])}" for k in ("recall_50", "recall_75", "recall_50_95")),
                  f"{pct(d['multiple_detections']['M(I)']['share_gt_with_2plus_masks'], 1)} → "
                  f"{pct(d['multiple_detections']['M*']['share_gt_with_2plus_masks'], 1)}", num(d["rho_sec"], 2)]
                 for n, lab in SUBSETS_4_4 for d in [r0["rho"]["by_subset"][n]]]), "",
          "Кратные детекции с учётом ŷ (доля рамок с двумя и больше детекциями своей метки при IoU ≥ 0,5, справочно), "
          "120 тестовых сцен: " + ", ".join(
              f"{name(r)} — {pct(d['M(I)']['share_gt_with_2plus_detections_of_its_label'], 1)} → "
              f"{pct(d['M*']['share_gt_with_2plus_detections_of_its_label'], 1)} %"
              for r in rs for d in [r["rho"]["by_subset"]["test/all"]["multiple_detections"]]) + ".", ""]
    if RHO_ENCODE_CHECK.is_file():
        chk = json.loads(RHO_ENCODE_CHECK.read_text())
        L += ["Время кодирования — отдельный замер на 40 калибровочных сценах (`experiments/rho_encode_check.json`), не прогон "
              "сетки, тестовые сцены не кодировались:", "",
              table(["энкодер, φ", "M(I): масок", "с", "масок/с", "M*: масок", "с", "масок/с", "отношение времени",
                     "медиана косинуса M* с сохранённым, по бинам", "max |Δs*|", "изменилось ŷ"],
                    [[f"{e}, {c['variant_label']}", str(c["timing"]["all"]["n_masks"]), num(c["timing"]["all"]["sec"], 1),
                      num(c["timing"]["all"]["masks_per_sec"], 2), str(c["timing"]["rho"]["n_masks"]), num(c["timing"]["rho"]["sec"], 1),
                      num(c["timing"]["rho"]["masks_per_sec"], 2), num(c["time_ratio_rho_to_all"], 3),
                      " / ".join(num(v["cos_median"], 6) for v in c["rho_vs_saved"]["by_area_bin"].values()),
                      f"{c['rho_vs_saved']['max_abs_delta_s_star']:.1e}".replace(".", ","), str(c["rho_vs_saved"]["n_y_hat_changed"])]
                     for e, c in chk["by_encoder"].items()]), ""]
    L += ["## Таблица 4.4-3 — ориентиры отбора после поиска (кодируются все маски M(I)); вердикта нет", "",
          f"`chain_max` — наибольшее s* среди сравнимых по вложенности масок (θ = 0,90); `box_nms` — NMS по описывающим рамкам, порог "
          f"{num(rs[0]['post_search']['params']['post_nms_iou'], 1)} — константа по [13], без подбора. Полная галерея, 120 тестовых сцен.", "",
          table(["энкодер, φ", "отбор", "детекций (160 сцен)", "AP", "AP50", "AP75", "рамок с ≥ 2 масками, %", "ΔAP к ρ, п."],
                [row for r in rs for row in (
                    [[name(r), "без отбора", str(r["baseline_check"]["metrics"]["full"]["n_detections"]),
                      *(pct(r["baseline_check"]["metrics"]["full"]["subsets"]["test/all"]["by_area"]["all"][m]) for m in ("ap", "ap50", "ap75")),
                      pct(r["rho"]["by_subset"]["test/all"]["multiple_detections"]["M(I)"]["share_gt_with_2plus_masks"], 1), "—"],
                     [name(r), "ρ (до поиска)", str(r["rho"]["metrics"]["full"]["n_detections"]),
                      *(pct(r["rho"]["metrics"]["full"]["subsets"]["test/all"]["by_area"]["all"][m]) for m in ("ap", "ap50", "ap75")),
                      pct(r["rho"]["by_subset"]["test/all"]["multiple_detections"]["M*"]["share_gt_with_2plus_masks"], 1), "—"]]
                    + [[name(r), rule, str(r["post_search"][rule]["n_detections"]),
                        *(pct(r["post_search"][rule]["ap"]["test/all"][m]) for m in ("ap", "ap50", "ap75")),
                        pct(r["post_search"][rule]["multiple_detections"]["test/all"]["share_gt_with_2plus_masks"], 1),
                        _diff(r["post_search"][rule]["paired_diff_minus_rho"]["test/all"]["ap"])] for rule in RU.POST_SEARCH])]), "",
          "## Таблица 4.4-4 — избыточность галереи при ρ: полная против дедуплицированной; вердикта нет", "",
          table(["энкодер, φ", "η", "N", "доля от полной, %", "n_max", "k", "AP, 120 тестовых", "ΔAP, п.", "q0,95 / q0,99 s* дистракторов (калибр.)",
                 "точный перебор, мс на запрос", "HNSW, мс на запрос", "полнота HNSW (1 поток; 30 построений: мин–макс)"],
                [[name(r), num(d["eta"]) if proto == "dedup" else "—", str(d["gallery"][proto]["N"]),
                  pct(d["gallery"][proto]["N"] / d["gallery"]["full"]["N"], 1), str(d["gallery"][proto]["n_max"]),
                  str(d["gallery"][proto]["k"]), ci(d["ap"][proto]["test/all"], "ap"),
                  _diff(d["paired_diff_dedup_minus_full"]["test/all"]["ap"]) if proto == "dedup" else "—",
                  " / ".join(num(v, 4) for v in d["distractor_s_star_quantiles_cal"][proto]),
                  *(num(v["single_query"]["ms_per_call_median"], 3)
                    for k, v in d["search_latency"][proto]["threads_default"].items() if isinstance(v, dict)),
                  (f"{num(h['single_thread']['recall'], 5)}; {num(h['multi_thread']['min'], 5)}–{num(h['multi_thread']['max'], 5)}"
                   if proto == "dedup" else "— (этап grid, по M(I))")]
                 for r in rs for d in [r["dedup"]] for h in [d["hnsw_recall_dedup"]] for proto in ("full", "dedup")]), "",
          f"Запросы задержки и полноты HNSW — маски M* 40 калибровочных сцен; критерий полноты замера HNSW — {num(rs[0]['dedup']['hnsw_recall_dedup']['criterion_recall_min'], 3)}, "
          "здесь приводится описательно. Квантили s* — только по дистракторам калибровочных сцен; порог для дедуплицированной "
          "галереи в 4.4 не ставится (точка с калибровкой — в 4.5).", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------- 4.5

RULE_LABELS_4_5 = {"quantile_recal": "квантиль с пересчётом", "model": "по модели", "quantile_frozen": "квантиль без пересчёта",
                   "fixed": "фиксированный 0,4", "top_k": "top-K (K = 100)"}


def _mm(vals, nd: int = 2, as_pct: bool = True) -> str:
    """Медиана по цепочкам и размах: «медиана (мин–макс)»; пропуски (κ вне отрезка) считаются отдельно."""
    import numpy as np

    v = [x for x in vals if x is not None]
    if not v:
        return "—"
    f = (lambda x: pct(x, nd)) if as_pct else (lambda x: num(x, nd))
    out = f(float(np.median(v))) if min(v) == max(v) else f"{f(float(np.median(v)))} ({f(min(v))}–{f(max(v))})"
    return out + (f" [нет у {len(vals) - len(v)}]" if len(v) != len(vals) else "")


def _med(vals, nd: int = 2) -> str:
    """Только медиана по цепочкам, в процентах (печатная сводка 4.5-3m)."""
    import numpy as np

    v = [x for x in vals if x is not None]
    return "—" if not v else pct(float(np.median(v)), nd)


def _cells_4_5(rec: dict, pair: str, eps: str, rule: str, field: str) -> list:
    return [(rec["cells"][r][pair][eps][rule] or {}).get(field) for r in sorted(rec["cells"], key=int)]


def _pairs_4_5(rec: dict) -> list[str]:
    return list(rec["cells"]["0"])


def _common_tables_4_5(r: dict, unit: str, with_holds: bool) -> list[str]:
    from src.eval import rules as RU

    eps_levels = list(r["cells"]["0"][_pairs_4_5(r)[0]])
    L = []
    for e in eps_levels:
        L += [f"**ε = {e.replace('.', ',')}.** Частота ложных срабатываний на дистракторах, %: медиана по 9 цепочкам (мин–макс)"
              + ("; в скобках после — число цепочек, где `eps_holds` истинно" if with_holds else "") + ".", "",
              table([f"{unit}: калибровка → галерея", *(RULE_LABELS_4_5[x] for x in RU.REJ_RULES)],
                    [[pk.replace("->", " → "), *(
                        _mm(_cells_4_5(r, pk, e, x, "fpr")) + (f" ({sum(bool(h) for h in _cells_4_5(r, pk, e, x, 'eps_holds'))}/9)"
                                                                 if with_holds else "") for x in RU.REJ_RULES)]
                     for pk in _pairs_4_5(r)]), "",
              "Доля пропусков на известных масках, % (медиана по цепочкам, мин–макс):", "",
              table([f"{unit}: калибровка → галерея", *(RULE_LABELS_4_5[x] for x in RU.REJ_RULES)],
                    [[pk.replace("->", " → "), *(_mm(_cells_4_5(r, pk, e, x, "miss_rate")) for x in RU.REJ_RULES)]
                     for pk in _pairs_4_5(r)]), "",
              "Доля верных ответов на известных масках, % (ŷ ≠ ∅ и метка верна):", "",
              table([f"{unit}: калибровка → галерея", *(RULE_LABELS_4_5[x] for x in RU.REJ_RULES)],
                    [[pk.replace("->", " → "), *(_mm(_cells_4_5(r, pk, e, x, "correct_rate")) for x in RU.REJ_RULES)]
                     for pk in _pairs_4_5(r)]), ""]
    return L


def _calib_rows_4_5(r: dict, sizes: list[str]) -> list[list[str]]:
    eps_levels = list(r["cells"]["0"][_pairs_4_5(r)[0]])
    chains = sorted(r["calibrations"], key=int)
    rows = []
    for n in sizes:
        for e in eps_levels:
            c = [r["calibrations"][k][n]["by_eps"][e] for k in chains]
            rows.append([n, str(r["calibrations"][chains[0]][n]["N0"]), e.replace(".", ","), _mm([x["tau_q"] for x in c], 4, False),
                         _mm([x["kappa_N0"] for x in c], 1, False), str(sum(x["kappa_out_of_range"] for x in c)),
                         f"{c[0]['check_at_N0']['n_ge_tau']} / {c[0]['check_at_N0']['k_trigger']}"])
    return rows


def _calib_table_4_5(r: dict, sizes: list[str]) -> list[str]:
    return [table(["объём", "N₀", "ε", "τ_q", "κN₀", "κ вне [10⁻³; 1], цепочек", "масок D_cal над τ_q / тревога с"],
                  _calib_rows_4_5(r, sizes)), ""]


def _assumption_tables_4_5(named: list[tuple[str | None, dict]]) -> list[str]:
    """Проверка допущений и условие перекалибровки; при нескольких записях (HR-InsDet — два энкодера) — общая таблица со столбцом
    «энкодер, φ», при одной (PCB) — без него."""
    lab = [n for n, _ in named if n is not None]
    r0 = named[0][1]
    chains = sorted(r0["assumptions"]["new_exemplar_tail"], key=int)
    pairs = list(r0["assumptions"]["new_exemplar_tail"][chains[0]])
    levels = [k for k in r0["assumptions"]["new_exemplar_tail"][chains[0]][pairs[0]] if k != "n_new_rows"]
    head = ["энкодер, φ"] if lab else []
    L = ["Хвост пар «дистрактор — новый эталон» против хвоста при калибровке: отношение долей пар не ниже уровня (1 — допущение "
         "об общем хвосте F выполнено); медиана по цепочкам (мин–макс).", "",
         table([*head, "калибровка → галерея", *levels],
               [[*([n] if lab else []), pk.replace("->", " → "),
                 *(_mm([(r["assumptions"]["new_exemplar_tail"][k][pk][lv] or {}).get("ratio") for k in chains], 2, False) for lv in levels)]
                for n, r in named for pk in pairs]), ""]
    eps_levels = [k for k in r0["recalibration"]["by_chain"][chains[0]][pairs[0]] if k not in ("N0", "N", "growth_requires_recalibration")]
    L += ["Условие перекалибровки: контрольная проверка порога τ_q(n₀) по всей D_cal при выросшей галерее — число цепочек с "
          "тревогой; масок над порогом — медиана (мин–макс) при тревоге с k.", "",
          table([*head, "калибровка → галерея", "N/N₀ > 1 + δ",
                 *(f"ε = {e.replace('.', ',')}: тревога, цепочек; масок над порогом / k" for e in eps_levels)],
                [[*([n] if lab else []), pk.replace("->", " → "),
                  "да" if r["recalibration"]["by_chain"][chains[0]][pk]["growth_requires_recalibration"] else "нет",
                  *(f"{sum(r['recalibration']['by_chain'][k][pk][e]['recalibrate'] for k in chains)}/9; "
                    f"{_mm([r['recalibration']['by_chain'][k][pk][e]['n_ge_tau'] for k in chains], 0, False)} / "
                    f"{r['recalibration']['by_chain'][chains[0]][pk][e]['k_trigger']}" for e in eps_levels)]
                 for n, r in named for pk in pairs]), ""]
    return L


def tables_4_5(recs: dict[str, dict]) -> str:
    """Таблицы 4.5 на HR-InsDet по записям `4_5_<run_id>.json`; общий вывод — `rules.rejection_outcome_joint`."""
    from src.eval import rules as RU

    order = [x for x in RU.REJ_RUNS["hr_insdet"] if f"4_5_{x}" in recs]
    rs = [recs[f"4_5_{x}"] for x in order]
    name = lambda r: f"{r['config']['encoder']}, {r['variant_label']}"  # noqa: E731
    L = ["# Эксперимент 4.5 — отклонение и калибровка порога, HR-InsDet", "",
         "Источник чисел — `experiments/runs/4_5_*_hr_insdet.json`. Полная галерея, "
         "подвыборки 25 / 50 / 100 экземпляров (по 24 эталона), 9 вложенных цепочек; маски M*; порог — по дистракторам M* 40 "
         "калибровочных сцен, измерение — на дистракторах 120 тестовых сцен. «Правило удерживает ε»: нижняя граница 95 % "
         "бутстрэп-интервала частоты ложных срабатываний не выше ε(1+δ), δ = 0,1; в цепочке — во всех трёх парах роста; в целом — не "
         "меньше чем в 5 цепочках из 9. Протокол, трактовка исходов и константы записаны до прогона.", "",
         "## Таблица 4.5-1 — исход по сочетаниям энкодера и ε", "",
         table(["энкодер, φ", "ε", "по модели, цепочек", "квантиль с пересчётом", "квантиль без пересчёта", "фиксированный 0,4", "top-K",
                "исход (уточнённая формулировка 21.09)", "исход (формулировка 12.09)", "квантиль с пересчётом при N = N₀ (перенос): экземпляров — итог (цепочек)"],
               [[name(r), e.replace(".", ","), *(f"{o['n_chains_hold'][x]}/9" for x in ("model", "quantile_recal", "quantile_frozen", "fixed", "top_k")),
                 f"({o['refined_21sep']['code']}) {o['refined_21sep']['text']}",
                 f"({o['as_written_12sep']['code']}) {o['as_written_12sep']['text']}" if o["as_written_12sep"]["code"] else o["as_written_12sep"]["text"],
                 "; ".join(f"{n}: " + ("удержала" if v["holds"] else "не удержала") + (f" ({v['n_chains_hold']}/9)" if "n_chains_hold" in v else "")
                           for n, v in o["fails_at_n0"]["by_n0"].items())]
                for r in rs for e, o in r["outcome"].items()]), ""]
    if len(rs) == len(RU.REJ_RUNS["hr_insdet"]):
        j = RU.rejection_outcome_joint({f"{r['config']['encoder']}, ε = {e}": o["refined_21sep"]["code"] for r in rs for e, o in r["outcome"].items()})
        L += ["**Общий вывод:** " + (f"(а) — {j['text']}" if j["all_a"] else "исход называется по сочетаниям энкодера и ε (общий (а) — только "
                                     "при (а) во всех четырёх): " + "; ".join(f"{k} — ({v})" for k, v in j["by_combination"].items())) + ".", ""]
    else:
        L += ["**Общий вывод:** не выводится — есть запись только одного энкодера.", ""]
    L += ["Исход определяют только правило по модели и правило по квантили с пересчётом; контрольные варианты приводятся рядом. "
          "Классификация уточнена (формулировка 21.09 против первоначальной 12.09) после калибровки порога на калибровочных "
          "сценах и до любых частот ложных срабатываний на тестовых; поэтому исход приведён по обеим формулировкам. Провал уже при N = N₀ читается как провал "
          "переноса калибровочного набора (40 сцен уровня easy) на тестовые сцены, а не правила роста.", ""]
    # Дальше — таблицы 4.5-2…4.5-9: величины, общие для обоих энкодеров, сведены в одну таблицу со столбцом «энкодер, φ»;
    # блок частот, пропусков и верных ответов — у каждого энкодера свой (4.5-3 (dinov2), 4.5-3 (dinov3)).
    named = [(name(r), r) for r in rs]
    L += ["## Таблица 4.5-2 — калибровки подвыборок (медиана по цепочкам, мин–макс)", ""]
    for n_, r in named:
        c = r["composition"]
        L += [f"Состав, {n_}: D_cal — {c['n_D_cal']} масок; 120 тестовых сцен — {c['n_masks_M_star_test']} масок M* (на сцену: мин "
              f"{c['n_masks_M_star_per_scene_min_median_max'][0]}, медиана {num(c['n_masks_M_star_per_scene_min_median_max'][1], 1)}, макс "
              f"{c['n_masks_M_star_per_scene_min_median_max'][2]}; сцен, где масок не больше K = 100, — {c['n_scenes_with_M_star_le_top_k']}: "
              f"там top-K отвечает на все маски), дистракторов {c['n_distractors_test']}, оракульных масок {c['n_oracle_masks_test']}. "
              "Калибровка при 100 экземплярах совпала с `calib.json` точно.", ""]
    L += [table(["энкодер, φ", "объём", "N₀", "ε", "τ_q", "κN₀", "κ вне [10⁻³; 1], цепочек", "масок D_cal над τ_q / тревога с"],
                [[n_, *row] for n_, r in named for row in _calib_rows_4_5(r, ["25", "50", "100"])]), ""]
    for n_, r in named:
        L += [f"## Таблица 4.5-3 ({r['config']['encoder']}) — {n_}: частота ложных срабатываний, доля пропусков и доля верных ответов", "",
              *_common_tables_4_5(r, "экземпляров", True)]
    chains = sorted(rs[0]["cells"], key=int)
    eps_levels = list(rs[0]["outcome"])
    # Сводка таблиц 4.5-3 до медиан: полные таблицы по цепочкам — 4.5-3 выше.
    L += ["## Таблица 4.5-3m — частота ложных срабатываний и доля пропусков по правилам: медиана по 9 цепочкам, % (сводка таблиц 4.5-3; полные таблицы с размахом и числом согласных цепочек — `4_5_hr_insdet.md`)", "",
          table(["энкодер, φ", "ε", "экземпляров: калибровка → галерея", *(RULE_LABELS_4_5[x] for x in RU.REJ_RULES)],
                [[n_, e.replace(".", ","), pk.replace("->", " → "),
                  *(f"{_med(_cells_4_5(r, pk, e, x, 'fpr'))} / {_med(_cells_4_5(r, pk, e, x, 'miss_rate'))}" for x in RU.REJ_RULES)]
                 for n_, r in named for e in eps_levels for pk in _pairs_4_5(r)]), "",
          "В ячейке — частота ложных срабатываний на дистракторах / доля пропусков на известных масках; медиана по 9 цепочкам, без размаха.", ""]
    L += ["## Таблица 4.5-4 — рост частоты ложных срабатываний с объёмом галереи у порога, не зависящего от галереи (тезис §2.4)", "",
          table(["энкодер, φ", "ε", "правило", "FPR при 25, %", "при 50, %", "при 100, %", "отношение 100 / 25", "предсказано моделью"],
                [[n_, e.replace(".", ","), RULE_LABELS_4_5[x],
                  *(_mm([r["cells"][k][pk][e][x].get("fpr") for k in chains]) for pk in (("25->25", "25->50", "25->100") if x == "quantile_frozen" else ("25->25", "50->50", "100->100"))),
                  _mm([(r["cells"][k]["25->100"][e][x]["fpr"] / r["cells"][k]["25->25"][e][x]["fpr"]) if r["cells"][k]["25->25"][e][x].get("fpr") else None
                       for k in chains], 2, False),
                  num(r["cells"]["0"]["25->100"][e][x].get("predicted_ratio")) if x == "quantile_frozen" else "—"]
                 for n_, r in named for e in eps_levels for x in ("fixed", "quantile_frozen")]), "",
          "У фиксированного порога и у квантили без пересчёта порог один при всех объёмах (у второй — поставлен при 25 "
          "экземплярах); предсказание модели — (1 − (1 − ε)^{N/N₀})/ε. Медиана по 9 цепочкам (мин–макс).", "",
          "## Таблица 4.5-5 — объекты, чьих меток в подвыборке нет (без вердикта; граница применимости)", "",
          table(["энкодер, φ", "экземпляров в галерее", "ε", *(RULE_LABELS_4_5[x] for x in RU.REJ_RULES), "масок, медиана"],
                [[n_, n, e.replace(".", ","), *(_mm([(r["cells"][k][f"{n}->{n}"][e][x].get("unseen_objects") or {}).get("answered_rate") for k in chains])
                                                 for x in RU.REJ_RULES),
                  _mm([r["cells"][k][f"{n}->{n}"][e]["fixed"]["unseen_objects"]["n"] for k in chains], 0, False)]
                 for n_, r in named for n in ("25", "50") for e in eps_levels]), "",
          "Доля масок таких объектов, получивших ответ ŷ ≠ ∅, %; при 100 экземплярах таких масок нет.", "",
          "## Таблица 4.5-6 — AUROC «известный — неизвестный» по s* (маски M*)", "",
          table(["энкодер, φ", "экземпляров", "AUROC (медиана по цепочкам, мин–макс)", "известных масок, медиана"],
                [[n_, n, _mm([r["auroc"][k][n]["all"]["auroc"] for k in chains], 4, False),
                  _mm([r["auroc"][k][n]["all"]["n_known"] for k in chains], 0, False)] for n_, r in named for n in ("25", "50", "100")]), "",
          "## Таблица 4.5-7 — проверка допущений правила по модели и условие перекалибровки (описательно)", "",
          *_assumption_tables_4_5(named),
          "## Таблица 4.5-8 — точка на дедуплицированной галерее: порог по квантили (без вердикта)", "",
          table(["энкодер, φ", "ε", "галерея", "η", "N", "τ_q", "FPR, % [95 %]", "пропусков, %", "верных ответов, %"],
                [[n_, e.replace(".", ","), lab, num(d["eta"]) if lab != "полная" else "—",
                  str(d["calibration"]["N0"] if lab != "полная" else r["calibrations"]["0"]["100"]["N0"]), num(m["tau"], 4),
                  f"{pct(m['fpr'])} [{pct(m['fpr_ci95'][0])}; {pct(m['fpr_ci95'][1])}]", pct(m["miss_rate"]), pct(m["correct_rate"])]
                 for n_, r in named for d in [r["dedup_point"]] for e, v in d["by_eps"].items()
                 for lab, m in (("полная", v["full"]), ("дедуплицированная", v["dedup"]))]), "",
          "## Таблица 4.5-9 — справочные строки AP полного метода: ρ и порог τ_q при ε = 0,05 (без вердикта)", "",
          table(["энкодер, φ", "галерея", "τ_q", "детекций", "сцены", "AP", "AP50", "AP75"],
                [[n_, lab, num(v["tau_q"], 4), str(v["n_detections"]), sl, ci(v["ap"][sn], "ap"), ci(v["ap"][sn], "ap50"), ci(v["ap"][sn], "ap75")]
                 for n_, r in named for p, lab in (("full", "полная"), ("dedup", "дедуплицированная"))
                 for v in [r["full_method_ap"]["galleries"][p]] for sn, sl in SUBSETS_4_4[:4]]), ""]
    return "\n".join(L) + "\n"



# ---------------------------------------------------------------------- 4.6 и задержка по этапам

OWLV2_DTYPE_NOTE = "fp32"  # подпись задержки OWLv2
SUBSETS_4_6 = [("test/all", "120 тестовых"), ("all/all", "160 (набор [5])")]
LEVELS_4_6 = [("test/easy", "120: easy"), ("test/hard", "120: hard"), ("all/easy", "160: easy"), ("all/hard", "160: hard")]


def _ar_key(use_cats: bool, thr) -> str:
    return f"{'with' if use_cats else 'no'}_cats__{'no_threshold' if thr is None else f'threshold_{thr}'}"


def _signed(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}".replace(".", ",")


class _Row46:
    """Строка таблицы 4.6-1: откуда у неё AP, AR и AP без меток `PART_MASK_LABELS` (у строк «с порогом» — только AP)."""

    def __init__(self, label: str, kind: str, rec: dict, proto: str | None = None):
        self.label, self.kind, self.rec, self.proto = label, kind, rec, proto

    def _subset(self, g: str, n: str) -> dict | None:
        r = self.rec
        if self.kind == "thr":
            return None
        m = (r["rho"]["metrics"] if self.kind == "rho" else r["metrics"]).get(g)
        return None if m is None else m["subsets"][n]

    def ap(self, g: str, n: str, area: str = "all") -> dict | None:
        if self.kind == "thr":
            return self.rec["full_method_ap"]["galleries"][self.proto]["ap"][n] if g == "full" and area == "all" else None
        s = self._subset(g, n)
        return None if s is None else s["by_area"][area]

    def ar(self, g: str, n: str, key: str) -> dict | None:
        s = self._subset(g, n)
        v = None if s is None else s["ar_final_detections"].get(key)
        return None if v is None else v["by_area"]["all"]

    def part(self, g: str, n: str) -> dict | None:
        if self.kind == "thr":
            return None
        if self.kind == "rho":
            return self._subset(g, n)["ap_without_part_mask_labels"]
        return self.rec["ap_without_part_mask_labels"][g][n]

    def n_det(self) -> int | None:
        if self.kind == "thr":
            return self.rec["full_method_ap"]["galleries"][self.proto]["n_detections"]
        return (self.rec["rho"]["metrics"] if self.kind == "rho" else self.rec["metrics"])["full"]["n_detections"]


def _rows_4_6(base: dict[str, dict], exp44: dict[str, dict], exp45: dict[str, dict], owl: dict) -> list[_Row46]:
    """Строки таблицы 4.6-1 по порядку: контрольные прогоны, метод с ρ, справочные строки с порогом (записи 4.5), OWLv2."""
    rows = []
    first = RU.FIRST_RUN
    for run in (first, *RU.RHO_RUNS):
        b = base[run]
        tag = "ближайшая к [5]" if run == first else "лучший φ"
        rows.append(_Row46(f"{b['config']['encoder']}, {b['variant_label']} — без отбора ({tag})", "control", b))
    for run in RU.RHO_RUNS:
        r = exp44[f"4_4_{run}"]
        rows.append(_Row46(f"{r['config']['encoder']}, {r['variant_label']} — метод с ρ, без порога", "rho", r))
    for run in RU.RHO_RUNS:
        r = exp45[f"4_5_{run}"]
        for proto, lab in (("full", "полная галерея"), ("dedup", "дедуплицированная галерея")):
            rows.append(_Row46(f"{r['config']['encoder']}, {r['variant_label']} — с ρ и порогом τ_q (ε = "
                               f"{num(r['full_method_ap']['eps'])}), {lab}; справочно", "thr", r, proto))
    rows.append(_Row46("OWLv2 image-guided (обучен с текстом)", "owlv2", owl))
    return rows


def _pub_table_a() -> list[str]:
    from src.eval import published as PB

    yes = {True: "да", False: "нет"}
    return [table(["метод", "источник", "дообучение", "генератор предложений", "энкодер", "AP", "AP50", "AP75", "hard", "easy",
                   "small", "medium", "large", "AR@100", "примечание"],
                  [[p.method, f"{p.source}, {p.address}", yes[p.finetune], p.proposals, p.encoder,
                    *(PB.fmt(p, k) for k in PB.METRICS), p.note] for p in PB.GROUP_A])]


def _pub_table_a_short() -> list[str]:
    from src.eval import published as PB

    yes = {True: "да", False: "нет"}
    return [table(["метод", "источник", "дообучение", "генератор предложений", "AP"],
                  [[p.method, f"{p.source}, {p.address}", yes[p.finetune], p.proposals, PB.fmt(p, "ap")] for p in PB.GROUP_A])]


def tables_4_6_hr_insdet(base: dict[str, dict], exp44: dict[str, dict], exp45: dict[str, dict], owl: dict) -> str:
    from src.eval import published as PB

    rows = _rows_4_6(base, exp44, exp45, owl)
    ci_ = lambda d, k: "—" if d is None else ci(d, k)  # noqa: E731
    ar_ref = _ar_key(RU.AR_FOR_4_6["use_cats"], RU.AR_FOR_4_6["score_threshold"])
    ar_none = _ar_key(False, None)
    control = [r for r in rows if r.kind == "control"]
    p5 = next(p for p in PB.GROUP_A if p.key == "insdet_sam_dinov2")
    p5l = next(p for p in PB.GROUP_A if p.key == "insdet_sam_dinov2_readme_vitl")
    nids = next(p for p in PB.GROUP_A if p.key == "nidsnet_no_adapter")
    L = ["# Эксперимент 4.6 — сводное сравнение, HR-InsDet (таблица 4.6-1)", "",
         "Источник чисел — `experiments/runs/` (контрольные прогоны 4.3, записи 4.4, 4.5 и OWLv2) и "
         "`src/eval/published.py` (опубликованные числа с адресами в источниках). "
         "Все величины — в процентах; в скобках — 95 % перцентильный интервал бутстрэпа по сценам (1 000 повторов, "
         "общих у всех наших строк и OWLv2). Наши числа — при точном переборе, `maxDets` = 1 000 на пару «снимок — категория». "
         "Основное число «метода с ρ» — без порога τ (4.4); строки «с порогом» — справочные, из блока `full_method_ap` записей 4.5. "
         "На 160 сценах 40 калибровочных те же, на которых выбраны разрешение входа, φ, γ и η; основной набор — 120 тестовых.", "",
         "## Таблица 4.6-1-1 — опубликованные результаты instance detection на HR-InsDet: полный состав столбцов", "",
         "«Дообучение» — обучаются ли параметры детектора или энкодера по эталонам тестовых экземпляров или по данным, "
         "синтезированным из них; синтез галереи обученной по эталонам моделью (NeRF [8]) дообучением не считается и назван в "
         "пометке строки; наличие модели, обученной с текстом, — столбец «генератор предложений». Числа [5] и "
         "[7] — на всех 160 тестовых снимках; для [8] и [9] число снимков оценки не выписано; AR@100 — по итоговым детекциям после порога и сопоставления, а не по предложениям.", "",
         *_pub_table_a(), "",
         "## Таблица 4.6-1-1s — опубликованные результаты instance detection на HR-InsDet: метод, источник, дообучение, генератор предложений и AP", "",
         *_pub_table_a_short(), "",
         "## Таблица 4.6-1-2 — наши контрольные прогоны рядом с опубликованными: AP на 160 сценах, полная галерея", "",
         f"Опубликованная сторона — {PB.fmt(p5, 'ap')} [5] и {PB.fmt(p5l, 'ap')} "
         "(README кода [5], DINOv2 ViT-L/14, не статья); пороги контроля на входе (30 и 8 п.) — только у первого прогона "
         "C(0,1.0) на DINOv2 и только против 41,61.", "",
         table(["наша строка", "AP, 160 сцен", f"Δ к {PB.fmt(p5, 'ap')} [5], п.", f"Δ к {PB.fmt(p5l, 'ap')} (README [5]), п."],
               [[r.label, ci(r.ap("full", "all/all"), "ap"),
                 _signed(100 * r.ap("full", "all/all")["ap"] - p5.values["ap"]),
                 _signed(100 * r.ap("full", "all/all")["ap"] - p5l.values["ap"])] for r in control]), "",
         "## Отличия контрольного прогона от [5] — перечень для приложения, без номера таблицы", "",
         "Чем контрольный прогон отличается от протокола и кода [5]:", "",
         *(f"- {t};" for t in PB.DIFFERENCES_VS_5[:-1]), f"- {PB.DIFFERENCES_VS_5[-1]}.", ""]
    for i, (g, glab) in enumerate((("full", f"протокол «полная галерея» ({FULL_ROWS_NOTE})"), ("one_per_class", "один эталон на класс"))):
        rs = [r for r in rows if r.ap(g, "test/all") is not None]
        L += [f"## Таблица 4.6-1-{3 + i} — наши строки и OWLv2: AP, AP50, AP75, {glab}", "",
              table(["строка", *(f"{m}, {lab}" for lab in (s for _, s in SUBSETS_4_6) for m in ("AP", "AP50", "AP75"))],
                    [[r.label, *(ci_(r.ap(g, n), m) for n, _ in SUBSETS_4_6 for m in ("ap", "ap50", "ap75"))] for r in rs]), ""]
        if g == "full":
            # Текст главы: без справочных строк «с порогом» — они в полной 4.6-1-3 и в 4.5-9.
            L += ["## Таблица 4.6-1-3s — строки метода и OWLv2: AP, AP50, AP75, протокол «полная галерея» (24 эталона на экземпляр)", "",
                  table(["строка", *(f"{m}, {lab}" for lab in (s for _, s in SUBSETS_4_6) for m in ("AP", "AP50", "AP75"))],
                        [[r.label, *(ci_(r.ap(g, n), m) for n, _ in SUBSETS_4_6 for m in ("ap", "ap50", "ap75"))] for r in rs if r.kind != "thr"]), ""]
    L += ["## Таблица 4.6-1-5 — уровень трудности (протокол «полная галерея», AP)", "", f"Все строки галереи: {FULL_ROWS_NOTE}.", "",
          table(["строка", *(lab for _, lab in LEVELS_4_6)],
                [[r.label, *(ci_(r.ap("full", n), "ap") for n, _ in LEVELS_4_6)] for r in rows]), "",
          f"Опубликованные на 160 снимках: [5] — hard {PB.fmt(p5, 'hard')}, easy {PB.fmt(p5, 'easy')}; NIDS-Net без адаптера — hard "
          f"{PB.fmt(nids, 'hard')}, easy {PB.fmt(nids, 'easy')} (таблица группы (а)). В 120 тестовых сценах доля hard "
          "выше, чем в 160 (40 из 120 против 40 из 160).", "",
          "## Таблица 4.6-1-6 — размер объекта (полная галерея, AP; бины — по площади размеченной рамки в пикселях исходного снимка)", "",
          table(["строка", *(f"{lab}, {s}" for n, lab in SUBSETS_4_6 for s in ("< 200²", "200²–400²", "> 400²"))],
                [[r.label, *(ci_(r.ap("full", n, a), "ap") for n, _ in SUBSETS_4_6 for a in ("small", "medium", "large"))]
                 for r in rows if r.kind != "thr"]), "",
          "Бины [5] и [8] — «bounding box area», система координат в источниках не названа (снимки 6144×8192); сопоставление по "
          "бинам — с этой оговоркой.", "",
          f"## Таблица 4.6-1-7 — без меток {' и '.join(RU.PART_MASK_LABELS)} (AP, 98 меток; справочно)", "",
          "У этих меток маска эталона у метода — часть объекта, а OWLv2 по рамке промпта получает объект целиком (асимметрия в "
          "пользу OWLv2); строка делает пару равноправной.", "",
          table(["строка", *(f"{g_lab}, {lab}" for g_lab in ("полная", "один эталон") for _, lab in SUBSETS_4_6)],
                [[r.label, *(ci_(r.part(g, n), "ap") for g in ("full", "one_per_class") for n, _ in SUBSETS_4_6)]
                 for r in rows if r.kind != "thr"]), "",
          "## Таблица 4.6-1-8 — AR итоговых детекций без учёта категорий (AR@100, полная галерея)", "",
          f"Рядом с 63,06 [5] и 77,09 [8] встаёт вариант `rules.AR_FOR_4_6` — без учёта категорий, с порогом "
          f"{num(RU.AR_SCORE_THRESHOLD, 1)} на s*; порог взят по [8] на шкале CLS вырезки DINOv2, у DINOv3 шкала иная — строка "
          "с порогом у неё приводится, но как сопоставимая не обсуждается; у OWLv2 оценка — сигмоида логита, строк с порогом нет. "
          "AR в [5] усредняется по порогам IoU от 0,5 до 1,0, у нас — 0,5:0,95.", "",
          table(["строка", *(f"{lab}, с порогом {num(RU.AR_SCORE_THRESHOLD, 1)}" for _, lab in SUBSETS_4_6),
                 *(f"{lab}, без порога" for _, lab in SUBSETS_4_6)],
                [[r.label, *(ci_(r.ar("full", n, ar_ref), "ar_100") for n, _ in SUBSETS_4_6),
                  *(ci_(r.ar("full", n, ar_none), "ar_100") for n, _ in SUBSETS_4_6)] for r in rows if r.kind != "thr"]), ""]
    pdiff = owl["paired_diff"]
    L += ["## Таблица 4.6-1-9 — OWLv2 минус наша строка: парная бутстрэп-разность (п.)", "",
          f"Разность считается по общим повторам бутстрэпа. {pdiff['reading'][0].upper() + pdiff['reading'][1:].replace('; ', ', ')}. Контрольные прогоны — на 160 и 120 сценах, "
          "метод с ρ — на 120 тестовых.", "",
          table(["вычитаемая строка", "галерея", "сцены", "ΔAP", "ΔAP50", "ΔAP75"],
                [[f"{b['config']['encoder']}, {b['variant_label']} — " + ("без отбора" if row == "control" else "метод с ρ"),
                  {"full": "полная", "one_per_class": "один эталон"}[g], lab, *(_diff(d[m]) for m in ("ap", "ap50", "ap75"))]
                 for run in RU.RHO_RUNS for row in ("control", "rho") for b in [base[run]]
                 for g in ("full", "one_per_class") for n, lab in SUBSETS_4_6
                 for d in [pdiff[f"minus_{row}__{run}"][g].get(n)] if d is not None]), "",
          "## Таблица 4.6-1-10 — число детекций", "", f"Протокол «полная галерея»: {FULL_ROWS_NOTE}.", "",
          table(["строка", "детекций на 160 сцен", "в среднем на снимок"],
                [[r.label, str(r.n_det()), num(r.n_det() / 160, 1)] for r in rows]), "",
          "## Что стоит за строкой OWLv2 — без номера таблицы", "",
          f"OWLv2: на снимок после NMS — в среднем {num(owl['detections_per_image']['full']['mean'], 1)}, наибольшее "
          f"{owl['detections_per_image']['full']['max']} (полная галерея); кандидатов до NMS — до 5 184 рамок; рамок, вырожденных "
          f"после обрезки по кадру (остаются ложными детекциями), — {owl['n_detections_zero_area_after_clip']['full']} и "
          f"{owl['n_detections_zero_area_after_clip']['one_per_class']} (полная галерея / один эталон); эталонов без эмбеддинга "
          f"запроса — {owl['n_refs_without_query_embed']['full']}, меток без эталона в протоколе «один эталон» — "
          f"{len(owl['labels_without_ref_one_per_class'])}; модель — {owl['model']['id']}, {owl['model']['dtype']}, вход "
          f"{owl['input_size']}.", "",
          "**Как читать строку OWLv2**: разность с ним — "
          "разность с методом при штатном входе OWLv2; она смешивает текстовый надзор с архитектурой, обучением детектора и "
          f"разрешением входа (OWLv2 — штатный вход {owl['input_size']}, сцена сжимается в 8,1 раза; метод — сегментация на 2048 и "
          "вырезки из полного кадра) и чистой ценой текста не является; смешение по разрешению — против OWLv2. Отличия протокола OWLv2 (из записи `owlv2_hr_insdet.json`):", "",
          *(f"- {t};" for t in owl["differences_for_4_1_4_6"][:-1]), f"- {owl['differences_for_4_1_4_6'][-1]}.", "",
          "Независимые решения по маскам вместо взаимно-однозначного сопоставления [5] — условие сравнимости со всей группой (а).", ""]
    return "\n".join(L)


def tables_4_6_published() -> str:
    from src.eval import published as PB

    return "\n".join([
        "# Эксперимент 4.6 — опубликованные результаты top-down без текста на RoboTools (таблица 4.6-3)", "",
        "Источник чисел — `src/eval/published.py`, у каждого числа записан адрес в источнике. "
        "AP — в процентах, по [8], табл. 2.", "",
        "**Собственных прогонов на RoboTools в работе нет** (датасет не использовался из-за стоимости счёта): таблица — опубликованное свидетельство о направлении (вклад парадигмы при равном "
        "источнике надзора), на предложенный метод оно не переносится; строк метода здесь нет.", "",
        "## Таблица 4.6-3 — опубликованные результаты top-down методов без текста и bottom-up связок на RoboTools, AP по [8]", "",
        table(["метод", "источник", "надзор / вид", "AP", "примечание"],
              [[p.method, f"{p.source}, {p.address}", p.proposals, PB.fmt(p, "ap"), p.note] for p in PB.GROUP_B]), ""])


LATENCY_OWLV2 = [("hr_insdet", "HR-InsDet"), ("pcb", "PKU-Market-PCB")]


def tables_latency(exp44: dict[str, dict], owl: dict[str, dict]) -> str:
    """Задержка по этапам из отдельных замеров: 40 калибровочных сцен HR-InsDet."""
    import glob

    import numpy as np

    env = json.loads(Path("experiments/env.json").read_text())
    enc_chk = json.loads(RHO_ENCODE_CHECK.read_text())
    crop_chk = json.loads(Path("experiments/crop_pipeline_check.json").read_text())
    cal = json.loads(Path("splits/hr_insdet.json").read_text())["cal"]
    r0 = exp44[f"4_4_{RU.RHO_RUNS[0]}"]
    mdir = r0["mask_cache"]["scenes"]
    seg = [json.loads(Path(f"cache/masks/hr_insdet/{mdir}/{s.replace('/', '__')}.json").read_text())["info"] for s in cal]
    rdirs = glob.glob(f"cache/rho/hr_insdet/{mdir}__theta*")
    if len(rdirs) != 1:
        raise SystemExit(f"каталог cache/rho для {mdir}: {rdirs}")
    rho = [json.loads(Path(rdirs[0], f"{s.replace('/', '__')}.json").read_text())["sec"] for s in cal]
    n = len(cal)
    vlabel = {r["grid_run"]: r["variant_label"] for r in exp44.values()}
    seg_gen = np.array([i["sec_generate"] for i in seg])
    seg_all = np.array([i["sec_read"] + i["sec_resize"] + i["sec_generate"] for i in seg])
    f = lambda x, nd=2: num(float(x), nd)  # noqa: E731
    L = ["# Задержка по этапам (для 4.1 и 4.6)", "",
         "Источник чисел — отдельные замеры, а не журнал прогонов сетки: у парных проходов время "
         "кодирования общее на два энкодера. RTX 3050 6 ГБ, CPU контейнера. "
         f"Метод — {n} калибровочных сцен HR-InsDet (1536×2048 на входе S, 8192×6144 — источник вырезок); SAM 2 — fp32 под "
         "bf16-autocast, DINOv2 — fp16, DINOv3 — bf16. Итог на сцену — сумма средних этапов; медианы не складываются.", "",
         "## Таблица 4.1-T-1. Метод, HR-InsDet: время по этапам", "",
         table(["этап", "источник", "медиана на сцену, с", "среднее на сцену, с", f"всего на {n} сцен, с"],
               [["сегментация SAM 2 (`generate`, `crop_n_layers=1`)", f"`info` кеша масок `{mdir}`", f(np.median(seg_gen)),
                 f(seg_gen.mean()), f(seg_gen.sum(), 1)],
                ["сегментация вместе с чтением и уменьшением снимка", "там же", f(np.median(seg_all)), f(seg_all.mean()),
                 f(seg_all.sum(), 1)],
                ["отбор ρ (CPU)", "`cache/rho/`", f(np.median(rho), 4), f(np.mean(rho), 4), f(np.sum(rho), 2)],
                *([f"кодирование, {e}, {vlabel[c['run_id']]}: {lab} ({c['timing'][k]['n_masks']} масок)",
                   "`rho_encode_check.json`", f(c["timing"][k]["sec_per_scene_median"]), f(c["timing"][k]["sec"] / n),
                   f(c["timing"][k]["sec"], 1)]
                  for e, c in enc_chk["by_encoder"].items() for k, lab in (("all", "все маски M(I)"), ("rho", "только M*")))]), "",
         "Кодирование — время пайплайна целиком: вырезки готовятся на CPU в шести процессах параллельно проходу GPU; у "
         "C(blur,1.5) оно упирается в CPU (размытие фона).", ""]
    rows = []
    for c in crop_chk["rows"]:
        w = next(iter(c["pipelined"]))
        rows.append([c["encoder"].replace("encoder_", ""), c["variant"].replace("|", ","), str(c["n_masks"]),
                     f(1000 * c["sequential_sec"] / c["n_masks"], 1), f(1000 * c["pipelined"][w]["sec"] / c["n_masks"], 1)])
    cp = env["crop_precision"]["by_encoder"]
    pp = env["pool_precision"]["by_encoder"]
    L += ["### Составляющие кодирования (одна модель в памяти GPU)", "",
          "Подготовка вырезок и проход энкодера порознь — из отдельных замеров (у парных проходов сетки время кодирования "
          "общее на два энкодера): `experiments/crop_pipeline_check.json` (три калибровочные сцены, 334 маски) и `experiments/env.json` "
          "(`crop_precision`, `pool_precision`; десять калибровочных сцен сверки).", "",
          "## Таблица 4.1-T-2. Подготовка вырезок на CPU: последовательно и в шести процессах", "",
          table(["энкодер", "вариант", "масок", "мс на маску, последовательно", f"мс на маску, {next(iter(crop_chk['rows'][0]['pipelined']))} процессов"],
                rows), "",
          "## Таблица 4.1-T-3. Проход энкодера на GPU в рабочей точности и в fp32", "",
          table(["энкодер", "точность", "вырезок 448² в секунду на GPU (рабочая / fp32)", "проход кадра 1536×2048 для P, с (рабочая / fp32; справочно)"],
                [[e.replace("encoder_", ""), cp[e]["dtype_working"], f"{f(cp[e]['crops_per_sec']['working'], 1)} / {f(cp[e]['crops_per_sec']['fp32'], 1)}",
                  f"{f(pp[e]['sec_forward_median']['working'])} / {f(pp[e]['sec_forward_median']['fp32'])}"] for e in cp]), ""]
    srows = []
    for run in RU.RHO_RUNS:
        r = exp44[f"4_4_{run}"]
        d = r["dedup"]
        lat = d["search_latency"]
        hgrid = json.loads((RUNS / f"hnsw_{run}.json").read_text())["grid_check"]["single_thread"]["recall"]
        for proto in ("full", "dedup"):
            p = lat[proto]
            t = p["threads_default"]
            flat = t[f"flat_k{p['N']}"]["batch_per_scene"]
            hn = t[f"hnsw_k{p['k']}"]["batch_per_scene"]
            rec = hgrid if proto == "full" else d["hnsw_recall_dedup"]["single_thread"]["recall"]
            srows.append([f"{r['config']['encoder']}, {r['variant_label']}", proto, str(p["N"]),
                          f"{f(flat['ms_per_call_median'], 2)} / {f(flat['ms_per_call_mean'], 2)}",
                          f"{f(hn['ms_per_call_median'], 2)} / {f(hn['ms_per_call_mean'], 2)}",
                          f"{num(rec, 5)} ({'M(I), этап grid' if proto == 'full' else 'M*, блок dedup 4.4'})"])
    L += ["### Поиск (только `search`; пачка — маски M* одной сцены, 40 вызовов; потоков Faiss — по умолчанию)", "",
          "Рабочий режим поиска — точный перебор (`IndexFlatIP`, k = N): им считаются все метрики и калибровка. HNSW — средство "
          "масштабирования галереи; его задержка приводится только рядом с полнотой относительно точного перебора (однопоточное, "
          "воспроизводимое построение; критерий 0,999 не выдержан ни на одной галерее).", "",
          f"Условия замера задержки (из записи 4.4): {exp44[f'4_4_{RU.RHO_RUNS[0]}']['dedup']['search_latency']['protocol']}. "
          "Задержка и полнота HNSW в одной строке относятся к разным построениям графа: задержка — индекс, построенный в "
          f"{exp44[f'4_4_{RU.RHO_RUNS[0]}']['dedup']['search_latency']['full']['hnsw_build_threads']} потоков, запросы M*; полнота — "
          "однопоточное построение и запросы, названные в скобках.", "",
          "## Таблица 4.1-T-4. Задержка поиска на сцену: точный перебор и HNSW", "",
          table(["галерея", "строки", "N", "точный перебор, мс на сцену (медиана / среднее)", "HNSW, мс на сцену (медиана / среднее)",
                 "полнота HNSW (запросы)"], srows), "",
          f"Справочно, первый замер HNSW (галерея C(0,1.0) на DINOv2, N = 2 400, запросы — маски M(I), 40 сцен): точный перебор k = N — "
          f"{f(json.loads((RUNS / 'hnsw_dinov2_c_0_10_hr_insdet.json').read_text())['latency']['threads_default']['flat_k2400']['batch_per_scene']['ms_per_call_median'], 1)} мс на сцену (медиана).", ""]
    tot = []
    for e, c in enc_chk["by_encoder"].items():
        run = c["run_id"]
        lat = exp44[f"4_4_{run}"]["dedup"]["search_latency"]["full"]
        flat_s = lat["threads_default"][f"flat_k{lat['N']}"]["batch_per_scene"]["ms_per_call_mean"] / 1000
        enc_s = c["timing"]["rho"]["sec"] / n
        tot.append([f"{e}, {vlabel[run]}", f(seg_all.mean()), f(np.mean(rho), 3), f(enc_s), f(flat_s, 4),
                    f(seg_all.mean() + np.mean(rho) + enc_s + flat_s)])
    L += ["## Таблица 4.1-T-5. Итого на сцену, метод с ρ (среднее по 40 калибровочным сценам, с)", "",
          table(["энкодер, φ", "сегментация", "ρ", "кодирование M*", "поиск (точный, полная галерея)", "всего"], tot), "",
          f"Чтение снимка учтено дважды — в сегментации (в среднем {f(np.mean([i['sec_read'] for i in seg]))} с) и в процессах "
          "подготовки вырезок при кодировании, итог этим немного завышен. Прогрев и повторы по этапам такие. Сегментация — один проход "
          "на сцену при построении кеша, без прогрева. Кодирование — один проход, прогрев — первый батч первой сцены на "
          "энкодер. Поиск — прогрев полным проходом, медиана по пяти повторам. OWLv2 — один проход, без прогрева.", "",
          "Задержка метода на PKU-Market-PCB отдельно не замерялась.", ""]
    orows = []
    for ds, lab in LATENCY_OWLV2:
        o = owl.get(ds)
        if o is None:
            continue
        t = o["timing"]
        s, q = t["scenes"], t["queries"]
        orows.append([lab, str(s["n"]), f(s["sec_tower"] / s["n"]), f(s["sec_heads"] / s["n"], 4), f(s["sec_nms"] / s["n"], 3),
                      f(s["sec_prep"] / s["n"]), f(q["sec_tower"] / q["n"]),
                      f((s["sec_tower"] + s["sec_heads"] + s["sec_nms"]) / s["n"])])
    L += [f"## OWLv2 image-guided ({OWLV2_DTYPE_NOTE}; энкодеры метода — в fp16 и bf16)", "",
          "Из записей `experiments/runs/owlv2_<dataset>.json` — все снимки счёта, а не только калибровочные; башня и головки — "
          "GPU с синхронизацией, предобработка — CPU в параллельных процессах (сумма по процессам, не время стены), NMS и "
          "агрегация — CPU. Вход — 1008×1008; эмбеддинги эталонов-запросов считаются один раз на галерею (индексация).", "",
          "## Таблица 4.1-T-6. OWLv2: время на снимок и на эталон", "",
          table(["область", "снимков", "башня на снимок, с", "головки на снимок, с", "NMS на снимок, с",
                 "предобработка на снимок, с (CPU, сумма)",
                 "эталон-запрос целиком (башня, головки, выбор рамки), с (индексация)",
                 "итого на снимок без предобработки (башня + головки + NMS), с"], orows), "",
          "Итог OWLv2 на снимок не включает предобработку: она идёт на CPU в параллельных процессах, и её сумма по процессам "
          "не складывается с временем стены; итог метода «всего» выше включает чтение и уменьшение снимка.", ""]
    return "\n".join(L)


# Протокол `full` в 4.6-1 — все строки галереи прогона: у контрольных прогонов, метода с ρ и OWLv2 это 24 эталона на
# экземпляр, у справочных строк «дедуплицированная галерея» — её состав.
FULL_ROWS_NOTE = "24 эталона на экземпляр; у строк «дедуплицированная галерея» — её дедуплицированный состав"


# Сквозная нумерация таблиц: «Таблица 4.N» — таблицы в тексте главы 4 по порядку первого упоминания,
# «Таблица А.N» — печатное приложение: полнота и кратные детекции после ρ (4.4-2), выбор φ по калибровочным сценам (4.3-4),
# исход отклонения при пополнении галереи (4.5-1), рост частоты ложных срабатываний у непересчитанного порога (4.5-4),
# парные разности с OWLv2 (4.6-1-9), опубликованные результаты top-down без текста на RoboTools (4.6-3), итог задержки
# на сцену (4.1-T-5). Ключ — «файл:идентификатор таблицы в файле»; идентификаторы в файлах не меняются, заголовок печатает
# оба обозначения. Остальные таблицы в тексте работы не приводятся.
CHAPTER_TABLES = [
    "4_2_hr_insdet:4.2-1", "4_2_pcb:4.2-2", "4_3_hr_insdet:4.3-1 (full)", "4_4_hr_insdet:4.4-1",
    "4_6_hr_insdet:4.6-1-1s", "4_6_hr_insdet:4.6-1-3s",
]
APPENDIX_TABLES = [
    "4_4_hr_insdet:4.4-2", "4_3_hr_insdet:4.3-4", "4_5_hr_insdet:4.5-1", "4_5_hr_insdet:4.5-4", "4_6_hr_insdet:4.6-1-9",
    "4_6_published:4.6-3", "latency:4.1-T-5",
]
_HEADING = re.compile(r"^## Таблица (\d\.\d-[^ .—]+(?: \([^)]*\))?)(\.| —)", re.M)
_INDEX: dict[str, str] = {}


def table_label(key: str) -> str | None:
    if key in CHAPTER_TABLES:
        return f"Таблица 4.{CHAPTER_TABLES.index(key) + 1}"
    if key in APPENDIX_TABLES:
        return f"Таблица А.{APPENDIX_TABLES.index(key) + 1}"
    return None


def _number_tables(name: str, text: str) -> str:
    def sub(m):
        tid, sep = m.group(1), m.group(2)
        label = table_label(f"{name}:{tid}")
        if label is None:
            return m.group(0)
        _INDEX[label] = f"`{name}.md`, {tid} — {text[m.end():].split(chr(10), 1)[0].strip()}"
        return f"## {label} ({tid}){sep}"
    return _HEADING.sub(sub, text)


def _write(name: str, text: str) -> str:
    out = TABLES / f"{name}.md"
    tmp = out.with_suffix(".md.tmp")
    tmp.write_text(_number_tables(name, text))
    tmp.replace(out)
    return str(out)


def _table_section(name: str, tid: str) -> str:
    """Таблица в печатном виде: раздел файла от заголовка «## Таблица …» до следующего «## », с подписью и видом PRINT_VIEW."""
    text = (TABLES / f"{name}.md").read_text()
    m = re.search(r"^## (Таблица [^\n]*\(" + re.escape(tid) + r"\)[^\n]*)\n", text, re.M)
    if m is None:
        raise SystemExit(f"в {name}.md нет заголовка таблицы {tid}")
    body = text[m.end():]
    nxt = re.search(r"^## ", body, re.M)
    body = body[:nxt.start()] if nxt else body
    title = m.group(1)
    label, rest = title.split(" (", 1)
    depth, k = 1, 0                      # идентификатор может содержать скобки — «4.3-1 (full)»
    while k < len(rest) and depth:
        depth += {"(": 1, ")": -1}.get(rest[k], 0)
        k += 1
    caption = rest[k:].lstrip(".— ").strip()
    view = PRINT_VIEW.get(f"{name}:{tid}")
    if view:
        caption = view.get("caption", caption)
        body = _print_body(body, view)
    body = "\n".join(_print_cells(s) if s.startswith("|") else s for s in body.split("\n"))
    if view and view.get("thousands"):                     # разряды через неразрывный пробел, как в тексте работы
        body = "\n".join("|".join(re.sub(r"^(\s*)(\d{1,3})(\d{3})(\s*)$", "\\1\\2\u00a0\\3\\4", c) for c in s.split("|"))
                          if s.startswith("|") else s for s in body.split("\n"))
    return f"{label} — {caption}\n\n{body.strip()}\n"


# Обозначения в ячейках печатных таблиц — как в тексте работы: имена энкодеров, варианты φ формулой, минус вместо дефиса,
# «полнота» вместо «recall», без внутренних сокращений из статей-источников.
_VARIANT = {"0": "0", "x̄": "\\bar x", "blur": "\\mathrm{blur}"}
CELL_REPL = [
    ("без предложений: плотное сопоставление патчей + SAM2-L по точкам; модуль отбора — PE-Core-L14-336 (обучен с текстом)",
     "плотное сопоставление патчей и SAM 2, отбор — Perception Encoder (обучен с текстом)"),
    ("SAM + DINOv2 ViT-L/14 — README кода, не статья", "SAM + DINOv2 ViT-L/14"),
    ("[5], README кода", "[5], репозиторий авторов"), ("[5], табл. 2, 3", "[5], табл. 2"), (" (демо-код)", ""),
    ("SAM + DINOv2 (OTS-FM SAM в [8])", "SAM + DINOv2"), ("SAM + DINOv2 (OTS-FM SAM)", "SAM + DINOv2"),
    ("OTS-FM Grounding DINO", "Связка без обучения с Grounding DINO"), ("supp. табл. 6", "доп. материалы, табл. 6"),
    ("NIDS-Net без адаптера (FFA)", "NIDS-Net без адаптера"), ("NIDS-Net с адаптером WA", "NIDS-Net с адаптером"),
    ("PE-Core-L14-336", "Perception Encoder"), ("надзор / вид", "надзор"),
    ("без отбора (ближайшая к [5])", "контрольный прогон"), ("— метод с ρ, без порога", "— метод с ρ"),
    ("recall@", "полнота, IoU ≥ "), ("recall, IoU", "полнота, IoU"), ("recall ", "полнота "),
    ("top-1 на оракульных масках", "top-1"), ("время ρ, с", "время ρ на весь набор, с"), ("— без отбора |", "— контрольный прогон |"),
        ("NeRF-ракурсы в галерее при тесте", "синтезированные ракурсы в галерее"), ("доп. материалы, табл. 6", "прил., табл. 6"),
    ("SAM2-L", "SAM 2"), ("bottom-up, для сравнения внутри той же таблицы [8]", "bottom-up, из той же таблицы [8]"),
    ("bottom-up, для сравнения внутри [8]", "bottom-up, из той же таблицы [8]"), (" (§1.2)", ""),
    ("вычитаемая строка", "вариант метода"), ("рамок с ≥ 2 масками, % M(I) → M*", "доля рамок с двумя и больше масками, %"),
    ("полнота 0,5:0,95", "полнота, IoU 0,5:0,95"),
]


def _print_cells(line: str) -> str:
    for a, b in CELL_REPL:
        line = line.replace(a, b)
    line = re.sub(r"\bdinov([23])\b", r"DINOv\1", line)
    line = re.sub(r"C\((0|x̄|blur),1\.([05])\)", lambda m: f"$C({_VARIANT[m.group(1)]},1{{,}}{m.group(2)})$", line)
    line = line.replace("P⊥", "$P^\\perp$").replace("| P |", "| $P$ |")
    line = line.replace("M(I)", "$M(I)$").replace("M*", "$M^*$").replace("ŷ", "$\\hat y$")
    line = re.sub(r"(?<![\\$\w])ρ", r"$\\rho$", line)
    line = re.sub(r"(?<![\\$\w])φ", r"$\\varphi$", line)
    line = re.sub(r"(?<![\w.])-(\d)", r"−\1", line)
    cells = line.split("|")
    if len(cells) > 2 and cells[-2].strip() == "" and cells[1].strip() == "энкодер":
        cells[-2] = " выбор "
    return "|".join(cells)


# Печатный вид таблиц в тексте работы: из полной таблицы берутся нужные столбцы и строки, проза под таблицей снимается
# («text»: False) или заменяется («replace»). Строки отбираются по значениям столбцов: {"столбец": {допустимые значения}}.
PRINT_VIEW = {
    "4_2_hr_insdet:4.2-1": {
        "caption": "Полнота сегментатора по разрешению входа и набору сцен HR-InsDet",
        "drop": ["рамок", "сцен", "recall, 0,5:0,95", "AR@100"],
        "rows": {"сцены": {"калибровочные", "тестовые"}},
        "also": [{"`crop_n_layers` (эфф. разрешение)": "1 (2048)", "сцены": "все", "уровень": "все"}],
        "rename": {"`crop_n_layers` (эфф. разрешение)": "слой окон (разрешение)"},
    },
    "4_2_pcb:4.2-2": {
        "caption": "Полнота сегментатора по типу дефекта PKU-Market-PCB, тестовые платы, один слой окон",
        "drop": ["`crop_n_layers` (эфф. разрешение)", "снимков", "масок на снимок", "recall, 0,5:0,95", "AR@100", "AR@1000"],
        "rows": {"`crop_n_layers` (эфф. разрешение)": {"1 (2048)"}},
    },
    "4_3_hr_insdet:4.3-1 (full)": {
        "caption": "Сетка кодирования, тестовые сцены HR-InsDet, полная галерея",
        "drop": ["AP75", "AP при `maxDets`=100", "общий проход с"],
    },
    "4_4_hr_insdet:4.4-1": {
        "caption": "AP метода с отбором $\\rho$ и контрольного прогона без отбора при полной галерее, разности для easy и hard без интервалов",
        "drop": ["галерея", "ΔAP75, п."],
        "rows": {"галерея": {"full"}},
        "text": False,
    },
    "4_6_hr_insdet:4.6-1-1s": {
        "caption": "Опубликованные результаты instance detection на HR-InsDet",
    },
    "4_6_hr_insdet:4.6-1-3s": {
        "caption": "Контрольный прогон, варианты метода и OWLv2 при полной галерее",
        "rename": {"строка": "вариант"},
        "drop": ["AP75, 120 тестовых", "AP75, 160 (набор [5])"],
        "exclude": {"строка": {"dinov2, C(x̄,1.0) — без отбора (лучший φ)", "dinov3, C(blur,1.5) — без отбора (лучший φ)"}},
    },
    "4_4_hr_insdet:4.4-2": {
        "caption": "Число масок, полнота после отбора и кратные детекции",
        "replace": [
            ("Кратные детекции с учётом ŷ (доля рамок с двумя и больше детекциями своей метки при IoU ≥ 0,5, справочно), "
             "120 тестовых сцен: dinov2, C(x̄,1.0) — 10,5 → 0,1 %, dinov3, C(blur,1.5) — 13,0 → 0,1 %.",
             "Доля рамок с двумя и больше детекциями своей метки при IoU ≥ 0,5 на 120 тестовых сценах падает после отбора "
             "с 10,5 до 0,1 % у DINOv2 и с 13,0 до 0,1 % у DINOv3."),
            ("Время кодирования — отдельный замер на 40 калибровочных сценах (`experiments/rho_encode_check.json`), "
             "не прогон сетки, тестовые сцены не кодировались:",
             "При кодировании одних отобранных масок $M^*$ на 40 калибровочных сценах ни один ответ не изменился, а время "
             "кодирования сократилось примерно вдвое: с 289,6 до 156,3 с у DINOv2 и с 1217,7 до 557,1 с у DINOv3."),
        ],
        "first_table_only": True,
        "thousands": True,
    },
    "4_3_hr_insdet:4.3-4": {
        "caption": "Выбор варианта $\\varphi$ для каждого энкодера по калибровочным сценам",
        "replace": [
            ("Правило выбора: наибольший AP с оракулом при полной галерее на 40 калибровочных сценах, а в пределах 0,1 п. — "
             "больший AP50, затем вариант, стоящий выше в таблице 2.1. Выбор объявляется, когда посчитаны все прогоны энкодера: "
             "dinov2 — 7 из 7, лучший — C(x̄,1.0), dinov3 — 8 из 8, лучший — C(blur,1.5).",
             "Лучшим считается вариант с наибольшим AP с оракулом при полной галерее на 40 калибровочных сценах, "
             "а при разнице в пределах 0,1 п. вариант с большим AP50."),
        ],
    },
    "4_6_hr_insdet:4.6-1-9": {
        "caption": "Разность OWLv2 и вариантов метода по парному бутстрэпу, п.",
        "replace": [
            ("Разность считается по общим повторам бутстрэпа. Различие есть, если 95 % интервал парной разности не содержит "
             "нуля, других порогов нет. Контрольные прогоны — на 160 и 120 сценах, метод с ρ — на 120 тестовых.",
             "Разность считается по общим повторам бутстрэпа, и различие признаётся, если 95 % интервал парной разности "
             "не содержит нуля. Контрольные прогоны сравниваются на 160 и 120 сценах, варианты с $\\rho$ на 120 тестовых."),
        ],
    },
    "latency:4.1-T-5": {
        "caption": "Время на сцену у метода с $\\rho$, среднее по 40 калибровочным сценам, с",
        "replace": [
            ("Чтение снимка учтено дважды — в сегментации (в среднем 0,37 с) и в процессах подготовки вырезок при кодировании, "
             "итог этим немного завышен. Прогрев и повторы по этапам такие. Сегментация — один проход на сцену при построении "
             "кеша, без прогрева. Кодирование — один проход, прогрев — первый батч первой сцены на энкодер. Поиск — прогрев "
             "полным проходом, медиана по пяти повторам. OWLv2 — один проход, без прогрева.",
             "Чтение снимка, в среднем 0,37 с, учтено и в сегментации, и при подготовке вырезок, поэтому итог немного завышен. "
             "Сегментация и кодирование измерены одним проходом, поиск как медиана пяти повторов после пробного запуска."),
            ("\n\nЗадержка метода на PKU-Market-PCB отдельно не замерялась.", ""),
        ],
    },
    "4_5_hr_insdet:4.5-1": {
        "caption": "Отклонение при пополнении галереи: в скольких последовательностях подвыборок из девяти порог удерживает "
                   "заданную частоту ложных срабатываний $\\varepsilon$ при росте галереи с 25 до 100 экземпляров",
        "drop": ["по модели, цепочек", "top-K", "фиксированный 0,4", "исход (уточнённая формулировка 21.09)", "исход (формулировка 12.09)",
                 "квантиль с пересчётом при N = N₀ (перенос): экземпляров — итог (цепочек)"],
        "rename": {"энкодер, φ": "энкодер, $\\varphi$", "ε": "$\\varepsilon$"},
        "text": False,
    },
    "4_5_hr_insdet:4.5-4": {
        "caption": "Частота ложных срабатываний на дистракторах 120 тестовых сцен, %, у порога, поставленного при 25 экземплярах "
                   "и не пересчитанного при росте галереи: медиана по девяти последовательностям, в скобках наименьшее и наибольшее",
        "exclude": {"правило": {"фиксированный 0,4"}},
        "drop": ["правило"],
        "rename": {"энкодер, φ": "энкодер, $\\varphi$", "ε": "$\\varepsilon$", "FPR при 25, %": "при 25 экземплярах",
                   "при 50, %": "при 50", "при 100, %": "при 100", "отношение 100 / 25": "отношение 100 к 25",
                   "предсказано моделью": "по модели (§2.4)"},
        "text": False,
    },
    "4_6_published:4.6-3": {
        "caption": "Опубликованные результаты top-down методов без текста и bottom-up связок на RoboTools по [8]. Для top-down "
                   "методов это средняя полнота лучшего ответа на запрос из [6], которая при одном экземпляре на снимок совпадает с AP",
    },
}


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def also_one(r: list[str], a: dict, head: list[str]) -> bool:
    return all(r[head.index(c)] == v for c, v in a.items())


def _print_body(body: str, view: dict) -> str:
    """Первая таблица раздела в печатном виде: без столбцов `drop`, строки по `rows` / `exclude`; прочие части раздела
    остаются, кроме прозы после первой таблицы при `text: False`."""
    lines = body.strip("\n").split("\n")
    if view.get("first_table_only"):                      # таблицы после первой в печатный вид не входят
        start = next(i for i, s in enumerate(lines) if s.startswith("|"))
        end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
        lines = lines[:end] + [s for s in lines[end:] if not s.startswith("|")]
        while lines and not lines[-1].strip():
            lines.pop()
    if not any(k in view for k in ("drop", "rows", "exclude", "rename")) and view.get("text", True):
        out = "\n".join(lines)
        for a, b in view.get("replace", []):
            if a not in out:
                raise SystemExit(f"печатный вид: нет текста для замены: {a[:60]}")
            out = out.replace(a, b)
        return out
    start = next(i for i, s in enumerate(lines) if s.startswith("|"))
    end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
    head = _cells(lines[start])
    rows = [_cells(s) for s in lines[start + 2:end]]
    also = lambda r: any(all(r[head.index(c)] == v for c, v in a.items()) for a in view.get("also", []))
    for col, vals in [*view.get("rows", {}).items(), *view.get("exclude", {}).items()]:
        absent = set(vals) - {r[head.index(col)] for r in rows}
        if absent:                                         # значение фильтра не найдено — источник изменился
            raise SystemExit(f"печатный вид: в столбце «{col}» нет значений {sorted(absent)}")
    for a in view.get("also", []):
        if not any(also_one(r, a, head) for r in rows):
            raise SystemExit(f"печатный вид: нет строки {a}")
    for col, keep in view.get("rows", {}).items():
        rows = [r for r in rows if r[head.index(col)] in keep or also(r)]
    for col, skip in view.get("exclude", {}).items():
        rows = [r for r in rows if r[head.index(col)] not in skip]
    keep_idx = [i for i, h in enumerate(head) if h not in view.get("drop", [])]
    missing = [c for c in view.get("drop", []) if c not in head]
    if missing:
        raise SystemExit(f"печатный вид: нет столбцов {missing}")
    fmt = lambda r: "| " + " | ".join(r[i] for i in keep_idx) + " |"
    head = [view.get("rename", {}).get(h, h) for h in head]
    tbl = [fmt(head), "|" + "---|" * len(keep_idx), *map(fmt, rows)]
    tail = lines[end:] if view.get("text", True) else []
    out = "\n".join(lines[:start] + tbl + tail)
    for a, b in view.get("replace", []):
        if a not in out:
            raise SystemExit(f"печатный вид: нет текста для замены: {a[:60]}")
        out = out.replace(a, b)
    return out


PRINT_OUT = "print"


def write_print() -> str:
    """Таблицы 4.1–4.6 и А.1–А.7 в том виде, в каком они напечатаны в тексте работы, одним файлом `print.md`."""
    parts = ["# Таблицы в тексте работы", "",
             "Печатный вид таблиц главы 4 и приложения, порождается из файлов этого каталога функцией `write_print`; "
             "полные таблицы — в файлах, указанных у каждой.", ""]
    for key in CHAPTER_TABLES + APPENDIX_TABLES:
        name, tid = key.split(":", 1)
        parts += [_table_section(name, tid), f"Полная таблица: `{name}.md`, {tid}.", ""]
    out = TABLES / f"{PRINT_OUT}.md"
    tmp = out.with_suffix(".md.tmp")
    tmp.write_text("\n".join(parts))
    tmp.replace(out)
    return str(out)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    made = []
    for p in sorted(RUNS.glob("4_2_*.json")):
        rec = json.loads(p.read_text())
        made.append(_write(rec["run_id"], (tables_4_2_pcb if rec["dataset"] == "pcb" else tables_4_2)(rec)))
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
    recs = {r["run_id"]: r for r in runs if r.get("kind") == "grid_run" and r["dataset"] == "hr_insdet"}
    base_hr = {r["grid_run"]: r for r in runs if r.get("kind") == "control_run" and r["dataset"] == "hr_insdet"}
    if recs:
        made.append(_write("4_3_hr_insdet", tables_4_3("hr_insdet", recs, base_hr)))
    exp44 = {r["run_id"]: r for r in runs if r.get("kind") == "exp_4_4"}
    if exp44:
        made.append(_write("4_4_hr_insdet", tables_4_4(exp44)))
    hr45 = {r["run_id"]: r for r in runs if r.get("kind") == "exp_4_5" and r["dataset"] == "hr_insdet"}
    if hr45:
        made.append(_write("4_5_hr_insdet", tables_4_5(hr45)))
    owl = {r["dataset"]: r for r in runs if r.get("kind") == "comparison_run" and r.get("method") == "owlv2_image_guided"}
    if "hr_insdet" in owl and len(exp44) == len(RU.RHO_RUNS) and len(hr45) == len(RU.RHO_RUNS):
        made.append(_write("4_6_hr_insdet", tables_4_6_hr_insdet(base_hr, exp44, hr45, owl["hr_insdet"])))
        made.append(_write("latency", tables_latency(exp44, owl)))
    made.append(_write("4_6_published", tables_4_6_published()))
    if all((TABLES / f"{k.split(':')[0]}.md").is_file() for k in CHAPTER_TABLES + APPENDIX_TABLES):
        made.append(write_print())
    print("\n".join(made) if made else "в журнале нет записей")


if __name__ == "__main__":
    main()
