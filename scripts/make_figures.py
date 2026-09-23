"""Рисунки главы 4 — только из журнала прогонов `experiments/runs/*.json`, без нового счёта.

    python scripts/make_figures.py → experiments/figures/Рисунок_*.png

Рисунок 4.1 — полнота сегментатора при IoU ≥ 0,5 по бинам площади рамки без слоя окон и с ним, калибровочные, тестовые и все сцены
    (4.2; те же числа, что таблица 4.2-2 файла `experiments/tables/4_2_hr_insdet.md`).
Рисунок 4.2 — AP по бинам площади рамки для всех вариантов φ и обоих энкодеров (сетка 4.3, 120 тестовых сцен, полная галерея;
    те же числа, что таблица 4.3-2 файла `experiments/tables/4_3_hr_insdet.md`).
Рисунок 4.3 — строки метода и OWLv2 рядом с опубликованными результатами instance detection на HR-InsDet с пометкой дообучения
    (опубликованные — `src.eval.published`, как в таблице 4.6-1-1s; строки работы — AP на 160 сценах, как в таблице 4.6-1-3s).
Рисунок «кривые полноты сегментатора по порогу IoU 0,5–0,95» не строится: запись 4.2 хранит полноту только при IoU 0,5, 0,75
    и среднее по 0,5:0,95 (`src.eval.recall.summarize`), а пересчёт по кешу масок был бы новым счётом.

Каждое число рисунка сверяется с ячейкой соответствующей таблицы `experiments/tables/` (та же запись, то же округление);
расхождение останавливает сборку. Оформление: один масштаб оси на рисунок, серии различаются цветом и формой маркера или
штриховкой, палитра проверена на различимость при нарушениях цветовосприятия, шрифт — Liberation Serif (метрический аналог
Times New Roman; самого Times New Roman в образе нет).
"""
from __future__ import annotations

import json
import numpy as np
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.encode.variants import VARIANTS, grid  # noqa: E402
from src.eval import published as PB  # noqa: E402
from scripts import make_tables as MT  # noqa: E402

RUNS = Path("experiments/runs")
TABLES = Path("experiments/tables")
TEXT = Path("experiments/figures")
FIG_DATA = Path("experiments/figures")

for p in Path("/usr/share/fonts").rglob("LiberationSerif*.ttf"):
    fm.fontManager.addfont(str(p))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#6b6b6b", "axes.linewidth": 0.6, "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
                     "axes.labelcolor": "#1a1a1a", "grid.color": "#d9d9d9", "grid.linewidth": 0.5, "legend.frameon": False})

CM = 1 / 2.54
WIDTH = 16 * CM       # ширина полосы набора (A4, поля 3 и 1,5 см)
DPI = 300

# палитра (проверена валидатором dataviz, light, все пары): семейство заполнителя b → цвет, контекст α → штрих и маркер
HUE = {"0": "#3a6fb0", "mean": "#e08a2a", "blur": "#2a9d8f", None: "#c0392b"}
INK = "#1a1a1a"
AREAS = [("small", "< 200²"), ("medium", "200²–400²"), ("large", "> 400²")]


def _cell(text: str, tid: str, row_prefix: list[str], col: int) -> str:
    """Ячейка таблицы `tid` файла markdown: строка, начинающаяся с ячеек `row_prefix`, столбец `col` (0 — первый после «|»).
    Заголовок таблицы — «## Таблица <tid>.» либо «## Таблица 4.N (<tid>)» / «## Таблица А.N (<tid>)» у нумерованных."""
    m = re.search(r"^## Таблица (?:[^\n(]*\()?" + re.escape(tid) + r"\)?[.\s—]", text, re.M)
    if m is None:
        raise SystemExit(f"нет таблицы {tid}")
    sect = text[m.end():]
    for line in sect.split("\n"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[: len(row_prefix)] == row_prefix:
            return cells[col]
    raise SystemExit(f"в таблице нет строки {row_prefix}")


# ---------------------------------------------------------------- рисунок 4.1: полнота сегментатора по бинам площади, 4.2
def figure_recall_bins() -> dict:
    rec = json.loads((RUNS / "4_2_hr_insdet.json").read_text())
    table = (TABLES / "4_2_hr_insdet.md").read_text()
    subsets = [("cal/all", "калибровочные", "40 калибровочных сцен"), ("test/all", "тестовые", "120 тестовых сцен"),
               ("all/all", "все", "все 160 сцен")]
    data = {"source": "experiments/runs/4_2_hr_insdet.json → results[crop_n_layers][subset].by_area[bin].recall_50",
            "checked_against": "experiments/tables/4_2_hr_insdet.md, таблица 4.2-2", "unit": "recall при IoU ≥ 0,5, %", "bars": []}
    style = {"0": dict(color="#cfd8e3", edgecolor="#5b7fa6", label="без слоя окон (вход 1024)"),
             "1": dict(color="#3a6fb0", edgecolor="#3a6fb0", label="один слой окон 2×2 (вход 2048)")}
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 6.6 * CM), sharey=True)
    w = 0.36
    for ax, (key, tname, title) in zip(axes, subsets):
        ax.set_title(title, fontsize=9.5, color=INK, pad=6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        for j, cnl in enumerate(("0", "1")):
            ys, lo, hi = [], [], []
            for a, a_name in AREAS:
                m = rec["results"][cnl][key]["by_area"][a]
                ys.append(100 * m["recall_50"]); lo.append(100 * m["ci95"]["recall_50"][0]); hi.append(100 * m["ci95"]["recall_50"][1])
                shown = _cell(table, "4.2-2", [f"{cnl} ({MT.EFFECTIVE[cnl]})", tname, a_name], 4)
                if shown.split(" ")[0] != MT.pct(m["recall_50"]):
                    raise SystemExit(f"расхождение с таблицей 4.2-2: {cnl} {tname} {a_name}: {shown} против {MT.pct(m['recall_50'])}")
                data["bars"].append({"subset": key, "crop_n_layers": int(cnl), "bin": a, "recall_50": round(ys[-1], 2),
                                     "ci95": [round(lo[-1], 2), round(hi[-1], 2)]})
            x = [i + (j - 0.5) * (w + 0.04) for i in range(3)]
            st = style[cnl]
            ax.bar(x, ys, width=w, color=st["color"], edgecolor=st["edgecolor"], linewidth=0.7, label=st["label"])
            ax.errorbar(x, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]], fmt="none",
                        ecolor=INK, elinewidth=0.7, capsize=2)
        ax.set_xticks(range(3), [n for _, n in AREAS])
        ax.set_xlim(-0.6, 2.6)
    axes[0].set_ylabel("полнота при IoU ≥ 0,5, %", fontsize=9)
    axes[0].set_ylim(0, 100)
    axes[1].set_xlabel("площадь размеченной рамки, пикс.² исходного снимка", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, handlelength=1.8, columnspacing=2.0, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.3, wspace=0.08)
    out = TEXT / "Рисунок_4-1.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    data["file"] = str(out)
    return data


# ---------------------------------------------------------------- рисунок 4.2: AP по бинам площади, сетка 4.3
def figure_ap_bins() -> dict:
    recs = {}
    for enc, var, ds in grid():
        if ds != "hr_insdet":
            continue
        recs[(enc, var)] = json.loads((RUNS / f"{enc}_{var}_{ds}.json").read_text())
    table = (TABLES / "4_3_hr_insdet.md").read_text()
    data = {"source": "experiments/runs/<encoder>_<variant>_hr_insdet.json → metrics.full.subsets['test/all'].by_area[bin].ap",
            "checked_against": "experiments/tables/4_3_hr_insdet.md, таблица 4.3-2", "unit": "AP, %", "series": []}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 7.6 * CM), sharey=True)
    handles = {}
    for ax, enc in zip(axes, ("dinov2", "dinov3")):
        ax.set_title({"dinov2": "DINOv2 ViT-L/14", "dinov3": "DINOv3 ViT-L/16"}[enc], fontsize=9.5, color=INK, pad=6)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        for (e, var), rec in recs.items():
            if e != enc:
                continue
            v = VARIANTS[var]
            ys, lo, hi = [], [], []
            for a, a_name in AREAS:
                m = rec["metrics"]["full"]["subsets"]["test/all"]["by_area"][a]
                ys.append(100 * m["ap"]); lo.append(100 * m["ci95"]["ap"][0]); hi.append(100 * m["ci95"]["ap"][1])
                shown = _cell(table, "4.3-2", [enc, v.label, a_name], 5)
                if shown.split(" ")[0] != MT.pct(m["ap"]):
                    raise SystemExit(f"расхождение с таблицей 4.3-2: {enc} {v.label} {a_name}: {shown} против {MT.pct(m['ap'])}")
            data["series"].append({"encoder": enc, "variant": v.label, "ap": [round(y, 2) for y in ys],
                                   "ci95": [[round(a, 2), round(b, 2)] for a, b in zip(lo, hi)]})
            color = HUE[v.b]
            ls = {1.0: "-", 1.5: "--", None: "-" if not v.debias else ":"}[v.alpha]
            marker = {1.0: "o", 1.5: "s", None: "^" if not v.debias else "D"}[v.alpha]
            x = list(range(3))
            ax.errorbar(x, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                        fmt="none", ecolor=color, elinewidth=0.6, capsize=2, alpha=0.55)
            (line,) = ax.plot(x, ys, ls=ls, lw=1.4, color=color, marker=marker, ms=4.5, mfc="white" if v.alpha == 1.5 else color,
                              mew=1.2)
            handles.setdefault(v.label, line)
        ax.set_xticks(range(3), [n for _, n in AREAS])
        ax.set_xlim(-0.35, 2.35)
        ax.set_xlabel("площадь размеченной рамки, пикс.² исходного снимка", fontsize=8.5)
    axes[0].set_ylabel("AP, %", fontsize=9)
    axes[0].set_ylim(0, 90)
    order = [VARIANTS[v].label for v in VARIANTS]
    names = [k.replace("⊥", r"$^{\perp}$").replace(",1.", ",1,") for k in order]  # «⊥» — из mathtext (в Liberation Serif его нет)
    fig.legend([handles[k] for k in order], names, loc="lower center", ncol=4, fontsize=8, handlelength=2.6,
               columnspacing=1.6, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.91, bottom=0.33, wspace=0.08)
    out = TEXT / "Рисунок_4-2.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    data["file"] = str(out)
    return data


# ---------------------------------------------------------------- рисунок 4.3: строки работы рядом с опубликованными
# подпись строки и признак «в пайплайне есть модель, обученная с текстом»: такие подписи набираются курсивом, символы-сноски
# в подписях не используются
SHORT = {
    "insdet_sam_dinov2": ("SAM + DINOv2 [5]", False),
    "insdet_sam_dinov2_readme_vitl": ("SAM + DINOv2 ViT-L/14, репозиторий [5]", False),
    "otsfm_gdino": ("Связка без обучения с Grounding DINO [8]", True),
    "otsfm_gdino_nerf_gallery": ("Связка с Grounding DINO и синтезированными ракурсами [8]", True),
    "idow_sam": ("IDOW SAM [8]", False),
    "idow_gdino": ("IDOW Grounding DINO [8]", True),
    "nidsnet_gs_cls": ("NIDS-Net, CLS-токен [7]", True),
    "nidsnet_no_adapter": ("NIDS-Net без адаптера [7]", True),
    "nidsnet_wa": ("NIDS-Net с адаптером [7]", True),
    "l2gdet": ("L2G-Det [9]", True),
}
OURS = {  # подпись строки таблицы 4.6-1-3s → подпись на рисунке
    "dinov2, C(0,1.0) — без отбора (ближайшая к [5])": "C(0,1,0) на DINOv2, контрольный прогон",
    "dinov2, C(x̄,1.0) — без отбора (лучший φ)": "C(x̄,1,0) на DINOv2, контрольный прогон",
    "dinov3, C(blur,1.5) — без отбора (лучший φ)": "C(blur,1,5) на DINOv3, контрольный прогон",
    "dinov2, C(x̄,1.0) — метод с ρ, без порога": "метод: C(x̄,1,0) на DINOv2, с ρ",
    "dinov3, C(blur,1.5) — метод с ρ, без порога": "метод: C(blur,1,5) на DINOv3, с ρ",
    "OWLv2 image-guided (обучен с текстом)": "OWLv2 image-guided",
}


def figure_published() -> dict:
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
    base = {r["grid_run"]: r for r in runs if r.get("kind") == "control_run" and r["dataset"] == "hr_insdet"}
    exp44 = {r["run_id"]: r for r in runs if r.get("kind") == "exp_4_4"}
    exp45 = {r["run_id"]: r for r in runs if r.get("kind") == "exp_4_5" and r["dataset"] == "hr_insdet"}
    owl = next(r for r in runs if r.get("kind") == "comparison_run" and r.get("method") == "owlv2_image_guided"
               and r["dataset"] == "hr_insdet")
    table = (TABLES / "4_6_hr_insdet.md").read_text()
    bars = []  # (подпись, AP, lo, hi, вид)
    for row in MT._rows_4_6(base, exp44, exp45, owl):
        if row.kind == "thr":
            continue
        m = row.ap("full", "all/all")
        shown = _cell(table, "4.6-1-3s", [row.label], 4)
        if shown.split(" ")[0] != MT.pct(m["ap"]):
            raise SystemExit(f"расхождение с таблицей 4.6-1-3s: {row.label}: {shown} против {MT.pct(m['ap'])}")
        kind = "owlv2" if row.kind == "owlv2" else ("rho" if row.kind == "rho" else "control")
        bars.append((OURS[row.label], 100 * m["ap"], 100 * m["ci95"]["ap"][0], 100 * m["ci95"]["ap"][1], kind, kind == "owlv2"))
    for p in PB.GROUP_A:
        shown = _cell(table, "4.6-1-1s", [p.method], 4)
        if shown != PB.fmt(p, "ap"):
            raise SystemExit(f"расхождение с таблицей 4.6-1-1s: {p.method}: {shown} против {PB.fmt(p, 'ap')}")
        bars.append((SHORT[p.key][0], p.values["ap"], None, None, "pub_ft" if p.finetune else "pub", SHORT[p.key][1]))
    bars.sort(key=lambda b: b[1])
    data = {"source": "experiments/runs/*__baseline.json, 4_4_*.json (rho.metrics), owlv2_hr_insdet.json — AP на 160 сценах (all/all), "
                      "полная галерея; опубликованные — src/eval/published.py (GROUP_A)",
            "checked_against": "experiments/tables/4_6_hr_insdet.md, таблицы 4.6-1-3s и 4.6-1-1s", "unit": "AP, %",
            "bars": [{"label": b[0], "ap": round(b[1], 2), "ci95": None if b[2] is None else [round(b[2], 2), round(b[3], 2)],
                      "kind": b[4], "text_trained_model_in_pipeline": b[5]} for b in bars]}
    style = {"control": dict(color="#a9c3e3", edgecolor="#3a6fb0", hatch=None),
             "rho": dict(color="#3a6fb0", edgecolor="#3a6fb0", hatch=None),
             "owlv2": dict(color="#e08a2a", edgecolor="#e08a2a", hatch=None),
             "pub": dict(color="#d3d3d3", edgecolor="#8c8c8c", hatch=None),
             "pub_ft": dict(color="#d3d3d3", edgecolor="#5a5a5a", hatch="////")}
    plt.rcParams["hatch.linewidth"] = 0.6
    fig, ax = plt.subplots(figsize=(WIDTH, 9.4 * CM))
    y = list(range(len(bars)))
    for yi, (label, ap, lo, hi, kind, _) in zip(y, bars):
        st = style[kind]
        ax.barh(yi, ap, height=0.68, color=st["color"], edgecolor=st["edgecolor"], hatch=st["hatch"], linewidth=0.7)
        if lo is not None:
            ax.errorbar(ap, yi, xerr=[[ap - lo], [hi - ap]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
    ax.set_yticks(y, [b[0] for b in bars], fontsize=8)
    for lab, b in zip(ax.get_yticklabels(), bars):  # курсив — в пайплайне есть модель, обученная с текстом
        if b[5]:
            lab.set_fontstyle("italic")
    ax.set_xlim(0, 80)
    ax.set_xlabel("AP, %", fontsize=9)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    legend = [Patch(facecolor="#3a6fb0", edgecolor="#3a6fb0", label="метод с ρ (эта работа)"),
              Patch(facecolor="#a9c3e3", edgecolor="#3a6fb0", label="контрольный прогон без отбора (эта работа)"),
              Patch(facecolor="#e08a2a", edgecolor="#e08a2a", label="OWLv2 image-guided (эта работа)"),
              Patch(facecolor="#d3d3d3", edgecolor="#8c8c8c", label="опубликовано, без дообучения"),
              Patch(facecolor="#d3d3d3", edgecolor="#5a5a5a", hatch="////", label="опубликовано, с дообучением")]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=7.8, handlelength=1.8, columnspacing=1.4,
               bbox_to_anchor=(0.55, 0.0))
    fig.subplots_adjust(left=0.36, right=0.99, top=0.99, bottom=0.2)
    out = TEXT / "Рисунок_4-3.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    data["file"] = str(out)
    return data




# ---------------------------------------------------------------- рисунки главы 2: иллюстрации на одном объекте калибровочной сцены
# Строятся из снимка датасета и кеша масок SAM 2 на процессоре, без энкодера и без нового счёта; нужны каталоги data/ и cache/
# (в репозиторий не входят), без них шаг пропускается. Объект выбирается детерминированно: первая по порядку оракульная маска
# калибровочной сцены, у которой есть родитель-почти-дубликат и вложенная часть, а сторона рамки лежит в удобном для печати
# диапазоне.
CH2_MIN_SIDE, CH2_MAX_SIDE = 350, 1400  # пикселей исходного снимка


def _ch2_setup():
    from fractions import Fraction

    from src.data import hr_insdet as HR
    from src.eval.oracle import oracle_assign
    from src.segment import cache as MC
    from src.segment import sam2 as S
    from src.select import rho as R

    split = json.loads(Path("splits/hr_insdet.json").read_text())
    root = Path(split["root"])
    rec = json.loads((RUNS / "4_2_hr_insdet.json").read_text())
    key = rec["mask_cache"][str(rec["selection"]["selected_crop_n_layers"])]["key"]
    if not root.is_dir() or not Path("cache/masks").is_dir():
        return None
    cache = MC.MaskCache("hr_insdet", key)
    scenes = {s["id"]: s for s in HR.scenes(root)}
    theta, gamma = Fraction(9, 10), Fraction(4, 5)
    for sid in split["cal"]:
        if not cache.has(sid):
            continue
        entry = cache.load(sid)
        gt = HR.scene_gt(scenes[sid], split["labels"], root)
        gt_boxes = np.array([g["box"] for g in gt], float)
        assign = oracle_assign(gt_boxes, entry.boxes)
        area, inter = R.geometry([r["rle"] for r in entry.records])
        inside, _ = R.nesting(area, inter, theta)
        for g, m in zip(gt, assign):
            if m < 0:
                continue
            x0, y0, x1, y1 = g["box"]
            if not CH2_MIN_SIDE <= max(x1 - x0, y1 - y0) <= CH2_MAX_SIDE:
                continue
            parents = [p for p in inside[m] if gamma * area[p] <= area[m]]                       # почти-дубликаты
            parts = [c for c in range(len(area)) if m in inside[c] and area[c] < gamma * area[m]]  # существенно вложенные
            if parents and parts:
                p = max(parents, key=lambda k: area[m] / area[k])
                c = max(parts, key=lambda k: area[k])
                img = S.read_rgb(root / scenes[sid]["image"])
                return dict(scene=sid, label=g["label"], m=int(m), parent=int(p), part=int(c), entry=entry, img=img,
                            area=area, box=entry.records[m]["box"])
    return None


def _outline(ax, mask: np.ndarray, color: str, lw: float = 1.2, fill: bool = False) -> None:
    if fill:
        ax.imshow(np.dstack([np.full(mask.shape, 1.0)] * 3 + [mask * 0.35]) * np.array([*_rgb(color), 1.0]))
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _rgb(color: str) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb
    return to_rgb(color)


def figure_2_2() -> dict | None:
    """Рисунок 2.2 — входы энкодера у вариантов φ: шесть вырезок C(b, α) и окно карты целого кадра для z_pool."""
    from src.encode import crop as C
    from src.encode import pool as PL

    ex = _ch2_setup()
    if ex is None:
        print("рисунок 2.2 пропущен: нет data/ или cache/")
        return None
    img, entry, m = ex["img"], ex["entry"], ex["m"]
    box = ex["box"]
    win = entry.windower(m)
    fig = plt.figure(figsize=(WIDTH, 10.6 * CM))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.32, wspace=0.08)
    # сцена вокруг объекта с контуром маски
    qx, qy, side = C.square(box, 2.4)
    h, w = img.shape[:2]
    ax = fig.add_subplot(gs[0, 0:2])
    x0, y0, x1, y1 = max(qx, 0), max(qy, 0), min(qx + side, w), min(qy + side, h)
    ax.imshow(img[y0:y1, x0:x1])
    _outline(ax, win(x0, y0, x1, y1), "#e08a2a", 1.4)
    ax.set_title("объект в сцене и маска сегментатора", fontsize=8.5, color=INK)
    ax.set_axis_off()
    # окно карты целого кадра: вход энкодера 1536×2048, сетка патчей 14 px, патчи под маской
    patch = 14
    x_in, in_hw = PL.encoder_input(img, patch)
    m_seg = entry.decode(m)
    grid = PL.patch_grid_mask(m_seg, in_hw, patch)
    n_patches = int((grid >= 0.5).sum())
    sx, sy = entry.scale_xy
    bx0, by0, bx1, by1 = box[0] / sx, box[1] / sy, box[2] / sx, box[3] / sy  # scale_xy: вход S → исходный снимок
    pad = max(bx1 - bx0, by1 - by0) * 0.7
    gx0, gy0 = int(max(bx0 - pad, 0)) // patch * patch, int(max(by0 - pad, 0)) // patch * patch
    gx1, gy1 = min(int(bx1 + pad), x_in.shape[1]), min(int(by1 + pad), x_in.shape[0])
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(x_in[gy0:gy1, gx0:gx1])
    for gx in range(0, gx1 - gx0, patch):
        ax.axvline(gx - 0.5, color="white", lw=0.25, alpha=0.4)
    for gy in range(0, gy1 - gy0, patch):
        ax.axhline(gy - 0.5, color="white", lw=0.25, alpha=0.4)
    cells = np.kron(grid[gy0 // patch:(gy1 + patch - 1) // patch, gx0 // patch:(gx1 + patch - 1) // patch] >= 0.5,
                    np.ones((patch, patch)))[: gy1 - gy0, : gx1 - gx0]
    ax.imshow(np.dstack([np.ones_like(cells) * c for c in _rgb("#e08a2a")] + [cells * 0.45]))
    ax.set_title(f"$P$: сетка патчей входа 1536×2048,\nпод маской {n_patches} патчей", fontsize=8.5, color=INK)
    ax.set_axis_off()
    labels = {"0": "0", "mean": r"\bar x", "blur": r"\mathrm{blur}"}
    for r_, alpha in enumerate((1.0, 1.5), start=1):
        for c_, b in enumerate(("0", "mean", "blur")):
            ax = fig.add_subplot(gs[r_, c_])
            ax.imshow(C.make_crop(img, box, win, b, alpha))
            ax.set_title(f"$C({labels[b]},\\,{alpha:.1f})$".replace(".", "{,}"), fontsize=9, color=INK)
            ax.set_axis_off()
    out = TEXT / "Рисунок_2-2.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {"file": str(out), "scene": ex["scene"], "label": ex["label"], "mask": m, "box": box,
            "patches_under_mask_at_pool_input": n_patches, "source": "data/InsDet-FULL + cache/masks (SAM 2, один слой окон)"}


def figure_2_3() -> dict | None:
    """Рисунок 2.3 — лес вложенности на одном объекте: маска объекта, родитель-почти-дубликат и вложенная часть."""
    from src.encode import crop as C

    ex = _ch2_setup()
    if ex is None:
        print("рисунок 2.3 пропущен: нет data/ или cache/")
        return None
    img, entry, area = ex["img"], ex["entry"], ex["area"]
    m, p, c = ex["m"], ex["parent"], ex["part"]
    pbox = entry.records[p]["box"]
    qx, qy, side = C.square(pbox, 1.2)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(qx, 0), max(qy, 0), min(qx + side, w), min(qy + side, h)
    ratio_pm = area[m] / area[p]
    ratio_cm = area[c] / area[m]
    panels = [(m, "маска объекта $m$", "#3a6fb0"),
              (p, "родитель $m_b$, почти-дубликат\n" + f"$|m|/|m_b|={ratio_pm:.2f}$".replace(".", "{,}"), "#e08a2a"),
              (c, "часть $m_a$, существенное вложение\n" + f"$|m_a|/|m|={ratio_cm:.2f}$".replace(".", "{,}"), "#2a9d8f")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 6.4 * CM))
    for ax, (k, title, color) in zip(axes, panels):
        ax.imshow(img[y0:y1, x0:x1])
        _outline(ax, entry.window(k, x0, y0, x1, y1), color, 1.3, fill=True)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.set_axis_off()
    fig.subplots_adjust(wspace=0.05)
    out = TEXT / "Рисунок_2-3.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return {"file": str(out), "scene": ex["scene"], "label": ex["label"], "mask": m, "parent": p, "part": c,
            "area_ratio_object_to_parent": round(ratio_pm, 4), "area_ratio_part_to_object": round(ratio_cm, 4),
            "theta": "9/10", "gamma": "4/5", "source": "data/InsDet-FULL + cache/masks (SAM 2, один слой окон)"}

def _fit_width(path: str) -> None:
    """Разрешение в метаданных PNG: pandoc берёт из него размер картинки — ровно ширина полосы 16 см."""
    from PIL import Image

    with Image.open(path) as im:
        dpi = im.size[0] / (16 / 2.54)
        im.save(path, dpi=(dpi, dpi))


def main() -> None:
    FIG_DATA.mkdir(parents=True, exist_ok=True)
    for name, fn in (("2-2", figure_2_2), ("2-3", figure_2_3), ("4-1", figure_recall_bins), ("4-2", figure_ap_bins),
                     ("4-3", figure_published)):
        data = fn()
        if data is None:
            continue
        _fit_width(data["file"])
        print(data["file"])


if __name__ == "__main__":
    main()
