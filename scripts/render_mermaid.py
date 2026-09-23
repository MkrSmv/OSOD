"""Схемы 2.1 и 3.1 из Mermaid в картинки: experiments/figures/Рисунок_N-M.mmd → experiments/figures/Рисунок_N-M.png.

Штатный `mmdc` (@mermaid-js/mermaid-cli) требует headless Chromium, поэтому рендер идёт npm-пакетом `beautiful-mermaid`
(flowchart в SVG без браузера, раскладка elkjs), который ставится `npm install` в рабочий каталог WORK (переменная
MERMAID_WORK, по умолчанию `cache/mermaid`) и в репозиторий не входит; растр — `resvg-py` (обёртка resvg).

Сверх рендера: `beautiful-mermaid` выдаёт цвета CSS-переменными и `color-mix()`, которых resvg не понимает, поэтому они
заменяются явными значениями (ч/б оформление из директивы init в .mmd: чёрные линии и рамки, белый фон; заливки узлов из
блока classDef сохраняются); теги `<sub>`/`<sup>` в подписях, которые `beautiful-mermaid` выбрасывает, возвращаются как
tspan со сдвигом базовой линии; шрифт — Liberation Serif (метрически совместим с Times New Roman), для отсутствующих
глифов — DejaVu Serif.

    python scripts/render_mermaid.py            # все experiments/figures/Рисунок_*.mmd
    python scripts/render_mermaid.py 2-1        # один рисунок
"""
from __future__ import annotations

import html
import json
import os
import pathlib
import re
import subprocess
import sys

TEXT = pathlib.Path("experiments/figures")
WORK = pathlib.Path(os.environ.get("MERMAID_WORK", "cache/mermaid")).resolve()
PACKAGE = "beautiful-mermaid@1.1.3"
ZOOM = 3.0  # масштаб растра: ширина схемы 2.1 около 5 300 пикселей — достаточно для печати на ширину полосы

# ч/б оформление (как в директиве init файлов .mmd): все производные цвета из fg/bg заменяются явно
COLORS = {"bg": "#ffffff", "fg": "#000000", "line": "#000000", "accent": "#000000", "muted": "#000000",
          "surface": "#ffffff", "border": "#000000"}
RESOLVED = {"--_text": "#000000", "--_text-sec": "#000000", "--_text-muted": "#000000", "--_text-faint": "#404040",
            "--_line": "#000000", "--_arrow": "#000000", "--_node-fill": "#ffffff", "--_node-stroke": "#000000",
            "--_group-fill": "#ffffff", "--_group-hdr": "#f2f2f2", "--_inner-stroke": "#d9d9d9", "--_key-badge": "#e6e6e6",
            "--bg": "#ffffff", "--fg": "#000000", "--line": "#000000", "--accent": "#000000", "--muted": "#000000",
            "--surface": "#ffffff", "--border": "#000000"}

# Times New Roman в образе нет: Liberation Serif метрически совместим с ним, DejaVu Serif (в составе matplotlib) даёт глифы
# «∈», «⊥» и подобные, которых в Liberation нет; resvg подставляет их по семейству
FONT_STACK = "'Liberation Serif', 'DejaVu Serif', serif"  # без «Times New Roman»: fontdb подменяет его Liberation Sans
FONT_DIRS = ["/usr/share/fonts"] + [str(pathlib.Path(__import__("matplotlib").get_data_path()) / "fonts" / "ttf")]

NODE_JS = """
import { renderMermaid } from 'beautiful-mermaid';
import fs from 'fs';
const [src, out, opts] = process.argv.slice(2);
const svg = await renderMermaid(fs.readFileSync(src, 'utf8'), JSON.parse(opts));
fs.writeFileSync(out, svg);
"""


def ensure_package() -> None:
    if (WORK / "node_modules" / "beautiful-mermaid" / "package.json").is_file():
        return
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "package.json").is_file():
        (WORK / "package.json").write_text('{"name": "mermaid-render", "private": true, "type": "module"}\n')
    subprocess.run(["npm", "install", "--no-audit", "--no-fund", PACKAGE], cwd=WORK, check=True)


def resolve_css(svg: str) -> str:
    svg = re.sub(r"@import url\([^)]*\);", "", svg)
    for name, value in RESOLVED.items():
        svg = re.sub(r"var\(" + re.escape(name) + r"(?:,[^()]*(?:\([^()]*\))?[^()]*)?\)", value, svg)
    assert "var(--" not in svg, "остались нераскрытые CSS-переменные"
    return svg


def sub_sup_lines(mmd: str) -> dict[str, str]:
    """Строки подписей с <sub>/<sup>: текст без тегов → текст с tspan-разметкой (как их выдаёт beautiful-mermaid)."""
    out = {}
    for label in re.findall(r'"([^"]*)"', mmd):
        for line in re.split(r"<br\s*/?>", label):
            if "<sub>" not in line and "<sup>" not in line:
                continue
            plain = html.escape(re.sub(r"</?su[bp]>", "", line), quote=False)
            marked = re.sub(r"<sub>(.*?)</sub>", lambda m: f'<tspan baseline-shift="sub" font-size="75%">{m.group(1)}</tspan>', line)
            marked = re.sub(r"<sup>(.*?)</sup>", lambda m: f'<tspan baseline-shift="super" font-size="75%">{m.group(1)}</tspan>', marked)
            out[plain.strip()] = marked.strip()
    return out


def restore_sub_sup(svg: str, table: dict[str, str]) -> str:
    def sub(m):
        text = m.group(2).strip()
        return m.group(1) + table[text] + m.group(3) if text in table else m.group(0)
    return re.sub(r"(<tspan[^>]*>)([^<]*)(</tspan>)", sub, svg)


def center_group_labels(svg: str) -> str:
    """Заголовок группы (subgraph) — по центру полосы заголовка, а не у левого края: у схемы 2.1 стрелка «галерея → поиск» входит
    в группу «Инференс» слева и пересекала бы подпись."""
    pat = re.compile(r'(<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)" height="28"[^>]*/>\s*<text x=")[\d.]+(" [^>]*font-weight="600")')
    return pat.sub(lambda m: f'{m.group(1)}{float(m.group(2)) + float(m.group(3)) / 2:.2f}{m.group(4)} text-anchor="middle"', svg)


def render(stem: str) -> pathlib.Path:
    import resvg_py

    src = TEXT / f"{stem}.mmd"
    svg_path = WORK / f"{stem}.svg"
    opts = dict(COLORS, font="Liberation Serif", padding=24)
    (WORK / "run.mjs").write_text(NODE_JS)  # файлом, а не `node -e`: await верхнего уровня в `-e` не выполняется
    subprocess.run(["node", "run.mjs", str(src.resolve()), str(svg_path), json.dumps(opts)], cwd=WORK, check=True)
    svg = restore_sub_sup(resolve_css(svg_path.read_text()), sub_sup_lines(src.read_text()))
    svg = center_group_labels(svg)
    svg = re.sub(r"font-family:[^;}\"]*", "font-family: " + FONT_STACK, svg, count=1)
    svg = svg.replace("<svg ", f'<svg font-family="{FONT_STACK}" ', 1)
    assert FONT_STACK in svg.split("<defs>")[0]
    svg_path.write_text(svg)
    png = resvg_py.svg_to_bytes(svg_string=svg, zoom=ZOOM, background="#ffffff", font_family="Liberation Serif",
                                serif_family="Liberation Serif", font_dirs=FONT_DIRS)
    out = TEXT / f"{stem}.png"
    out.write_bytes(png)
    from PIL import Image  # разрешение в метаданных: pandoc берёт из него размер картинки — ширина полосы 16 см

    with Image.open(out) as im:
        dpi = im.size[0] / (16 / 2.54)
        im.save(out, dpi=(dpi, dpi))
    return out


def main() -> None:
    ensure_package()
    stems = [f"Рисунок_{a}" for a in sys.argv[1:]] or sorted(p.stem for p in TEXT.glob("Рисунок_*.mmd"))
    for stem in stems:
        out = render(stem)
        print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
