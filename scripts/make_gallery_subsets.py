"""Списки подвыборок галереи эксперимента 4.5 по seed разбиения.

    python scripts/make_gallery_subsets.py          # → splits/gallery_subsets_hr_insdet.json

Правило — `src.gallery.subsets`: 9 вложенных цепочек, 25 ⊂ 50 ⊂ 100 экземпляров HR-InsDet. Без GPU, без данных и без
галерей: читается только `splits/hr_insdet.json`. Файл закоммичен вместе с константами 4.5 до первого запуска
`scripts/run_4_5.py`; существующий файл не перезаписывается — повторный запуск только сверяет его с пересчётом.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gallery import subsets as SB  # noqa: E402


def main() -> None:
    for dataset in SB.DATASETS:
        split = json.loads(Path(f"splits/{dataset}.json").read_text())
        rec, p = SB.build(dataset, split), SB.path(dataset)
        if p.exists():
            same = json.loads(p.read_text()) == rec
            print(f"{p}: уже есть — {'совпадает с пересчётом по seed' if same else 'РАСХОДИТСЯ с пересчётом'}; не перезаписывается")
            if not same:
                raise SystemExit(1)
            continue
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
        print(f"{p}: цепочек {rec['n_chains']}, объёмы {rec['sizes']} ({rec['unit']}), seed {rec['seed']}; начало цепочки 0 — "
              f"{rec['chains'][0]['order'][:3]}")


if __name__ == "__main__":
    main()
