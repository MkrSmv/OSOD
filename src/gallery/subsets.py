"""Подвыборки галереи эксперимента 4.5 — HR-InsDet.

Перестановка 100 меток (в порядке `splits/hr_insdet.json`, `labels`) генератором `np.random.default_rng(seed + r)`,
первые 25 ⊂ первые 50 ⊂ все 100; у каждой метки — все её эталоны. Цепочек — `rules.GALLERY_CHAINS`, seed — из
`splits/hr_insdet.json`. Списки пишет `scripts/make_gallery_subsets.py` в `splits/gallery_subsets_hr_insdet.json`; они
закоммичены до первого запуска `scripts/run_4_5.py`, а прогон читает только файл и сверяет его с пересчётом по seed.
Подвыборка — подмножество строк той же галереи: `emb.npy` общий.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval import rules as RU

FORMAT = 1
DATASETS = ("hr_insdet",)
RULE = {
    "hr_insdet": "перестановка меток (порядок `labels` разбиения) генератором np.random.default_rng(seed + r); подвыборка "
                 "объёма n — первые n меток перестановки со всеми их эталонами",
}


def path(dataset: str) -> Path:
    return Path(f"splits/gallery_subsets_{dataset}.json")


def build(dataset: str, split: dict) -> dict:
    """Списки всех цепочек по разбиению `splits/<dataset>.json` (метки, seed)."""
    if dataset not in DATASETS:
        raise ValueError(f"подвыборки галереи — только {DATASETS}: {dataset}")
    seed, labels = int(split["seed"]), list(split["labels"])
    chains = []
    for r in range(RU.GALLERY_CHAINS):
        rng = np.random.default_rng(seed + r)
        chains.append({"r": r, "seed": seed + r, "order": [labels[i] for i in rng.permutation(len(labels))]})
    return {"format": FORMAT, "dataset": dataset, "split_file": f"splits/{dataset}.json", "split_written": split["written"],
            "seed": seed, "n_chains": RU.GALLERY_CHAINS, "sizes": list(RU.GALLERY_SUBSET_SIZES),
            "unit": "экземпляров (меток)", "rule": RULE[dataset], "chains": chains}


def load(dataset: str) -> dict:
    """Закоммиченные списки; расхождение с пересчётом по seed — ошибка (состав по результатам не выбирается)."""
    rec = json.loads(path(dataset).read_text())
    split = json.loads(Path(f"splits/{dataset}.json").read_text())
    if rec != build(dataset, split):
        raise ValueError(f"{path(dataset)}: списки подвыборок расходятся с пересчётом по seed разбиения")
    return rec


def chain_ids(rec: dict, chain: dict, size: int) -> list[str]:
    """Метки подвыборки объёма `size` цепочки."""
    if size not in rec["sizes"]:
        raise ValueError(f"объём {size} вне {rec['sizes']}")
    return list(chain["order"][:size])


def rows(gallery, rec: dict, chain: dict, size: int) -> np.ndarray:
    """Строки полной галереи, входящие в подвыборку (по возрастанию номера строки); помеченных `deleted` в ней нет."""
    full = gallery.rows("full")
    ids = set(chain_ids(rec, chain, size))
    out = np.array([int(i) for i in full if gallery.meta[i]["label"] in ids], np.int64)
    labels = [gallery.meta[i]["label"] for i in out]
    counts = np.unique(labels, return_counts=True)[1]
    if len(set(labels)) != size or len(set(counts)) != 1:
        raise ValueError(f"подвыборка {size} меток: в галерее найдено {len(set(labels))} меток, эталонов на метку {set(counts)}")
    return out


def name(r: int, size: int) -> str:
    """Имя протокола строк подвыборки — поле `gallery_rows` записи калибровки подвыборки."""
    return f"subset_r{r}_n{size}"
