"""Дедупликация галереи — §2.3. Параметры заморожены: η по результату не меняется.

Правило: эталоны просматриваются в порядке `insert_no`; эталон остаётся, если у его метки ещё нет оставленных эталонов
либо наибольшее скалярное произведение его эмбеддинга с оставленными эталонами той же метки строго меньше $\\eta$.
Арифметика: `emb.npy` в float32, произведение в float32, наибольшее значение приводится к `float` и сравнивается с
$\\eta$. Дедупликация «при записи» и подмножество строк полной галереи дают один состав: правило сравнивает только с
уже оставленными.

Значение $\\eta$ есть только у двух галерей лучших $\\varphi$ HR-InsDet (`DEDUP_ETA`); для остальных галерей запрос
дедупликации — ошибка: правило выбора $\\eta$ заново не запускается. На PKU-Market-PCB дедупликация не применяется.

Отдельного каталога дедуплицированная галерея не заводит: это протокол строк `Gallery.rows("dedup")`, состав
фиксируется файлом `dedup.json` каталога галереи.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.gallery import store

DEDUP_ETA = {("dinov2", "c_mean_10", "hr_insdet"): 0.96, ("dinov3", "c_blur_15", "hr_insdet"): 0.90}
DEDUP_RULE = "insert_order_same_label_max_cos_lt_eta_v1"
FILE = "dedup.json"
FORMAT = 1


def eta_for(passport: dict) -> float:
    key = (passport["encoder"], passport["variant"], passport["dataset"])
    if key not in DEDUP_ETA:
        raise ValueError(f"для галереи {key} значения η нет: дедупликация определена только для {sorted(DEDUP_ETA)}")
    return DEDUP_ETA[key]


def keep_mask(z: np.ndarray, label_ids: np.ndarray, insert_no: np.ndarray, eta: float) -> np.ndarray:
    """Булева маска оставленных эталонов; `z` — как есть (float32 у галереи), без приведения точности."""
    keep = np.zeros(len(z), bool)
    for j in np.argsort(insert_no, kind="stable"):
        same = np.flatnonzero(keep & (label_ids == label_ids[j]))
        if len(same) == 0 or float((z[same] @ z[j]).max()) < eta:
            keep[j] = True
    return keep


def build(g: store.Gallery) -> dict:
    """Содержимое `dedup.json` галереи `g`."""
    eta = eta_for(g.passport)
    if g.emb.dtype != np.float32:
        raise ValueError("эмбеддинги галереи — не float32")
    if g.deleted.any():
        raise ValueError("в галерее есть помеченные эталоны: состав дедупликации определён для полной галереи")
    insert_no = np.array([m["insert_no"] for m in g.meta], np.int64)
    keep = keep_mask(g.emb, g.label_ids, insert_no, eta)
    rows = np.flatnonzero(keep)
    rows = rows[np.argsort(insert_no[rows], kind="stable")]
    per = np.bincount(g.label_ids[rows], minlength=len(g.labels))
    return {"format": FORMAT, "eta": eta, "rule": DEDUP_RULE, "gallery_emb_sha256": g.passport["emb_sha256"],
            "n_full": len(g), "n_kept": int(len(rows)), "kept_ids": [g.meta[i]["id"] for i in rows],
            "rows_sha256": store.rows_sha256(g, rows),
            "n_per_label": {"min": int(per.min()), "median": float(np.median(per)), "max": int(per.max()),
                            "n_labels_with_one": int((per == 1).sum())},
            "n_max": g.n_max(rows), "k": g.k(rows)}


def write(g: store.Gallery, rec: dict) -> Path:
    p = g.path / FILE
    store._atomic_text(p, json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def read_rows(g: store.Gallery) -> np.ndarray:
    """Строки протокола `dedup` по `dedup.json`; запись сверяется с галереей и константами кода."""
    p = g.path / FILE
    if not p.is_file():
        raise FileNotFoundError(f"{p}: состав дедупликации не записан — python scripts/dedup_gallery.py")
    rec = json.loads(p.read_text())
    if rec["format"] != FORMAT or rec["rule"] != DEDUP_RULE or rec["eta"] != eta_for(g.passport):
        raise ValueError(f"{p}: правило или η записи расходятся с кодом")
    if rec["gallery_emb_sha256"] != g.passport.get("emb_sha256") or rec["n_full"] != len(g):
        raise ValueError(f"{p}: состав записан для другой галереи")
    row_of = {m["id"]: i for i, m in enumerate(g.meta)}
    rows = np.array([row_of[i] for i in rec["kept_ids"]], np.int64)
    if len(rows) != rec["n_kept"] or store.rows_sha256(g, rows) != rec["rows_sha256"]:
        raise ValueError(f"{p}: список оставленных эталонов повреждён")
    return np.sort(rows)


def file_sha256(g: store.Gallery) -> str:
    return hashlib.sha256((g.path / FILE).read_bytes()).hexdigest()
