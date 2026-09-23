"""Дедупликация галереи лучшего $\\varphi$ → состав протокола строк `dedup`.

    python scripts/dedup_gallery.py --config configs/dinov2_c_mean_10_hr_insdet.yaml
    python scripts/dedup_gallery.py --config configs/dinov3_c_blur_15_hr_insdet.yaml

Состав протокола строк `dedup` → `gallery/<encoder>/<variant>/<dataset>/dedup.json`; сводка (те же поля без списка
эталонов, с sha256 `dedup.json`) → `experiments/dedup_<run_id>.json`: каталог галереи не под контролем версий.
Значение $\\eta$ — константа `src.gallery.dedup.DEDUP_ETA`; для галереи без значения — отказ. Сверка: число оставленных
эталонов обязано совпасть с аналитикой §2.3 (`experiments/analysis/gallery_dedup.json`) при том же $\\eta$; рядом — тот
же состав при приведении эмбеддингов к float64 (так считала аналитика; правило задаёт float32). Существующая запись не
перезаписывается: повторный запуск сверяет, что состав воспроизводится побайтно. Только CPU, секунды; сцены не читаются.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as CFG  # noqa: E402
from src import env  # noqa: E402
from src.gallery import dedup as DD  # noqa: E402
from src.gallery import store  # noqa: E402
from src.segment import cache as MC  # noqa: E402

OUT = Path("experiments")
ANALYSIS = Path("experiments/analysis/gallery_dedup.json")
_STAMP = env.code_stamp()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = CFG.load(ap.parse_args().config)
    g = store.Gallery.open(store.gallery_dir(cfg.encoder, cfg.variant, cfg.dataset),
                           store.expected(cfg.to_dict(), MC.box_key(cfg.seg_long_side)))
    rec = DD.build(g)                       # галерея без значения η — отказ здесь
    data = (json.dumps(rec, ensure_ascii=False) + "\n").encode()
    p = g.path / DD.FILE
    if p.exists():
        if p.read_bytes() != data:
            raise SystemExit(f"{p}: записанный состав не воспроизводится — не перезаписывается")
        print(f"{p}: состав уже записан и воспроизводится побайтно")
    else:
        DD.write(g, rec)
        if p.read_bytes() != data:
            raise AssertionError("dedup.json записан не теми байтами")
    rows = g.rows("dedup")

    insert_no = np.array([m["insert_no"] for m in g.meta], np.int64)
    keep64 = DD.keep_mask(g.emb.astype(np.float64), g.label_ids, insert_no, rec["eta"])
    ref = json.loads(ANALYSIS.read_text())["gallery_dedup"][cfg.encoder]
    n_ref = ref["by_t"][str(rec["eta"])]["n_kept"]
    check = {"analysis_record": str(ANALYSIS), "analysis_t_star": ref["selection"]["t_star"], "analysis_n_kept": n_ref,
             "n_kept_equal": n_ref == rec["n_kept"], "eta_equal": ref["selection"]["t_star"] == rec["eta"],
             "same_rows_in_float64": bool(np.array_equal(np.flatnonzero(keep64), rows))}
    summary = {"run_id": cfg.run_id, "kind": "gallery_dedup", "step": "дедупликация галереи лучшего φ", **_STAMP,
               "gallery": str(g.path), **{k: v for k, v in rec.items() if k != "kept_ids"},
               "dedup_json_sha256": hashlib.sha256(data).hexdigest(), "check_vs_analysis": check,
               "passed": bool(check["n_kept_equal"] and check["eta_equal"] and check["same_rows_in_float64"])}
    sp = OUT / f"dedup_{cfg.run_id}.json"
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(sp)
    print(f"{cfg.run_id}: η = {rec['eta']}, эталонов {rec['n_kept']} из {rec['n_full']} (аналитика — {n_ref}), "
          f"на метку {rec['n_per_label']}, n_max = {rec['n_max']}, k = {rec['k']}; в float64 тот же состав — "
          f"{check['same_rows_in_float64']}; {'СВЕРКА ПРОЙДЕНА' if summary['passed'] else 'СВЕРКА НЕ ПРОЙДЕНА'}: {sp}")
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
