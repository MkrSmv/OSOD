"""Конфигурации прогонов: на диске ровно сетка, значения — константы кода."""

import pytest

from src import config as CF
from src.encode import pipeline as PL
from src.encode.variants import grid


def test_configs_on_disk_are_the_grid():
    files = sorted(p.name for p in CF.CONFIG_DIR.glob("*.yaml"))
    assert files == sorted(f"{e}_{v}_{ds}.yaml" for e, v, ds in grid()) and len(files) == 22
    for e, v, ds in grid():
        cfg = CF.default(e, v, ds)
        assert CF.load(CF.path(cfg)) == cfg, f"{cfg.run_id}: файл расходится с константами — python -m src.config"


def test_batch_is_part_of_config():
    cfg = CF.default("dinov2", "c_0_10", "hr_insdet")
    assert cfg.encoder_batch == PL.BATCH == 16 and cfg.batch_rule == PL.BATCH_RULE
    assert cfg.crop_size == 448 and cfg.pool_long_side == cfg.seg_long_side == 2048 and cfg.crop_n_layers == 1
    assert "crop_workers" not in cfg.to_dict()  # число процессов — не конфигурация: на эмбеддинги не влияет


def test_outside_grid_rejected():
    for e, v, ds in (("dinov2", "p_perp", "hr_insdet"), ("dinov3", "p", "pcb"), ("dinov3", "p_perp", "pcb"),
                     ("dinov2", "c_white_10", "hr_insdet")):
        with pytest.raises((ValueError, KeyError)):
            CF.default(e, v, ds)
