"""Конфигурация прогона — один yaml на прогон: `configs/<encoder>_<variant>_<dataset>.yaml`.

Все значения — константы модулей пайплайна; ни одно не подбирается по данным. Файлы порождаются `default()` и
сверяются с ним тестом (`tests/test_config.py`), так что сетка на диске — ровно 22 прогона. Код прогона читает
значения из конфигурации, а не из констант модулей.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from src.encode import crop as C
from src.encode import debias as D
from src.encode import pipeline as PL
from src.encode import pool as PO
from src.encode.variants import ENCODERS, VARIANTS, grid
from src.segment import sam2 as S

CONFIG_DIR = Path("configs")
SEED = 20260918  # seed разбиений (`splits/*.json`); им же задаются бутстрэп и состав $\mathcal D_{\mathrm{chk}}$


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    encoder: str
    variant: str
    seg_long_side: int = S.LONG_SIDE                 # вход $S$
    crop_n_layers: int = 1                           # выбран в 4.2 по калибровочным сценам
    points_per_batch: int = S.POINTS_PER_BATCH
    crop_size: int = C.CROP_SIZE                     # вырезка для $z_{\mathrm{crop}}$
    blur_sigma_frac: float = C.SIGMA_FRAC
    blur_work_side: int = C.BLUR_WORK_SIDE
    pool_long_side: int = PO.POOL_LONG_SIDE          # вход $E$ для $z_{\mathrm{pool}}$
    debias_r: int = D.R                              # только для `p_perp`
    debias_noise_seed: int = D.NOISE_SEED
    encoder_batch: int = PL.BATCH                    # от состава батча зависит округление
    batch_rule: str = PL.BATCH_RULE
    gallery_protocols: tuple[str, ...] = ("full", "one_per_class")
    hnsw: dict = field(default_factory=lambda: {"M": 32, "efConstruction": 200, "efSearch_min": 64})
    eps: tuple[float, ...] = (0.05, 0.01)
    delta: float = 0.1
    max_dets: int = 1000
    bootstrap: int = 1000
    seed: int = SEED

    def __post_init__(self):
        if (self.encoder, self.variant, self.dataset) not in grid():
            raise ValueError(f"прогона ({self.encoder}, {self.variant}, {self.dataset}) нет в сетке")

    @property
    def run_id(self) -> str:
        return f"{self.encoder}_{self.variant}_{self.dataset}"

    @property
    def model_key(self) -> str:
        return ENCODERS[self.encoder]

    @property
    def phi(self):
        return VARIANTS[self.variant]

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}


def default(encoder: str, variant: str, dataset: str) -> RunConfig:
    if dataset == "pcb":  # интервалов на PCB нет — повторы бутстрэпа не считаются
        from src.eval.rules import PCB_BOOTSTRAP

        return RunConfig(dataset=dataset, encoder=encoder, variant=variant, bootstrap=PCB_BOOTSTRAP)
    return RunConfig(dataset=dataset, encoder=encoder, variant=variant)


def path(cfg: RunConfig, root: Path = CONFIG_DIR) -> Path:
    return root / f"{cfg.run_id}.yaml"


def dump(cfg: RunConfig) -> str:
    return yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)


def load(p: Path) -> RunConfig:
    d = yaml.safe_load(Path(p).read_text())
    for k in ("gallery_protocols", "eps"):
        d[k] = tuple(d[k])
    cfg = RunConfig(**d)
    if Path(p).name != f"{cfg.run_id}.yaml":
        raise ValueError(f"{p}: имя файла расходится с содержимым ({cfg.run_id})")
    return cfg


def write_all(root: Path = CONFIG_DIR) -> list[Path]:
    root.mkdir(exist_ok=True)
    out = []
    for e, v, ds in grid():
        cfg = default(e, v, ds)
        path(cfg, root).write_text(dump(cfg))
        out.append(path(cfg, root))
    return out


if __name__ == "__main__":
    for p in write_all():
        print(p)
