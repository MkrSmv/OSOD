"""Семейство $\\varphi$ по таблице 2.1 и сетка прогонов: имена для конфигураций, журнала и каталогов.

Сетка фиксирована: 7 вариантов × 2 энкодера + $P^\\perp$ на DINOv3 для HR-InsDet,
7 вариантов на DINOv2 для PKU-Market-PCB — всего 22; других сочетаний код не принимает.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str            # имя в конфигурации и каталогах
    label: str           # обозначение по таблице 2.1
    kind: str            # "crop" | "pool"
    b: str | None = None       # заполнитель: "0" | "mean" | "blur"
    alpha: float | None = None
    debias: bool = False


VARIANTS: dict[str, Variant] = {v.name: v for v in (
    Variant("c_0_10", "C(0,1.0)", "crop", "0", 1.0),
    Variant("c_0_15", "C(0,1.5)", "crop", "0", 1.5),
    Variant("c_mean_10", "C(x̄,1.0)", "crop", "mean", 1.0),
    Variant("c_mean_15", "C(x̄,1.5)", "crop", "mean", 1.5),
    Variant("c_blur_10", "C(blur,1.0)", "crop", "blur", 1.0),
    Variant("c_blur_15", "C(blur,1.5)", "crop", "blur", 1.5),
    Variant("p", "P", "pool"),
    Variant("p_perp", "P⊥", "pool", debias=True),
)}

ENCODERS = {"dinov2": "encoder_dinov2", "dinov3": "encoder_dinov3"}  # имя в конфигурации → ключ `env.MODEL_IDS`
DATASETS = ("hr_insdet", "pcb")


def grid() -> list[tuple[str, str, str]]:
    """22 прогона в порядке выполнения: (энкодер, вариант, датасет)."""
    seven = [n for n in VARIANTS if n != "p_perp"]
    return ([("dinov2", v, "hr_insdet") for v in seven]
            + [("dinov3", v, "hr_insdet") for v in seven] + [("dinov3", "p_perp", "hr_insdet")]
            + [("dinov2", v, "pcb") for v in seven])
