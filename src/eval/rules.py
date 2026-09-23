"""Пороги и правила оценки, записанные до первого прогона 4.3; по результату не пересматриваются.

Здесь только то, что решено заранее: значения — константы кода, а не аргументы командной строки. Каждый блок
констант коммитился отдельно, до кода своего прогона и до самого прогона.
"""

from __future__ import annotations

from src.encode.variants import VARIANTS

# --- контроль на входе: первый прогон, C(0,1.0) на DINOv2, HR-InsDet, протокол baseline
FIRST_RUN = "dinov2_c_0_10_hr_insdet"
FIRST_RUN_SCENES = "all"       # оба порога — по AP контрольного прогона на всех 160 сценах, наборе опубликованного числа
FIRST_RUN_GALLERY = "full"     # полная галерея — протокол [5]
PUBLISHED_AP = 41.61           # SAM + DINOv2 [5], табл. 2 (`src.eval.published`)
AP_MIN_FIRST_RUN = 30.0        # ниже — ошибка пайплайна, а не результат: прогоны останавливаются
AP_DIFF_MAX = 8.0              # |AP − 41,61| больше — разбирается до продолжения, в любую сторону
PUBLISHED_AP_README_VIT_L = 43.33  # README кода [5], DINOv2 ViT-L/14 — не статья; в 4.6 приводится рядом, пороги — против 41,61

# Разбор расхождения больше `AP_DIFF_MAX` (записан до первого прогона): пороги не меняются; проверки ниже выполняются
# все, итог записывается в `FIRST_RUN_DIFF_RESOLUTION`; новое кодирование сверх сетки
# при разборе запрещено при любом исходе — только диагностика по уже посчитанным эмбеддингам, блоком `post_hoc` записи.
# Ошибок нет — число остаётся как есть и подаётся с перечнем отличий от [5]; ошибка найдена — старая запись не
# удаляется, а переименовывается с коммитом исправления (причина коммитится до пересчёта). Продолжение открывает
# записанное решение — без него сторож `scripts/run_4_3.py` остальные прогоны не пускает.
# AP ниже `AP_MIN_FIRST_RUN` разрешением не снимается: это ошибка пайплайна, она исправляется.
FIRST_RUN_REVIEW = {
    "up": ("истина — 3 078 рамок и 100 категорий, рамок вне 100 объектов нет",
           "метка детекции берётся из галереи, а не из истины; номера меток галереи совпадают с category_id",
           "эталоны галереи и сцены-запросы не пересекаются (профили объектов против снимков сцен)",
           "число детекций контрольного прогона равно числу масок кеша на тех же сценах",
           "AP с оракулом на 40 калибровочных сценах (`selection_cal`) воспроизводит виденные до прогона 61,9",
           "AP при maxDets 100 и 1000 совпадают (предел не действует)"),
    "down": ("геометрия вырезки: квадрат, центр, сторона, порядок «заполнение → масштабирование»",
             "масштаб маски: окно маски в исходных координатах против рамки маски",
             "нормировка входа энкодера и нормировка эмбеддингов",
             "протокол оценки: рамки x, y, w, h, исключающий край, id аннотаций с 1, бины площади"),
}
FIRST_RUN_DIFF_RESOLUTION: str | None = (  # итог разбора; без него остальные прогоны не идут
    "AP 50,32 против 41,61 (+8,71 п.), разбор по перечню ошибок не нашёл, продолжение разрешено")

# --- метки, у которых маска эталона — часть объекта: AP без них — справочно, всегда
PART_MASK_LABELS = ("051_truffettes", "055_candle_beast")

# --- AR итоговых детекций контрольного прогона: порог на s* — по [8]; в тексте [5] он не назван
AR_SCORE_THRESHOLD = 0.4
AR_MAX_DETS = (100, 1000)
AR_FOR_4_6 = {"use_cats": False, "score_threshold": AR_SCORE_THRESHOLD}  # вариант рядом с 63,06 [5] и 77,09 [8]

# --- выбор лучшего варианта φ энкодера: только по калибровочным сценам
SELECT_SPLIT = "cal"
SELECT_GALLERY = "full"
SELECT_TIE_AP = 0.1            # пунктов AP: варианты, отстающие от наибольшего AP не больше чем на столько, равны
TABLE_2_1_ORDER = tuple(VARIANTS)  # развязка последней очереди — вариант, стоящий выше в таблице 2.1

# --- критерий «слабого результата» DINOv3 (вопрос о PCA-whitening; не сработал, вопрос закрыт): записан
# до первого прогона 4.3; без записи прогоны DINOv3 не начинаются
WEAK_DINOV3_CRITERION: str | None = (
    "120 тестовых сцен, оракульный протокол, полная галерея, AP при maxDets из конфигурации; лучший φ каждого энкодера — "
    "по калибровочным сценам (`best_variant`); результат DINOv3 слабый, если 95 % перцентильный интервал парной "
    "бутстрэп-разности AP(лучший φ DINOv2) − AP(лучший φ DINOv3) целиком выше нуля и то же верно для разности "
    "AP(лучший φ DINOv2) − AP(P⊥ на DINOv3); повторы бутстрэпа по сценам общие у всех прогонов")
BOOT_AP_SUBSET = "test/all"    # повторы бутстрэпа AP этого подмножества пишутся в запись прогона — для парных разностей

# --- критерий осмысленности галереи PCB (записан до сборки любой галереи PCB и до кодирования эталонов PCB) — вопрос
# о 4.5 на PCB. На сетку 4.3 не влияет.
# Считается один раз, по эмбеддингам эталонов первой собранной галереи `PCB_GALLERY_RUN`; пороги не пересматриваются.
PCB_GALLERY_RUN = "dinov2_c_mean_10_pcb"  # C(x̄,1.0) на DINOv2; галерея полная
PCB_GALLERY_BOARDS = ("01", "04")         # эталоны платы 01 ищутся среди эталонов платы 04 и наоборот
PCB_GALLERY_N_TYPES = 6                   # типов дефекта; случайный уровень top-1 — около 1/6
PCB_GALLERY_MEAN_TOP1_MIN = 0.5           # среднее top-1 по шести типам — не ниже
PCB_GALLERY_TYPE_TOP1_MIN = 0.33          # top-1 типа — не ниже …
PCB_GALLERY_TYPES_MIN = 4                 # … не меньше чем у стольких типов из шести

# --- 4.3 на PKU-Market-PCB: порядок, состав оценки и признаки ошибки пайплайна (записаны отдельным коммитом
# до кода оценки и индексации PCB, до любой галереи и любого эмбеддинга PCB; к этому моменту были известны только
# запись 4.2 на PCB и число масок в кеше)
PCB_FIRST_RUN = PCB_GALLERY_RUN           # первым и отдельно, с контрольным прогоном baseline; P — последним
PCB_RUN_ORDER = ("c_mean_10", "c_0_10", "c_0_15", "c_mean_15", "c_blur_10", "c_blur_15", "p")
PCB_BASELINE_VARIANT = "c_mean_10"        # выбора на PCB нет: лучший φ DINOv2 переносится с HR-InsDet и сверяется с журналом
PCB_BOOTSTRAP = 0                         # интервалов на PCB нет: повторы бутстрэпа не считаются (поле конфигурации)
PCB_ZONES_AP_CLOSE = 0.5                  # пунктов AP: «AP без зон близок»
# состав, с которым сверяется каждый прогон PCB; расхождение — ошибка, записи журнала нет
PCB_RECALL_RECORD = "experiments/runs/4_2_pcb.json"   # число масок, рамок и полнота по типам — из неё (`crop_n_layers` прогона)
PCB_N_TEST_IMAGES, PCB_N_GT, PCB_N_ZONES = 361, 1770, 32
PCB_N_REFS, PCB_N_MAX, PCB_K = 716, 131, 262          # k = 2·n_max; точный перебор идёт при k = N
# потолок AP: найденной может быть только рамка, у которой среди масок M(I) есть маска с IoU не ниже порога, поэтому при
# каждом пороге AP категории ≤ (⌊100·R⌋ + 1)/101 ≤ R + 1/101, где R — полнота 4.2 этого типа дефекта при том же пороге
PCB_AP_CEILING_SLACK = 1 / 101
# нижний признак — только первый прогон: тип, у которого и эталон, и маска запроса — сам объект (площадка с отверстием)
PCB_SANITY_TYPE = "Missing_hole"
PCB_SANITY_TOP1_MIN = 1 / 6               # top-1 на оракульных масках этого типа при полной галерее ниже случайного уровня
PCB_SANITY_AUROC_MIN = 0.5                # AUROC по s* при полной галерее ниже случайного уровня (записано
                                          # до записи первого прогона PCB): перепутанные
                                          # метки либо знак объясняют такое значение, близость дистракторов к галерее — нет
PCB_FIRST_RUN_REVIEW = (
    "снимок платы читается по пути раздачи, его размер совпадает с `hw` записи кеша масок (у каждой платы свой)",
    "масштаб маски: окно маски в исходных координатах против рамки маски — у плат масштаб входа S разный",
    "номера меток галереи совпадают с category_id истины; эталоны — только платы 01 и 04, запросы — только тестовые платы",
    "вырезки пяти оракульных масок и пяти эталонов этого типа просмотрены глазами (черновой каталог, не репозиторий)",
    "top-1 эталонов этого типа между платами 01 и 04 (критерий осмысленности галереи) — рядом: низок и он — это галерея",
    "AUROC: оценка детекции — s*, «известные» — оракульные маски, «неизвестные» — дистракторы; знак и состав не перепутаны",
)
PCB_REJECTED_SUFFIX = "__rejected_"       # запись-заглушка прогона, остановленного жёстким признаком: `<run_id>__rejected_<коммит>.json`
PCB_SANITY_RESOLUTION: str | None = None  # по итогу разбора; без него остальные шесть прогонов не идут

# --- диагностика после результата сетки HR-InsDet: усреднение под маской по патч-токенам вырезки (протокол записан
# до кода диагностики и до любого её счёта).
# Не прогон сетки: в таблицы 4.3 не идёт, тестовые сцены не читаются, лучший φ по ней не выбирается. Числа, с которыми
# она сравнивается (`selection_cal` прогонов P и C(0,1.0)), известны до записи.
POOL_CROP_DATASET = "hr_insdet"
POOL_CROP_SPLIT = "cal"                    # только 40 калибровочных сцен
POOL_CROP_ENCODERS = ("dinov2", "dinov3")  # без P⊥: U_r оценён для входа 1536×2048 и к вырезке 448 не относится
POOL_CROP_ALPHA = 1.0                      # квадрат по большей стороне рамки маски, из исходного снимка
POOL_CROP_SIZE = 448
POOL_CROP_FILL = None                      # фон не заполняется; часть квадрата за кадром — x̄, вес маски там нулевой
POOL_CROP_BINDING = "soft"                 # доля пикселей патча вырезки под маской, как в P
POOL_CROP_GALLERY = "full"
POOL_CROP_COMPARE = {"crop": "c_0_10", "pool": "p"}  # `selection_cal` этих прогонов того же энкодера
POOL_CROP_MARGIN_AP = 10.0                 # пунктов AP с оракулом на калибровочных сценах

# --- эксперимент 4.4 — отбор гранулярности. Протокол и трактовка исходов записаны до любого кода 4.4; константы
# закоммичены отдельным коммитом до первого запуска `scripts/run_4_4.py`. До записи были известны AP контрольных
# прогонов лучших φ на 120 тестовых сценах (51,70 и 54,69), числа аналитики §2.3 по калибровочным сценам, итог сверки ρ
# с аналитикой, составы `dedup` (1 142 и 743 эталона) и число масок M* по сценам (2 498 из 4 534 калибровочных, 8 964
# из 22 112 тестовых — вывод `scripts/select_rho.py`); AP с ρ не считался ни на каких сценах. Параметры ρ и η ни при
# каком исходе не меняются.
RHO_RUNS = ("dinov2_c_mean_10_hr_insdet", "dinov3_c_blur_15_hr_insdet")  # лучшие φ; скрипт сверяет с `best_variant`
RHO_MAIN = {"subset": "test/all", "gallery": "full", "metric": "ap"}     # главное сравнение: ρ минус без отбора, парно
RHO_DIFF_METRICS = ("ap", "ap50", "ap75")                                # справочно — AP50, AP75, `one_per_class`, `all/all`
RHO_DIFF_SUBSETS = ("test/all", "all/all")                               # интервалы — только здесь; easy / hard — точечно
RHO_GALLERIES = ("full", "one_per_class")                                # `one_per_class` нужен 4.6
RHO_OUTCOMES = {
    1: "ρ помогает: нижняя граница 95 % перцентильного интервала парной бутстрэп-разности AP выше нуля",
    2: "разница не выявлена: интервал содержит ноль (приводится с шириной интервала и долей некодируемых масок тестовых сцен)",
    3: "ρ вредит: верхняя граница интервала ниже нуля — цена отбора до поиска",
}
RHO_OUTCOME_BY_ENCODERS = "по энкодерам"   # сводный исход при разных исходах у энкодеров; при одинаковых — общий
RHO_BASELINE_TOL = 1e-9                    # точечные AP, AP50, AP75 пересчитанного контрольного прогона против журнала
# ориентиры отбора после поиска — по эмбеддингам всех масок M(I), без вердикта; параметры — константы без подбора
POST_SEARCH = ("chain_max", "box_nms")
POST_NMS_IOU = 0.5                         # порог NMS в [13], разд. 3.3; маска подавляется при IoU выше
POST_SEARCH_GALLERY = "full"               # ориентиры и блок `dedup` считаются при полной галерее (в 4.6 они не идут)
RHO_RECORD_BLOCKS = ("baseline_check", "rho", "paired_diff", "outcome", "post_search", "dedup")
# описательные величины рядом с AP (внесены до первого запуска `run_4_4.py`)
RHO_MULT_IOU = 0.5                         # §2.3: рамке «соответствует» маска при IoU рамок не ниже 0,5 (кратные детекции)
RHO_DESCRIBE_SUBSETS = ("cal/all", "test/all", "test/easy", "test/hard", "all/all", "all/easy", "all/hard")
RHO_DEDUP_QUANTILES = (0.95, 0.99)         # квантили s* дистракторов M* — только калибровочные сцены; то, чем был бы τ_q
# предусловия запуска 4.4: обе записи обязаны существовать с `passed: true`
RHO_PRECONDITIONS = ("experiments/rho_verify_cal.json", "experiments/rho_encode_check.json")

# --- эксперимент 4.5 — отклонение и калибровка порога при росте галереи. Протокол записан до любого кода 4.5 и
# заморожен с первого запуска `run_4_4.py`; константы закоммичены вместе со списками подвыборок
# (`splits/gallery_subsets_hr_insdet.json`) отдельным коммитом до первого запуска `scripts/run_4_5.py`. До записи были
# известны числа пересчитанной калибровки с ρ на калибровочных сценах (τ_q, κN0 — `experiments/calib_*.json`), записи
# 4.3 и 4.4; частоты ложных срабатываний на тестовых сценах при калиброванных порогах не считались.
EPS_HOLDS_RULE = ("нижняя граница 95 % перцентильного бутстрэп-интервала (по сценам) наблюдаемой частоты ложных "
                  "срабатываний на дистракторах тестовых сцен не выше ε(1+δ)")
REJ_RUNS = {"hr_insdet": RHO_RUNS}        # лучшие φ обоих энкодеров
GALLERY_CHAINS = 9                        # вложенных цепочек подвыборок: перестановки `default_rng(seed + r)`, r = 0..8
GALLERY_SUBSET_SIZES = (25, 50, 100)      # экземпляров (по 24 эталона)
GROWTH_PAIRS = ((25, 50), (25, 100), (50, 100))   # (n0, n): правило, откалиброванное на меньшей, — на большей (§2.4)
CHAIN_MAJORITY = 5                        # правило «удерживает», если так не меньше чем в 5 цепочках из 9
TOP_K = 100                               # [13], разд. 3.3; при равных s* — меньший номер маски
FIXED_TAU = 0.4                           # [5, 8]
REJ_RULES = ("quantile_recal", "model", "quantile_frozen", "fixed", "top_k")
REJ_VERDICT_RULES = ("model", "quantile_recal")   # исход определяют только они
REJ_TAIL_QUANTILES = (0.99, 0.999)        # хвост пар «дистрактор — новый эталон» против F_n0 — описательно
REJ_FULL_AP_EPS = 0.05                    # справочные строки AP полного метода: τ_q при этом ε
REJ_BOOTSTRAP = 1000                      # повторов бутстрэпа по сценам у частоты ложных срабатываний
REJ_OUTCOMES = {                          # уточнённая формулировка — до любых частот на тестовых сценах
    "a": "правило по модели удерживает ε при росте галереи",
    "b": "правило по модели не удерживает, правило по квантили с пересчётом удерживает",
    "c": "не удерживает ни одно из двух правил",
}
REJ_OUTCOMES_12SEP = {                    # первоначальная формулировка: исход (б) и (в) зависят ещё от фиксированного 0,4
    "a": "правило по модели удерживает ε на 25/50/100",
    "b": "модель не удерживает, квантиль с пересчётом удерживает, фиксированный 0,4 — нет",
    "c": "не удерживает ничего",
    None: "сочетание вне записанных 12 сентября исходов (фиксированный 0,4 удерживает при не удержавшей модели)",
}
REJ_RECORD_BLOCKS = {"hr_insdet": ("calibration_check", "cells", "verdict", "outcome", "assumptions", "recalibration",
                                   "dedup_point", "full_method_ap")}


def eps_holds(fpr_ci95_low: float, eps: float, delta: float) -> bool:
    """`EPS_HOLDS_RULE`: правило калибровки удерживает уровень ε на данной галерее."""
    return bool(fpr_ci95_low <= eps * (1 + delta))


def chain_holds(holds_by_pair: dict) -> bool:
    """Правило «удерживает» в цепочке, если `eps_holds` истинно во всех трёх парах роста (ячейки N = N0 не входят)."""
    if set(holds_by_pair) != set(GROWTH_PAIRS):
        raise ValueError(f"пары роста {sorted(holds_by_pair)}, ожидаются {GROWTH_PAIRS}")
    return all(bool(v) for v in holds_by_pair.values())


def majority_holds(chain_flags: list[bool]) -> dict:
    """Вердикт по большинству цепочек: не меньше `CHAIN_MAJORITY` из `GALLERY_CHAINS`; число согласных — в запись."""
    if len(chain_flags) != GALLERY_CHAINS:
        raise ValueError(f"цепочек {len(chain_flags)}, ожидается {GALLERY_CHAINS}")
    n = int(sum(bool(f) for f in chain_flags))
    return {"holds": n >= CHAIN_MAJORITY, "n_chains_hold": n, "n_chains": GALLERY_CHAINS, "majority": CHAIN_MAJORITY}


def transfer_failure(holds_at_n0: dict[int, list[bool]]) -> dict:
    """Признак `fails_at_n0` (записан до первого запуска `run_4_5.py`): «провал переноса» калибровочного набора на
    тестовые сцены говорится по ячейкам N = N0 правила по квантили с пересчётом — при 25 и при 50 экземплярах по тому же
    правилу большинства цепочек, что у вердикта (уровень не удержан, если `eps_holds` истинно меньше чем в
    `CHAIN_MAJORITY` цепочках), при 100 экземплярах — по единственной ячейке (вся галерея, от цепочки не зависит).
    В классификацию исходов не входит."""
    if set(holds_at_n0) != set(GALLERY_SUBSET_SIZES):
        raise ValueError(f"объёмы {sorted(holds_at_n0)}, ожидаются {GALLERY_SUBSET_SIZES}")
    full = GALLERY_SUBSET_SIZES[-1]
    out = {}
    for n, flags in holds_at_n0.items():
        if n == full:
            if len(set(bool(f) for f in flags)) != 1:
                raise ValueError("ячейка полной галереи обязана быть одной и той же у всех цепочек")
            out[n] = {"holds": bool(flags[0]), "n_distinct_cells": 1}
        else:
            m = majority_holds(flags)
            out[n] = {"holds": m["holds"], "n_chains_hold": m["n_chains_hold"], "n_distinct_cells": GALLERY_CHAINS}
    return {"by_n0": out, "any": bool(not all(v["holds"] for v in out.values()))}


def rejection_outcome(model: bool, quantile_recal: bool, fixed: bool) -> dict:
    """Исход 4.5 у одного сочетания энкодера и ε по обеим формулировкам."""
    code = "a" if model else "b" if quantile_recal else "c"
    if model:
        old = "a"
    elif quantile_recal:
        old = "b" if not fixed else None
    else:
        old = "c" if not fixed else None
    return {"refined_21sep": {"code": code, "text": REJ_OUTCOMES[code]},
            "as_written_12sep": {"code": old, "text": REJ_OUTCOMES_12SEP[old]},
            "holds": {"model": bool(model), "quantile_recal": bool(quantile_recal), "fixed": bool(fixed)}}


def rejection_outcome_joint(codes: dict[str, str]) -> dict:
    """Общий вывод (а) — только если (а) во всех четырёх сочетаниях энкодера и ε; иначе исход называется по сочетаниям."""
    if len(codes) != 4 or set(codes.values()) - set(REJ_OUTCOMES):
        raise ValueError(f"нужны исходы четырёх сочетаний энкодера и ε: {codes}")
    all_a = all(c == "a" for c in codes.values())
    return {"all_a": all_a, "text": REJ_OUTCOMES["a"] if all_a else "по сочетаниям энкодера и ε", "by_combination": dict(codes)}


def rho_outcome(ci95: list[float]) -> dict:
    """Исход 4.4 у одного энкодера по интервалу парной разности AP «ρ минус без отбора» (`RHO_MAIN`). Граница, равная
    нулю, — исход (2): (1) и (3) требуют строгого неравенства."""
    lo, hi = float(ci95[0]), float(ci95[1])
    if not lo <= hi:
        raise ValueError(f"интервал {ci95}")
    code = 1 if lo > 0 else 3 if hi < 0 else 2
    return {"code": code, "text": RHO_OUTCOMES[code], "ci95": [lo, hi], "ci95_width": hi - lo}


def rho_outcome_joint(codes: dict[str, int]) -> dict:
    """Сводный исход по двум энкодерам: одинаковые исходы — общий, разные — «по энкодерам»."""
    if set(codes.values()) - set(RHO_OUTCOMES) or not codes:
        raise ValueError(f"исходы {codes}")
    same = len(set(codes.values())) == 1
    code = next(iter(codes.values())) if same else None
    return {"same": same, "code": code, "text": RHO_OUTCOMES[code] if same else RHO_OUTCOME_BY_ENCODERS,
            "by_encoder": dict(codes)}


def paired_diff(boot_a, boot_b, point_a: float, point_b: float) -> dict:
    """Парная разность «a минус b» по общим повторам бутстрэпа: точечная оценка и перцентильный интервал 95 %."""
    import numpy as np

    d = np.asarray(boot_a, float) - np.asarray(boot_b, float)
    if d.ndim != 1 or not len(d) or np.isnan(d).any():
        raise ValueError("повторы бутстрэпа — два ряда равной длины без пропусков")
    lo, hi = (float(v) for v in np.quantile(d, [0.025, 0.975]))
    return {"diff": float(point_a) - float(point_b), "ci95": [lo, hi], "n_boot": int(len(d))}


def first_run_check(ap_percent: float) -> dict:
    """Вердикт контроля на входе по AP контрольного прогона (в пунктах, 0–100)."""
    diff = ap_percent - PUBLISHED_AP
    return {"ap": ap_percent, "published_ap": PUBLISHED_AP, "diff": diff,
            "ap_min": AP_MIN_FIRST_RUN, "diff_max": AP_DIFF_MAX,
            "ap_ge_min": bool(ap_percent >= AP_MIN_FIRST_RUN), "diff_within_max": bool(abs(diff) <= AP_DIFF_MAX),
            "passed": bool(ap_percent >= AP_MIN_FIRST_RUN and abs(diff) <= AP_DIFF_MAX)}


def paired_diff_above_zero(boot_a, boot_b) -> dict:
    """95 % перцентильный интервал парной разности по общим повторам бутстрэпа и признак «целиком выше нуля»."""
    import numpy as np

    d = np.asarray(boot_a, float) - np.asarray(boot_b, float)
    lo, hi = (float(v) for v in np.quantile(d, [0.025, 0.975]))
    return {"ci95": [lo, hi], "above_zero": bool(lo > 0)}


def weak_dinov3(boot_best_dinov2, boot_best_dinov3, boot_p_perp_dinov3) -> dict:
    """Критерий `WEAK_DINOV3_CRITERION` по повторам бутстрэпа AP трёх прогонов (`BOOT_AP_SUBSET`, полная галерея)."""
    a = paired_diff_above_zero(boot_best_dinov2, boot_best_dinov3)
    b = paired_diff_above_zero(boot_best_dinov2, boot_p_perp_dinov3)
    return {"best_dinov2_minus_best_dinov3": a, "best_dinov2_minus_p_perp_dinov3": b,
            "weak": bool(a["above_zero"] and b["above_zero"])}


def best_variant(cal: dict[str, dict]) -> str:
    """Лучший вариант энкодера. `cal` — вариант → `{"ap", "ap50"}` (доли, 0–1) на калибровочных сценах с оракулом
    при полной галерее, `maxDets` из конфигурации.

    Наибольший AP; варианты в пределах `SELECT_TIE_AP` пункта от него равны — среди них больший AP50, затем
    стоящий выше в таблице 2.1. $P^\\perp$ участвует наравне с остальными.
    """
    if not cal or set(cal) - set(TABLE_2_1_ORDER):
        raise ValueError(f"варианты вне таблицы 2.1: {sorted(set(cal) - set(TABLE_2_1_ORDER))}")
    top = max(v["ap"] for v in cal.values())
    tied = [n for n, v in cal.items() if 100 * (top - v["ap"]) <= SELECT_TIE_AP]
    return min(tied, key=lambda n: (-cal[n]["ap50"], TABLE_2_1_ORDER.index(n)))


def pool_crop_verdict(ap_diag: float, ap_crop: float, ap_pool: float) -> dict:
    """Трактовка диагностики «усреднение под маской по патч-токенам вырезки» (AP в пунктах, 0–100, калибровочные сцены).

    «Возвращается к уровню вырезок» — не ниже AP C(0,1.0) минус `POOL_CROP_MARGIN_AP`; «не возвращается» — не выше
    AP P плюс `POOL_CROP_MARGIN_AP`; между ними — «частично». Если зоны пересекаются, трактовка не определена — ошибка.
    """
    lo, hi = ap_pool + POOL_CROP_MARGIN_AP, ap_crop - POOL_CROP_MARGIN_AP
    if not lo < hi:
        raise ValueError(f"зоны трактовки пересекаются: P + {POOL_CROP_MARGIN_AP} = {lo}, C − {POOL_CROP_MARGIN_AP} = {hi}")
    verdict = "returns_to_crop_level" if ap_diag >= hi else "stays_at_pool_level" if ap_diag <= lo else "partial"
    return {"ap_diag": ap_diag, "ap_crop": ap_crop, "ap_pool": ap_pool, "margin_ap": POOL_CROP_MARGIN_AP,
            "returns_if_ge": hi, "stays_if_le": lo, "verdict": verdict}


def pcb_gallery_verdict(per_type: dict[str, tuple[int, int]]) -> dict:
    """Вердикт критерия осмысленности галереи PCB по счётчикам `тип → (совпадений, эталонов типа с обеих плат)`.

    Сравнения с порогами — в рациональных числах: среднее ровно 0,5 и top-1 ровно 0,33 порог выдерживают («не ниже»).
    """
    from fractions import Fraction

    if len(per_type) != PCB_GALLERY_N_TYPES or any(n <= 0 or not 0 <= h <= n for h, n in per_type.values()):
        raise ValueError(f"нужны счётчики {PCB_GALLERY_N_TYPES} типов с эталонами: {per_type}")
    top1 = {t: Fraction(h, n) for t, (h, n) in per_type.items()}
    mean = sum(top1.values()) / PCB_GALLERY_N_TYPES
    n_ok = sum(v >= Fraction(str(PCB_GALLERY_TYPE_TOP1_MIN)) for v in top1.values())
    mean_ok, types_ok = mean >= Fraction(str(PCB_GALLERY_MEAN_TOP1_MIN)), n_ok >= PCB_GALLERY_TYPES_MIN
    return {"top1_by_type": {t: float(v) for t, v in top1.items()},
            "hits_by_type": {t: [int(h), int(n)] for t, (h, n) in per_type.items()},
            "mean_top1": float(mean), "mean_top1_min": PCB_GALLERY_MEAN_TOP1_MIN, "mean_ok": bool(mean_ok),
            "n_types_ge_min": int(n_ok), "type_top1_min": PCB_GALLERY_TYPE_TOP1_MIN,
            "types_min": PCB_GALLERY_TYPES_MIN, "types_ok": bool(types_ok),
            "run_4_5_on_pcb": bool(mean_ok and types_ok)}


def pcb_gallery_criterion(emb, labels, boards) -> dict:
    """Критерий осмысленности галереи PCB по эмбеддингам эталонов галереи `PCB_GALLERY_RUN` (строки нормированы).

    Для каждого эталона платы 01 — ближайший по скалярному произведению среди эталонов платы 04 и наоборот, точный
    перебор; совпадение — тип дефекта ближайшего равен типу эталона. Top-1 типа — доля совпадений среди всех эталонов
    типа с обеих плат, оба направления поиска вместе; заглавная величина — среднее по шести типам.
    """
    import numpy as np

    z, labels, boards = np.asarray(emb, np.float32), list(labels), list(boards)
    if z.ndim != 2 or not len(z) == len(labels) == len(boards):
        raise ValueError(f"эмбеддинги {z.shape}, меток {len(labels)}, плат {len(boards)}")
    if set(boards) != set(PCB_GALLERY_BOARDS):
        raise ValueError(f"платы эталонов {sorted(set(boards))}, а не {PCB_GALLERY_BOARDS}")
    if np.abs(np.linalg.norm(z, axis=1) - 1).max() > 1e-4:
        raise ValueError("эмбеддинги эталонов не нормированы")
    lab, brd = np.asarray(labels), np.asarray(boards)
    types = sorted(set(labels))
    side = [brd == b for b in PCB_GALLERY_BOARDS]
    if len(types) != PCB_GALLERY_N_TYPES or any(set(lab[s]) != set(types) for s in side):
        raise ValueError(f"на каждой из плат {PCB_GALLERY_BOARDS} нужны эталоны всех {PCB_GALLERY_N_TYPES} типов")
    hit = np.zeros(len(z), bool)
    for q, g in (side, side[::-1]):
        hit[q] = lab[g][(z[q] @ z[g].T).argmax(1)] == lab[q]
    out = pcb_gallery_verdict({t: (int(hit[lab == t].sum()), int((lab == t).sum())) for t in types})
    return {"run_id": PCB_GALLERY_RUN, "boards": list(PCB_GALLERY_BOARDS), "search": "exact_ip_between_boards",
            "n_refs_by_board": {b: int(s.sum()) for b, s in zip(PCB_GALLERY_BOARDS, side)}, **out}


def pcb_ap_ceiling(recall_by_type: dict[str, dict]) -> dict:
    """Потолок AP категории по полноте 4.2 её типа дефекта: `recall_by_type` — тип → `{"recall_50", "recall_75",
    "recall_50_95"}` (доли, все маски $M(I)$, без взаимной однозначности). Возвращает тип → потолки AP, AP50, AP75."""
    keys = {"ap": "recall_50_95", "ap50": "recall_50", "ap75": "recall_75"}
    return {t: {m: min(1.0, r[k] + PCB_AP_CEILING_SLACK) for m, k in keys.items()} for t, r in recall_by_type.items()}


def pcb_hard_checks(ap_by_type: dict[str, dict], ceiling: dict[str, dict], composition: dict, expected: dict) -> dict:
    """Признаки ошибки пайплайна на PCB, при которых записи журнала нет: AP категории выше потолка по полноте 4.2
    (в любом протоколе оценки и галереи) и расхождение состава оценки с ожидаемым. `failed` — перечень сработавшего."""
    failed = [f"{t}: {m} {ap_by_type[t][m]:.4f} выше потолка {ceiling[t][m]:.4f}"
              for t in sorted(ceiling) for m in ("ap", "ap50", "ap75")
              if ap_by_type[t].get(m) is not None and ap_by_type[t][m] > ceiling[t][m] + 1e-12]
    failed += [f"{k}: {composition.get(k)} вместо {v}" for k, v in expected.items() if composition.get(k) != v]
    return {"ceiling": ceiling, "composition_expected": expected, "failed": failed, "passed": not failed}


def pcb_first_run_check(top1_sanity_type: float | None, n_oracle_sanity_type: int, auroc: float | None) -> dict:
    """Нижние признаки первого прогона PCB при полной галерее: top-1 на оракульных масках `PCB_SANITY_TYPE` и AUROC по $s^*$."""
    top1_ok = top1_sanity_type is not None and n_oracle_sanity_type > 0 and top1_sanity_type >= PCB_SANITY_TOP1_MIN
    auroc_ok = auroc is not None and auroc >= PCB_SANITY_AUROC_MIN
    return {"type": PCB_SANITY_TYPE, "top1": top1_sanity_type, "n_oracle": n_oracle_sanity_type,
            "top1_min": PCB_SANITY_TOP1_MIN, "top1_passed": bool(top1_ok), "auroc": auroc,
            "auroc_min": PCB_SANITY_AUROC_MIN, "auroc_passed": bool(auroc_ok), "passed": bool(top1_ok and auroc_ok),
            "review_if_failed": list(PCB_FIRST_RUN_REVIEW), "resolution": PCB_SANITY_RESOLUTION}
