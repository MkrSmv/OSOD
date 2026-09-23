"""OWLv2 в режиме image-guided — метод сравнения (глава 3, §4.6).

Единственное исключение из запрета текстовой модальности, и то без текста в счёте: вызываются только зрительная
башня и головки детекции — `image_embedder` → `embed_image_query` (запрос) и `class_predictor`, `box_predictor`
(сцена). Текстовая башня, токенизатор и `forward` с `input_ids` не вызываются нигде; текстовая башня и её проекция
остаются на CPU (`load_model`), а `trace_modules` записывает, какие модули модели сработали, и отказывает, если среди
них текстовые.

Все параметры — штатные значения установленной версии transformers (4.57.6: `modeling_owlv2.py`,
`image_processing_owlv2.py`) либо константы ниже, записанные до счёта, без подбора по каким-либо данным. Разрешение
входа не константа: оно читается из конфигурации установленного процессора и модели (`input_size`) и пишется в журнал.

Отличия от штатного пути (объявляются в 4.1 и 4.6): штатная постобработка `post_process_image_guided_detection`
не используется — она перенормирует оценки внутри запроса; оценка пары «рамка — эталон» —
`sigmoid(logit)`, оценка рамки по метке — максимум по эталонам метки, одна метка на рамку ($\\arg\\max$, при равенстве —
меньший номер), затем NMS без учёта меток со штатным порогом; эмбеддинг запроса считается один раз на эталон.
"""

from __future__ import annotations

import contextlib
import hashlib
import json

import numpy as np

from src.encode.crop import square

MODEL_KEY = "owlv2"          # `env.MODEL_IDS`
DTYPE = "float32"            # веса fp32: в fp16 сверка с fp32 не пройдена (`env.json`, `owlv2_precision`)
ATTN = "eager"               # SDPA у `Owlv2ForObjectDetection` установленной версии не поддерживается
QUERY_ALPHA = 1.0            # квадрат `src.encode.crop.square` при α = 1,0 вокруг `prompt_box`
PAD_VALUE = 0.5              # серый штатного паддинга `Owlv2ImageProcessor.pad` (в долях после `rescale`)
NMS_IOU = 0.3                # `nms_threshold` по умолчанию `post_process_image_guided_detection`
SCORE_MIN = 0.0              # `threshold` по умолчанию там же, то есть порога нет
PROTOCOLS = ("full", "one_per_class")   # оба протокола галереи — из одних логитов
# у OWLv2 своей галереи нет — берутся `id`, метка, снимок и `prompt_box` строк галереи лучшего φ DINOv2
GALLERY = {"hr_insdet": ("dinov2", "c_mean_10"), "pcb": ("dinov2", "c_mean_10")}
TEXT_MODULES = ("owlv2.text_model", "owlv2.text_projection")

# сверка fp16 с fp32 до счёта детекций. Части: (а) HR-InsDet, 10 калибровочных сцен пилота, по
# одному эталону 100 меток; (б) PCB, 10 снимков калибровочных плат пилота, все 716 эталонов; в обеих — 100 лучших рамок
# снимка по оценке в fp32 и не больше `PRECISION_MAX_CHANGED_PER_IMAGE` рамок со сменившейся меткой на каждом снимке;
# (в) полная галерея HR-InsDet — оценка сверху: не больше того же числа рамок из 100 лучших (по fp16) с разрывом между
# наибольшим и вторым логитом меток не больше `MARGIN_FACTOR`·δ, δ — наибольшее |Δ логита| части (а) по её 100 лучшим
# рамкам сцены и всем запросам.
PRECISION_TOP = 100
PRECISION_MAX_CHANGED_PER_IMAGE = 1
MARGIN_FACTOR = 2.0
# Сверка со штатным путём: логиты прямого вызова головок против штатного `image_guided_detection` на той же паре «сцена —
# эталон» — допуск порядка округления fp16, записан до сверки: |Δ| ≤ ATOL + RTOL·|штатный логит|; 2⁻⁸ — четыре
# относительных шага fp16 (2⁻¹⁰). Рамки обязаны совпасть точно: вход и вычисление у них одни.
STOCK_RTOL = 2.0 ** -8
STOCK_ATOL = 2.0 ** -8
STOCK_REFS_PER_SCENE = 5     # пары сверки: на сцене k пилота — эталоны `one_per_class` с номерами (10k + i) mod n, i < 5
# Сверка со штатным идёт при той же форме вызова `class_predictor`, что счёт: столбцов столько же, сколько строк полной
# галереи (2 400 и 716), — запросы `one_per_class` повторены до этого числа (ядро умножения матриц в fp16 выбирается
# по форме); на обеих областях — калибровочные снимки пилота.


def constants() -> dict:
    """Константы, от которых зависят детекции, — в ключ рабочих файлов и в запись."""
    return {"dtype": DTYPE, "attn_implementation": ATTN, "query_alpha": QUERY_ALPHA, "pad_value": PAD_VALUE,
            "nms_iou": NMS_IOU, "score_min": SCORE_MIN, "protocols": list(PROTOCOLS),
            "aggregation": "max sigmoid(logit) по эталонам метки; одна метка на рамку, при равенстве — меньший номер",
            "nms": "жадный, без учёта меток, по убыванию оценки; подавляется IoU > nms_iou (как в штатном)"}


# ---------------------------------------------------------------------- модель и предобработка


def load_model(dtype: str = DTYPE, revision: str | None = None):
    """Модель на GPU без текстовой башни (она и её проекция остаются на CPU: случайный вызов упал бы на устройстве),
    штатный процессор и разрешение входа — из конфигурации установленных процессора и модели."""
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2ImageProcessor

    from src import env

    mid = env.MODEL_IDS[MODEL_KEY]
    model = Owlv2ForObjectDetection.from_pretrained(mid, revision=revision, dtype=getattr(torch, dtype),
                                                    attn_implementation=ATTN).eval()
    model.to("cuda")
    for name in TEXT_MODULES:
        model.get_submodule(name).to("cpu")
    processor = Owlv2ImageProcessor.from_pretrained(mid, revision=revision)
    return model, processor, input_size(model, processor)


def input_size(model, processor) -> int:
    """Сторона входа: `size` процессора, сверенный с `vision_config.image_size` модели (при расхождении — отказ)."""
    s = processor.size
    side = model.config.vision_config.image_size
    if not (s["height"] == s["width"] == side) or side % model.config.vision_config.patch_size:
        raise ValueError(f"вход процессора {s} расходится с image_size модели {side}")
    return int(side)


def query_image(img: np.ndarray, box) -> np.ndarray:
    """Изображение-запрос: квадрат `src.encode.crop.square` при α = 1,0 вокруг рамки из сырого снимка, без замены фона;
    часть за кадром — серый штатного паддинга. float32 в шкале 0–255: процессор делит на 255, и 127,5 даёт ровно 0,5."""
    qx, qy, side = square(box, QUERY_ALPHA)
    out = np.full((side, side, 3), PAD_VALUE * 255.0, np.float32)
    h, w = img.shape[:2]
    a0, a1, b0, b1 = max(qx, 0), min(qx + side, w), max(qy, 0), min(qy + side, h)
    if a1 > a0 and b1 > b0:
        out[b0 - qy:b1 - qy, a0 - qx:a1 - qx] = img[b0:b1, a0:a1]
    return out


def preprocess(processor, image: np.ndarray) -> np.ndarray:
    """Штатная предобработка: паддинг до квадрата снизу и справа серым, приведение к стороне входа, нормировка."""
    return processor(images=image, return_tensors="np", input_data_format="channels_last")["pixel_values"][0]


_PROC = {}


def _proc():
    if "p" not in _PROC:
        from transformers import Owlv2ImageProcessor

        from src import env

        _PROC["p"] = Owlv2ImageProcessor.from_pretrained(env.MODEL_IDS[MODEL_KEY], revision=_PROC.get("revision"))
    return _PROC["p"]


def init_worker(revision: str | None) -> None:
    """Процесс подготовки входа: свой экземпляр штатного процессора той же ревизии."""
    _PROC["revision"] = revision


def prep_scene(path: str) -> tuple[np.ndarray, tuple[int, int], float]:
    """Снимок → вход модели, (h, w) снимка, секунды на чтение и предобработку (в процессе подготовки)."""
    import time

    from src.segment.sam2 import read_rgb

    t0 = time.perf_counter()
    img = read_rgb(path)
    return preprocess(_proc(), img), img.shape[:2], time.perf_counter() - t0


def prep_query(path: str, box) -> tuple[np.ndarray, float]:
    import time

    from src.segment.sam2 import read_rgb

    t0 = time.perf_counter()
    return preprocess(_proc(), query_image(read_rgb(path), box)), time.perf_counter() - t0


# ---------------------------------------------------------------------- вызовы модели (без текстовой башни)


def image_features(model, pixel_values):
    """Проход зрительной башни: карта признаков (B, h, w, D) и она же построчно (B, h·w, D)."""
    fm = model.image_embedder(pixel_values=pixel_values)[0]
    return fm, fm.reshape(fm.shape[0], -1, fm.shape[-1])


def query_embed(model, pixel_values):
    """Эмбеддинг запроса по одному изображению-запросу (батч 1: штатный `embed_image_query` пропускает запрос без
    отобранной рамки молча, и в батче номера бы съехали). Возвращает (D,) и номер рамки либо (None, None)."""
    if pixel_values.shape[0] != 1:
        raise ValueError("изображение-запрос — по одному")
    fm, feats = image_features(model, pixel_values)
    q, idx, _ = model.embed_image_query(feats, fm)
    if q is None:
        return None, None
    return q.reshape(-1), int(idx.reshape(-1)[0])


def scene_heads(model, feature_map, feats, queries):
    """Головки по одной сцене: логиты (P, N) пар «рамка — запрос» и рамки (P, 4) cxcywh в долях паддированного
    квадрата. `queries` — (N, D) в dtype модели; батч сцен — 1."""
    if feats.shape[0] != 1:
        raise ValueError("сцена — по одной")
    logits = model.class_predictor(image_feats=feats, query_embeds=queries[None])[0][0]
    boxes = model.box_predictor(feats, feature_map)[0]
    return logits, boxes


def columns(rows: list[dict], prot: dict, labels: list[str], has) -> dict:
    """Столбцы логитов: запросы строк полной галереи, у которых есть эмбеддинг, в порядке строк; для протокола —
    номера столбцов его строк (`sub`) и номера меток этих столбцов (`col_label`)."""
    has = np.asarray(has, bool)
    label_of = {y: k for k, y in enumerate(labels)}
    full = np.array([i for i in range(len(rows)) if has[i]], np.int64)
    pos = {int(i): k for k, i in enumerate(full)}                          # строка галереи → столбец логитов
    sub = {p: np.array([pos[int(i)] for i in prot[p] if has[i]], np.int64) for p in PROTOCOLS}
    return {"full_rows": full, "pos": pos, "sub": sub,
            "col_label": {p: np.array([label_of[rows[int(full[c])]["label"]] for c in sub[p]], np.int64) for p in PROTOCOLS}}


def protocol_logits(logits, cols: dict, n_labels: int) -> dict:
    """Оба протокола галереи из одних логитов: для каждого — наибольший логит по его запросам метки, (P, L)."""
    import torch

    return {p: label_max_logits(logits[:, torch.as_tensor(cols["sub"][p], device=logits.device)], cols["col_label"][p],
                                n_labels) for p in PROTOCOLS}


def label_max_logits(logits, col_label, n_labels: int) -> np.ndarray:
    """Наибольший логит по запросам метки: (P, L) float64; метка без запросов — −inf. Сигмоида монотонна, так что
    максимум сигмоид по эталонам метки — сигмоида этого максимума."""
    import torch

    idx = torch.as_tensor(np.asarray(col_label), device=logits.device, dtype=torch.long)
    out = torch.full((logits.shape[0], n_labels), float("-inf"), device=logits.device, dtype=torch.float32)
    out.scatter_reduce_(1, idx[None].expand(logits.shape[0], -1), logits.float(), reduce="amax", include_self=True)
    return out.cpu().numpy().astype(np.float64)


@contextlib.contextmanager
def trace_modules(model, into: set):
    """Имена сработавших модулей до второго уровня вложенности (`owlv2.vision_model`, `class_head`, …); после
    выхода — отказ, если среди них текстовые."""
    hooks = [m.register_forward_hook(lambda *_, n=name: into.add(n))
             for name, m in model.named_modules() if name and name.count(".") <= 1]
    try:
        yield into
    finally:
        for h in hooks:
            h.remove()
    text = sorted(n for n in into if n.startswith(TEXT_MODULES) or n == "owlv2")
    if text:
        raise AssertionError(f"вызваны модули текстовой башни либо полный forward модели: {text}")


# ---------------------------------------------------------------------- детекции: агрегация, NMS, рамки


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def assign(label_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Одна метка на рамку: оценка по метке — сигмоида наибольшего логита, метка — $\\arg\\max$ (первая, то
    есть с меньшим номером, при равенстве), оценка детекции — этот максимум. Метки без запросов не выбираются."""
    s = sigmoid(label_logits)
    s[np.isneginf(label_logits)] = -np.inf
    lab = np.argmax(s, axis=1)
    return lab.astype(np.int64), s[np.arange(len(s)), lab]


def label_margin(label_logits: np.ndarray) -> np.ndarray:
    """Разрыв между наибольшим и вторым логитом меток рамки (метки без запросов не считаются), (P,)."""
    x = np.where(np.isneginf(label_logits), np.nan, np.asarray(label_logits, np.float64))
    top2 = -np.sort(-np.nan_to_num(x, nan=-np.inf), axis=1)[:, :2]
    return top2[:, 0] - top2[:, 1]


def corners(cxcywh: np.ndarray) -> np.ndarray:
    b = np.asarray(cxcywh, np.float64).reshape(-1, 4)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def nms(xyxy: np.ndarray, score: np.ndarray, iou_thr: float = NMS_IOU) -> np.ndarray:
    """Жадный NMS без учёта меток: номера оставленных рамок по убыванию оценки (равные — по номеру). Подавляется
    рамка с IoU строго больше порога, как в штатной постобработке; IoU вырожденной пары — 0."""
    b = np.asarray(xyxy, np.float64).reshape(-1, 4)
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = np.argsort(-np.asarray(score, np.float64), kind="stable")
    alive = np.ones(len(b), bool)
    keep = []
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        iw = np.clip(np.minimum(b[i, 2], b[:, 2]) - np.maximum(b[i, 0], b[:, 0]), 0, None)
        ih = np.clip(np.minimum(b[i, 3], b[:, 3]) - np.maximum(b[i, 1], b[:, 1]), 0, None)
        inter = iw * ih
        union = area[i] + area - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        iou[i] = -1.0
        alive &= ~(iou > iou_thr)
    return np.asarray(keep, np.int64)


def to_pixels(cxcywh: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Рамки из долей паддированного квадрата в пиксели снимка: умножение на $\\max(H,W)$, затем обрезка по
    кадру. Рамки исключающие, как у истины (`src.eval.boxes`)."""
    h, w = hw
    b = corners(cxcywh) * max(h, w)
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
    return b


def detections(scene: dict, protocol: str) -> dict:
    """Детекции снимка по рабочему файлу (`scene` — содержимое `.npz`): рамки в пикселях, номер метки, оценка."""
    keep = scene[f"{protocol}_keep"]
    return {"box": to_pixels(scene["boxes"][keep], tuple(scene["hw"])),
            "label_id": scene[f"{protocol}_label"][keep].astype(np.int64),
            "score": scene[f"{protocol}_score"][keep].astype(np.float64)}


def fingerprint() -> str:
    """sha256 исходника этого модуля: путь счёта и константы. Сверки пишут его в `env.json`, счёт и оценка сверяют
    с текущим — сверка, пройденная другим кодом, прогон не открывает."""
    from pathlib import Path

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def refs_sha256(rows: list[dict]) -> str:
    """Состав эталонов-запросов: `id`, метка, снимок и `prompt_box` в порядке строк галереи."""
    key = [[r["id"], r["label"], r["image"], [float(v) for v in r["prompt_box"]]] for r in rows]
    return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------- эталоны-запросы и снимки


def image_root(dataset: str):
    from src.data import hr_insdet, pcb

    return {"hr_insdet": hr_insdet.ROOT, "pcb": pcb.ROOT}[dataset]


def refs(dataset: str) -> tuple[list[dict], dict[str, np.ndarray], list[str], str]:
    """Эталоны-запросы: строки полной галереи лучшего φ DINOv2 в порядке `meta.jsonl` — `id`, метка, путь
    снимка, `prompt_box`; номера строк каждого протокола; метки (номер метки — индекс, как у `Gallery.labels`)."""
    from src.gallery import store

    enc, var = GALLERY[dataset]
    g = store.Gallery.open(store.gallery_dir(enc, var, dataset))
    root = image_root(dataset)
    rows = [{"id": m["id"], "label": m["label"], "image": m["image"], "path": str(root / m["image"]),
             "prompt_box": [float(v) for v in m["prompt_box"]]} for m in g.meta]
    full = g.rows("full")
    if not np.array_equal(full, np.arange(len(rows))):
        raise ValueError(f"{g.path}: в полной галерее есть помеченные удалёнными — состав запросов не определён")
    return rows, {p: g.rows(p) for p in PROTOCOLS}, g.labels, str(g.path)
