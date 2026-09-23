"""Подготовка вырезок на CPU параллельно проходу GPU.

Это способ счёта, а не параметр метода. Единица работы — батч масок одного изображения в порядке записи кеша;
батч не пересекает границу изображения. Основной процесс забирает батчи строго в порядке задач, поэтому порядок
и состав батчей те же, что в последовательном пути (`workers=0`), и эмбеддинги обязаны совпадать побитово
(проверено на калибровочных сценах, `experiments/crop_pipeline_check.json`). От размера и состава батча в половинной точности зависит округление — они поля
конфигурации и журнала; число процессов — нет.
"""

from __future__ import annotations

import collections
import multiprocessing as mp
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from src.encode import crop as C

BATCH = 16          # размер батча энкодера
BATCH_RULE = "маски одного изображения в порядке записи кеша; батч не пересекает границу изображения"
WORKERS = 6         # при рабочей стороне размытия 1 792 шесть процессов на C(blur,1.5) быстрее четырёх
                    # на 10 %, восемь — нет; на C(0,1.0) разницы нет (`crop_pipeline_check.json`, `workers_scan`)
MAX_IN_FLIGHT = 8   # батчей в полёте: 8 × 16 × 448² × 3 байт ≈ 77 МБ


@dataclass(frozen=True)
class ImageRef:
    """Снимок и его запись в кеше масок — всё, что нужно процессу подготовки, чтобы открыть их самому."""
    dataset: str
    image_id: str
    path: str
    key_json: str                      # ключ кеша масок, JSON (`segment.cache.auto_key` | `box_key`)
    prompt_boxes_json: str | None = None
    cache_root: str | None = None


@dataclass(frozen=True)
class Task:
    ref: ImageRef
    start: int
    stop: int
    b: str
    alpha: float
    size: int = C.CROP_SIZE
    sigma_frac: float = C.SIGMA_FRAC
    work_side: int = C.BLUR_WORK_SIDE
    indices: tuple[int, ...] | None = None   # номера масок батча при отборе $\\rho$; без отбора — `start:stop`

    @property
    def mask_numbers(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop)) if self.indices is None else self.indices


def image_ref(dataset: str, image_id: str, path, key: dict, prompt_boxes: list | None = None,
              cache_root=None) -> ImageRef:
    import json

    return ImageRef(dataset, image_id, str(path), json.dumps(key, sort_keys=True),
                    None if prompt_boxes is None else json.dumps(prompt_boxes),
                    None if cache_root is None else str(cache_root))


def load_entry(ref: ImageRef):
    import json

    from src.segment import cache as MC

    boxes = None if ref.prompt_boxes_json is None else json.loads(ref.prompt_boxes_json)
    return MC.MaskCache(ref.dataset, json.loads(ref.key_json), root=ref.cache_root).load(ref.image_id, boxes)


def batch_tasks(ref: ImageRef, n_masks: int, b: str, alpha: float, batch: int = BATCH, selected=None,
                **crop_kw) -> list[Task]:
    """Батчи одного изображения: маски в порядке записи кеша, последний батч — неполный.

    `selected` — номера масок $M^*$: кодируются только они, правило состава батчей применяется к
    отобранным маскам; `None` — все маски. `crop_kw` — `size`, `sigma_frac`, `work_side` из конфигурации прогона.
    """
    if selected is None:
        return [Task(ref, k, min(k + batch, n_masks), b, alpha, **crop_kw) for k in range(0, n_masks, batch)]
    sel = [int(i) for i in selected]
    if sel != sorted(set(sel)) or (sel and not 0 <= sel[0] <= sel[-1] < n_masks):
        raise ValueError(f"{ref.image_id}: номера отобранных масок — по возрастанию, без повторов, в пределах записи кеша")
    return [Task(ref, k, min(k + batch, len(sel)), b, alpha, indices=tuple(sel[k:k + batch]), **crop_kw)
            for k in range(0, len(sel), batch)]


_state: dict = {}


def make_batch(task: Task) -> np.ndarray:
    """Вырезки `records[start:stop]` снимка, (n, size, size, 3) uint8. Снимок и запись кеша держатся в процессе,
    пока снимок не сменится."""
    from src.segment import sam2 as S

    if _state.get("ref") != task.ref:
        _state.clear()
        _state.update(ref=task.ref, entry=load_entry(task.ref), img=S.read_rgb(task.ref.path))
    e, img = _state["entry"], _state["img"]
    if tuple(img.shape[:2]) != tuple(e.hw):
        raise ValueError(f"{task.ref.image_id}: размер снимка {img.shape[:2]} расходится с записью кеша {e.hw}")
    return np.stack([C.make_crop(img, e.records[i]["box"], e.windower(i), task.b, task.alpha, task.size,
                                 sigma_frac=task.sigma_frac, work_side=task.work_side)
                     for i in task.mask_numbers])


def _init_worker(cv2_threads: int) -> None:
    import cv2

    cv2.setNumThreads(cv2_threads)


class CropFeeder:
    """Батчи вырезок в порядке задач. `workers=0` — последовательный путь в основном процессе.

    Процессы создаются способом `spawn` и, в скриптах прогона, до инициализации CUDA:
    объект создаётся раньше загрузки модели.
    """

    def __init__(self, workers: int = WORKERS, max_in_flight: int = MAX_IN_FLIGHT, cpu_threads: int | None = None):
        self.workers, self.max_in_flight = workers, max_in_flight
        self.cv2_threads = None
        self._pool = None
        if workers > 0:
            self.cv2_threads = max(1, (cpu_threads or os.cpu_count() or 1) // workers)
            self._pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(self.cv2_threads,))

    def __enter__(self) -> "CropFeeder":
        return self

    def __exit__(self, exc_type, *_) -> None:
        self.close(abort=exc_type is not None)

    def close(self, abort: bool = False) -> None:
        if self._pool is not None:
            if abort:  # при ошибке не ждать батчей, оставшихся в полёте
                self._pool.terminate()
            else:
                self._pool.close()
            self._pool.join()
            self._pool = None

    def batches(self, tasks: Iterable[Task]) -> Iterator[tuple[Task, np.ndarray]]:
        if self._pool is None:
            for task in tasks:
                yield task, make_batch(task)
            return
        pending: collections.deque = collections.deque()
        it = iter(tasks)
        while True:
            while len(pending) < self.max_in_flight and (task := next(it, None)) is not None:
                pending.append((task, self._pool.apply_async(make_batch, (task,))))
            if not pending:
                return
            task, res = pending.popleft()  # строго в порядке задач: состав батчей тот же, что в последовательном пути
            yield task, res.get()

    def journal(self, batch: int = BATCH) -> dict:
        """Поля журнала прогона. Число процессов — справочно: на эмбеддинги оно не влияет."""
        return {"encoder_batch": batch, "batch_rule": BATCH_RULE, "crop_workers": self.workers,
                "crop_max_in_flight": self.max_in_flight, "cv2_threads_per_worker": self.cv2_threads}


def encode_variant_multi(models: dict, feeder: CropFeeder, images: list[tuple], b: str, alpha: float,
                         batch: int = BATCH, **crop_kw) -> Iterator[tuple[ImageRef, dict[str, np.ndarray]]]:
    """$z_{\\mathrm{crop}}(b,\\alpha)$ всех масок каждого снимка для каждой модели из `models` (имя → модель).

    Вырезка от энкодера не зависит, поэтому батч готовится один раз и кодируется каждой моделью по очереди;
    состав и порядок батчей у каждой модели те же, что в её отдельном прогоне. Выдаёт (снимок, {имя: (n, d)
    float32}) в порядке `images`; снимок без масок даёт пустые массивы. Элемент `images` — (снимок, число масок) либо
    (снимок, число масок, номера масок $M^*$): во втором случае кодируются только отобранные маски, строки — в их порядке.
    """
    from src.encode import model as M

    images = [(im[0], im[1], im[2] if len(im) > 2 else None) for im in images]
    tasks = [t for ref, n, sel in images for t in batch_tasks(ref, n, b, alpha, batch, selected=sel, **crop_kw)]
    stream = feeder.batches(tasks)
    for ref, n, sel in images:
        parts: dict[str, list] = {name: [] for name in models}
        for _ in range(0, n if sel is None else len(sel), batch):
            task, crops = next(stream)
            assert task.ref == ref
            for name, model in models.items():
                parts[name].append(M.encode_crops(model, crops).cpu().numpy())
        yield ref, {name: (np.concatenate(v) if v else np.zeros((0, models[name].config.hidden_size), np.float32))
                    for name, v in parts.items()}


def encode_variant(model, feeder: CropFeeder, images: list[tuple[ImageRef, int]], b: str, alpha: float,
                   batch: int = BATCH, **crop_kw) -> Iterator[tuple[ImageRef, np.ndarray]]:
    """То же для одной модели: (снимок, (n, d) float32) в порядке `images`."""
    for ref, z in encode_variant_multi({"m": model}, feeder, images, b, alpha, batch, **crop_kw):
        yield ref, z["m"]
