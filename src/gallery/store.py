"""Галерея эталонов на диске; §2.5.

Каталог на прогон сетки: `gallery/<encoder>/<variant>/<dataset>/` — датасет в пути, потому что у семи вариантов
DINOv2 пара «энкодер — вариант» одна на HR-InsDet и PCB. Файлы: `gallery.json` —
паспорт, `emb.npy` — float32 $N\\times d$, нормированные, `meta.jsonl` — по строке на эталон, `calib.json` — запись
калибровки (сама калибровка — `scripts/calibrate.py`; здесь формат, запись и чтение). `U_r.npy` лежит уровнем выше: он один на энкодер.

Паспорт сверяется при чтении, расхождение — ошибка, как у кеша масок: чужая галерея по ошибке пути не читается.
Все файлы пишутся во временный и переименовываются; паспорт с числом эталонов пишется последним,
поэтому оборванная запись обнаруживается при чтении по расхождению длин.

Удаление эталонов не реализуется: есть поле `deleted`, помеченные эталоны исключаются из поиска
(`active`, `src.search`), операции пометки и перестройки индекса нет.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("gallery")
FORMAT = 1
MODES = ("instance", "category")
ORIGINS = ("profile", "scene")
PROTOCOLS = ("full", "one_per_class", "dedup")  # `dedup` — состав из `dedup.json`
# поля строки `meta.jsonl`; `insert_no` и `deleted` ставит `add`
ENTRY_FIELDS = ("id", "label", "mode", "label2", "image_id", "mask_rle", "origin")
PASSPORT_KEYS = ("dataset", "encoder", "variant", "config_sha1", "mask_key")
NORM_TOL = 1e-4

F_TOP = 200_000       # точно хранятся наибольшие сходства «дистрактор — эталон»…
F_QUANTILES = 10_001  # …плюс сетка квантилей


def gallery_dir(encoder: str, variant: str, dataset: str, root: Path | None = None) -> Path:
    return (ROOT if root is None else Path(root)) / encoder / variant / dataset


def config_sha1(config: dict) -> str:
    return hashlib.sha1(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def passport(config: dict, mask_key: dict, d: int) -> dict:
    """`config` — `RunConfig.to_dict()`; `mask_key` — ключ кеша масок эталонов (режим `box`)."""
    config = json.loads(json.dumps(config, ensure_ascii=False))
    return {"format": FORMAT, "dataset": config["dataset"], "encoder": config["encoder"], "variant": config["variant"],
            "d": int(d), "config_sha1": config_sha1(config), "config": config, "mask_key": mask_key}


def expected(config: dict, mask_key: dict) -> dict:
    """То, с чем сверяется паспорт при чтении, когда размерность заранее не известна."""
    return {k: v for k, v in passport(config, mask_key, 0).items() if k in PASSPORT_KEYS}


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    tmp.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    _atomic(path, lambda fh: fh.write(text.encode()))


class Gallery:
    def __init__(self, path: Path, passport: dict, emb: np.ndarray, meta: list[dict]):
        self.path, self.passport, self.emb, self.meta = Path(path), passport, emb, meta

    # ------------------------------------------------------------------ создание, чтение, запись

    @classmethod
    def create(cls, path: Path, passport: dict) -> "Gallery":
        return cls(path, dict(passport), np.zeros((0, passport["d"]), np.float32), [])

    @classmethod
    def open(cls, path: Path, expect: dict | None = None) -> "Gallery":
        """`expect` — паспорт (`passport` либо `expected`), с которым галерея обязана совпасть по `PASSPORT_KEYS`."""
        path = Path(path)
        pp = json.loads((path / "gallery.json").read_text())
        if pp["format"] != FORMAT:
            raise ValueError(f"{path}: формат галереи {pp['format']}, ожидается {FORMAT}")
        if expect is not None:
            diff = [k for k in (*PASSPORT_KEYS, "d") if k in expect and pp[k] != expect[k]]
            if diff:
                raise ValueError(f"{path}: галерея собрана при другом {diff} — не та галерея либо устаревшая")
        emb = np.load(path / "emb.npy")
        meta = [json.loads(line) for line in (path / "meta.jsonl").read_text().splitlines()]
        if emb.dtype != np.float32 or emb.ndim != 2 or emb.shape[1] != pp["d"]:
            raise ValueError(f"{path}: emb.npy — {emb.dtype} {emb.shape}, ожидается float32 (N, {pp['d']})")
        if "emb_sha256" in pp and hashlib.sha256((path / "emb.npy").read_bytes()).hexdigest() != pp["emb_sha256"]:
            raise ValueError(f"{path}: emb.npy не тот, что записан в паспорте")
        if not len(emb) == len(meta) == pp["n"]:
            raise ValueError(f"{path}: запись оборвана — emb {len(emb)}, meta {len(meta)}, паспорт {pp['n']}")
        g = cls(path, pp, np.ascontiguousarray(emb), meta)
        g._check_meta()
        return g

    def save(self) -> None:
        self._check_meta()
        self.path.mkdir(parents=True, exist_ok=True)
        _atomic(self.path / "emb.npy", lambda fh: np.save(fh, self.emb))
        self.passport["emb_sha256"] = hashlib.sha256((self.path / "emb.npy").read_bytes()).hexdigest()
        _atomic_text(self.path / "meta.jsonl", "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in self.meta))
        _atomic_text(self.path / "gallery.json",
                     json.dumps({**self.passport, "n": len(self.meta)}, ensure_ascii=False, indent=1) + "\n")

    def _check_meta(self) -> None:
        ids = [m["id"] for m in self.meta]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.path}: повторяющиеся идентификаторы эталонов")
        if [m["insert_no"] for m in self.meta] != list(range(len(self.meta))):
            raise ValueError(f"{self.path}: номера вставки не идут подряд с нуля")

    # ------------------------------------------------------------------ пополнение

    def add(self, z: np.ndarray, entries: list[dict], rho_selected: dict[str, set[int]] | None = None) -> list[int]:
        """Пополнение $G_N\\to G_{N+n}$: эмбеддинги (n, d), нормированные, и записи с полями `ENTRY_FIELDS`
        (прочие поля сохраняются как есть). Возвращает номера вставки.

        Отбор при индексации из сцены: эталон `origin=scene`, взятый по маске авторежима
        (поле `auto_mask_no` — номер маски в записи кеша авторежима снимка `image_id`), допускается, только если эта
        маска входит в $M^*$ своего изображения: `rho_selected` — снимок → номера масок $M^*$. Эталон, заданный
        рамкой, поля не несёт, и $\\rho$ к нему не применяется."""
        z = np.ascontiguousarray(z, dtype=np.float32)  # из fp16-модели — явное приведение
        if z.ndim != 2 or z.shape[1] != self.passport["d"] or len(z) != len(entries):
            raise ValueError(f"эмбеддинги {z.shape} при {len(entries)} записях и d = {self.passport['d']}")
        if not np.isfinite(z).all():  # Faiss на NaN молча возвращает бессмысленных соседей
            raise FloatingPointError("не-конечный эмбеддинг эталона")
        if len(z) and np.abs(np.linalg.norm(z, axis=1) - 1).max() > NORM_TOL:
            raise ValueError("эмбеддинги эталонов не нормированы на единицу")
        known = {m["id"] for m in self.meta}
        out = []
        for e in entries:
            missing = [k for k in ENTRY_FIELDS if k not in e]
            if missing or e["mode"] not in MODES or e["origin"] not in ORIGINS:
                raise ValueError(f"запись эталона {e.get('id')}: нет полей {missing} либо mode / origin вне списка")
            if e["id"] in known or "insert_no" in e or "deleted" in e:
                raise ValueError(f"эталон {e['id']}: уже в галерее либо несёт служебные поля")
            if e.get("auto_mask_no") is not None:
                if e["origin"] != "scene":
                    raise ValueError(f"эталон {e['id']}: маска авторежима — только у эталона из сцены")
                if rho_selected is None or e["image_id"] not in rho_selected:
                    raise ValueError(f"эталон {e['id']}: маска авторежима без M* её изображения")
                if int(e["auto_mask_no"]) not in rho_selected[e["image_id"]]:
                    raise ValueError(f"эталон {e['id']}: маска {e['auto_mask_no']} не входит в M* — в галерею не идёт")
            known.add(e["id"])
            out.append(len(self.meta))
            self.meta.append({**e, "insert_no": out[-1], "deleted": False})
        self.emb = np.ascontiguousarray(np.concatenate([self.emb, z]))
        return out

    # ------------------------------------------------------------------ состав

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def deleted(self) -> np.ndarray:
        return np.array([m["deleted"] for m in self.meta], bool)

    @property
    def labels(self) -> list[str]:
        """Множество меток $\\mathcal Y$ в отсортированном порядке; номер метки — индекс в этом списке."""
        return sorted({m["label"] for m in self.meta})

    @property
    def label_ids(self) -> np.ndarray:
        idx = {y: i for i, y in enumerate(self.labels)}
        return np.array([idx[m["label"]] for m in self.meta], np.int64)

    def rows(self, protocol: str = "full", upto_insert_no: int | None = None) -> np.ndarray:
        """Строки галереи, участвующие в поиске: без `deleted`; `upto_insert_no` восстанавливает состояние
        $G_{N_0}$ на момент калибровки (§2.5); `one_per_class` — первый эталон метки по отсортированному
        идентификатору, то есть по имени файла — подмножество полной галереи, те же векторы;
        `dedup` — состав `dedup.json`, тоже подмножество строк."""
        if protocol not in PROTOCOLS:
            raise ValueError(f"протокол галереи {protocol!r} вне {PROTOCOLS}")
        if protocol == "dedup":
            from src.gallery import dedup

            if upto_insert_no is not None:
                raise ValueError("состав дедупликации определён для полной галереи, без среза по номеру вставки")
            rows = dedup.read_rows(self)
            return rows[~self.deleted[rows]]
        keep = ~self.deleted
        if upto_insert_no is not None:
            keep &= np.array([m["insert_no"] <= upto_insert_no for m in self.meta], bool)
        rows = np.flatnonzero(keep)
        if protocol == "one_per_class":
            first: dict[str, int] = {}
            for i in sorted(rows, key=lambda i: self.meta[i]["id"]):
                first.setdefault(self.meta[i]["label"], int(i))
            rows = np.array(sorted(first.values()), np.int64)
        return rows

    def n_max(self, rows: np.ndarray) -> int:
        """$n_{\\max}=\\max_y n_y$ по строкам `rows`."""
        return int(np.bincount(self.label_ids[rows]).max()) if len(rows) else 0

    def k(self, rows: np.ndarray) -> int:
        """$k=2n_{\\max}$, но не больше $N$."""
        return min(2 * self.n_max(rows), len(rows))


def rows_sha256(g: Gallery, rows) -> str:
    """sha256 списка `id` строк в порядке `insert_no` — состав строк галереи: у полной галереи,
    дедуплицированной и подвыборок `emb.npy` один, и хеш файла смену состава не видит."""
    ids = [g.meta[i]["id"] for i in sorted((int(r) for r in rows), key=lambda i: g.meta[i]["insert_no"])]
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


# ---------------------------------------------------------------------- calib.json


def eps_key(eps: float) -> str:
    return repr(float(eps))


def check_calib(rec: dict, eps: tuple[float, ...]) -> None:
    """Формат записи калибровки: `N0`, `F`, `delta`, `check_set_ids`, `by_eps` — по записи на каждый уровень
    $\\varepsilon$ с `tau_q`, `tau_m`, `kappa`; `check_rule` — правило контрольной проверки. `F` — функция выживания сходства «дистрактор — эталон» при $N_0$:
    `n_pairs`, `top` — наибольшие сходства по убыванию (не больше `F_TOP`), `quantiles` — сетка из `F_QUANTILES`
    квантилей по возрастанию; от $\\varepsilon$ не зависит."""
    missing = [k for k in ("format", "N0", "F", "delta", "check_set_ids", "check_rule", "by_eps") if k not in rec]
    if missing:
        raise ValueError(f"calib.json: нет полей {missing}")
    if rec["format"] != FORMAT or not (isinstance(rec["N0"], int) and rec["N0"] > 0) or not rec["delta"] > 0:
        raise ValueError("calib.json: формат, N0 или delta")
    f = rec["F"]
    top, q = np.asarray(f["top"], float), np.asarray(f["quantiles"], float)
    if not (0 < len(top) <= min(F_TOP, f["n_pairs"]) and (np.diff(top) <= 0).all()):
        raise ValueError("calib.json: F.top — наибольшие сходства по убыванию")
    if len(q) != F_QUANTILES or (np.diff(q) < 0).any() or q[-1] != top[0]:
        raise ValueError(f"calib.json: F.quantiles — {F_QUANTILES} квантилей по возрастанию, последний — максимум")
    if not all(isinstance(i, str) for i in rec["check_set_ids"]):
        raise ValueError("calib.json: check_set_ids — идентификаторы масок")
    if set(rec["by_eps"]) != {eps_key(e) for e in eps}:
        raise ValueError(f"calib.json: by_eps — уровни {sorted(rec['by_eps'])}, ожидаются {[eps_key(e) for e in eps]}")
    for e, r in rec["by_eps"].items():
        if set(r) != {"tau_q", "tau_m", "kappa"} or not 0 < r["kappa"] <= 1:
            raise ValueError(f"calib.json: by_eps[{e}] — tau_q, tau_m, kappa ∈ (0, 1]")
        if not all(np.isfinite(r[k]) for k in r):
            raise ValueError(f"calib.json: by_eps[{e}] — не-конечное значение")


ROWS_FIELDS = ("gallery_rows", "gallery_rows_sha256")  # состав строк, по которому поставлен порог


def write_calib(gallery_path: Path, rec: dict, eps: tuple[float, ...]) -> None:
    rec = {"format": FORMAT, **rec}
    check_calib(rec, eps)
    if not all(isinstance(rec.get(k), str) and rec[k] for k in ROWS_FIELDS):
        raise ValueError(f"calib.json: запись калибровки несёт состав строк — поля {ROWS_FIELDS}")
    _atomic_text(Path(gallery_path) / "calib.json", json.dumps(rec, ensure_ascii=False) + "\n")


def read_calib(gallery_path: Path, eps: tuple[float, ...], gallery: Gallery, protocol: str = "full") -> dict:
    """Порог действителен только для того состава строк, по которому поставлен: `gallery_rows_sha256` записи сверяется
    с составом строк `protocol` галереи `gallery`, расхождение — отказ. Старая запись без этого поля читается
    как `gallery_rows = "full"` только после сверки `N0` с числом строк полной галереи."""
    rec = json.loads((Path(gallery_path) / "calib.json").read_text())
    check_calib(rec, eps)
    rows = gallery.rows(protocol)
    if "gallery_rows_sha256" not in rec:
        if protocol != "full" or rec["N0"] != len(gallery.rows("full")):
            raise ValueError("calib.json без состава строк: действителен только для полной галереи при том же N0")
        return {**rec, "gallery_rows": "full"}
    if rec["gallery_rows"] != protocol or rec["gallery_rows_sha256"] != rows_sha256(gallery, rows) or rec["N0"] != len(rows):
        raise ValueError(f"calib.json поставлен по составу строк «{rec['gallery_rows']}», запрошен «{protocol}» — "
                         "порог для другого состава недействителен")
    return rec
