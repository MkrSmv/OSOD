"""Метод сравнения OWLv2 в режиме image-guided → рабочие файлы и запись журнала.

    python scripts/run_owlv2.py --dataset pcb                 # эмбеддинги запросов, затем детекции по снимкам (GPU, прерываемо)
    python scripts/run_owlv2.py --dataset pcb --limit 5       # то же, но не больше 5 новых снимков
    python scripts/run_owlv2.py --dataset pcb --status        # что посчитано
    python scripts/run_owlv2.py --dataset pcb --queries-only  # только эмбеддинги запросов (нужны сверке точности)
    python scripts/run_owlv2.py --dataset pcb --evaluate      # оценка и запись experiments/runs/owlv2_pcb.json (без GPU)

Детекции (GPU): эталоны-запросы — строки полной галереи лучшего φ DINOv2 (`src.baselines.owlv2.GALLERY`); на каждый —
вырезка по `prompt_box` и один проход зрительной башни → `cache/owlv2/<dataset>/queries/` кусками по `QUERY_CHUNK`;
затем по снимку — один проход башни, логиты по всем запросам, оба протокола галереи из одних логитов →
`cache/owlv2/<dataset>/scenes/<image_id>.npz`. Снимки: HR-InsDet — все 160 сцен (калибровочные и тестовые), PCB —
361 снимок тестовых плат. Посчитанное пропускается; ключ рабочих файлов (модель и ревизия, версия transformers,
разрешение входа, константы, состав эталонов) пишется в `key.json` и сверяется: расхождение — отказ, а не смешение.
Предобработка штатным процессором идёт на CPU в `PREP_WORKERS` процессах параллельно GPU.

Оценка (`--evaluate`, без GPU, при чистом дереве и всех посчитанных снимках): тем же кодом, что контрольный прогон
метода (`protocol.evaluate_baseline` с готовыми детекциями), с теми же сценами, seed, числом повторов бутстрэпа и
`maxDets`, что у контрольного прогона `dinov2_c_mean_10_<dataset>`; AR итоговых детекций — только без порога.
Запись не перезаписывается. Не прогон сетки: `kind: comparison_run`.

Предусловия (`scripts/check_owlv2.py`; записи `experiments/env.json` пройдены при чистом дереве и тем же `owlv2.py`):
запросы и детекции — `owlv2_direct_vs_stock`; `owlv2_precision` — только при половинной точности (модель считается
в fp32, и сверка точности не требуется). Отпечаток `owlv2.py` в записи сверки обязан совпасть с текущим файлом; после
любой правки `owlv2.py` сверку повторяют: `python scripts/check_owlv2.py stock`. Вывод — ещё и в
`logs/owlv2_<dataset>.log`.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as CFG  # noqa: E402
from src import env  # noqa: E402
from src.baselines import owlv2 as O  # noqa: E402
from src.eval import protocol as PR  # noqa: E402
from src.eval import rules as RU  # noqa: E402

WORK = Path("cache/owlv2")
RUNS = Path("experiments/runs")
QUERY_CHUNK = 50      # эталонов в одном рабочем файле запросов
PREP_WORKERS = 3      # процессов штатной предобработки на CPU; на числа не влияет (сцена HR-InsDet — около 7 с и 3 ГБ ОЗУ)
PAIR_CONFIG = {d: Path(f"configs/dinov2_c_mean_10_{d}.yaml") for d in O.GALLERY}  # seed, bootstrap, max_dets — как у пары
_STAMP = env.code_stamp()


def _logger(dataset: str):
    Path("logs").mkdir(exist_ok=True)
    log = open(Path("logs") / f"owlv2_{dataset}.log", "a")

    def say(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [owlv2_{dataset}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    return say


def record_path(dataset: str) -> Path:
    return RUNS / f"owlv2_{dataset}.json"


def scene_list(dataset: str) -> list[tuple[str, str, str]]:
    """(идентификатор, роль, путь снимка) в порядке оценки: калибровочные, затем тестовые (у PCB — только тестовые)."""
    sp = json.loads(Path(f"splits/{dataset}.json").read_text())
    if dataset == "hr_insdet":
        from src.data import hr_insdet as H

        return [(i, r, str(H.ROOT / "Scenes" / f"{i}.jpg")) for r in ("cal", "test") for i in sp[r]]
    from src.data import pcb as P

    img = {i["id"]: i["image"] for i in P.images()}
    return [(i, "test", str(P.ROOT / img[i])) for i in sp["test"]]


def make_key(dataset: str, rows: list[dict], gallery_path: str, size: int) -> dict:
    models = json.loads(env.ENV_JSON.read_text())["models"]
    return {"dataset": dataset, "model_id": env.MODEL_IDS[O.MODEL_KEY], "revision": models[O.MODEL_KEY]["revision"],
            "transformers": env.package_versions()["transformers"], "input_size": size, "constants": O.constants(),
            "gallery": gallery_path, "n_refs": len(rows), "refs_sha256": O.refs_sha256(rows)}


def key_sha(key: dict) -> str:
    return hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def check_key(dataset: str, key: dict) -> str:
    """`key.json` рабочих файлов: при первом счёте пишется, дальше сверяется — чужие файлы не смешиваются."""
    p = WORK / dataset / "key.json"
    if p.is_file():
        old = json.loads(p.read_text())
        if old != key:
            diff = sorted(k for k in set(old) | set(key) if old.get(k) != key.get(k))
            raise SystemExit(f"{p}: рабочие файлы посчитаны при другом ключе ({diff}) — удаляются только отдельным решением")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n")
    return key_sha(key)


def preconditions(sections=("owlv2_precision", "owlv2_direct_vs_stock")) -> dict:
    """Сверки `sections` пройдены при чистом дереве и тем же путём счёта: отпечаток `owlv2.py`, ревизия модели и разрешение
    входа в записи сверки совпадают с текущими."""
    data = json.loads(env.ENV_JSON.read_text())
    rev = data["models"][O.MODEL_KEY]["revision"]
    out = {}
    for sec in sections:
        r = data.get(sec)
        if not r or not r.get("passed") or r.get("code_dirty"):
            raise SystemExit(f"env.json, {sec}: сверка не пройдена либо не выполнена при чистом дереве — "
                             "сначала python scripts/check_owlv2.py stock")
        if r.get("owlv2_py_sha256") != O.fingerprint() or r.get("model", {}).get("revision") != rev:
            raise SystemExit(f"env.json, {sec}: сверка пройдена другим кодом owlv2.py либо другой ревизией модели — "
                             "повторить python scripts/check_owlv2.py stock")
        out[sec] = {k: r[k] for k in ("written", "code_commit", "passed", "owlv2_py_sha256", "input_size")}
    return out


def required_checks(queries_only: bool = False) -> tuple[str, ...]:
    """Сверка со штатным — всегда; сверка точности — только при половинной точности (в fp32 сверять не с чем:
    fp32 и есть эталон)."""
    if queries_only or O.DTYPE == "float32":
        return ("owlv2_direct_vs_stock",)
    return ("owlv2_precision", "owlv2_direct_vs_stock")


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _stamp_fields(ksha: str, modules: set) -> dict:
    return {"key_sha1": np.array(ksha), "code_commit": np.array(_STAMP["code_commit"]),
            "code_dirty": np.array(_STAMP["code_dirty"]), "modules": np.array(json.dumps(sorted(modules)))}


def _valid(f: dict, ksha: str) -> bool:
    return str(f["key_sha1"]) == ksha


def query_path(dataset: str, c: int) -> Path:
    return WORK / dataset / "queries" / f"chunk_{c:04d}.npz"


def scene_path(dataset: str, image_id: str) -> Path:
    return WORK / dataset / "scenes" / f"{image_id}.npz"


# ---------------------------------------------------------------------- счёт на GPU


def _pool(revision: str):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    return ProcessPoolExecutor(PREP_WORKERS, mp_context=mp.get_context("spawn"), initializer=O.init_worker,
                               initargs=(revision,))


def run_queries(dataset, model, rows, ksha, revision, say) -> None:
    import torch

    n_chunks = (len(rows) + QUERY_CHUNK - 1) // QUERY_CHUNK
    todo = [c for c in range(n_chunks) if not (query_path(dataset, c).is_file() and _valid(_load_npz(query_path(dataset, c)), ksha))]
    say(f"запросы: {len(rows)} эталонов, кусков {n_chunks}, посчитано {n_chunks - len(todo)}")
    if not todo:
        return
    dt = getattr(torch, O.DTYPE)
    t_start, done = time.perf_counter(), 0
    with _pool(revision) as pool:
        for c in todo:
            part = rows[c * QUERY_CHUNK:(c + 1) * QUERY_CHUNK]
            emb, box_idx, prep, tower, called = [], [], 0.0, 0.0, set()
            for px, sec in pool.map(O.prep_query, [r["path"] for r in part], [r["prompt_box"] for r in part]):
                prep += sec
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode(), O.trace_modules(model, called):
                    e, i = O.query_embed(model, torch.from_numpy(px)[None].to("cuda", dt))
                    if e is not None and not torch.isfinite(e).all():
                        raise FloatingPointError(f"не-конечный эмбеддинг запроса, кусок {c}")
                    e = None if e is None else e.float().cpu().numpy()
                torch.cuda.synchronize()
                tower += time.perf_counter() - t0
                emb.append(e), box_idx.append(-1 if i is None else i)
            if not any(e is not None for e in emb):
                raise SystemExit(f"кусок запросов {c}: ни у одного эталона нет эмбеддинга запроса — разобрать до счёта")
            d = next(e.shape[0] for e in emb if e is not None)
            _save_npz(query_path(dataset, c), ids=np.array([r["id"] for r in part]),
                      emb=np.stack([np.full(d, np.nan, np.float32) if e is None else e for e in emb]),
                      has=np.array([e is not None for e in emb]), box_index=np.array(box_idx, np.int32),
                      sec_prep=np.array(prep), sec_tower=np.array(tower), **_stamp_fields(ksha, called))
            done += 1
            el = time.perf_counter() - t_start
            say(f"запросы: кусок {c + 1}/{n_chunks}, башня {tower:.0f} с; осталось ~{el / done * (len(todo) - done) / 60:.0f} мин")


def load_queries(dataset: str, rows: list[dict], ksha: str) -> dict:
    n_chunks = (len(rows) + QUERY_CHUNK - 1) // QUERY_CHUNK
    fs = []
    for c in range(n_chunks):
        p = query_path(dataset, c)
        if not p.is_file():
            raise SystemExit(f"нет {p}: сначала запросы")
        f = _load_npz(p)
        if not _valid(f, ksha):
            raise SystemExit(f"{p}: другой ключ")
        fs.append(f)
    ids = np.concatenate([f["ids"] for f in fs])
    if ids.tolist() != [r["id"] for r in rows]:
        raise SystemExit("состав запросов расходится с галереей")
    return {"emb": np.concatenate([f["emb"] for f in fs]), "has": np.concatenate([f["has"] for f in fs]),
            "files": fs}


def run_scenes(dataset, model, rows, prot, labels, q, ksha, revision, limit, say) -> None:
    import torch

    scenes = scene_list(dataset)
    todo = [s for s in scenes if not (scene_path(dataset, s[0]).is_file() and _valid(_load_npz(scene_path(dataset, s[0])), ksha))]
    say(f"снимки: {len(scenes)}, посчитано {len(scenes) - len(todo)}" + (f"; в этом запуске — не больше {limit}" if limit else ""))
    if limit:
        todo = todo[:limit]
    if not todo:
        return
    dt = getattr(torch, O.DTYPE)
    cols = O.columns(rows, prot, labels, q["has"])
    Q = torch.from_numpy(np.ascontiguousarray(q["emb"][cols["full_rows"]])).to("cuda", dt)
    t_start = time.perf_counter()
    with _pool(revision) as pool:
        for k, ((sid, role, _), (px, hw, sec_prep)) in enumerate(zip(todo, pool.map(O.prep_scene, [s[2] for s in todo]))):
            called: set = set()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode(), O.trace_modules(model, called):
                fm, feats = O.image_features(model, torch.from_numpy(px)[None].to("cuda", dt))
                if not torch.isfinite(fm).all():
                    raise FloatingPointError(f"{sid}: не-конечный выход башни")
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                logits, boxes = O.scene_heads(model, fm, feats, Q)
                lm = O.protocol_logits(logits, cols, len(labels))
                boxes = boxes.float().cpu().numpy()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            out = {"boxes": boxes.astype(np.float32), "hw": np.array(hw, np.int64)}
            xyxy = O.corners(out["boxes"])
            for p in O.PROTOCOLS:
                lab, sc = O.assign(lm[p])
                out[f"{p}_label"], out[f"{p}_score"] = lab.astype(np.int16), sc
                out[f"{p}_keep"] = O.nms(xyxy, sc).astype(np.int32)
            t3 = time.perf_counter()
            _save_npz(scene_path(dataset, sid), **out, sec_prep=np.array(sec_prep), sec_tower=np.array(t1 - t0),
                      sec_heads=np.array(t2 - t1), sec_nms=np.array(t3 - t2), **_stamp_fields(ksha, called))
            el = time.perf_counter() - t_start
            say(f"снимок {k + 1}/{len(todo)} {sid}: башня {t1 - t0:.2f} с, головки {t2 - t1:.2f} с, NMS {t3 - t2:.2f} с, "
                f"детекций {len(out['full_keep'])} / {len(out['one_per_class_keep'])}; осталось ~{el / (k + 1) * (len(todo) - k - 1) / 60:.0f} мин")


def status(dataset: str, say) -> None:
    kp = WORK / dataset / "key.json"
    if not kp.is_file():
        say("рабочих файлов нет")
        return
    ksha = key_sha(json.loads(kp.read_text()))
    rows, *_ = O.refs(dataset)
    n_chunks = (len(rows) + QUERY_CHUNK - 1) // QUERY_CHUNK
    qd = sum(query_path(dataset, c).is_file() and _valid(_load_npz(query_path(dataset, c)), ksha) for c in range(n_chunks))
    sc = scene_list(dataset)
    sd = sum(scene_path(dataset, s[0]).is_file() and _valid(_load_npz(scene_path(dataset, s[0])), ksha) for s in sc)
    say(f"запросы: кусков {qd}/{n_chunks}; снимки: {sd}/{len(sc)}; запись: "
        f"{'есть' if record_path(dataset).is_file() else 'нет'} ({record_path(dataset)})")


def compute(dataset: str, limit: int | None, say, queries_only: bool = False) -> None:
    """Запросы требуют пройденной сверки со штатным; детекции по снимкам — ещё и сверки точности, которая сама
    читает эмбеддинги запросов fp16 из этих рабочих файлов."""
    import torch

    pre = preconditions(required_checks(queries_only))
    if _STAMP["code_dirty"]:
        raise SystemExit("дерево src/ или scripts/ не чистое — рабочие файлы OWLv2 считаются только закоммиченным кодом")
    rows, prot, labels, gpath = O.refs(dataset)
    rev = json.loads(env.ENV_JSON.read_text())["models"][O.MODEL_KEY]["revision"]
    model, processor, size = O.load_model(O.DTYPE, rev)
    ksha = check_key(dataset, make_key(dataset, rows, gpath, size))
    say(f"вход {size}×{size} (из конфигурации процессора и модели); сверки: {pre}; ключ {ksha[:10]}")
    run_queries(dataset, model, rows, ksha, rev, say)
    q = load_queries(dataset, rows, ksha)
    say(f"эталонов без эмбеддинга запроса: {int((~q['has']).sum())}")
    if queries_only:
        status(dataset, say)
        return
    run_scenes(dataset, model, rows, prot, labels, q, ksha, rev, limit, say)
    say(f"пик памяти GPU {torch.cuda.max_memory_allocated() // 2 ** 20} МиБ")
    status(dataset, say)


# ---------------------------------------------------------------------- оценка и запись (без GPU)


PAIRED_SUBSETS = {"control": RU.RHO_DIFF_SUBSETS, "rho": ("test/all",)}   # контрольные прогоны — 160 и 120 сцен, с ρ — 120
PAIRED_READING = "различие есть, если 95 % интервал парной разности не содержит нуля; других порогов нет"


def _point(ev: dict, subset: str) -> dict:
    return ev["subsets"][subset]["by_area"]["all"]


def paired_vs_method(owl: dict, method: dict, subsets: tuple[str, ...]) -> dict:
    """Парная разность «OWLv2 минус строка метода» по общим повторам бутстрэпа: `owl` и `method` — оценки
    `evaluate_baseline` одного протокола галереи с повторами (`boot`) на `subsets`."""
    return {n: {m: RU.paired_diff(owl["boot"][n][m], method["boot"][n][m], _point(owl, n)[m], _point(method, n)[m])
                for m in RU.RHO_DIFF_METRICS} for n in subsets}


def method_boots(say) -> tuple[dict, dict]:
    """Повторы бутстрэпа строк метода из рабочих файлов 4.4 (`cache/exp_4_4/<run_id>/`: блоки `baseline_check` —
    контрольный прогон без отбора, `rho` — метод с ρ): посчитаны тем же `evaluate_baseline` с тем же seed, числом
    повторов, `maxDets` и списком сцен. Точечные AP, AP50, AP75 сверяются с записями 4.4 точно; расхождение — отказ."""
    import pickle

    boots, sources = {}, {}
    for run in RU.RHO_RUNS:
        cfg = CFG.load(Path(f"configs/{run}.yaml"))
        pair = CFG.load(PAIR_CONFIG["hr_insdet"])
        if (cfg.seed, cfg.bootstrap, cfg.max_dets) != (pair.seed, pair.bootstrap, pair.max_dets):
            raise SystemExit(f"{run}: seed, bootstrap или max_dets расходятся с конфигурацией пары — повторы не общие")
        rec = json.loads(Path(f"experiments/runs/4_4_{run}.json").read_text())
        if (rec["seed"], rec["config"]["bootstrap"], rec["config"]["max_dets"]) != (pair.seed, pair.bootstrap, pair.max_dets):
            raise SystemExit(f"4_4_{run}.json: seed, bootstrap или max_dets записи 4.4 расходятся с конфигурацией пары")
        for block, key in (("baseline_check", "control"), ("rho", "rho")):
            path = Path(f"cache/exp_4_4/{run}/{block}.pkl")
            raw = path.read_bytes()
            pk = pickle.loads(raw)
            if pk["code_dirty"]:
                raise SystemExit(f"{path}: посчитан при нечистом дереве")
            for proto in O.PROTOCOLS:
                for n in PAIRED_SUBSETS[key]:
                    for m in RU.RHO_DIFF_METRICS:
                        if _point(pk["value"][proto], n)[m] != _point(rec[block]["metrics"][proto], n)[m]:
                            raise SystemExit(f"{path}: {proto} {n} {m} расходится с записью 4.4")
                    if len(pk["value"][proto]["boot"][n]["ap"]) != pair.bootstrap:
                        raise SystemExit(f"{path}: число повторов не {pair.bootstrap}")
            boots[(run, key)] = pk["value"]
            sources[f"{run}:{block}"] = {"file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                         "code_commit": pk["code_commit"], "record": f"experiments/runs/4_4_{run}.json"}
    say(f"повторы бутстрэпа метода: {len(boots)} блоков 4.4, точечные значения совпали с записями")
    return boots, sources


class _NoMasks:
    """OWLv2 масок не читает: сцены загружаются без рамок масок (оракул и дистракторы пусты и не используются)."""

    def load(self, _):
        return SimpleNamespace(boxes=np.zeros((0, 4)))


DIFFERENCES = [
    "не штатный вызов image_guided_detection с его постобработкой, а прямой вызов тех же головок (логиты равны штатным — "
    "env.json, owlv2_direct_vs_stock): штатная постобработка перенормирует оценки внутри запроса",
    "одна метка на рамку: оценка по метке — max sigmoid(logit) по эталонам метки, метка — argmax",
    "штатный порог NMS 0,3 — по всем рамкам снимка после назначения метки, а не внутри одного запроса",
    "эталон без эмбеддинга запроса пропускается; в протоколе one_per_class его метка остаётся без эталона",
    "вход 1008 (вся сцена приводится к нему) против 2048 у сегментатора и вырезок из полного кадра у метода",
    "эталон — квадратная вырезка по рамке промпта без маски; у меток PART_MASK_LABELS OWLv2 получает объект целиком",
    "рамка детекции предсказывается моделью, а не описывает маску; на PCB OWLv2 перенимает рамку эталона",
    "оценка — сигмоида логита, а не косинус: строки AR с порогом 0,4 не приводятся",
    "NMS после сопоставления; модель обучена с текстом (таблица 1.2)",
    "рамки обрезаются по кадру (в штатном пути не обрезаются); рамки, вырожденные после обрезки (целиком в сером "
    "паддинге), остаются ложными детекциями — их число: n_detections_zero_area_after_clip",
    "кандидатов на снимок у OWLv2 — до 5 184 рамок до NMS, у метода — маски M(I) (в среднем около 166 на сцену "
    "HR-InsDet); порога оценки нет ни у кого, maxDets = 1000 — число детекций на снимок: detections_per_image",
]


def evaluate(dataset: str, say) -> Path:
    out = record_path(dataset)
    if out.exists():
        raise SystemExit(f"запись уже есть и не перезаписывается: {out}")
    if _STAMP["code_dirty"]:
        raise SystemExit("запись OWLv2 пишется только при чистом дереве src/ и scripts/")
    pre = preconditions(required_checks())
    key = json.loads((WORK / dataset / "key.json").read_text())
    ksha = key_sha(key)
    rows, prot, labels, gpath = O.refs(dataset)
    if key != {**key, "refs_sha256": O.refs_sha256(rows), "gallery": gpath, "constants": O.constants()}:
        raise SystemExit("ключ рабочих файлов расходится с текущими константами либо галереей")
    q = load_queries(dataset, rows, ksha)
    cfg = CFG.load(PAIR_CONFIG[dataset])
    scenes, cat_names, edges, sp = PR.load_scenes(dataset, _NoMasks())
    if cat_names != labels:
        raise SystemExit("метки галереи расходятся с категориями истины")
    files = {}
    for s in scenes:
        p = scene_path(dataset, s.id)
        if not p.is_file():
            raise SystemExit(f"не посчитан снимок {s.id}: сначала python scripts/run_owlv2.py --dataset {dataset}")
        f = _load_npz(p)
        if not _valid(f, ksha) or bool(f["code_dirty"]):
            raise SystemExit(f"{p}: другой ключ либо посчитан при нечистом дереве")
        if tuple(int(v) for v in f["hw"]) != (s.wh[1], s.wh[0]):
            raise SystemExit(f"{s.id}: размер снимка {f['hw']} расходится с истиной {s.wh}")
        files[s.id] = f
    if len(files) != len(scene_list(dataset)):
        raise SystemExit("число снимков оценки расходится со списком счёта")
    t0 = time.perf_counter()
    is_pcb = dataset == "pcb"
    gt = PR.coco_truth(scenes, labels)  # у PCB — с зонами игнорирования, как у метода
    dets = {p: [O.detections(files[s.id], p) for s in scenes] for p in O.PROTOCOLS}
    metrics, no_zones = {}, {}
    head = "test/all" if is_pcb else "all/all"
    for p in O.PROTOCOLS:
        metrics[p] = PR.evaluate_baseline(scenes, labels, edges, gt, labels, None, None, cfg.max_dets, cfg.bootstrap, cfg.seed,
                                          names=PR.baseline_subsets(dataset), levels_=PR.levels(dataset), by_type=is_pcb,
                                          dets=dets[p], ar_thresholds=(None,),
                                          boot_subsets=() if is_pcb else RU.RHO_DIFF_SUBSETS)
        a = metrics[p]["subsets"][head]["by_area"]["all"]
        say(f"галерея «{p}»: AP {100 * a['ap']:.2f} AP50 {100 * a['ap50']:.2f} AP75 {100 * a['ap75']:.2f} ({head})")
        if is_pcb:
            for v in metrics[p]["subsets"].values():  # оракульных масок у OWLv2 нет — поле метода не переносится
                for t in v.get("by_type", {}).values():
                    t.pop("n_oracle", None)
            m = PR.evaluate_baseline(scenes, labels, edges, PR.coco_truth(scenes, labels, zones=False), labels, None, None,
                                     cfg.max_dets, cfg.bootstrap, cfg.seed, names=("test/all",), levels_=PR.levels(dataset),
                                     by_type=True, with_ar=False, dets=dets[p], ar_thresholds=(None,))["subsets"]["test/all"]
            b = m["by_area"]["all"]
            no_zones[p] = {"ap": b["ap"], "ap50": b["ap50"], "ap75": b["ap75"], "ap_with_zones": a["ap"],
                           "diff_ap_points": round(100 * (b["ap"] - a["ap"]), 4),
                           "by_type": {t: {k: v[k] for k in ("ap", "ap50", "ap75")} for t, v in m["by_type"].items()}}
    paired = None
    if not is_pcb:
        boots, sources = method_boots(say)
        paired = {"what": "OWLv2 минус строка пары 4.6 по общим повторам бутстрэпа по сценам: "
                          "контрольные прогоны лучших φ — на 160 и 120 сценах, метод с ρ — на 120 тестовых",
                  "reading": PAIRED_READING, "sources": sources, "metrics": list(RU.RHO_DIFF_METRICS)}
        for run in RU.RHO_RUNS:
            for row in ("control", "rho"):
                paired[f"minus_{row}__{run}"] = {p: paired_vs_method(metrics[p], boots[(run, row)][p], PAIRED_SUBSETS[row])
                                                for p in O.PROTOCOLS}
        d = paired[f"minus_rho__{RU.RHO_RUNS[0]}"]["full"]["test/all"]["ap"]
        say(f"парная разность AP, test/all, полная галерея: OWLv2 минус метод с ρ ({RU.RHO_RUNS[0]}) {100 * d['diff']:+.2f} п.")
        for p in O.PROTOCOLS:
            metrics[p].pop("boot")
    per_image = {p: [len(d["score"]) for d in dets[p]] for p in O.PROTOCOLS}
    zero_area = {p: int(sum(((d["box"][:, 2] <= d["box"][:, 0]) | (d["box"][:, 3] <= d["box"][:, 1])).sum() for d in dets[p]))
                 for p in O.PROTOCOLS}
    has = q["has"]
    opc_labels_without = sorted({rows[i]["label"] for i in prot["one_per_class"] if not has[i]})
    timing = {"queries": {"sec_prep_cpu": round(float(sum(f["sec_prep"] for f in q["files"])), 1),
                          "sec_tower": round(float(sum(f["sec_tower"] for f in q["files"])), 1), "n": len(rows)},
              "scenes": {k: round(float(sum(f[k] for f in files.values())), 1)
                         for k in ("sec_prep", "sec_tower", "sec_heads", "sec_nms")} | {"n": len(files)},
              "evaluate_sec": round(time.perf_counter() - t0, 1),
              "note": "башня, головки — GPU с синхронизацией; предобработка — CPU в параллельных процессах (сумма по "
                      "процессам, не стена); NMS и агрегация — CPU"}
    modules = sorted(set().union(*(json.loads(str(f["modules"])) for f in [*q["files"], *files.values()])))
    rec = {
        "run_id": f"owlv2_{dataset}", "kind": "comparison_run", "method": "owlv2_image_guided", "experiment": "4.6",
        "dataset": dataset, "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"], "started_utc": _STAMP["written_utc"],
        "detection_code_commits": sorted({str(f["code_commit"]) for f in [*q["files"], *files.values()]}),
        "what": "OWLv2 image-guided без текстовой башни: запрос — квадратная вырезка эталона по prompt_box, "
                "эмбеддинг запроса — штатный embed_image_query; по снимку — один проход башни, логиты class_predictor по "
                "всем запросам, рамки box_predictor; max sigmoid по эталонам метки, одна метка на рамку, NMS без учёта "
                "меток; без порога оценки; не прогон сетки",
        "model": {"id": key["model_id"], "revision": key["revision"], "dtype": O.DTYPE, "attn_implementation": O.ATTN,
                  "transformers": key["transformers"]},
        "input_size": key["input_size"], "input_size_source": "Owlv2ImageProcessor.size чекпойнта = vision_config.image_size",
        "constants": O.constants(), "work_key": key, "checks": pre,
        "modules_called": modules, "text_modules": list(O.TEXT_MODULES),
        "text_modules_called": [m for m in modules if m.startswith(O.TEXT_MODULES)],
        "gallery": {"source": gpath, "N": {p: int(len(prot[p])) for p in O.PROTOCOLS}, "n_labels": len(labels),
                    "refs_sha256": key["refs_sha256"]},
        "n_refs_without_query_embed": {p: int((~has[prot[p]]).sum()) for p in O.PROTOCOLS},
        "labels_without_ref_one_per_class": opc_labels_without,
        "pair_config": {"file": str(PAIR_CONFIG[dataset]), "seed": cfg.seed, "bootstrap": cfg.bootstrap,
                        "max_dets": cfg.max_dets},
        "splits": {"file": f"splits/{dataset}.json", "written": sp["written"], "seed": sp["seed"],
                   "n_cal": len(sp["cal"]), "n_test": len(sp["test"]), "n_evaluated": len(scenes)},
        "rules": {"area_edges_px2": edges[:2], "ar": "без учёта и с учётом категорий, только без порога"},
        "n_detections_zero_area_after_clip": zero_area,
        "detections_per_image": {p: {"mean": float(np.mean(v)), "max": int(np.max(v)), "min": int(np.min(v))}
                                 for p, v in per_image.items()},
        "scene_order_sha256": hashlib.sha256("\n".join(s.id for s in scenes).encode()).hexdigest(),
        "differences_for_4_1_4_6": DIFFERENCES,
        "timing": timing,
        "metrics_note": ("доли (0–1); " + ("test — 361 снимок тестовых плат, разбивка по платам и по типам дефекта; зоны "
                                           "игнорирования — в истине с iscrowd=1 в каждой категории; интервалов нет"
                                           if is_pcb else "all — 160 сцен (набор опубликованных чисел), test — 120 тестовых; "
                                           f"интервалы 95 % — бутстрэп по сценам, {cfg.bootstrap} повторов, те же, что у "
                                           "метода") + f"; AP — при maxDets={cfg.max_dets}, справочно — при 100"),
        "metrics": metrics,
    }
    if paired is not None:
        rec["paired_diff"] = paired
    if is_pcb:
        rec["ap_without_zones"] = {"what": "справочно: истина без зон игнорирования; подмножество test/all", **no_zones}
    else:
        rec["ap_without_part_mask_labels"] = {
            "labels": list(RU.PART_MASK_LABELS),
            **{p: {n: v.pop("ap_without_part_mask_labels") for n, v in metrics[p]["subsets"].items()} for p in metrics}}
    if rec["text_modules_called"]:
        raise SystemExit(f"в рабочих файлах отмечены вызовы текстовой башни: {rec['text_modules_called']}")
    RUNS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(out)
    say(f"ГОТОВО: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(O.GALLERY))
    ap.add_argument("--limit", type=int, default=None, help="не больше стольких новых снимков за запуск")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true")
    g.add_argument("--evaluate", action="store_true")
    g.add_argument("--queries-only", action="store_true", help="только эмбеддинги запросов (до сверки точности)")
    a = ap.parse_args()
    say = _logger(a.dataset)
    if a.status:
        status(a.dataset, say)
    elif a.evaluate:
        evaluate(a.dataset, say)
    else:
        compute(a.dataset, a.limit, say, a.queries_only)


if __name__ == "__main__":
    main()
