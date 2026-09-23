"""Сверки OWLv2 до прогонов → `experiments/env.json`.

    python scripts/check_owlv2.py precision   # fp16 против fp32 → owlv2_precision (GPU, минуты)
    python scripts/check_owlv2.py stock       # прямой вызов головок против штатного image_guided_detection, обе области → owlv2_direct_vs_stock

Только калибровочные снимки сверок (`splits/check_samples.json`): сверка точности — 10 сцен HR-InsDet и по одному эталону
каждой из 100 меток (строки `one_per_class` галереи, из которой OWLv2 берёт запросы); сверка со штатным — ещё и 10
снимков калибровочных плат PCB с шестью запросами `one_per_class`, при форме вызова счёта. Пороги — константы
`src.baselines.owlv2`, записаны до сверки. OWLv2 считается в fp32 (сверка fp16 с fp32 не пройдена): этап
`precision` отказывает — непройденная сверка fp16 остаётся в `env.json` как основание решения; `run_owlv2.py`
проверяет только сверку со штатным. Каждый этап заменяет только свой раздел `env.json`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import env  # noqa: E402
from src.baselines import owlv2 as O  # noqa: E402
from src.data import hr_insdet as H  # noqa: E402
from src.segment.sam2 import read_rgb  # noqa: E402

_STAMP = env.code_stamp()


def _write_env(section: str, value: dict) -> None:
    data = json.loads(env.ENV_JSON.read_text())
    data[section] = {"written": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                     "code_commit": _STAMP["code_commit"], "code_dirty": _STAMP["code_dirty"],
                     "started_utc": _STAMP["written_utc"], "owlv2_py_sha256": O.fingerprint(), **value}
    tmp = env.ENV_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    tmp.replace(env.ENV_JSON)
    print(f"env.json: {section}", flush=True)


def revision() -> str:
    return json.loads(env.ENV_JSON.read_text())["models"][O.MODEL_KEY]["revision"]


def inputs(processor):
    """Входы сверок: 10 калибровочных сцен сверки и 100 запросов `one_per_class` HR-InsDet (предобработка на CPU —
    один раз для обеих точностей)."""
    pilot = json.loads(Path("splits/check_samples.json").read_text())["hr_scenes"]
    sp = json.loads(Path("splits/hr_insdet.json").read_text())
    if not set(pilot) <= set(sp["cal"]) or len(pilot) != 10:
        raise SystemExit("сцены сверки — не 10 калибровочных сцен")
    rows, prot, labels, gpath = O.refs("hr_insdet")
    opc = [rows[i] for i in prot["one_per_class"]]
    if [r["label"] for r in opc] != labels:
        raise SystemExit("one_per_class — не по эталону на каждую метку в порядке меток")
    t0 = time.perf_counter()
    scenes = {s: O.preprocess(processor, read_rgb(H.ROOT / "Scenes" / f"{s}.jpg")) for s in pilot}
    queries = [O.preprocess(processor, O.query_image(read_rgb(r["path"]), r["prompt_box"])) for r in opc]
    print(f"предобработка: {time.perf_counter() - t0:.0f} с", flush=True)
    return pilot, scenes, opc, queries, gpath


def _load(dtype: str):
    import torch

    torch.cuda.reset_peak_memory_stats()
    return O.load_model(dtype, revision())


def _run_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_owlv2", Path(__file__).resolve().parent / "run_owlv2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_owlv2"] = mod
    spec.loader.exec_module(mod)
    return mod


def _embed(model, queries: list, dtype) -> np.ndarray:
    """Эмбеддинги запросов в точности модели → float32 (P, D); без эмбеддинга — отказ (в сверке не пропускается)."""
    import torch

    out = []
    for q in queries:
        e, _ = O.query_embed(model, torch.from_numpy(q)[None].to("cuda", dtype))
        if e is None:
            raise SystemExit("эталон сверки без эмбеддинга запроса — разобрать до сверки")
        out.append(e.float().cpu().numpy())
    return np.stack(out)


def _scene_logits(model, px: np.ndarray, Q: np.ndarray, dtype) -> np.ndarray:
    """Логиты (P, N) float32 снимка по запросам `Q` (N, D) в точности модели — тот же путь, что в счёте."""
    import torch

    fm, feats = O.image_features(model, torch.from_numpy(px)[None].to("cuda", dtype))
    if not torch.isfinite(fm).all():
        raise FloatingPointError(f"не-конечный выход башни в {dtype}")
    logits, _ = O.scene_heads(model, fm, feats, torch.from_numpy(Q).to("cuda", dtype))
    return logits.float().cpu().numpy()


def _pcb_inputs(processor):
    samples = json.loads(Path("splits/check_samples.json").read_text())["pcb_images"]
    sp = json.loads(Path("splits/pcb.json").read_text())
    if not set(samples) <= set(sp["cal"]) or len(samples) != 10:
        raise SystemExit("снимки сверки PCB — не 10 снимков калибровочных плат")
    from src.data import pcb as P

    img = {i["id"]: i["image"] for i in P.images()}
    return samples, {i: O.preprocess(processor, read_rgb(P.ROOT / img[i])) for i in samples}


def _fp32_queries_pcb(model, rows: list[dict], processor, ksha: str) -> np.ndarray:
    """Эмбеддинги всех запросов PCB в fp32 — кусками в `cache/owlv2/pcb/queries_fp32/` (прерываемо)."""
    import torch

    R = _run_script()
    d = R.WORK / "pcb" / "queries_fp32"
    out = []
    for c in range((len(rows) + R.QUERY_CHUNK - 1) // R.QUERY_CHUNK):
        f = d / f"chunk_{c:04d}.npz"
        part = rows[c * R.QUERY_CHUNK:(c + 1) * R.QUERY_CHUNK]
        if f.is_file():
            z = R._load_npz(f)
            if str(z["key_sha1"]) == ksha and z["ids"].tolist() == [r["id"] for r in part]:
                out.append(z["emb"])
                continue
        t0 = time.perf_counter()
        qs = [O.preprocess(processor, O.query_image(read_rgb(r["path"]), r["prompt_box"])) for r in part]
        with torch.inference_mode():
            emb = _embed(model, qs, torch.float32)
        R._save_npz(f, ids=np.array([r["id"] for r in part]), emb=emb, key_sha1=np.array(ksha))
        out.append(emb)
        print(f"fp32, запросы PCB: кусок {c + 1}, {time.perf_counter() - t0:.0f} с", flush=True)
    return np.concatenate(out)


def _changed(l16: np.ndarray, l32: np.ndarray) -> dict:
    """Метки рамок по логитам меток (P, L) в двух точностях: сменившиеся среди 100 лучших рамок по fp32."""
    lab16, _ = O.assign(l16)
    lab32, s32 = O.assign(l32)
    top = np.argsort(-s32, kind="stable")[:O.PRECISION_TOP]
    return {"changed_top": int((lab16[top] != lab32[top]).sum()), "n_top": int(len(top))}


PRECISION_WORK = Path("cache/owlv2/precision")


def _precision_inputs():
    R = _run_script()
    hr_rows, hr_prot, hr_labels, hr_g = O.refs("hr_insdet")
    pcb_rows, pcb_prot, pcb_labels, pcb_g = O.refs("pcb")
    hr_key, pcb_key = [R.key_sha(json.loads((R.WORK / ds / "key.json").read_text())) if (R.WORK / ds / "key.json").is_file()
                       else None for ds in ("hr_insdet", "pcb")]
    if hr_key is None or pcb_key is None:
        raise SystemExit("нет эмбеддингов запросов fp16: сначала run_owlv2.py --dataset <dataset> --queries-only")
    hr_q16, pcb_q16 = R.load_queries("hr_insdet", hr_rows, hr_key), R.load_queries("pcb", pcb_rows, pcb_key)
    if not (hr_q16["has"].all() and pcb_q16["has"].all()):
        raise SystemExit("в рабочих файлах есть эталоны без эмбеддинга запроса — разобрать до сверки")
    return {"R": R, "hr": (hr_rows, hr_prot, hr_labels, hr_g, hr_key, hr_q16), "pcb": (pcb_rows, pcb_prot, pcb_labels, pcb_g, pcb_key, pcb_q16),
            "hr_cols": O.columns(hr_rows, hr_prot, hr_labels, hr_q16["has"]),
            "pcb_cols": O.columns(pcb_rows, pcb_prot, pcb_labels, pcb_q16["has"])}


def _part_key(inp: dict, dtype: str) -> str:
    import hashlib

    src = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashlib.sha1(json.dumps([dtype, O.fingerprint(), src, revision(), inp["hr"][4], inp["pcb"][4]]).encode()).hexdigest()


def stage_precision_part(dtype: str) -> None:
    """Проход одной точности — в своём процессе: две модели в одном процессе с CUDA дважды обрывались сбоем драйвера
    («device not ready») в начале прохода fp32, отдельный процесс fp32 проходит. Итог — `cache/owlv2/precision/`."""
    import torch

    if O.DTYPE == "float32":
        raise SystemExit("модель в fp32: сверка fp16 с fp32 не нужна")
    inp = _precision_inputs()
    R = inp["R"]
    hr_rows, hr_prot, hr_labels, hr_g, hr_key, hr_q16 = inp["hr"]
    pcb_rows, pcb_prot, pcb_labels, pcb_g, pcb_key, pcb_q16 = inp["pcb"]
    model, processor, size = _load(dtype)
    dt = getattr(torch, dtype)
    pilot, scenes, opc, queries, gpath = inputs(processor)
    pcb_ids, pcb_scenes = _pcb_inputs(processor)
    t0 = time.perf_counter()
    called: set = set()
    arrays = {}
    with torch.inference_mode(), O.trace_modules(model, called):
        q_opc = _embed(model, queries, dt)                                       # (а): запросы в своей точности
        pcb_q = pcb_q16["emb"] if dtype == "float16" else _fp32_queries_pcb(model, pcb_rows, processor, pcb_key)
        for k, s in enumerate(pilot):
            arrays[f"a_{k}"] = _scene_logits(model, scenes[s], q_opc, dt)
            if dtype == "float16":                                               # (в): полная галерея, запросы счёта
                lg = torch.from_numpy(_scene_logits(model, scenes[s], hr_q16["emb"], dt))
                arrays[f"c_{k}"] = O.protocol_logits(lg, inp["hr_cols"], len(hr_labels))["full"]
        for k, s in enumerate(pcb_ids):                                          # (б)
            lg = torch.from_numpy(_scene_logits(model, pcb_scenes[s], pcb_q, dt))
            arrays[f"b_{k}"] = O.protocol_logits(lg, inp["pcb_cols"], len(pcb_labels))["full"]
    info = {"sec": round(time.perf_counter() - t0, 1), "vram_peak_mib": torch.cuda.max_memory_allocated() // 2 ** 20,
            "modules_called": sorted(called)}
    meta = {"key": _part_key(inp, dtype), "pilot": pilot, "pcb_ids": pcb_ids, "opc": [r["id"] for r in opc],
            "gallery": gpath, "input_size": size, "info": info}
    R._save_npz(PRECISION_WORK / f"{dtype}.npz", meta=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    print(f"{dtype}: {info}", flush=True)


def stage_precision() -> None:
    """Три части сверки: (а) HR-InsDet, один эталон на метку; (б) PCB, полная галерея;
    (в) полная галерея HR-InsDet — оценка сверху по разрыву между метками. Каждая точность — отдельным процессом
    (`precision-part`); посчитанная той же версией проход не повторяется."""
    import subprocess

    if O.DTYPE == "float32":
        raise SystemExit("модель в fp32: сверка fp16 с fp32 не нужна, запись owlv2_precision не перезаписывается")
    inp = _precision_inputs()
    R = inp["R"]
    hr_rows, hr_prot, hr_labels, hr_g, hr_key, hr_q16 = inp["hr"]
    pcb_rows, pcb_prot, pcb_labels, pcb_g, pcb_key, pcb_q16 = inp["pcb"]
    res, info = {}, {}
    for dtype in ("float16", "float32"):
        f = PRECISION_WORK / f"{dtype}.npz"
        z = R._load_npz(f) if f.is_file() else None
        if z is None or json.loads(str(z["meta"]))["key"] != _part_key(inp, dtype):
            subprocess.run([sys.executable, "-W", "ignore::UserWarning", __file__, "precision-part", dtype], check=True)
            z = R._load_npz(f)
        meta = json.loads(str(z["meta"]))
        if meta["key"] != _part_key(inp, dtype):
            raise SystemExit(f"{f}: ключ прохода расходится")
        res[dtype], info[dtype] = z, meta["info"]
    pilot, pcb_ids, size = meta["pilot"], meta["pcb_ids"], meta["input_size"]
    opc, gpath = meta["opc"], meta["gallery"]
    h = {**{("a", s): res["float16"][f"a_{k}"] for k, s in enumerate(pilot)},
         **{("c", s): res["float16"][f"c_{k}"] for k, s in enumerate(pilot)},
         **{("b", s): res["float16"][f"b_{k}"] for k, s in enumerate(pcb_ids)}}
    f = {**{("a", s): res["float32"][f"a_{k}"] for k, s in enumerate(pilot)},
         **{("b", s): res["float32"][f"b_{k}"] for k, s in enumerate(pcb_ids)}}
    part_a, deltas = [], []
    for s in pilot:
        lh, lf = h[("a", s)], f[("a", s)]
        top = np.argsort(-lf.max(1), kind="stable")[:O.PRECISION_TOP]
        delta = float(np.abs(lh[top] - lf[top]).max())
        deltas.append(delta)
        part_a.append({"scene": s, **_changed(lh.astype(np.float64), lf.astype(np.float64)), "max_abs_diff_logit_top": delta,
                       "max_abs_diff_logit": float(np.abs(lh - lf).max()),
                       "max_abs_diff_sigmoid": float(np.abs(O.sigmoid(lh) - O.sigmoid(lf)).max())})
    part_b = [{"image": s, **_changed(h[("b", s)], f[("b", s)]),
               "max_abs_diff_label_logit": float(np.abs(h[("b", s)] - f[("b", s)]).max())} for s in pcb_ids]
    delta = max(deltas)
    part_c = []
    for s in pilot:
        lm = h[("c", s)]
        _, sc = O.assign(lm)
        top = np.argsort(-sc, kind="stable")[:O.PRECISION_TOP]
        m = O.label_margin(lm)[top]
        part_c.append({"scene": s, "n_top_margin_le_bound": int((m <= O.MARGIN_FACTOR * delta).sum()),
                       "min_margin_top": float(m.min()), "n_top": int(len(top))})
    ok_a = all(r["changed_top"] <= O.PRECISION_MAX_CHANGED_PER_IMAGE for r in part_a)
    ok_b = all(r["changed_top"] <= O.PRECISION_MAX_CHANGED_PER_IMAGE for r in part_b)
    ok_c = all(r["n_top_margin_le_bound"] <= O.PRECISION_MAX_CHANGED_PER_IMAGE for r in part_c)
    passed = ok_a and ok_b and ok_c
    _write_env("owlv2_precision", {
        "what": "OWLv2 fp16 против fp32: (а) HR-InsDet, 10 калибровочных сцен "
                "сверки, по одному эталону 100 меток — запросы и логиты в обеих точностях; (б) PCB, 10 снимков "
                "калибровочных плат сверки, все эталоны — запросы fp16 из рабочих файлов счёта, fp32 — отдельно; (в) полная "
                "галерея HR-InsDet, fp16 по запросам счёта — оценка сверху по разрыву между метками",
        "input_size": size, "input_size_source": "Owlv2ImageProcessor.size чекпойнта = vision_config.image_size",
        "model": {"id": env.MODEL_IDS[O.MODEL_KEY], "revision": revision(), "attn_implementation": O.ATTN},
        "rule": f"(а), (б): на каждом снимке не больше {O.PRECISION_MAX_CHANGED_PER_IMAGE} рамок со сменившейся меткой "
                f"среди {O.PRECISION_TOP} лучших по fp32; (в): на каждой сцене не больше "
                f"{O.PRECISION_MAX_CHANGED_PER_IMAGE} рамок из {O.PRECISION_TOP} лучших по fp16 с разрывом меток ≤ "
                f"{O.MARGIN_FACTOR:g}·δ, δ — наибольшее |Δ логита| (а) по её лучшим рамкам; записано до сверки",
        "delta": delta, "bound": O.MARGIN_FACTOR * delta,
        "part_a_hr_one_per_class": {"passed": ok_a, "gallery": hr_g, "n_queries": len(opc), "images": part_a},
        "part_b_pcb_full": {"passed": ok_b, "gallery": pcb_g, "n_queries": int(len(pcb_q16["emb"])), "images": part_b},
        "part_c_hr_full_bound": {"passed": ok_c, "gallery": hr_g, "n_queries": int(len(hr_q16["emb"])), "images": part_c},
        "queries_fp16_keys": {"hr_insdet": hr_key, "pcb": pcb_key},
        "passed": bool(passed), "by_dtype": info})
    print(f"owlv2_precision: (а) {'да' if ok_a else 'НЕТ'}, (б) {'да' if ok_b else 'НЕТ'}, (в) {'да' if ok_c else 'НЕТ'}; "
          f"δ = {delta:.4g} — {'выдержан' if passed else 'НЕ ВЫДЕРЖАН: детекции не считаются'}", flush=True)


def _stock_inputs(dataset: str, processor):
    """Калибровочные снимки сверки области и запросы `one_per_class`; строки полной галереи — для формы вызова."""
    samples = json.loads(Path("splits/check_samples.json").read_text())
    sp = json.loads(Path(f"splits/{dataset}.json").read_text())
    if dataset == "hr_insdet":
        ids, paths = samples["hr_scenes"], [H.ROOT / "Scenes" / f"{s}.jpg" for s in samples["hr_scenes"]]
    else:
        from src.data import pcb as P

        img = {i["id"]: i["image"] for i in P.images()}
        ids = samples["pcb_images"]
        paths = [P.ROOT / img[i] for i in ids]
    if not set(ids) <= set(sp["cal"]):
        raise SystemExit(f"{dataset}: снимки сверки — не калибровочные")
    rows, prot, labels, gpath = O.refs(dataset)
    opc = [int(i) for i in prot["one_per_class"]]
    scenes = {i: O.preprocess(processor, read_rgb(pa)) for i, pa in zip(ids, paths)}
    queries = [O.preprocess(processor, O.query_image(read_rgb(rows[i]["path"]), rows[i]["prompt_box"])) for i in opc]
    return ids, scenes, rows, prot, labels, opc, queries, gpath


def stage_stock() -> None:
    import torch

    model, processor, size = _load(O.DTYPE)
    dt = getattr(torch, O.DTYPE)
    pairs, called, by_ds = [], set(), {}
    t0 = time.perf_counter()
    for dataset in ("hr_insdet", "pcb"):
        ids, scenes, rows, prot, labels, opc, queries, gpath = _stock_inputs(dataset, processor)
        with torch.inference_mode():
            with O.trace_modules(model, called):
                # прямой путь — как в run_owlv2.py: запрос считается один раз, хранится во float32, в модель идёт в O.DTYPE;
                # столбцы — как у счёта (`O.columns`), с числом строк полной галереи: запросы one_per_class повторены
                own = {}
                for i, q in zip(opc, queries):
                    e, _ = O.query_embed(model, torch.from_numpy(q)[None].to("cuda", dt))
                    own[i] = e.float().cpu().numpy()
                d = len(next(iter(own.values())))
                emb = np.stack([own[opc[r % len(opc)]] for r in range(len(rows))]).astype(np.float32)
                for i in opc:
                    emb[i] = own[i]
                cols = O.columns(rows, prot, labels, np.ones(len(rows), bool))
                Q = torch.from_numpy(np.ascontiguousarray(emb[cols["full_rows"]])).to("cuda", dt)
            for k, s in enumerate(ids):
                px = torch.from_numpy(scenes[s])[None].to("cuda", dt)
                with O.trace_modules(model, called):
                    fm, feats = O.image_features(model, px)
                    ours, boxes = O.scene_heads(model, fm, feats, Q)
                for i in range(O.STOCK_REFS_PER_SCENE):
                    j = opc[(10 * k + i) % len(opc)]
                    st = model.image_guided_detection(pixel_values=px,
                                                      query_pixel_values=torch.from_numpy(queries[opc.index(j)])[None].to("cuda", dt))
                    a, b = ours[:, cols["pos"][j]].float().cpu().numpy(), st.logits[0, :, 0].float().cpu().numpy()
                    dd = np.abs(a - b)
                    pairs.append({"dataset": dataset, "scene": s, "ref": rows[j]["id"], "max_abs_diff_logit": float(dd.max()),
                                  "n_boxes_over_tolerance": int((dd > O.STOCK_ATOL + O.STOCK_RTOL * np.abs(b)).sum()),
                                  "frac_exactly_equal": float((dd == 0).mean()),
                                  "boxes_equal": bool(torch.equal(boxes, st.target_pred_boxes[0])),
                                  "max_abs_logit": float(np.abs(b).max())})
        by_ds[dataset] = {"gallery": gpath, "n_columns": int(Q.shape[0]), "d": d, "scenes": ids}
    passed = all(p["n_boxes_over_tolerance"] == 0 and p["boxes_equal"] for p in pairs)
    _write_env("owlv2_direct_vs_stock", {
        "what": "логиты class_predictor по посчитанным один раз эмбеддингам запросов против штатного "
                "image_guided_detection для той же пары «сцена — эталон», до штатной постобработки; вызов прямого пути — "
                "той же формы, что в счёте (столбцов — сколько строк полной галереи; запросы one_per_class повторены), "
                f"столбец эталона — через src.baselines.owlv2.columns; калибровочные снимки сверки обеих областей × "
                f"{O.STOCK_REFS_PER_SCENE} эталонов one_per_class; рамки box_predictor",
        "input_size": size, "model": {"id": env.MODEL_IDS[O.MODEL_KEY], "revision": revision(), "dtype": O.DTYPE,
                                      "attn_implementation": O.ATTN},
        "tolerance": {"rule": "|Δ логита| ≤ atol + rtol·|штатный логит| для каждой рамки; рамки — точное равенство; "
                              "записано до сверки (константы src.baselines.owlv2)",
                      "atol": O.STOCK_ATOL, "rtol": O.STOCK_RTOL},
        "by_dataset": by_ds, "max_abs_diff_logit": max(p["max_abs_diff_logit"] for p in pairs),
        "n_pairs": len(pairs), "n_pairs_failed": sum(p["n_boxes_over_tolerance"] > 0 or not p["boxes_equal"] for p in pairs),
        "modules_called_direct": sorted(called), "passed": bool(passed), "sec": round(time.perf_counter() - t0, 1),
        "vram_peak_mib": torch.cuda.max_memory_allocated() // 2 ** 20, "pairs": pairs})
    print(f"owlv2_direct_vs_stock: {len(pairs)} пар, наибольшее |Δ| {max(p['max_abs_diff_logit'] for p in pairs):.3g} — "
          f"{'сошлось' if passed else 'РАСХОЖДЕНИЕ: ошибка, прогон не начинается'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["precision", "stock", "precision-part"])
    ap.add_argument("dtype", nargs="?", choices=["float16", "float32"])
    a = ap.parse_args()
    if a.stage == "precision-part":
        stage_precision_part(a.dtype)
    else:
        {"precision": stage_precision, "stock": stage_stock}[a.stage]()


if __name__ == "__main__":
    main()
