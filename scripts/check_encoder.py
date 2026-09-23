"""Сверки кодирования на калибровочных сценах → `experiments/env.json`.

Этапы (каждый заменяет только свой раздел `env.json`):

    python scripts/check_encoder.py u_r         # debias: U_r для DINOv3 проходом в fp32 → gallery/dinov3/p_perp/U_r.npy
    python scripts/check_encoder.py memory      # pool_memory: пик памяти P (у DINOv3 — вместе с P⊥), обе точности
    python scripts/check_encoder.py onepass     # pool_onepass: P и P⊥ из одного прохода против раздельного счёта
    python scripts/check_encoder.py blur        # blur_large_masks: размытие на уменьшенном окне против буквального
    python scripts/check_encoder.py precision   # pool_precision: z_pool в рабочей точности против fp32 по бинам площади
    python scripts/check_encoder.py crop_precision   # crop_precision: CLS вырезок в рабочей точности против fp32
    python scripts/check_encoder.py blur_refs   # blur_refs: досчёт сверки размытия на эталонах и масках среднего размера

Составы и пороги записаны до замера и здесь — константы;
по результату не пересматриваются. Сцены — 10 калибровочных сцен сверки (`splits/check_samples.json`,
`hr_scenes`); тестовые сцены не открываются — проверяется по `splits/hr_insdet.json`. Маски — из кеша масок
при выбранном `crop_n_layers`. Этап `blur` считает буквальное размытие на квадратах до 12 288 пикс. —
десятки минут CPU; промежуточные вырезки лежат в `cache/checks/encoder/`, повторный запуск продолжает с места.
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

from src import env
from src.data import hr_insdet
from src.encode import crop as C
from src.encode import debias as D
from src.encode import pool as PO
from src.eval.oracle import ORACLE_IOU, oracle_assign
from src.segment import cache as MC
from src.segment import sam2 as S

# --- записано до замера, по результату не пересматривается ---
BLUR_N_LARGEST = 8           # крупнейшие маски 10 сцен сверки — по стороне квадрата при α = 1,0 (большая сторона
                             # описывающей рамки); при равенстве — по порядку сцен сверки, затем по номеру маски
BLUR_COS_MIN = 0.998         # косинус CLS с буквальным путём у каждой из 16 вырезок, у обоих энкодеров
BLUR_WORK_SIDE_FIRST, BLUR_WORK_SIDE_RETRY = 896, 1792  # при провале рабочая сторона удваивается, сверка повторяется
POOL_COS_MEDIAN_MIN = 0.999  # медиана косинуса z_pool «рабочая точность — fp32» в каждом бине площади, у каждого
                             # энкодера; при провале режим не меняется, результат требует разбора
# Бины — по площади размеченной рамки w·h, как в 4.3; маски — оракульные.
# P⊥ в рабочей точности против fp32 пишется справочно: порог записан для z_pool.
# «Совпадают» для одного прохода против раздельного счёта — побитово (`np.array_equal`); иначе — числа без вердикта.
VRAM_TOTAL_MIB = 6144
# Сверка CLS вырезок:
# те же 10 сцен и оракульные маски, варианты C(0, 1,0) — чёрный фон даёт самые нетипичные активации — и
# C(blur, 1,5); батчи по 16 масок сцены в порядке номеров масок, одни и те же в обеих точностях; порог тот же,
# что у z_pool, — медиана косинуса в каждом бине площади не ниже 0,999 у каждого энкодера и варианта;
# при провале режим не меняется, результат требует разбора.
# Досчёт сверки размытия: 12 эталонов
# HR-InsDet со стороной квадрата при α = 1,0 больше рабочей (1 792 пикс.) и 6 масок 10 калибровочных сцен сверки со
# стороной квадрата при α = 1,0 от 2 300 до 5 066 пикс. (при α = 1,5 — до 7 600), выбор — `default_rng(seed разбиения)`
# из отсортированных списков; оба α. Порог тот же — `BLUR_COS_MIN` на каждой вырезке у каждого энкодера.
# Правило при провале: рабочая сторона НЕ меняется (удвоение до 3 584 удорожило бы прогоны с b = blur в разы),
# в 3.1 и 4.1 идут оговорка и числа; минимум ниже `BLUR_REFS_ESCALATE` требует разбора до первого прогона
# с b = blur: это уже не погрешность интерполяции.
BLUR_REFS_N, BLUR_REFS_CAL_N = 12, 6
BLUR_REFS_CAL_SIDE = (2300.0, 7600.0 / 1.5)
BLUR_REFS_ESCALATE = 0.99
CROP_PRECISION_VARIANTS = (("0", 1.0), ("blur", 1.5))
CROP_COS_MEDIAN_MIN = POOL_COS_MEDIAN_MIN

ENCODERS = ("encoder_dinov2", "encoder_dinov3")
U_R_PATH = Path("gallery/dinov3/p_perp/U_r.npy")
WORK = Path("cache/checks/encoder")
AREA_BINS = ("<200²", "200²–400²", ">400²")


_STAMP = env.code_stamp()  # версия кода — на момент запуска процесса, а не записи: скрипт могли править, пока он считал


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def _write_env(section: str, value: dict) -> None:
    data = json.loads(env.ENV_JSON.read_text())
    data[section] = {"written": _today(), **{**env.code_stamp(), "code_commit": _STAMP["code_commit"],
                                              "code_dirty": _STAMP["code_dirty"]}, **value}
    tmp = env.ENV_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")  # отступ — как у файла в репозитории
    tmp.replace(env.ENV_JSON)
    print(f"env.json: {section}", flush=True)


def _setup() -> dict:
    sp = json.loads(Path("splits/hr_insdet.json").read_text())
    pilot = json.loads(Path("splits/check_samples.json").read_text())
    scenes = pilot["hr_scenes"]
    if len(scenes) != 10 or not set(scenes) <= set(sp["cal"]):
        raise SystemExit("сцены сверки — 10 калибровочных сцен из splits/check_samples.json; тестовые сцены не открываются")
    cnl = json.loads(Path("experiments/runs/4_2_hr_insdet.json").read_text())["selection"]["selected_crop_n_layers"]
    by_id = {s["id"]: s for s in hr_insdet.scenes()}
    ref = next(r for r in sp["gallery"] if r["id"] == pilot["hr_refs"][0])
    return {"scenes": scenes, "labels": sp["labels"], "by_id": by_id, "cnl": cnl, "ref": ref,
            "cache": MC.MaskCache("hr_insdet", MC.auto_key(cnl)), "box_cache": MC.MaskCache("hr_insdet", MC.box_key())}


def _scene_img(scene_id: str) -> np.ndarray:
    return S.read_rgb(hr_insdet.ROOT / "Scenes" / f"{scene_id}.jpg")


# ---------------------------------------------------------------- memory


def stage_memory() -> None:
    """Пик памяти прохода P: сцена HR-InsDet и наибольший вход — эталон 2048², рабочая точность и fp32."""
    import torch

    from src.encode import model as M

    st = _setup()
    inputs = {"scene": (_scene_img(st["scenes"][0]), st["cache"].load(st["scenes"][0])),
              "reference": (S.read_rgb(hr_insdet.ROOT / st["ref"]["image"]),
                            st["box_cache"].load(st["ref"]["id"], [st["ref"]["box"]]))}
    out = {"what": "torch.cuda.max_memory_allocated за вызов pool_image (веса модели входят), МиБ",
           "vram_total_mib": VRAM_TOTAL_MIB, "scene": st["scenes"][0], "reference": st["ref"]["id"], "by_encoder": {}}
    for name in ENCODERS:
        rows = {}
        for prec, fp32 in (("working", False), ("fp32", True)):
            model = M.load(name, fp32=fp32)
            # у DINOv3 прогон считает P и P⊥ одним проходом — пик меряется вместе с проектором
            u_r = torch.from_numpy(D.load(U_R_PATH)).cuda() if name == "encoder_dinov3" else None
            for tag, (img, entry) in inputs.items():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                try:
                    r = PO.pool_image(model, img, entry, u_r)
                    torch.cuda.synchronize()
                    rows[f"{prec}_{tag}"] = {
                        "dtype": str(next(model.parameters()).dtype), "tokens": r["grid"][0] * r["grid"][1],
                        "n_masks": len(entry), "with_p_perp": u_r is not None,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() // 2**20,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20, "fits": True}
                except RuntimeError as e:  # нехватка памяти; под WSL — иногда ошибкой драйвера CUDA, а не OOM
                    rows[f"{prec}_{tag}"] = {"fits": False, "error": str(e)[:200]}
                print(name, prec, tag, rows[f"{prec}_{tag}"], flush=True)
            del model, u_r
            torch.cuda.empty_cache()
        out["by_encoder"][name] = rows
    out["working_fits_6gb"] = all(r["fits"] and r["peak_reserved_mib"] < VRAM_TOTAL_MIB
                                  for e in out["by_encoder"].values() for k, r in e.items() if k.startswith("working"))
    _write_env("pool_memory", out)


# ---------------------------------------------------------------- U_r


def stage_u_r() -> None:
    import torch

    from src.encode import model as M

    model = M.load("encoder_dinov3", fp32=True)
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    u_r, info = D.estimate_u_r(model)
    sec = time.perf_counter() - t
    again, _ = D.estimate_u_r(model)
    D.save(U_R_PATH, u_r)
    _write_env("debias", {"encoder": env.MODEL_IDS["encoder_dinov3"], "path": str(U_R_PATH), "shape": list(u_r.shape),
                          "sha256": D.sha256(u_r), "repeat_bitwise_identical": bool(np.array_equal(u_r, again)),
                          "sec": round(sec, 1), "peak_allocated_mib": torch.cuda.max_memory_allocated() // 2**20,
                          **info})


# ---------------------------------------------------------------- один проход против раздельного


def stage_onepass() -> None:
    import torch

    from src.encode import model as M

    st = _setup()
    model = M.load("encoder_dinov3")
    u_r = torch.from_numpy(D.load(U_R_PATH)).cuda()
    rows = []
    for sid in st["scenes"]:
        img, entry = _scene_img(sid), st["cache"].load(sid)
        both = PO.pool_image(model, img, entry, u_r)
        only_p = PO.pool_image(model, img, entry, None, outputs=("p",))
        only_perp = PO.pool_image(model, img, entry, u_r, outputs=("p_perp",))
        row = {"scene": sid, "n_masks": len(entry), "n_empty": int(both["empty"].sum())}
        for k, sep in (("p", only_p), ("p_perp", only_perp)):
            row[k] = {"bitwise_identical": bool(np.array_equal(both[k], sep[k])),
                      "max_abs_diff": float(np.abs(both[k] - sep[k]).max()),
                      "cos_min": float((both[k] * sep[k]).sum(1).min())}
        # насколько дебиасинг меняет эмбеддинг — справочно, к сверке не относится
        row["cos_p_vs_p_perp_median"] = float(np.median((both["p"] * both["p_perp"]).sum(1)))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
    _write_env("pool_onepass", {
        "what": "DINOv3, рабочая точность: P и P⊥ из одного прохода против двух раздельных проходов; все маски сцены",
        "criterion": "побитовое совпадение", "crop_n_layers": st["cnl"], "u_r_sha256": D.sha256(u_r.cpu().numpy()),
        "n_scenes": len(rows), "n_masks": sum(r["n_masks"] for r in rows),
        "all_bitwise_identical": all(r[k]["bitwise_identical"] for r in rows for k in ("p", "p_perp")), "rows": rows})


# ---------------------------------------------------------------- размытие на крупнейших масках


def _largest(st: dict) -> list[dict]:
    cand = []
    for si, sid in enumerate(st["scenes"]):
        e = st["cache"].load(sid)
        for i, r in enumerate(e.records):
            b = r["box"]
            cand.append((-max(b[2] - b[0], b[3] - b[1]), si, i, sid))
    return [{"scene": sid, "mask": i, "box_side": -s} for s, _, i, sid in sorted(cand)[:BLUR_N_LARGEST]]


def stage_blur(work_side: int) -> None:
    import torch

    from src.encode import model as M

    st = _setup()
    WORK.mkdir(parents=True, exist_ok=True)
    picks = _largest(st)
    pairs = []
    img_cache: dict = {}
    for k, pk in enumerate(picks):
        box = st["cache"].load(pk["scene"]).records[pk["mask"]]["box"]
        for a in C.ALPHAS:
            fe, ff = WORK / f"blur_exact_{k}_{a}.npz", WORK / f"blur_fast_ws{work_side}_{k}_{a}.npz"
            if not (fe.exists() and ff.exists()):
                if img_cache.get("id") != pk["scene"]:
                    img_cache = {"id": pk["scene"], "img": _scene_img(pk["scene"]),
                                 "entry": st["cache"].load(pk["scene"])}
                img, win = img_cache["img"], img_cache["entry"].windower(pk["mask"])
                for path, kw in ((ff, {"work_side": work_side}), (fe, {"exact_blur": True})):
                    if path.exists():
                        continue
                    t = time.perf_counter()
                    crop = C.make_crop(img, box, win, "blur", a, **kw)
                    sec = time.perf_counter() - t
                    tmp = path.with_suffix(".tmp.npz")
                    np.savez(tmp, crop=crop, sec=sec)
                    tmp.replace(path)
                    print(f"маска {k} α={a} {'точно' if 'exact_blur' in kw else 'быстро'}: {sec:.1f} с", flush=True)
            fast, exact = np.load(ff), np.load(fe)
            d = np.abs(fast["crop"].astype(int) - exact["crop"].astype(int))
            pairs.append({**pk, "alpha": a, "square_side": C.square(box, a)[2],
                          "sec_fast": float(fast["sec"]), "sec_exact": float(exact["sec"]),
                          "abs_diff_max_uint8": int(d.max()), "abs_diff_mean": float(d.mean()),
                          "_fast": fast["crop"], "_exact": exact["crop"]})
    img_cache.clear()
    fast = np.stack([p.pop("_fast") for p in pairs])
    exact = np.stack([p.pop("_exact") for p in pairs])
    by_enc = {}
    for name in ENCODERS:
        model = M.load(name)
        cos = (M.encode_crops(model, fast) * M.encode_crops(model, exact)).sum(1).cpu().numpy()
        for p, c in zip(pairs, cos):
            p[f"cls_cos_{name}"] = float(c)
        by_enc[name] = {"cls_cos_min": float(cos.min()), "cls_cos_median": float(np.median(cos)),
                        "passed": bool(cos.min() >= BLUR_COS_MIN)}
        del model
        torch.cuda.empty_cache()
    passed = all(v["passed"] for v in by_enc.values())
    data = json.loads(env.ENV_JSON.read_text()).get("blur_large_masks", {})
    attempts = {k: v for k, v in data.get("attempts", {}).items()}
    attempts[str(work_side)] = {"work_side": work_side, "by_encoder": by_enc, "passed": passed, "pairs": pairs}
    _write_env("blur_large_masks", {
        "what": "размытие на уменьшенном окне против буквального: 8 крупнейших масок 10 калибровочных сцен сверки, оба α",
        "threshold_cls_cos_min": BLUR_COS_MIN, "rule_on_fail": f"рабочая сторона {BLUR_WORK_SIDE_RETRY}, сверка повторяется",
        "crop_n_layers": st["cnl"], "attempts": attempts})
    print("ПРОЙДЕНО" if passed else f"НЕ ПРОЙДЕНО при рабочей стороне {work_side}", by_enc, flush=True)


# ---------------------------------------------------------------- точность z_pool


def stage_precision() -> None:
    import torch

    from src.encode import model as M

    st = _setup()
    picks = {}
    for sid in st["scenes"]:
        gt = hr_insdet.scene_gt(st["by_id"][sid], st["labels"])
        e = st["cache"].load(sid)
        gtb = np.array([g["box"] for g in gt], float).reshape(-1, 4)
        a = oracle_assign(gtb, e.boxes)
        area = (gtb[:, 2] - gtb[:, 0]) * (gtb[:, 3] - gtb[:, 1])
        picks[sid] = {"mask": a[a >= 0], "bin": np.digitize(area[a >= 0], hr_insdet.AREA_EDGES[:-1]),
                      "n_gt": len(gt)}
    u_r_np = D.load(U_R_PATH)
    out = {"what": "z_pool в рабочей точности против fp32: косинус по оракульным маскам, бины — площадь рамки разметки",
           "threshold_median_cos_per_bin": POOL_COS_MEDIAN_MIN, "oracle_iou": ORACLE_IOU, "crop_n_layers": st["cnl"],
           "scenes": st["scenes"], "n_gt": sum(p["n_gt"] for p in picks.values()),
           "n_oracle_masks": int(sum(len(p["mask"]) for p in picks.values())), "bins": list(AREA_BINS),
           "by_encoder": {}}
    for name in ENCODERS:
        z, secs, peak = {}, {}, {}
        for prec, fp32 in (("working", False), ("fp32", True)):
            model = M.load(name, fp32=fp32)
            u_r = torch.from_numpy(u_r_np).cuda() if name == "encoder_dinov3" else None
            PO.pool_image(model, _scene_img(st["scenes"][0]), st["cache"].load(st["scenes"][0]), u_r)  # прогрев
            z[prec], secs[prec] = {}, []
            torch.cuda.reset_peak_memory_stats()
            for sid in st["scenes"]:
                img, entry = _scene_img(sid), st["cache"].load(sid)
                x, _ = PO.encoder_input(img, M.patch_size(model))
                torch.cuda.synchronize()
                t = time.perf_counter()
                M.encode_full(model, x)  # время одного прохода энкодера, без привязки масок
                torch.cuda.synchronize()
                secs[prec].append(time.perf_counter() - t)
                r = PO.pool_image(model, img, entry, u_r)
                z[prec][sid] = {k: r[k][picks[sid]["mask"]] for k in ("p", "p_perp") if k in r}
            peak[prec] = torch.cuda.max_memory_allocated() // 2**20
            del model, u_r
            torch.cuda.empty_cache()
        bins = np.concatenate([picks[s]["bin"] for s in st["scenes"]])
        res = {"dtype_working": env.ENCODER_DTYPE[name],
               "sec_forward_median": {k: float(np.median(v)) for k, v in secs.items()},
               "sec_forward_min_max": {k: [float(min(v)), float(max(v))] for k, v in secs.items()},
               "peak_allocated_mib": {k: int(v) for k, v in peak.items()}}
        for k in z["working"][st["scenes"][0]]:
            cos = np.concatenate([(z["working"][s][k] * z["fp32"][s][k]).sum(1) for s in st["scenes"]])
            res[k] = {"by_bin": {AREA_BINS[b]: {"n": int((bins == b).sum()),
                                               "cos_median": float(np.median(cos[bins == b])),
                                               "cos_min": float(cos[bins == b].min())}
                                 for b in range(3) if (bins == b).any()},
                      "cos_median_all": float(np.median(cos)), "cos_min_all": float(cos.min())}
        res["passed"] = all(v["cos_median"] >= POOL_COS_MEDIAN_MIN for v in res["p"]["by_bin"].values())
        if "p_perp" in res:
            res["p_perp"]["note"] = "справочно: порог записан для z_pool"
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
        out["by_encoder"][name] = res
    out["passed"] = all(v["passed"] for v in out["by_encoder"].values())
    _write_env("pool_precision", out)


def stage_blur_refs() -> None:
    import torch

    from src.encode import model as M

    st = _setup()
    sp = json.loads(Path("splits/hr_insdet.json").read_text())
    rng = np.random.default_rng(sp["seed"])
    side = lambda b: max(b[2] - b[0], b[3] - b[1])
    refs = sorted((r for r in sp["gallery"] if side(r["box"]) > C.BLUR_WORK_SIDE), key=lambda r: r["id"])
    ref_pick = [refs[i] for i in sorted(rng.choice(len(refs), BLUR_REFS_N, replace=False))]
    cal = [(sid, i) for sid in st["scenes"] for i, r in enumerate(st["cache"].load(sid).records)
           if BLUR_REFS_CAL_SIDE[0] < side(r["box"]) <= BLUR_REFS_CAL_SIDE[1]]
    cal_pick = [cal[i] for i in sorted(rng.choice(len(cal), BLUR_REFS_CAL_N, replace=False))]
    items = [("reference", r["id"], 0, hr_insdet.ROOT / r["image"],
              lambda r=r: st["box_cache"].load(r["id"], [r["box"]])) for r in ref_pick]
    items += [("cal_mask", sid, i, hr_insdet.ROOT / "Scenes" / f"{sid}.jpg", lambda sid=sid: st["cache"].load(sid))
              for sid, i in cal_pick]
    WORK.mkdir(parents=True, exist_ok=True)
    pairs, fast_all, exact_all = [], [], []
    for k, (kind, iid, mi, path, load) in enumerate(items):
        f = WORK / f"blur_refs_ws{C.BLUR_WORK_SIDE}_{k}.npz"
        if not f.exists():
            img, e = S.read_rgb(path), load()
            win, box = e.windower(mi), e.records[mi]["box"]
            out = {}
            for a in C.ALPHAS:
                for tag, kw in (("fast", {}), ("exact", {"exact_blur": True})):
                    t = time.perf_counter()
                    out[f"{tag}_{a}"] = C.make_crop(img, box, win, "blur", a, **kw)
                    out[f"sec_{tag}_{a}"] = time.perf_counter() - t
                out[f"side_{a}"] = C.square(box, a)[2]
            tmp = f.with_suffix(".tmp.npz")
            np.savez(tmp, **out)
            tmp.replace(f)
            print(f"{k + 1}/{len(items)} {kind} {iid}", flush=True)
        with np.load(f) as z:
            for a in C.ALPHAS:
                d = np.abs(z[f"fast_{a}"].astype(int) - z[f"exact_{a}"].astype(int))
                pairs.append({"kind": kind, "image": iid, "mask": mi, "alpha": a, "square_side": int(z[f"side_{a}"]),
                              "sec_fast": float(z[f"sec_fast_{a}"]), "sec_exact": float(z[f"sec_exact_{a}"]),
                              "abs_diff_max_uint8": int(d.max())})
                fast_all.append(z[f"fast_{a}"])
                exact_all.append(z[f"exact_{a}"])
    fast_all, exact_all = np.stack(fast_all), np.stack(exact_all)
    by_enc = {}
    for name in ENCODERS:
        model = M.load(name)
        enc = lambda x: np.concatenate([M.encode_crops(model, x[k:k + 16]).cpu().numpy() for k in range(0, len(x), 16)])
        cos = (enc(fast_all) * enc(exact_all)).sum(1)
        for p_, c in zip(pairs, cos):
            p_[f"cls_cos_{name}"] = float(c)
        kinds = np.array([p_["kind"] for p_ in pairs])
        by_enc[name] = {kd: {"n": int((kinds == kd).sum()), "cls_cos_min": float(cos[kinds == kd].min()),
                             "cls_cos_median": float(np.median(cos[kinds == kd]))} for kd in ("reference", "cal_mask")}
        by_enc[name]["passed"] = bool(cos.min() >= BLUR_COS_MIN)
        by_enc[name]["escalate"] = bool(cos.min() < BLUR_REFS_ESCALATE)
        del model
        torch.cuda.empty_cache()
    _write_env("blur_refs", {
        "what": "размытие на уменьшенном окне против буквального: эталоны и маски калибровочных сцен среднего размера, оба α",
        "work_side": C.BLUR_WORK_SIDE, "threshold_cls_cos_min": BLUR_COS_MIN, "escalate_below": BLUR_REFS_ESCALATE,
        "rule_on_fail": "рабочая сторона не меняется; оговорка и числа в 3.1 и 4.1; минимум ниже escalate_below — требует разбора",
        "seed": sp["seed"], "n_candidates": {"reference": len(refs), "cal_mask": len(cal)},
        "by_encoder": by_enc, "passed": all(v["passed"] for v in by_enc.values()),
        "escalate": any(v["escalate"] for v in by_enc.values()), "pairs": pairs})
    print("ПРОЙДЕНО" if all(v["passed"] for v in by_enc.values()) else "НЕ ПРОЙДЕНО", by_enc, flush=True)


def _oracle_picks(st: dict) -> dict:
    picks = {}
    for sid in st["scenes"]:
        gt = hr_insdet.scene_gt(st["by_id"][sid], st["labels"])
        gtb = np.array([g["box"] for g in gt], float).reshape(-1, 4)
        a = oracle_assign(gtb, st["cache"].load(sid).boxes)
        area = (gtb[:, 2] - gtb[:, 0]) * (gtb[:, 3] - gtb[:, 1])
        picks[sid] = {"mask": a[a >= 0], "bin": np.digitize(area[a >= 0], hr_insdet.AREA_EDGES[:-1]), "n_gt": len(gt)}
    return picks


def stage_crop_precision() -> None:
    import torch

    from src.encode import model as M
    from src.encode import pipeline as PL

    st = _setup()
    picks = _oracle_picks(st)
    WORK.mkdir(parents=True, exist_ok=True)
    crops = {}
    for v in CROP_PRECISION_VARIANTS:  # вырезки готовятся один раз: от точности и энкодера они не зависят
        f = WORK / f"crop_precision_{v[0]}_{v[1]}.npz"
        if not f.exists():
            out = {}
            for sid in st["scenes"]:
                img, e = _scene_img(sid), st["cache"].load(sid)
                out[sid] = np.stack([C.make_crop(img, e.records[i]["box"], e.windower(i), *v) for i in picks[sid]["mask"]])
            tmp = f.with_suffix(".tmp.npz")
            np.savez(tmp, **{s.replace("/", "__"): a for s, a in out.items()})
            tmp.replace(f)
        with np.load(f) as z:
            crops[v] = {sid: z[sid.replace("/", "__")] for sid in st["scenes"]}
    bins = np.concatenate([picks[s]["bin"] for s in st["scenes"]])
    out = {"what": "CLS вырезок 448 в рабочей точности против fp32: косинус по оракульным маскам, бины — площадь рамки разметки",
           "threshold_median_cos_per_bin": CROP_COS_MEDIAN_MIN, "oracle_iou": ORACLE_IOU, "crop_n_layers": st["cnl"],
           "scenes": st["scenes"], "n_oracle_masks": int(len(bins)), "bins": list(AREA_BINS), "batch": PL.BATCH,
           "blur_work_side": C.BLUR_WORK_SIDE, "by_encoder": {}}
    for name in ENCODERS:
        z, rate = {}, {}
        for prec, fp32 in (("working", False), ("fp32", True)):
            model = M.load(name, fp32=fp32)
            M.encode_crops(model, np.zeros((PL.BATCH, 448, 448, 3), np.uint8))  # прогрев
            torch.cuda.synchronize()
            t, n = time.perf_counter(), 0
            for v in CROP_PRECISION_VARIANTS:
                parts = []
                for sid in st["scenes"]:
                    x = crops[v][sid]
                    parts += [M.encode_crops(model, x[k:k + PL.BATCH]).cpu().numpy() for k in range(0, len(x), PL.BATCH)]
                    n += len(x)
                z[prec, v] = np.concatenate(parts)
            torch.cuda.synchronize()
            rate[prec] = n / (time.perf_counter() - t)
            del model
            torch.cuda.empty_cache()
        res = {"dtype_working": env.ENCODER_DTYPE[name], "crops_per_sec": {k: round(v, 2) for k, v in rate.items()},
               "by_variant": {}}
        for v in CROP_PRECISION_VARIANTS:
            cos = (z["working", v] * z["fp32", v]).sum(1)
            res["by_variant"][f"C({v[0]}|{v[1]})"] = {
                "by_bin": {AREA_BINS[b]: {"n": int((bins == b).sum()), "cos_median": float(np.median(cos[bins == b])),
                                          "cos_min": float(cos[bins == b].min())} for b in range(3) if (bins == b).any()},
                "cos_median_all": float(np.median(cos)), "cos_min_all": float(cos.min())}
        res["passed"] = all(b["cos_median"] >= CROP_COS_MEDIAN_MIN
                            for r in res["by_variant"].values() for b in r["by_bin"].values())
        print(name, json.dumps(res, ensure_ascii=False), flush=True)
        out["by_encoder"][name] = res
    out["passed"] = all(v["passed"] for v in out["by_encoder"].values())
    _write_env("crop_precision", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["memory", "u_r", "onepass", "blur", "precision", "crop_precision", "blur_refs"])
    ap.add_argument("--work-side", type=int, default=BLUR_WORK_SIDE_FIRST, choices=[BLUR_WORK_SIDE_FIRST, BLUR_WORK_SIDE_RETRY],
                    help="только этап blur: рабочая сторона размытия; 1792 — повтор после провала при 896")
    args = ap.parse_args()
    if args.stage == "blur":
        stage_blur(args.work_side)
    else:
        {"memory": stage_memory, "u_r": stage_u_r, "onepass": stage_onepass, "precision": stage_precision,
         "crop_precision": stage_crop_precision, "blur_refs": stage_blur_refs}[args.stage]()


if __name__ == "__main__":
    main()
