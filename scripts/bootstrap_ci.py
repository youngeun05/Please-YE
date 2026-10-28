#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for the benchmark results (no new predictions).

    python scripts/bootstrap_ci.py --kitti-root D:/datasets/KITTI --split eval_val

It reads the prediction folders that run_benchmark.py already wrote
(runs/benchmark/<split>/predictions_<label>/) and answers the questions the
report's section 6.5 has to answer with numbers instead of adjectives:

  * Is the INT8 drop real, or within sampling noise of a 1,000-image test set?
  * Is percentile calibration really better than minmax / entropy?
  * Is the YOLOv8n vs YOLO11n accuracy gap larger than sampling noise?

Method: draw the split's images with replacement (same draw for every model,
so comparisons are paired), score every model on the draw with the same pinned
KITTI evaluator, repeat --resamples times, report the 2.5/97.5 percentiles.
Paired differences cancel the "which images were picked" noise that both
models share, which is why a drop can be significant even when each model's
own interval is +-1 point wide.

It also writes the full Easy / Moderate / Hard AP40 table for every model
(point estimates on the whole split), which run_benchmark.py does not print.

Nothing is predicted and no model is loaded; eval_val is only re-scored from
predictions of the models already locked in FINAL_LOCK.json.

Speed: one KITTI evaluation is ~5-25 s in pure Python. With numba installed it
is far faster after the first call. Recommended: a separate small venv so the
verified project venv stays untouched (see 실행가이드1029.md):
    py -3.13 -m venv .venv-stats; .venv-stats\\Scripts\\pip install numpy==2.4.6 numba
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
CLASSES = ("Car", "Pedestrian", "Cyclist")
LEVELS = ("easy", "moderate", "hard")
SPLITS = {"internal_val": ROOT / "splits/internal_val.txt", "eval_val": ROOT / "eval_val.txt"}
DEFAULT_PAIRS = [
    ("baseline", "optimized", "YOLO11n INT8 하락 (FP32 − INT8 percentile)"),
    ("optimized", "y11n_int8_minmax", "percentile − minmax (YOLO11n INT8)"),
    ("optimized", "y11n_int8_entropy", "percentile − entropy (YOLO11n INT8)"),
    ("y8n_fp32", "y8n_int8_percentile", "YOLOv8n INT8 하락 (FP32 − INT8 percentile)"),
    ("y8n_fp32", "baseline", "YOLOv8n − YOLO11n (FP32)"),
    ("y8n_int8_percentile", "optimized", "YOLOv8n − YOLO11n (INT8 percentile)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SPLITS), default="eval_val")
    parser.add_argument("--bench-dir", type=Path, default=None, help="default: runs/benchmark/<split>")
    parser.add_argument("--pair", action="append", default=[], metavar="A:B",
                        help="difference A - B to test (repeatable). Default: the standard pairs that exist")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--reference-evaluator", type=Path,
                        default=ROOT / "third_party/mmdetection3d/mmdet3d/evaluation/functional/kitti_utils/eval.py")
    return parser.parse_args()


# ----------------------------------------------------------------- scoring (also runs in workers)
_STATE: dict = {}


def _init(kitti_root: str, split_file: str, folders: dict[str, str], evaluator: str) -> None:
    from evaluate_kitti import load_reference, parse_label, read_ids

    ids = read_ids(Path(split_file))
    gt_dir = Path(kitti_root) / "training/label_2"
    _STATE["gt"] = [parse_label(gt_dir / f"{i}.txt", False) for i in ids]
    _STATE["dt"] = {label: [parse_label(Path(folder) / f"{i}.txt", True) for i in ids]
                    for label, folder in folders.items()}
    _STATE["ref"], _STATE["accel"] = load_reference(Path(evaluator))


def _score(indices, labels) -> dict[str, dict[str, float]]:
    """Moderate AP40 per class and mean for each label on one image draw."""
    gt = [_STATE["gt"][i] for i in indices]
    out = {}
    for label in labels:
        dt = [_STATE["dt"][label][i] for i in indices]
        _, metrics = _STATE["ref"].kitti_eval(gt, dt, list(CLASSES), ["bbox"])
        ap = {c: float(metrics[f"KITTI/{c}_2D_AP40_moderate_strict"]) for c in CLASSES}
        ap["mean"] = sum(ap.values()) / 3
        out[label] = ap
    return out


def _full_table(labels) -> dict[str, dict[str, dict[str, float]]]:
    n = len(_STATE["gt"])
    table = {}
    for label in labels:
        _, metrics = _STATE["ref"].kitti_eval(_STATE["gt"], _STATE["dt"][label], list(CLASSES), ["bbox"])
        table[label] = {lv: {c: float(metrics[f"KITTI/{c}_2D_AP40_{lv}_strict"]) for c in CLASSES} for lv in LEVELS}
        for lv in LEVELS:
            table[label][lv]["mean"] = sum(table[label][lv][c] for c in CLASSES) / 3
    assert n
    return table


def _work(task):
    chunk, labels = task
    return [_score(indices, labels) for indices in chunk]


# ----------------------------------------------------------------- main
def find_folders(bench: Path, split_ids: int) -> dict[str, Path]:
    folders = {}
    for folder in sorted(bench.glob("predictions_*")):
        stamp = folder / ".complete.json"
        if not stamp.is_file():
            continue
        info = json.loads(stamp.read_text(encoding="utf-8"))
        if info.get("files") == split_ids and len(list(folder.glob("*.txt"))) == split_ids:
            folders[folder.name.removeprefix("predictions_")] = folder
    return folders


def ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    split_file = SPLITS[args.split]
    bench = (args.bench_dir or ROOT / "runs/benchmark" / args.split).resolve()
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not args.reference_evaluator.is_file():
        raise SystemExit(f"Pinned evaluator not found: {args.reference_evaluator} (run scripts/setup_evaluator.ps1)")
    if not (args.kitti_root / "training/label_2").is_dir():
        raise SystemExit(f"KITTI labels not found under {args.kitti_root}")
    folders = find_folders(bench, len(ids))
    if not folders:
        raise SystemExit(f"No complete predictions_<label> folders in {bench}. Run scripts/run_benchmark.py first.")

    if args.pair:
        pairs = []
        for item in args.pair:
            if ":" not in item:
                raise SystemExit(f"--pair expects A:B, got {item!r}")
            a, b = item.split(":", 1)
            pairs.append((a, b, f"{a} − {b}"))
    else:
        pairs = [p for p in DEFAULT_PAIRS if p[0] in folders and p[1] in folders]
    missing = sorted({x for a, b, _ in pairs for x in (a, b)} - folders.keys())
    if missing:
        raise SystemExit(f"No complete predictions for {missing}; available: {sorted(folders)}")
    if not pairs:
        raise SystemExit(f"None of the default pairs is available; available labels: {sorted(folders)}")
    labels = sorted({x for a, b, _ in pairs for x in (a, b)})

    print(f"split     : {args.split} ({len(ids)} images), predictions from {bench}")
    print(f"models    : {', '.join(labels)}")
    print(f"pairs     : {len(pairs)}, resamples {args.resamples}, workers {args.workers}, seed {args.seed}")

    init_args = (str(args.kitti_root.resolve()), str(split_file), {k: str(v) for k, v in folders.items()},
                 str(args.reference_evaluator.resolve()))
    _init(*init_args)
    print(f"evaluator : {_STATE['accel']}" + ("" if _STATE["accel"] == "numba" else
          "  (slow; numba in a separate venv makes this much faster — see the docstring)"))

    started = time.perf_counter()
    full = _full_table(sorted(folders))
    point = {label: full[label]["moderate"] for label in labels}
    print(f"full Easy/Moderate/Hard table for {len(folders)} models: {time.perf_counter() - started:.0f} s")

    # identical draws for every model and every worker count
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(ids), size=(args.resamples, len(ids)))
    chunk = max(1, min(20, args.resamples // (args.workers * 4) or 1))
    tasks = [(draws[i:i + chunk].tolist(), labels) for i in range(0, args.resamples, chunk)]
    results: list[dict] = []
    started = time.perf_counter()
    if args.workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(args.workers, initializer=_init, initargs=init_args) as pool:
            for part in pool.imap(_work, tasks):
                results.extend(part)
                done = len(results)
                if done % max(1, args.resamples // 20) < chunk or done == args.resamples:
                    rate = (time.perf_counter() - started) / done
                    print(f"  {done}/{args.resamples} resamples, ~{rate * (args.resamples - done) / 60:.1f} min left",
                          flush=True)
    else:
        for task in tasks:
            results.extend(_work(task))
            done = len(results)
            rate = (time.perf_counter() - started) / done
            print(f"  {done}/{args.resamples} resamples, ~{rate * (args.resamples - done) / 60:.1f} min left", flush=True)

    keys = (*CLASSES, "mean")
    samples = {label: {k: np.array([r[label][k] for r in results]) for k in keys} for label in labels}
    model_ci = {label: {k: {"point": point[label][k], "ci95": ci(samples[label][k])} for k in keys} for label in labels}
    pair_ci = []
    for a, b, title in pairs:
        entry = {"a": a, "b": b, "title": title}
        for k in keys:
            diff = samples[a][k] - samples[b][k]
            entry[k] = {"point": point[a][k] - point[b][k], "ci95": ci(diff),
                        "p_positive": float((diff > 0).mean())}
        pair_ci.append(entry)

    report = {"split": args.split, "images": len(ids), "resamples": args.resamples, "seed": args.seed,
              "evaluator_execution": _STATE["accel"], "created": datetime.now().astimezone().isoformat(timespec="seconds"),
              "prediction_folders": {k: str(v) for k, v in folders.items()},
              "easy_moderate_hard": full, "models": model_ci, "pairs": pair_ci}
    (bench / "bootstrap_ci.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fmt(x):
        return f"{x['point']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"

    lines = [f"# Bootstrap 신뢰구간 — {args.split} ({len(ids)}장, 재표본 {args.resamples}회, 95%)", "",
             "이미지를 복원추출해 모든 모델을 **같은 표본**으로 다시 채점한 결과입니다 (짝지은 비교). "
             "값은 `점추정 [2.5%, 97.5%]`. P(>0)은 재표본 중 차이가 양수였던 비율입니다.", "",
             "## 1. 비교 (A − B)", "", "| 비교 | 평균 | Car | Pedestrian | Cyclist | P(>0), 평균 | 판정 |", "|---|---|---|---|---|---|---|"]
    for e in pair_ci:
        low, high = e["mean"]["ci95"]
        verdict = "0보다 큼 (유의)" if low > 0 else "0보다 작음 (유의)" if high < 0 else "0 포함 (구분 안 됨)"
        lines.append(f"| {e['title']} | {fmt(e['mean'])} | {fmt(e['Car'])} | {fmt(e['Pedestrian'])} | "
                     f"{fmt(e['Cyclist'])} | {e['mean']['p_positive']:.3f} | {verdict} |")
    lines += ["", "## 2. 모델별 Moderate AP40", "", "| 모델 | 평균 | Car | Pedestrian | Cyclist |", "|---|---|---|---|---|"]
    for label in labels:
        m = model_ci[label]
        lines.append(f"| {label} | {fmt(m['mean'])} | {fmt(m['Car'])} | {fmt(m['Pedestrian'])} | {fmt(m['Cyclist'])} |")
    lines += ["", "## 3. Easy / Moderate / Hard AP40 (전체 표본, 점추정)", "",
              "| 모델 | 난이도 | 평균 | Car | Pedestrian | Cyclist |", "|---|---|---|---|---|---|"]
    for label in sorted(full):
        for lv in LEVELS:
            t = full[label][lv]
            lines.append(f"| {label} | {lv} | {t['mean']:.2f} | {t['Car']:.2f} | {t['Pedestrian']:.2f} | {t['Cyclist']:.2f} |")
    lines += ["", f"평가기 실행: {_STATE['accel']} (numba와 순수 Python 경로는 같은 값을 냅니다)."]
    (bench / "bootstrap_ci.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[:8 + len(pair_ci)]))
    print(f"\nsaved: {bench / 'bootstrap_ci.md'}  (+ .json)  in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
