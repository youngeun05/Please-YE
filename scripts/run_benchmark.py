#!/usr/bin/env python3
"""One command that reproduces the report: accuracy (and optionally speed) of the
baseline and the optimized model, written as the report's tables.

    # sanity check on the internal split (free to run any number of times)
    python scripts/run_benchmark.py --kitti-root D:/datasets/KITTI

    # the official numbers (eval_val.txt). See "eval_val lock" below.
    python scripts/run_benchmark.py --kitti-root D:/datasets/KITTI --split eval_val --confirm-final

Default models are the packaged submission files in models/ (see
scripts/package_models.py and README.md):

    models/yolo11n_kitti_1248x384_fp32.onnx   baseline  (FP32, ONNX Runtime)
    models/yolo11n_kitti_1248x384_int8.onnx   optimized (INT8 PTQ, ONNX Runtime)

Steps per model:
  1. predict_kitti.py on CPU, batch 1, at the model's own static input shape,
     writing one KITTI result file per image (skipped when a previous run with
     the same model hash and split is complete).
  2. evaluate_kitti.py: KITTI 2D Moderate AP40, strict IoU, pinned evaluator.
Then, unless --no-speed: benchmark_onnx.py on the ONNX models (size,
parameters, FLOPs, latency, peak memory). Speed numbers depend on the machine;
--speed-json reuses an earlier benchmark_onnx.py result instead (matched to the
models by SHA-256), which is how the report keeps one measurement machine.

Output (runs/benchmark/<split>/):
  summary.md     the report tables: section 2 (summary), 6.3 (accuracy), 6.4 (speed)
  summary.json   the same numbers with model hashes and versions
  metrics_<label>.json, predictions_<label>/

eval_val lock: the official split is scored only with --confirm-final. The first
such run writes runs/benchmark/eval_val/FINAL_LOCK.json with the model hashes.
Re-running with the SAME models (or a subset of them) is allowed (a crash, or a
reviewer reproducing the numbers); running with a new or changed model is refused, because choosing a model
after seeing eval_val scores would turn the test set into a validation set.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from onnx_common import ROOT, file_sha256, ids_sha256

SCRIPTS = Path(__file__).resolve().parent
SPLITS = {"internal_val": ROOT / "splits/internal_val.txt", "eval_val": ROOT / "eval_val.txt"}
DEFAULT_BASELINE = ROOT / "models/yolo11n_kitti_1248x384_fp32.onnx"
DEFAULT_OPTIMIZED = ROOT / "models/yolo11n_kitti_1248x384_int8.onnx"
CLASSES = ("Car", "Pedestrian", "Cyclist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, required=True,
                        help="folder that contains training/image_2 and training/label_2")
    parser.add_argument("--split", choices=tuple(SPLITS), default="internal_val")
    parser.add_argument("--confirm-final", action="store_true",
                        help="required for --split eval_val (the official, one-time evaluation)")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--optimized", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--also", action="append", default=[], metavar="LABEL=PATH",
                        help="extra model to score and compare with the baseline (repeatable)")
    parser.add_argument("--no-speed", action="store_true", help="accuracy only")
    parser.add_argument("--speed-json", type=Path, action="append", default=[],
                        help="reuse benchmark_onnx.py output(s) instead of measuring; several "
                             "files are averaged (models matched by SHA-256)")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime threads for speed")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--reference-evaluator", type=Path,
                        default=ROOT / "third_party/mmdetection3d/mmdet3d/evaluation/functional/kitti_utils/eval.py")
    parser.add_argument("--output", type=Path, default=None, help="default: runs/benchmark/<split>")
    return parser.parse_args()


# ----------------------------------------------------------------- inputs
def model_list(args: argparse.Namespace) -> list[dict]:
    models = [{"label": "baseline", "role": "baseline", "path": args.baseline},
              {"label": "optimized", "role": "optimized", "path": args.optimized}]
    for item in args.also:
        if "=" not in item:
            raise SystemExit(f"--also expects LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        if not label.replace("_", "").replace("-", "").isalnum():
            raise SystemExit(f"--also label must be letters, digits, - or _: {label!r}")
        models.append({"label": label, "role": "extra", "path": Path(path)})
    labels = [m["label"] for m in models]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"Duplicate model labels: {labels}")
    for model in models:
        model["path"] = model["path"].resolve()
        if not model["path"].is_file():
            hint = " (run scripts/package_models.py, or download the models; see README.md)" \
                if model["path"].parent == (ROOT / "models").resolve() else ""
            raise SystemExit(f"{model['label']} model not found: {model['path']}{hint}")
        model["sha256"] = file_sha256(model["path"])
        model["format"] = model["path"].suffix.lstrip(".").lower()
    return models


def input_size(path: Path) -> str:
    """WIDTHxHEIGHT from a static ONNX input; .pt models use the training shape."""
    if path.suffix.lower() != ".onnx":
        return "1248x384"
    from onnx_common import static_input

    _, (_, _, height, width) = static_input(path)
    return f"{width}x{height}"


def check_lock(output: Path, split: str, models: list[dict], confirm: bool) -> Path | None:
    if split != "eval_val":
        return None
    if not confirm:
        raise SystemExit(
            "eval_val.txt is the official test split. Score it once, after every model and setting\n"
            "is final, by adding --confirm-final. Use --split internal_val for everything else.")
    lock = output / "FINAL_LOCK.json"
    if lock.is_file():
        locked = json.loads(lock.read_text(encoding="utf-8"))
        locked_hashes = {m["label"]: m["sha256"] for m in locked["models"]}
        current = {m["label"]: m["sha256"] for m in models}
        # Allowed: the same models, or a subset of them (a reviewer re-running only the
        # baseline and the optimized model). Refused: any new or changed model.
        if any(locked_hashes.get(label) != sha for label, sha in current.items()):
            raise SystemExit(
                f"eval_val was already scored on {locked['created']} with different models.\n"
                f"  locked : {json.dumps(locked_hashes, indent=2)}\n  current: {json.dumps(current, indent=2)}\n"
                "Re-scoring with other models would make eval_val a validation set. Refusing.")
        print(f"eval_val lock: same models as the run of {locked['created']}; re-running is allowed")
    return lock


# ----------------------------------------------------------------- steps
def predict(model: dict, args: argparse.Namespace, split_file: Path, output: Path) -> Path:
    folder = output / f"predictions_{model['label']}"
    stamp_path = folder / ".complete.json"
    split_ids = [line.strip() for line in split_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    stamp = {"model_sha256": model["sha256"], "split_sha256": ids_sha256(split_ids), "files": len(split_ids)}
    previous = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.is_file() else {}
    if all(previous.get(key) == value for key, value in stamp.items()):
        print(f"[{model['label']}] predictions already complete for this model and split; reusing")
        return folder
    if folder.exists():
        shutil.rmtree(folder)  # stale or partial output from another model/split
    size = input_size(model["path"])
    command = [sys.executable, str(SCRIPTS / "predict_kitti.py"),
               "--weights", str(model["path"]), "--kitti-root", str(args.kitti_root.resolve()),
               "--split", str(split_file), "--output", str(folder),
               "--imgsz", size, "--device", "cpu", "--batch", "1"]
    print(f"[{model['label']}] predicting {len(split_ids)} images at {size} on CPU ...", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"[{model['label']}] predict_kitti.py failed (exit {result.returncode})")
    written = len(list(folder.glob("*.txt")))
    if written != len(split_ids):
        raise SystemExit(f"[{model['label']}] expected {len(split_ids)} prediction files, found {written}")
    stamp["seconds"] = round(time.perf_counter() - started, 1)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return folder


def evaluate(model: dict, args: argparse.Namespace, split_file: Path, predictions: Path, output: Path) -> dict:
    from evaluate_kitti import evaluate as kitti_evaluate

    if not args.reference_evaluator.is_file():
        raise SystemExit(f"Pinned evaluator not found: {args.reference_evaluator}\n"
                         "Run scripts/setup_evaluator.ps1 (Windows) or scripts/setup_evaluator.sh (Linux/macOS).")
    result = kitti_evaluate(argparse.Namespace(
        kitti_root=args.kitti_root.resolve(), split=split_file, predictions=predictions,
        reference_evaluator=args.reference_evaluator, output=None))
    result["model"] = {"path": str(model["path"]), "sha256": model["sha256"]}
    (output / f"metrics_{model['label']}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ap = result["class_ap40_moderate"]
    print(f"[{model['label']}] Moderate AP40  Car {ap['Car']:.2f}  Ped {ap['Pedestrian']:.2f}  "
          f"Cyc {ap['Cyclist']:.2f}  mean {result['mean_ap40_moderate']:.2f}")
    return result


def measure_speed(models: list[dict], args: argparse.Namespace, output: Path) -> list[Path]:
    onnx_models = [m for m in models if m["format"] == "onnx"]
    if not onnx_models:
        return []
    target = output / "speed_benchmark_onnx.json"
    command = [sys.executable, str(SCRIPTS / "benchmark_onnx.py"),
               "--models", *[str(m["path"]) for m in onnx_models],
               "--threads", str(args.threads), "--warmup", str(args.warmup),
               "--repeats", str(args.repeats), "--output", str(target)]
    print("measuring speed with benchmark_onnx.py (close other programs; results depend on this machine) ...")
    if subprocess.run(command, cwd=str(ROOT)).returncode != 0:
        raise SystemExit("benchmark_onnx.py failed")
    return [target]


def load_speed(files: list[Path], models: list[dict]) -> dict[str, dict]:
    """Average size/params/FLOPs/latency/memory per model across benchmark files."""
    by_hash: dict[str, list[dict]] = {}
    environments = []
    for path in files:
        report = json.loads(path.read_text(encoding="utf-8"))
        environments.append(report.get("environment", {}))
        for row in report["models"]:
            by_hash.setdefault(row["sha256"], []).append(row)
    speed: dict[str, dict] = {}
    for model in models:
        rows = by_hash.get(model["sha256"])
        if not rows:
            continue

        def mean(values):
            return sum(values) / len(values)

        latency = [r["runtime"]["latency_ms"] for r in rows]
        speed[model["label"]] = {
            "bytes": rows[0]["bytes"], "mib": rows[0]["bytes"] / 1024 ** 2, "mb": rows[0]["bytes"] / 1e6,
            "parameters": rows[0]["parameters"], "gflops": rows[0]["flops"]["gflops"],
            "latency_median_ms": mean([x["median"] for x in latency]),
            "latency_mean_ms": mean([x["mean"] for x in latency]),
            "latency_p95_ms": mean([x["p95"] for x in latency]),
            "peak_memory_mib": mean([r["runtime"]["memory"]["peak_rss_mib"] for r in rows]),
            "runs_averaged": len(rows),
            "threads": rows[0]["runtime"]["threads"], "warmup": rows[0]["runtime"]["warmup"],
            "repeats": rows[0]["runtime"]["repeats"],
        }
        speed[model["label"]]["fps"] = 1000.0 / speed[model["label"]]["latency_median_ms"]
    return {"models": speed, "environments": environments, "files": [str(p.resolve()) for p in files]}


# ----------------------------------------------------------------- report
def pct(new: float, old: float) -> str:
    return f"{(new - old) / old * 100:+.1f}%" if old else "n/a"


def build_summary(split: str, models: list[dict], metrics: dict[str, dict], speed: dict) -> str:
    base, opt = metrics["baseline"], metrics["optimized"]
    sb, so = speed["models"].get("baseline"), speed["models"].get("optimized")
    lines = [f"# 벤치마크 결과 — {split} ({base['samples']}장)", "",
             f"생성: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}  ",
             "정확도: KITTI 2D Moderate AP40, strict IoU (Car 0.7 / Ped·Cyc 0.5), MMDetection3D `fe25f7a`  ",
             "모델 크기 MiB = 2^20 바이트. Latency는 중앙값(p50), Throughput = 1000 / Latency (배치 1).", ""]
    for model in models:
        lines.append(f"- {model['label']}: `{Path(model['path']).name}` sha256 `{model['sha256'][:16]}…`")
    lines += ["", "## 2. 결과 요약", "", "| 항목 | Baseline | 최적화 후 | 변화율 |", "|---|---|---|---|"]
    if sb and so:
        lines += [
            f"| 모델 크기 (MiB) | {sb['mib']:.2f} | {so['mib']:.2f} | {pct(so['mib'], sb['mib'])} |",
            f"| 파라미터 수 (M) | {sb['parameters'] / 1e6:.3f} | {so['parameters'] / 1e6:.3f} | "
            f"{pct(so['parameters'], sb['parameters'])} |",
            f"| 연산량 (GFLOPs) | {sb['gflops']:.2f} | {so['gflops']:.2f} | {pct(so['gflops'], sb['gflops'])} |",
            f"| 추론 Latency (ms) | {sb['latency_median_ms']:.1f} | {so['latency_median_ms']:.1f} | "
            f"{pct(so['latency_median_ms'], sb['latency_median_ms'])} |",
            f"| Throughput (FPS) | {sb['fps']:.1f} | {so['fps']:.1f} | {pct(so['fps'], sb['fps'])} |",
            f"| 피크 메모리 사용량 (MiB) | {sb['peak_memory_mib']:.0f} | {so['peak_memory_mib']:.0f} | "
            f"{pct(so['peak_memory_mib'], sb['peak_memory_mib'])} |"]
    else:
        lines.append("| (속도·크기 측정 없음: --no-speed 또는 해시가 맞는 --speed-json 없음) | | | |")
    lines.append(f"| 정확도 (Moderate AP40 평균) | {base['mean_ap40_moderate']:.2f} | "
                 f"{opt['mean_ap40_moderate']:.2f} | "
                 f"{opt['mean_ap40_moderate'] - base['mean_ap40_moderate']:+.2f}%p |")

    lines += ["", "## 6.3 정확도 — Moderate AP40", ""]
    header = "| 클래스 | Baseline | 최적화 후 | Drop (%p) |"
    extras = [m for m in models if m["role"] == "extra"]
    for extra in extras:
        header += f" {extra['label']} | {extra['label']} Drop (%p) |"
    lines += [header, "|" + "---|" * (4 + 2 * len(extras))]
    for cls in (*CLASSES, "mean"):
        def ap(result):
            return result["mean_ap40_moderate"] if cls == "mean" else result["class_ap40_moderate"][cls]
        name = "평균 (mAP)" if cls == "mean" else cls
        row = f"| {name} | {ap(base):.2f} | {ap(opt):.2f} | {ap(base) - ap(opt):.2f} |"
        for extra in extras:
            row += f" {ap(metrics[extra['label']]):.2f} | {ap(base) - ap(metrics[extra['label']]):.2f} |"
        lines.append(row)
    drop = base["mean_ap40_moderate"] - opt["mean_ap40_moderate"]
    lines += ["", f"평균 Drop: {drop:.2f}%p (상대 {drop / base['mean_ap40_moderate'] * 100:.2f}%, "
                  f"유지율 {opt['mean_ap40_moderate'] / base['mean_ap40_moderate'] * 100:.2f}%)"]

    lines += ["", "## 6.4 속도 및 리소스", ""]
    if sb and so:
        lines += ["| 항목 | Baseline | 최적화 후 | 개선율 (%) |", "|---|---|---|---|",
                  f"| 추론 Latency (ms, p50) | {sb['latency_median_ms']:.1f} | {so['latency_median_ms']:.1f} | "
                  f"{(sb['latency_median_ms'] - so['latency_median_ms']) / sb['latency_median_ms'] * 100:.1f} |",
                  f"| Throughput (FPS) | {sb['fps']:.1f} | {so['fps']:.1f} | "
                  f"{(so['fps'] - sb['fps']) / sb['fps'] * 100:.1f} |",
                  f"| 모델 크기 (MiB) | {sb['mib']:.2f} | {so['mib']:.2f} | "
                  f"{(sb['mib'] - so['mib']) / sb['mib'] * 100:.1f} |",
                  f"| 피크 메모리 (MiB) | {sb['peak_memory_mib']:.0f} | {so['peak_memory_mib']:.0f} | "
                  f"{(sb['peak_memory_mib'] - so['peak_memory_mib']) / sb['peak_memory_mib'] * 100:.1f} |",
                  "", f"속도 향상: {sb['latency_median_ms'] / so['latency_median_ms']:.2f}배 (p50 기준). "
                  f"p95: {sb['latency_p95_ms']:.1f} → {so['latency_p95_ms']:.1f} ms. "
                  f"측정 {sb['runs_averaged']}회 평균, {sb['threads']}스레드, warmup {sb['warmup']} / 반복 {sb['repeats']}."]
        env = speed["environments"][0] if speed["environments"] else {}
        if env:
            lines.append(f"측정 환경: {env.get('cpu')}, {env.get('platform')}, ONNX Runtime {env.get('onnxruntime')}, "
                         f"{env.get('provider', '').upper()} provider.")
    else:
        lines.append("(속도 측정 없음)")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    split_file = SPLITS[args.split]
    output = (args.output or ROOT / "runs/benchmark" / args.split).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for sub in ("training/image_2", "training/label_2"):
        if not (args.kitti_root / sub).is_dir():
            raise SystemExit(f"KITTI folder missing: {args.kitti_root / sub}")
    models = model_list(args)
    lock = check_lock(output, args.split, models, args.confirm_final)

    print(f"split  : {args.split} ({split_file.name})")
    for model in models:
        print(f"model  : {model['label']:<10} {model['path'].name}  sha256 {model['sha256'][:16]}…")

    metrics = {}
    for model in models:
        predictions = predict(model, args, split_file, output)
        metrics[model["label"]] = evaluate(model, args, split_file, predictions, output)

    if lock is not None and not lock.is_file():
        lock.write_text(json.dumps({
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "models": [{"label": m["label"], "file": m["path"].name, "sha256": m["sha256"]} for m in models],
            "mean_ap40_moderate": {k: v["mean_ap40_moderate"] for k, v in metrics.items()},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"eval_val lock written: {lock}")

    speed_files = [p.resolve() for p in args.speed_json]
    if not speed_files and not args.no_speed:
        speed_files = measure_speed(models, args, output)
    speed = load_speed(speed_files, models) if speed_files else {"models": {}, "environments": [], "files": []}
    missing = [m["label"] for m in models if m["format"] == "onnx" and m["label"] not in speed["models"]]
    if speed_files and missing:
        print(f"WARNING: no speed numbers for {missing}: their SHA-256 is not in {speed['files']}")

    summary_md = build_summary(args.split, models, metrics, speed)
    (output / "summary.md").write_text(summary_md, encoding="utf-8")
    import onnxruntime

    (output / "summary.json").write_text(json.dumps({
        "split": args.split, "split_file": str(split_file),
        "models": [{k: str(v) if isinstance(v, Path) else v for k, v in m.items()} for m in models],
        "accuracy": {k: {"class_ap40_moderate": v["class_ap40_moderate"],
                         "mean_ap40_moderate": v["mean_ap40_moderate"]} for k, v in metrics.items()},
        "speed": speed,
        "versions": {"python": platform.python_version(), "onnxruntime": onnxruntime.__version__},
        "platform": platform.platform(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n" + summary_md)
    print(f"saved: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
