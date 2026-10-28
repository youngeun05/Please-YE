#!/usr/bin/env python3
"""Experiment A: YOLO11n at the notice's light preset 640x192, kept fully separate
from the submitted 1248x384 results.

    python scripts/run_640.py --kitti-root D:/datasets/KITTI --affinity 0-15

One command runs every step and skips the ones already finished, so it can be
re-run after an interruption:

  1. train      YOLO11n, same recipe as the 1248 baseline, only --imgsz 640
                -> runs/baseline/yolo11n_exact3_640/
  2. export     FP32 ONNX, input 1x3x192x640  -> best_640x192_fp32.onnx
  3. quantize   INT8 percentile, same settings and the same 1,024 calibration images
  4. speed      benchmark_onnx.py on four models in one session, twice, CPU pinned
                (1248 FP32, 1248 INT8, 640 FP32, 640 INT8) -> runs/benchmark_640_pcores_run{1,2}.json
  5. internal   accuracy on internal_val -> runs/benchmark/internal_val_640/
  6. register   writes PREREGISTRATION.json before eval_val is scored: the 640
                result will be reported whatever it is, and it does not replace
                the submitted model
  7. eval_val   accuracy on eval_val -> runs/benchmark/eval_val_640/ (its own lock)
  8. package    models_640/ (files, SHA-256 manifest, zip)
  9. results    copies the summaries to results/640/ for git

In every table the baseline is the ORIGINAL submitted baseline (1248 FP32), so
the change column shows the combined effect of resolution + INT8; the 640 FP32
column separates the two effects.

Protection of the existing results: before step 1 the script hashes every
existing result file (results/, models/, runs/benchmark/{internal_val,eval_val}/,
the 10/14 speed files, the 1248 weights folder). Nothing here writes to those
paths; at the end the hashes are compared again and any change is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
RUN = "yolo11n_exact3_640"
W640 = ROOT / "runs/baseline" / RUN / "weights"
PT640 = W640 / "best.pt"
FP32_640 = W640 / "best_640x192_fp32.onnx"
INT8_640 = W640 / "best_640x192_int8_percentile.onnx"
FP32_1248 = ROOT / "models/yolo11n_kitti_1248x384_fp32.onnx"
INT8_1248 = ROOT / "models/yolo11n_kitti_1248x384_int8.onnx"
SPEED = [ROOT / "runs/benchmark_640_pcores_run1.json", ROOT / "runs/benchmark_640_pcores_run2.json"]
BENCH = {"internal_val": ROOT / "runs/benchmark/internal_val_640", "eval_val": ROOT / "runs/benchmark/eval_val_640"}
RESULTS = ROOT / "results/640"
MODELS_OUT = ROOT / "models_640"
PROTECTED = [ROOT / "results", ROOT / "models", ROOT / "runs/benchmark/internal_val", ROOT / "runs/benchmark/eval_val",
             ROOT / "runs/benchmark_onnx_pcores.json", ROOT / "runs/benchmark_onnx_pcores_run2.json",
             ROOT / "runs/baseline/yolo11n_exact3_1248/weights"]
STEPS = ("train", "export", "quantize", "speed", "internal", "register", "eval_val", "package", "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--affinity", default=None,
                        help="logical CPUs for the speed step, e.g. 0-15 (PC00 P-cores). Default: no pinning")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--cache", default="ram", choices=("ram", "none"),
                        help="RAM cache of 640-wide images (~3 GB): removes the PNG-decoding bottleneck")
    parser.add_argument("--stop-after", choices=STEPS, default=None, help="stop after this step")
    parser.add_argument("--skip-eval-val", action="store_true", help="stop before registering and scoring eval_val")
    return parser.parse_args()


# ----------------------------------------------------------------- helpers
def banner(step: str, text: str) -> None:
    print(f"\n{'=' * 72}\n[{step}] {text}\n{'=' * 72}", flush=True)


def call(args: list[str]) -> None:
    command = [sys.executable, *args]
    print("$ python " + " ".join(args), flush=True)
    if subprocess.run(command, cwd=str(ROOT)).returncode != 0:
        raise SystemExit(f"step failed: {' '.join(args[:1])}")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot() -> dict[str, str]:
    files: dict[str, str] = {}
    for base in PROTECTED:
        paths = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else []
        for path in paths:
            if "predictions_" in path.as_posix() or path.as_posix().startswith((ROOT / "results/640").as_posix()):
                continue  # thousands of small files; their summaries and locks are covered
            files[str(path.relative_to(ROOT))] = sha(path)
    return files


def parse_cpus(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-", 1)
            cpus.extend(range(int(low), int(high) + 1))
        else:
            cpus.append(int(part))
    return cpus


def benchmark_args(split: str, speed: list[Path]) -> list[str]:
    args = ["scripts/run_benchmark.py", "--kitti-root", str(KITTI), "--split", split,
            "--baseline", str(FP32_1248), "--optimized", str(INT8_640),
            "--also", f"y640_fp32={FP32_640}", "--also", f"y1248_int8={INT8_1248}",
            "--output", str(BENCH[split])]
    for path in speed:
        args += ["--speed-json", str(path)]
    if split == "eval_val":
        args.append("--confirm-final")
    return args


# ----------------------------------------------------------------- main
def main() -> None:
    global KITTI
    args = parse_args()
    KITTI = args.kitti_root.resolve()
    for path in (FP32_1248, INT8_1248):
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run scripts/package_models.py first (실행가이드1028.md 1단계).")
    if not (KITTI / "training/image_2").is_dir():
        raise SystemExit(f"KITTI not found: {KITTI}")

    started = time.perf_counter()
    before = snapshot()
    print(f"protected existing files: {len(before)} (hashed; checked again at the end)")

    def done(step: str) -> bool:
        if args.stop_after == step:
            finish(before, started, stopped=step)
            raise SystemExit(0)
        return False

    # 1. train
    if PT640.is_file():
        banner("1 train", f"skip: {PT640.relative_to(ROOT)} exists")
    else:
        if (ROOT / "runs/baseline" / RUN).exists() and any((ROOT / "runs/baseline" / RUN).iterdir()):
            raise SystemExit(f"runs/baseline/{RUN} exists but has no best.pt (interrupted training?). "
                             "Rename or delete that folder, then run again.")
        banner("1 train", f"YOLO11n @ 640, {args.epochs} epochs, same recipe as the 1248 baseline (GPU)")
        call(["scripts/train_baseline.py", "--model", "yolo11n.pt", "--imgsz", "640", "--epochs", str(args.epochs),
              "--cache", args.cache, "--name", RUN])
        if not PT640.is_file():
            raise SystemExit(f"training finished without {PT640}")
    done("train")

    # 2. export
    if FP32_640.is_file():
        banner("2 export", f"skip: {FP32_640.name} exists")
    else:
        banner("2 export", "FP32 ONNX 640x192 + PyTorch parity check")
        call(["scripts/export_onnx.py", "--weights", str(PT640), "--imgsz", "640x192", "--kitti-root", str(KITTI)])
    done("export")

    # 3. quantize
    if INT8_640.is_file():
        banner("3 quantize", f"skip: {INT8_640.name} exists")
    else:
        banner("3 quantize", "INT8 percentile, 1,024 calibration images (same settings as the submitted model)")
        call(["scripts/quantize_ort.py", "--model", str(FP32_640), "--kitti-root", str(KITTI), "--method", "percentile"])
    done("quantize")

    # 4. speed
    banner("4 speed", "4 models in one session, measured twice (close other programs)")
    import psutil

    process = psutil.Process()
    original = process.cpu_affinity() if hasattr(process, "cpu_affinity") else None
    if args.affinity:
        process.cpu_affinity(parse_cpus(args.affinity))  # child processes inherit this mask
        print(f"pinned to logical CPUs {args.affinity}")
    try:
        for path in SPEED:
            if path.is_file():
                print(f"skip: {path.name} exists")
                continue
            call(["scripts/benchmark_onnx.py", "--models", str(FP32_1248), str(INT8_1248), str(FP32_640),
                  str(INT8_640), "--threads", str(args.threads), "--output", str(path)])
    finally:
        if args.affinity and original is not None:
            process.cpu_affinity(original)
    done("speed")

    # 5. internal_val
    banner("5 internal", "accuracy on internal_val (4 models)")
    call(benchmark_args("internal_val", SPEED))
    done("internal")
    if args.skip_eval_val:
        finish(before, started, stopped="internal (--skip-eval-val)")
        return

    # 6. pre-registration
    BENCH["eval_val"].mkdir(parents=True, exist_ok=True)
    prereg = BENCH["eval_val"] / "PREREGISTRATION.json"
    if prereg.is_file():
        banner("6 register", "skip: PREREGISTRATION.json exists")
    else:
        banner("6 register", "commit to reporting the 640 result before eval_val is scored")
        internal = json.loads((BENCH["internal_val"] / "summary.json").read_text(encoding="utf-8"))
        prereg.write_text(json.dumps({
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "statement": ("Experiment A (YOLO11n 640x192, INT8 percentile) is reported in the final report as an "
                          "additional operating point whatever its eval_val score is. It does not replace the "
                          "submitted model (YOLO11n 1248x384 INT8 percentile), whose eval_val result was locked on "
                          "2026-10-28. No model, setting or threshold is changed after this eval_val run."),
            "models": {m["label"]: {"file": Path(m["path"]).name, "sha256": m["sha256"]} for m in internal["models"]},
            "internal_val_mean_ap40": {k: v["mean_ap40_moderate"] for k, v in internal["accuracy"].items()},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"written: {prereg.relative_to(ROOT)}")
    done("register")

    # 7. eval_val
    banner("7 eval_val", "accuracy on eval_val (4 models, separate lock in eval_val_640)")
    call(benchmark_args("eval_val", SPEED))
    done("eval_val")

    # 8. package
    banner("8 package", "models_640/ (separate from the submitted models/)")
    call(["scripts/package_models.py", "--run", RUN, "--imgsz", "640x192", "--method", "percentile",
          "--output", str(MODELS_OUT)])
    done("package")

    # 9. results for git
    banner("9 results", "copy summaries to results/640/")
    RESULTS.mkdir(parents=True, exist_ok=True)
    copies = {BENCH["internal_val"] / "summary.md": "internal_val_summary.md",
              BENCH["internal_val"] / "summary.json": "internal_val_summary.json",
              BENCH["eval_val"] / "summary.md": "eval_val_summary.md",
              BENCH["eval_val"] / "summary.json": "eval_val_summary.json",
              BENCH["eval_val"] / "FINAL_LOCK.json": "eval_val_FINAL_LOCK.json",
              prereg: "PREREGISTRATION.json",
              SPEED[0]: "benchmark_640_pcores_run1.json", SPEED[1]: "benchmark_640_pcores_run2.json",
              MODELS_OUT / "MODELS.md": "MODELS.md", MODELS_OUT / "manifest.json": "manifest.json"}
    for source, name in copies.items():
        shutil.copy2(source, RESULTS / name)
        print(f"  {source.relative_to(ROOT)} -> results/640/{name}")
    finish(before, started)


def finish(before: dict[str, str], started: float, stopped: str | None = None) -> None:
    after = snapshot()
    changed = sorted(k for k in before if after.get(k) != before[k])
    added = sorted(set(after) - set(before))
    print(f"\n{'=' * 72}")
    if stopped:
        print(f"stopped after step: {stopped}")
    if changed:
        print(f"WARNING: {len(changed)} existing result files changed or disappeared:")
        for name in changed:
            print(f"  {name}")
    else:
        print(f"existing results unchanged: {len(before)} protected files have the same SHA-256")
    if added:
        print(f"note: {len(added)} new files appeared inside protected folders: {added[:5]}")
    print(f"elapsed {(time.perf_counter() - started) / 60:.1f} min")
    if not stopped:
        print("next: git add results/640 && git commit -m \"Add 640x192 experiment\" && git push (실행가이드_640.md)")


KITTI: Path

if __name__ == "__main__":
    main()
