#!/usr/bin/env python3
"""Measure FP32 and INT8 ONNX models under identical ONNX Runtime conditions.

    python scripts/benchmark_onnx.py ^
        --models runs/.../best_1248x384_fp32.onnx runs/.../best_1248x384_int8_minmax.onnx ^
        --threads 4 --output runs/benchmark_onnx.json

The first model is the baseline; every other row is also reported relative to it.

Per model it records the five quantities the benchmark notice scores
(accuracy comes from predict_kitti.py + evaluate_kitti.py, not from here):

  parameters  elements of the weight and bias tensors feeding Conv /
              ConvTranspose / Gemm / MatMul (through DequantizeLinear in INT8
              graphs). Head constants (anchors, strides), scales and zero
              points are not counted, so FP32 and INT8 report the same number.
              The export folds BatchNorm into the convolutions, so compare with
              Ultralytics' fused model summary, not the unfused training one.
  FLOPs       2 x MACs of Conv / QLinearConv / ConvTranspose / MatMul / Gemm,
              from ONNX shape inference. Elementwise ops are not counted.
              Quantization does not change it; resolution and pruning do.
  size        bytes of the .onnx file.
  memory      resident memory of a fresh process after loading the session and
              its peak during the timed runs (each model runs in its own
              subprocess so one model's allocations cannot hide in another's).
  latency     session.run on one resident (1, 3, H, W) tensor: warmup, then
              repeats timed with perf_counter. Excludes image decoding,
              letterbox and NMS, exactly like benchmark_model.py.

Use the CPU provider for the INT8 comparison. ONNX Runtime runs QDQ models
with real INT8 kernels on CPU; the CUDA provider mostly dequantizes back to
float, so an INT8 speed-up measured there would be meaningless.
Keep the machine otherwise idle and pin --threads so the run is reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from onnx_common import ROOT, file_sha256, preprocess, static_input

WEIGHTED_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul", "QLinearConv", "QLinearMatMul"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, nargs="+", required=True, help="baseline first")
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="intra-op threads (recorded)")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--image", type=Path, default=None,
                        help="optional real image as input (default: seeded random tensor)")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/benchmark_onnx.json")
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    return parser.parse_args()


# ----------------------------------------------------------------- static analysis
def count_parameters(model) -> int:
    """Count weights/biases of weighted layers only.

    Counting every initializer would also count the Detect head's anchor and
    stride tables (~30k values at 1248x384), which are constants, not learned
    parameters. In a QDQ graph a weight reaches its Conv through
    DequantizeLinear, so each input is traced back to the initializer.
    """
    initializers = {init.name: init for init in model.graph.initializer}
    producer = {output: node for node in model.graph.node for output in node.output}

    def source_initializer(name: str):
        seen = 0
        while name not in initializers and seen < 4:
            node = producer.get(name)
            if node is None or node.op_type not in ("DequantizeLinear", "Identity", "Cast"):
                return None
            name, seen = node.input[0], seen + 1
        return initializers.get(name)

    counted: set[str] = set()
    for node in model.graph.node:
        if node.op_type not in WEIGHTED_OPS:
            continue
        if node.op_type == "QLinearConv":
            candidates = [node.input[3]] + ([node.input[8]] if len(node.input) > 8 else [])
        elif node.op_type == "QLinearMatMul":
            candidates = [node.input[0], node.input[3]]
        else:
            candidates = list(node.input[:3])
        for name in candidates:
            init = source_initializer(name) if name else None
            if init is not None and len(init.dims) > 0:
                counted.add(init.name)
    return int(sum(math.prod(initializers[name].dims) for name in counted))


def count_flops(model) -> dict[str, object]:
    import onnx
    from onnx import shape_inference

    inferred = shape_inference.infer_shapes(model)
    shapes: dict[str, list[int]] = {}
    for value in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = value.type.tensor_type.shape.dim
        shapes[value.name] = [d.dim_value if d.HasField("dim_value") else -1 for d in dims]
    for init in inferred.graph.initializer:
        shapes[init.name] = list(init.dims)

    def known(name: str) -> list[int] | None:
        shape = shapes.get(name)
        return shape if shape and min(shape) > 0 else None

    macs, by_op, unresolved = 0, {}, 0
    for node in inferred.graph.node:
        node_macs = None
        if node.op_type in ("Conv", "ConvTranspose", "QLinearConv"):
            weight = known(node.input[3] if node.op_type == "QLinearConv" else node.input[1])
            out = known(node.output[0])
            if weight and out:
                if node.op_type == "ConvTranspose":
                    inp = known(node.input[0])
                    node_macs = math.prod(inp) * math.prod(weight[1:]) if inp else None
                else:
                    node_macs = math.prod(out) * math.prod(weight[1:])
        elif node.op_type in ("MatMul", "Gemm"):
            a, out = known(node.input[0]), known(node.output[0])
            if a and out:
                transposed = node.op_type == "Gemm" and any(
                    attr.name == "transA" and attr.i for attr in node.attribute)
                k = a[0] if transposed else a[-1]
                node_macs = math.prod(out) * k
        else:
            continue
        if node_macs is None:
            unresolved += 1
            continue
        macs += node_macs
        by_op[node.op_type] = by_op.get(node.op_type, 0) + 2 * node_macs
    return {"flops": int(2 * macs), "gflops": 2 * macs / 1e9, "by_op": by_op,
            "unresolved_nodes": unresolved, "onnx_version": onnx.__version__,
            "scope": "2 x MACs of Conv/QLinearConv/ConvTranspose/MatMul/Gemm; elementwise ops excluded"}


def static_report(path: Path) -> dict[str, object]:
    import onnx

    model = onnx.load(str(path))
    ops: dict[str, int] = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    input_name, shape = static_input(path)
    return {
        "model": str(path.resolve()), "sha256": file_sha256(path),
        "bytes": path.stat().st_size, "mib": path.stat().st_size / 1024 ** 2,
        "input_name": input_name, "input_shape": shape,
        "parameters": count_parameters(model), "flops": count_flops(model),
        "qdq_nodes": ops.get("QuantizeLinear", 0) + ops.get("DequantizeLinear", 0),
        "node_count": len(model.graph.node),
        "metadata": {p.key: p.value for p in model.metadata_props if p.key == "please_ye_quantization"},
    }


# ----------------------------------------------------------------- timed worker
def rss_bytes() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def peak_rss_bytes() -> int:
    import psutil

    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return int(info.peak_wset)
    import resource  # Linux / macOS

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def worker(config: dict) -> dict:
    import numpy as np
    import onnxruntime as ort

    path = Path(config["model"])
    input_name, shape = static_input(path)
    if config["image"]:
        sample = preprocess(Path(config["image"]), shape[3], shape[2])
    else:
        sample = np.random.default_rng(20260917).random(shape, dtype=np.float32)

    before_load = rss_bytes()
    options = ort.SessionOptions()
    options.intra_op_num_threads = config["threads"]
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if config["provider"] == "cuda" \
        else ["CPUExecutionProvider"]
    started = time.perf_counter()
    session = ort.InferenceSession(str(path), options, providers=providers)
    load_seconds = time.perf_counter() - started
    if config["provider"] == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise SystemExit("CUDAExecutionProvider requested but not available (install onnxruntime-gpu)")
    after_load = rss_bytes()

    feed = {input_name: sample}
    for _ in range(config["warmup"]):
        session.run(None, feed)
    durations = []
    for _ in range(config["repeats"]):
        start = time.perf_counter()
        session.run(None, feed)
        durations.append((time.perf_counter() - start) * 1000)
    ordered = sorted(durations)
    mean = statistics.mean(durations)
    return {
        "providers": session.get_providers(), "threads": config["threads"],
        "warmup": config["warmup"], "repeats": config["repeats"],
        "input": "image " + config["image"] if config["image"] else "seeded random tensor",
        "session_load_seconds": load_seconds,
        "memory": {
            "rss_before_load_mib": before_load / 1024 ** 2,
            "rss_after_load_mib": after_load / 1024 ** 2,
            "session_load_delta_mib": (after_load - before_load) / 1024 ** 2,
            "peak_rss_mib": peak_rss_bytes() / 1024 ** 2,
            "scope": "whole worker process (python + numpy + onnxruntime + session)",
        },
        "latency_ms": {
            "mean": mean, "median": statistics.median(durations),
            "stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "min": ordered[0], "max": ordered[-1],
        },
        "images_per_second": 1000 / mean,
        "latency_scope": "session.run on a resident tensor; excludes decode, letterbox, NMS",
        "onnxruntime": ort.__version__,
    }


def run_worker(path: Path, args: argparse.Namespace) -> dict:
    # The worker runs with cwd=scripts/, so every path it receives must be absolute.
    config = {"model": str(path.resolve()), "provider": args.provider, "threads": args.threads,
              "warmup": args.warmup, "repeats": args.repeats,
              "image": str(args.image.resolve()) if args.image else None}
    command = [sys.executable, str(Path(__file__).resolve()), "--models", str(path.resolve()),
               "--_worker", json.dumps(config)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        raise SystemExit(f"benchmark worker failed for {path}:\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------- report
def cpu_name() -> str:
    if sys.platform == "win32":
        return platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    args = parse_args()
    if args._worker:
        print(json.dumps(worker(json.loads(args._worker))))
        return
    if args.warmup < 0 or args.repeats < 2 or args.threads <= 0:
        raise SystemExit("--warmup >= 0, --repeats >= 2 and --threads > 0 are required")
    args.models = [path.resolve() for path in args.models]
    for path in args.models:
        if not path.is_file():
            raise SystemExit(f"Model not found: {path}")
    if args.image is not None and not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")
    shapes = {tuple(static_input(path)[1]) for path in args.models}
    if len(shapes) > 1:
        print(f"NOTE: models have different input shapes {sorted(shapes)}; "
              "latency differences include the resolution change")

    import onnxruntime as ort

    rows = []
    for path in args.models:
        print(f"measuring {path.name} ...", flush=True)
        row = static_report(path)
        row["runtime"] = run_worker(path, args)
        rows.append(row)

    base = rows[0]
    for row in rows:
        row["relative_to_baseline"] = {
            "size": row["bytes"] / base["bytes"],
            "parameters": row["parameters"] / base["parameters"],
            "flops": row["flops"]["flops"] / base["flops"]["flops"] if base["flops"]["flops"] else None,
            "latency_mean": row["runtime"]["latency_ms"]["mean"] / base["runtime"]["latency_ms"]["mean"],
            "speedup": base["runtime"]["latency_ms"]["mean"] / row["runtime"]["latency_ms"]["mean"],
            "peak_rss": row["runtime"]["memory"]["peak_rss_mib"] / base["runtime"]["memory"]["peak_rss_mib"],
        }

    report = {
        "baseline": rows[0]["model"],
        "environment": {
            "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(), "onnxruntime": ort.__version__,
            "provider": args.provider, "threads": args.threads,
        },
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    header = (f"{'model':<44} {'MiB':>7} {'params':>10} {'GFLOPs':>7} "
              f"{'mean ms':>8} {'p95 ms':>8} {'peak MiB':>9} {'speedup':>8}")
    print("\n" + header + "\n" + "-" * len(header))
    for row in rows:
        runtime, rel = row["runtime"], row["relative_to_baseline"]
        print(f"{Path(row['model']).name[:44]:<44} {row['mib']:>7.2f} {row['parameters']:>10,} "
              f"{row['flops']['gflops']:>7.2f} {runtime['latency_ms']['mean']:>8.2f} "
              f"{runtime['latency_ms']['p95']:>8.2f} {runtime['memory']['peak_rss_mib']:>9.1f} "
              f"{rel['speedup']:>7.2f}x")
    print(f"\n{args.provider.upper()} provider, {args.threads} threads, {cpu_name()}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
