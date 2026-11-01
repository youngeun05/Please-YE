#!/usr/bin/env python3
"""Where does the INT8 (QDQ) error of the submitted YOLO11n model arise?  Per-tensor SQNR vs FP32.

    cd <repo root>
    .venv\\Scripts\\python.exe -B <this file> [n_images=16] [chunk=4]

Method: onnxruntime.quantization.qdq_loss_debug
  modify_model_output_intermediate_tensors -> collect_activations -> create_activation_matching
  -> compute_activation_error (qdq_err = SQNR of a tensor's own Q/DQ, xmodel_err = SQNR of the INT8
  model's dequantized tensor vs the FP32 tensor = accumulated error), create_weight_matching ->
  compute_weight_error.

Memory: one image of both augmented models is ~0.9 GB of saved tensors, so the official
collect/match functions run on chunks of `chunk` images and the SQNR over all images is
accumulated from squared norms: 20*log10(||x||/||x-y||) over the concatenation of all images
== 10*log10(sum ||x_i||^2 / sum ||x_i-y_i||^2), i.e. exactly compute_signal_to_quantization_noice_ratio
on the full lists. The first chunk is also run through the official compute_activation_error and
compared (printed as 'equivalence check').

Writes only next to this file (never into the repo): the repo models are copied here first because
qdq_loss_debug's load_model_with_shape_infer writes '<model>-inferred.onnx' next to its input.
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # importing the repo's scripts must not write .pyc files into the repo
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "4")  # another CPU job is running

import csv
import gc
import json
import math
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

REPO = Path(r"C:\Users\PC00\Desktop\이영은\이영은 모커톤")
sys.path.insert(0, str(REPO / "scripts"))

from onnxruntime.quantization import CalibrationDataReader  # noqa: E402
from onnxruntime.quantization.qdq_loss_debug import (  # noqa: E402
    collect_activations,
    compute_activation_error,
    compute_weight_error,
    create_activation_matching,
    create_weight_matching,
    modify_model_output_intermediate_tensors,
)

from onnx_common import file_sha256, static_input  # noqa: E402  (repo helpers)
from quantize_ort import KittiCalibrationReader  # noqa: E402  (repo calibration reader = same preprocess)

OUT = Path(__file__).resolve().parent
FP32_SRC = REPO / "models" / "yolo11n_kitti_1248x384_fp32.onnx"
INT8_SRC = REPO / "models" / "yolo11n_kitti_1248x384_int8.onnx"
FP32 = OUT / "fp32.onnx"
INT8 = OUT / "int8.onnx"
IMAGE_DIR = Path(r"D:\datasets\KITTI\training\image_2")
EPS = float(np.finfo("float").eps)  # same clamp as compute_signal_to_quantization_noice_ratio
BLOCK = re.compile(r"^/model\.(\d+)/")


class Reader(KittiCalibrationReader, CalibrationDataReader):
    """quantize_ort.KittiCalibrationReader (onnx_common.preprocess letterbox) made iterable:
    collect_activations does `for input_d in reader`, which needs CalibrationDataReader.__iter__/__next__."""


def session_options() -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL  # collect_activations' own default
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    return so


def sq(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.dot(x, x))


def sq_diff(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64).ravel() - np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(d, d))


def sqnr_db(signal_sq: float, noise_sq: float) -> float:
    return 20 * math.log10(max(math.sqrt(signal_sq), EPS) / max(math.sqrt(noise_sq), EPS))


def block_of(node_name: str) -> int:
    match = BLOCK.match(node_name or "")
    return int(match.group(1)) if match else -1


def part_of(node_name: str) -> tuple[str, str]:
    block = block_of(node_name)
    if block == 23:
        match = re.search(r"/(cv[23])\.(\d)/", node_name)
        if match:
            return ("head-box(cv2)" if match.group(1) == "cv2" else "head-cls(cv3)"), f"P{3 + int(match.group(2))}"
        return "head-decode(float)", ""
    if 0 <= block <= 10:
        return "backbone", ""
    if 11 <= block <= 22:
        return "neck", ""
    return "input", ""


def fmt(value) -> str:
    return "" if value is None else f"{value:.2f}"


def main() -> None:
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    started = time.perf_counter()
    print(f"onnxruntime {ort.__version__}, onnx {onnx.__version__}, numpy {np.__version__}")

    # 0. byte-identical copies in the scratch folder (see module docstring)
    for src, dst in ((FP32_SRC, FP32), (INT8_SRC, INT8)):
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size or file_sha256(dst) != file_sha256(src):
            shutil.copyfile(src, dst)
    sha = {"fp32": file_sha256(FP32), "int8": file_sha256(INT8)}
    print("sha256", sha)

    ids = [line.strip() for line in (REPO / "splits" / "internal_val.txt").read_text().splitlines()
           if line.strip()][:n_images]
    input_name, shape = static_input(FP32)
    _, _, height, width = shape
    print(f"images: {len(ids)} first ids of internal_val ({ids[0]}..{ids[-1]}), input {input_name} {shape}, "
          f"chunk {chunk}")

    # 1. augmented copies (all float tensors become outputs)
    aug_fp32, aug_int8 = OUT / "fp32_augmented.onnx", OUT / "int8_augmented.onnx"
    modify_model_output_intermediate_tensors(str(FP32), str(aug_fp32))
    modify_model_output_intermediate_tensors(str(INT8), str(aug_int8))
    print(f"augmented models written ({time.perf_counter() - started:.0f}s)")

    # graph bookkeeping on the FP32 graph
    fmodel = onnx.shape_inference.infer_shapes(onnx.load(str(FP32)))
    shapes = {}
    for vi in list(fmodel.graph.value_info) + list(fmodel.graph.output) + list(fmodel.graph.input):
        shapes[vi.name] = [d.dim_value if d.HasField("dim_value") else -1 for d in vi.type.tensor_type.shape.dim]
    producer, topo = {}, {input_name: -1}
    consumers = defaultdict(list)
    for index, node in enumerate(fmodel.graph.node):
        for name in node.output:
            producer[name], topo[name] = node, index
        for name in node.input:
            consumers[name].append(node)
    graph_outputs = [o.name for o in fmodel.graph.output]
    block_outputs, block_inputs = defaultdict(list), defaultdict(set)
    for node in fmodel.graph.node:
        block = block_of(node.name)
        for name in node.output:
            if name in graph_outputs or any(block_of(c.name) != block for c in consumers[name]):
                block_outputs[block].append(name)
        for name in node.input:
            if name == input_name or (name in producer and block_of(producer[name].name) != block):
                block_inputs[block].add(name)

    # quantization parameters of every activation QuantizeLinear in the INT8 model
    qmodel = onnx.load(str(INT8))
    qinit = {init.name: init for init in qmodel.graph.initializer}
    qparams = {}
    for node in qmodel.graph.node:
        if node.op_type == "QuantizeLinear" and node.input[0] not in qinit and node.input[1] in qinit:
            scale = numpy_helper.to_array(qinit[node.input[1]])
            zp = numpy_helper.to_array(qinit[node.input[2]]) if len(node.input) > 2 and node.input[2] in qinit else None
            if scale.size == 1:
                info = np.iinfo(zp.dtype) if zp is not None else np.iinfo(np.uint8)
                z = int(zp) if zp is not None else 0
                qparams[node.input[0]] = (float(scale), z, (info.min - z) * float(scale), (info.max - z) * float(scale))
    weight_dq_total = sum(1 for n in qmodel.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in qinit)
    del qmodel

    # 2. activations, chunk by chunk
    acc = defaultdict(lambda: defaultdict(float))
    matched_names, without_float, common = None, set(), None
    official_chunk0, equivalence = None, None
    fake_out0 = []
    out_name = graph_outputs[0]
    out_shape = shapes[out_name]
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        tick = time.perf_counter()
        q_acts = collect_activations(str(aug_int8), Reader(IMAGE_DIR, part, input_name, width, height),
                                     session_options=session_options())
        f_acts = collect_activations(str(aug_fp32), Reader(IMAGE_DIR, part, input_name, width, height),
                                     session_options=session_options())
        match = create_activation_matching(q_acts, f_acts)
        if matched_names is None:
            matched_names = sorted(match)
            without_float = {k for k, v in match.items() if "float" not in v}
            common = sorted(set(f_acts) & set(q_acts))
            float_names, qdq_names = set(f_acts), set(q_acts)
            official_chunk0 = compute_activation_error({k: v for k, v in match.items() if "float" in v})
        for name, m in match.items():
            a = acc[name]
            qp = qparams.get(name)
            for pre, post in zip(m["pre_qdq"], m["post_qdq"], strict=True):
                a["pre_sq"] += sq(pre)
                a["qdq_noise_sq"] += sq_diff(pre, post)
                if qp is not None:
                    s, _, lo, hi = qp
                    a["clip_n"] += float(np.count_nonzero((pre < lo - s / 2) | (pre > hi + s / 2)))
                    a["n"] += float(pre.size)
            if "float" in m:
                for fl, post in zip(m["float"], m["post_qdq"], strict=True):
                    a["float_sq"] += sq(fl)
                    a["x_noise_sq"] += sq_diff(fl, post)
        for name in common:  # direct comparison of same-named tensors (also covers the float decode tail)
            a = acc[name]
            for fl, qv in zip(f_acts[name], q_acts[name], strict=True):
                a["direct_sq"] += sq(fl)
                a["direct_noise_sq"] += sq_diff(fl, qv)
        for fl, qv in zip(f_acts[out_name], q_acts[out_name], strict=True):
            f2, q2 = fl.reshape(out_shape[1], -1), qv.reshape(out_shape[1], -1)
            fake_out0.append(q2.copy())
            for key, rows in (("box", slice(0, 4)), ("cls", slice(4, None))):
                acc[f"{out_name}[{key}]"]["direct_sq"] += sq(f2[rows])
                acc[f"{out_name}[{key}]"]["direct_noise_sq"] += sq_diff(f2[rows], q2[rows])
        if start == 0:
            diffs = []
            for name, err in official_chunk0.items():
                a = acc[name]
                diffs.append(abs(err["qdq_err"] - sqnr_db(a["pre_sq"], a["qdq_noise_sq"])))
                diffs.append(abs(err["xmodel_err"] - sqnr_db(a["float_sq"], a["x_noise_sq"])))
            equivalence = {"tensors": len(official_chunk0), "max_abs_diff_db": max(diffs)}
            print(f"equivalence check (chunk 0, official compute_activation_error vs accumulated): {equivalence}")
        del q_acts, f_acts, match
        gc.collect()
        print(f"chunk {start // chunk + 1}/{math.ceil(len(ids) / chunk)} done ({time.perf_counter() - tick:.0f}s)",
              flush=True)

    # 3. deployed INT8 kernels (default graph optimizations = fused QLinear ops) vs FP32 and vs fake-quant
    real = defaultdict(float)
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 4, 1
    fs = ort.InferenceSession(str(FP32), so, providers=["CPUExecutionProvider"])
    qs = ort.InferenceSession(str(INT8), so, providers=["CPUExecutionProvider"])
    for i, feed in enumerate(Reader(IMAGE_DIR, ids, input_name, width, height)):
        f2 = fs.run(None, feed)[0].reshape(out_shape[1], -1)
        q2 = qs.run(None, feed)[0].reshape(out_shape[1], -1)
        for key, rows in (("all", slice(None)), ("box", slice(0, 4)), ("cls", slice(4, None))):
            real[f"{key}_sq"] += sq(f2[rows])
            real[f"{key}_noise"] += sq_diff(f2[rows], q2[rows])
            real[f"{key}_fake_sq"] += sq(fake_out0[i][rows])
            real[f"{key}_fake_noise"] += sq_diff(fake_out0[i][rows], q2[rows])
    deployed = {key: {"int8_kernels_vs_fp32_db": round(sqnr_db(real[f"{key}_sq"], real[f"{key}_noise"]), 2),
                      "int8_kernels_vs_qdq_simulation_db":
                          round(sqnr_db(real[f"{key}_fake_sq"], real[f"{key}_fake_noise"]), 2)}
                for key in ("all", "box", "cls")}
    del fs, qs

    # 4. weights
    wmatch = create_weight_matching(str(FP32), str(INT8))
    scalar_weights = set()
    for name, wm in wmatch.items():
        # ORT helper bug: a 0-d initializer dequantizes to a numpy scalar, which compute_weight_error's
        # SQNR (len()/concatenate) cannot take; 1-element arrays give the identical SQNR.
        if np.asarray(wm["float"]).size == 1:
            scalar_weights.add(name)
        wm["float"] = np.atleast_1d(np.asarray(wm["float"]))
        wm["dequantized"] = np.atleast_1d(np.asarray(wm["dequantized"]))
    werr = compute_weight_error(wmatch)
    weight_user = {}
    for node in fmodel.graph.node:
        for name in node.input:
            weight_user.setdefault(name, node.name)
    weight_rows = []
    for name, value in sorted(werr.items(), key=lambda kv: kv[1]):
        arr = wmatch[name]["float"]
        kind = "scalar" if name in scalar_weights else ("bias" if arr.ndim == 1 else "weight")
        weight_rows.append({"weight": name, "kind": kind,
                            "node": weight_user.get(name, ""), "shape": "x".join(map(str, arr.shape)),
                            "sqnr_db": round(value, 2), "block": block_of(weight_user.get(name, ""))})

    # 5. tables
    rows = []
    names = sorted(set(matched_names) | set(common), key=lambda n: (topo.get(n, 10 ** 6), n))
    for name in names:
        a = acc[name]
        node = producer.get(name)
        part_name, scale = part_of(node.name if node else "")
        qp = qparams.get(name)
        quantized = name in matched_names
        rows.append({
            "topo": topo.get(name, ""), "tensor": name, "block": block_of(node.name) if node else -1,
            "part": part_name, "scale": scale, "producer_op": node.op_type if node else "graph input",
            "producer_node": node.name if node else "", "shape": "x".join(map(str, shapes.get(name, []))),
            "qdq_in_int8": "yes" if quantized else "no",
            "qdq_err_db": round(sqnr_db(a["pre_sq"], a["qdq_noise_sq"]), 2) if quantized else None,
            "xmodel_err_db": round(sqnr_db(a["float_sq"], a["x_noise_sq"]), 2)
            if quantized and name not in without_float else None,
            "xmodel_pre_qdq_db": round(sqnr_db(a["direct_sq"], a["direct_noise_sq"]), 2) if a["direct_sq"] else None,
            "act_scale": qp[0] if qp else None, "act_zero_point": qp[1] if qp else None,
            "act_range_lo": round(qp[2], 4) if qp else None, "act_range_hi": round(qp[3], 4) if qp else None,
            "clipped_frac": a["clip_n"] / a["n"] if a["n"] else None,
            "block_output": "yes" if any(name in v for v in block_outputs.values()) else "",
        })
    by_name = {r["tensor"]: r for r in rows}
    with (OUT / "per_tensor_sqnr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "weights_sqnr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weight_rows[0]))
        writer.writeheader()
        writer.writerows(weight_rows)

    def acc_err(name: str):
        r = by_name.get(name)
        if r is None:
            return None
        return r["xmodel_err_db"] if r["xmodel_err_db"] is not None else r["xmodel_pre_qdq_db"]

    block_rows = []
    for block in sorted(b for b in block_outputs if b >= 0):
        outs = block_outputs[block]
        ins = sorted(block_inputs[block])
        out_err = [acc_err(n) for n in outs]
        in_err = [acc_err(n) for n in ins]
        in_min = min((e for e in in_err if e is not None), default=None)
        out_min = min((e for e in out_err if e is not None), default=None)
        block_rows.append({
            "block": block, "first_op": next(n.op_type for n in fmodel.graph.node if block_of(n.name) == block),
            "output_tensors": ";".join(outs), "xmodel_err_out_db": out_min,
            "input_tensors": ";".join(ins), "xmodel_err_in_min_db": in_min,
            "delta_db": round(out_min - in_min, 2) if out_min is not None and in_min is not None else None,
            "n_tensors_in_block": sum(1 for r in rows if r["block"] == block),
            "min_qdq_err_in_block_db": min((r["qdq_err_db"] for r in rows
                                           if r["block"] == block and r["qdq_err_db"] is not None), default=None),
        })
    with (OUT / "block_sqnr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(block_rows[0]))
        writer.writeheader()
        writer.writerows(block_rows)

    head = {}
    for i, scale in enumerate(("P3", "P4", "P5")):
        for branch, tag in (("box", f"/model.23/cv2.{i}/cv2.{i}.2/Conv_output_0"),
                            ("cls", f"/model.23/cv3.{i}/cv3.{i}.2/Conv_output_0")):
            r = by_name.get(tag, {})
            head[f"{scale}_{branch}"] = {"tensor": tag, "xmodel_err_db": r.get("xmodel_err_db"),
                                         "qdq_err_db": r.get("qdq_err_db"), "clipped_frac": r.get("clipped_frac"),
                                         "act_range": [r.get("act_range_lo"), r.get("act_range_hi")]}
    output0 = {"all": acc_err(out_name),
               "box_rows": round(sqnr_db(acc[f"{out_name}[box]"]["direct_sq"], acc[f"{out_name}[box]"]["direct_noise_sq"]), 2),
               "cls_rows": round(sqnr_db(acc[f"{out_name}[cls]"]["direct_sq"], acc[f"{out_name}[cls]"]["direct_noise_sq"]), 2),
               "qdq_in_int8": by_name[out_name]["qdq_in_int8"]}

    quantized_rows = [r for r in rows if r["xmodel_err_db"] is not None]
    lowest = sorted(quantized_rows, key=lambda r: r["xmodel_err_db"])[:10]
    lowest_qdq = sorted((r for r in rows if r["qdq_err_db"] is not None), key=lambda r: r["qdq_err_db"])[:10]
    unquantized = [r["tensor"] for r in rows if r["qdq_in_int8"] == "no"]
    summary = {
        "images": ids, "sha256": sha, "onnxruntime": ort.__version__, "chunk": chunk,
        "float_tensors_saved": len(float_names), "qdq_tensors_saved": len(qdq_names),
        "common_names": len(common), "float_only_names": sorted(float_names - qdq_names),
        "matched_qdq_pairs": len(matched_names), "matched_with_float": len(matched_names) - len(without_float),
        "matched_without_float": sorted(without_float), "unquantized_common_tensors": unquantized,
        "equivalence_check_chunk0": equivalence,
        "weights_matched": len(wmatch), "weight_dq_nodes_in_int8": weight_dq_total,
        "lowest_xmodel": [{k: r[k] for k in ("tensor", "producer_node", "block", "part", "xmodel_err_db", "qdq_err_db")}
                          for r in lowest],
        "lowest_qdq": [{k: r[k] for k in ("tensor", "block", "part", "qdq_err_db", "xmodel_err_db", "clipped_frac")}
                       for r in lowest_qdq],
        "blocks": block_rows, "head_last_convs": head, "output0_qdq_simulation": output0,
        "output0_deployed_int8_kernels": deployed,
        "lowest_weights": [r for r in weight_rows if r["kind"] == "weight"][:5],
        "lowest_biases": [r for r in weight_rows if r["kind"] == "bias"][:3],
        "scalar_initializers": [r for r in weight_rows if r["kind"] == "scalar"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # console report
    print(f"\nmatched QDQ pairs: {summary['matched_qdq_pairs']} (with FP32 counterpart: {summary['matched_with_float']}); "
          f"FP32 float tensors {len(float_names)}, same-name in INT8 {len(common)}, unquantized {len(unquantized)}")
    print("\n10 lowest xmodel_err (accumulated):")
    for r in lowest:
        print(f"  {r['xmodel_err_db']:7.2f} dB  qdq {fmt(r['qdq_err_db']):>6}  [{r['block']:>2} {r['part']}]  {r['tensor']}")
    print("\n10 lowest qdq_err (own quantization):")
    for r in lowest_qdq:
        print(f"  {r['qdq_err_db']:7.2f} dB  x {fmt(r['xmodel_err_db']):>6}  clip {r['clipped_frac'] or 0:.2e}  "
              f"[{r['block']:>2} {r['part']}]  {r['tensor']}")
    print("\nblock outputs (xmodel_err, dB):")
    for b in block_rows:
        print(f"  model.{b['block']:<2} {b['first_op']:<8} out {fmt(b['xmodel_err_out_db']):>6}  in(min) "
              f"{fmt(b['xmodel_err_in_min_db']):>6}  delta {fmt(b['delta_db']):>6}  minQDQ {fmt(b['min_qdq_err_in_block_db']):>6}"
              f"  {b['output_tensors']}")
    print("\nhead last convs:", json.dumps(head, indent=1))
    print("output0 (QDQ simulation):", output0)
    print("output0 (deployed INT8 kernels):", deployed)
    print("\nweights matched", len(wmatch), "of", weight_dq_total, "DQ-on-initializer nodes")
    for r in summary["lowest_weights"] + summary["lowest_biases"] + summary["scalar_initializers"]:
        print(f"  {r['sqnr_db']:7.2f} dB  {r['kind']:<6} {r['shape']:<14} {r['weight']}  ({r['node']})")
    print(f"\ndone in {summary['seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
