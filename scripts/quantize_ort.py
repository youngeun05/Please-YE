#!/usr/bin/env python3
"""INT8 post-training static quantization of the FP32 ONNX model (ONNX Runtime).

    python scripts/quantize_ort.py ^
        --model runs/baseline/yolo11n_exact3_1248/weights/best_1248x384_fp32.onnx ^
        --kitti-root D:/datasets/KITTI

Design decisions (each one is something the report should state):

* Calibration data = splits/calibration.txt (1,024 images, a subset of train).
  The script refuses to run if any calibration id is in internal_val or in the
  official eval_val split, or is not in train.
* Preprocessing = onnx_common.preprocess, identical to Ultralytics' letterbox at
  the exported shape, so ranges are measured on the inference distribution.
* Format = QDQ (QuantizeLinear/DequantizeLinear pairs). It is the format ONNX
  Runtime CPU runs with real INT8 kernels and the one most toolchains read.
* Weights INT8 symmetric, per output channel by default; activations UINT8
  asymmetric. Per-channel weights matter for conv nets whose channels have very
  different ranges after BN folding.
* The box-decoding tail of the Detect head stays float by default (DFL,
  Sigmoid, Concat, the anchor arithmetic); the head's own convolutions are INT8.
  Reason: the final Concat packs pixel coordinates (0..~1300) and class scores
  (0..1) into one tensor. A single UINT8 scale over that range is ~5 px per
  step, so every score rounds to 0 and the model detects nothing (reproduced on
  a small model with the same structure: scores 0.07..0.47 became all 0; with
  the tail in float they stayed 0.07..0.46). This is also how Hailo deploys
  YOLO: convolutions on the NPU, decoding/NMS on the host CPU.
  --quantize-head quantizes the tail too; keep it only as the control
  experiment that shows this failure.

Calibration method: minmax (default), entropy or percentile.
ONNX Runtime keeps every intermediate tensor of every image it has seen in RAM
until it merges them; at 1248x384 that is tens of MB per image, far too much
for 1,024 images at once. The reader therefore feeds the calibrator in chunks
of --stride images (ONNX Runtime's CalibStridedMinMax option, which works for
all three methods): min/max ranges and histograms are merged after each chunk,
so memory stays flat while every calibration image still contributes.
(Its other option, CalibMaxIntermediateOutputs, discards data in onnxruntime
1.24 and is deliberately not used.)
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import tempfile
import time
from pathlib import Path

from baseline_common import check_splits, read_ids
from onnx_common import ROOT, file_sha256, ids_sha256, preprocess, static_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="FP32 ONNX from export_onnx.py")
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=ROOT / "splits/calibration.txt")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: model name with _fp32 replaced by _int8 (plus method/head tags)")
    parser.add_argument("--method", choices=("minmax", "entropy", "percentile"), default="minmax")
    parser.add_argument("--percentile", type=float, default=99.999, help="only for --method percentile")
    parser.add_argument("--per-tensor", action="store_true", help="per-tensor weights instead of per-channel")
    parser.add_argument("--quantize-head", action="store_true",
                        help="ALSO quantize the Detect head's decoding tail (control experiment only: "
                             "scores collapse to 0, see the module docstring)")
    parser.add_argument("--exclude-nodes", nargs="*", default=[], help="extra node names to keep in float")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N calibration ids (sorted), for quick trials")
    parser.add_argument("--stride", type=int, default=16,
                        help="images per calibration chunk; bounds RAM (lowered to a divisor of the count)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="skip onnxruntime quant_pre_process (shape inference + graph optimization)")
    return parser.parse_args()


def calibration_ids(path: Path, limit: int | None) -> list[str]:
    """Load calibration ids and prove they cannot leak evaluation data."""
    train, internal_val, official = check_splits(ROOT)
    ids = read_ids(path)
    if not set(ids) <= train:
        raise SystemExit(f"Leakage guard: {len(set(ids) - train)} calibration ids are not in splits/train.txt")
    if set(ids) & official:
        raise SystemExit("Leakage guard: calibration overlaps the official eval_val split")
    if set(ids) & internal_val:
        raise SystemExit("Leakage guard: calibration overlaps internal_val")
    ids = sorted(ids)
    if limit is not None:
        if limit <= 0:
            raise SystemExit("--limit must be positive")
        ids = ids[:limit]
    return ids


class KittiCalibrationReader:
    """onnxruntime CalibrationDataReader over KITTI PNGs, one image per batch."""

    def __init__(self, image_dir: Path, ids: list[str], input_name: str, width: int, height: int):
        self.paths = [image_dir / f"{image_id}.png" for image_id in ids]
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} calibration images missing, e.g. {missing[:3]}")
        self.input_name, self.width, self.height = input_name, width, height
        self.index, self.end = 0, len(self.paths)
        self.started = time.perf_counter()

    def __len__(self) -> int:
        return len(self.paths)

    def set_range(self, start_index: int, end_index: int) -> None:
        """Called by ONNX Runtime before each chunk when CalibStridedMinMax is set."""
        self.index, self.end = start_index, min(end_index, len(self.paths))

    def get_next(self):
        if self.index >= self.end:
            return None
        tensor = preprocess(self.paths[self.index], self.width, self.height)
        self.index += 1
        if self.index % 128 == 0 or self.index == len(self.paths):
            elapsed = time.perf_counter() - self.started
            print(f"  calibrated {self.index}/{len(self.paths)} images ({elapsed:.0f}s)", flush=True)
        return {self.input_name: tensor}

    def rewind(self):
        self.index, self.end = 0, len(self.paths)
        self.started = time.perf_counter()


def head_decode_nodes(model_path: Path) -> list[str]:
    """Nodes of the last '/model.N/' module that are not its cv2/cv3 conv branches.

    In Ultralytics ONNX exports the Detect head is the highest-numbered module.
    Its cv2 (box) and cv3 (class) branches are ordinary convolutions and are
    kept INT8; everything else in that module (DFL, Sigmoid, Concat, Split,
    anchor arithmetic) decodes boxes and is where INT8 error lands directly on
    coordinates.
    """
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    pattern = re.compile(r"^/model\.(\d+)/")
    indices = [int(match.group(1)) for node in model.graph.node if (match := pattern.match(node.name))]
    if not indices:
        raise SystemExit("Detect head: node names do not follow the Ultralytics '/model.N/' scheme; "
                         "pass the decoding node names with --exclude-nodes plus --quantize-head")
    prefix = f"/model.{max(indices)}/"
    conv_branch = re.compile(r"/(one2one_)?cv[23]\.")  # box/class conv branches (YOLO11 and YOLO26 names)
    return [node.name for node in model.graph.node
            if node.name.startswith(prefix) and not conv_branch.search(node.name)]


def default_output(model: Path, args: argparse.Namespace) -> Path:
    stem = model.stem.replace("_fp32", "")
    tags = ["int8", args.method]
    if args.per_tensor:
        tags.append("pertensor")
    if args.quantize_head:
        tags.append("fullhead")
    if args.limit:
        tags.append(f"n{args.limit}")
    return model.with_name(f"{stem}_{'_'.join(tags)}.onnx")


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    ids = calibration_ids(args.calibration, args.limit)
    input_name, shape = static_input(args.model)
    _, _, height, width = shape
    output = args.output or default_output(args.model, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    stride = max(d for d in range(1, min(args.stride, len(ids)) + 1) if len(ids) % d == 0)
    if stride != args.stride:
        print(f"stride      : {args.stride} does not divide {len(ids)}; using {stride}")
    print(f"model       : {args.model}  input {input_name} {shape}")
    print(f"calibration : {len(ids)} images from {args.calibration.name} (sha256 {ids_sha256(ids)[:12]})")
    print(f"method      : {args.method}, weights {'per-tensor' if args.per_tensor else 'per-channel'} INT8, "
          "activations UINT8, QDQ")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as workdir:
        source = args.model
        preprocessed = False
        if not args.skip_preprocess:
            from onnxruntime.quantization.shape_inference import quant_pre_process

            candidate = Path(workdir) / "preprocessed.onnx"
            try:
                quant_pre_process(str(args.model), str(candidate))
                source, preprocessed = candidate, True
            except Exception as error:  # the original graph is still valid input
                print(f"WARNING: quant_pre_process failed ({type(error).__name__}: {error}); "
                      "quantizing the exported graph as is")

        excluded = list(args.exclude_nodes)
        if args.quantize_head:
            print("head        : decoding tail quantized too (--quantize-head, control experiment)")
        else:
            head = head_decode_nodes(source)
            excluded.extend(head)
            print(f"head        : {len(head)} decoding nodes stay float (convolutions are INT8)")

        reader = KittiCalibrationReader(args.kitti_root / "training" / "image_2", ids, input_name, width, height)
        method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
                  "percentile": CalibrationMethod.Percentile}[args.method]
        extra = {"CalibStridedMinMax": stride}
        if args.method == "percentile":
            extra["CalibPercentile"] = args.percentile
        quantize_static(
            model_input=str(source),
            model_output=str(output),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=not args.per_tensor,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            nodes_to_exclude=excluded,
            extra_options=extra,
        )
    seconds = time.perf_counter() - started

    # Carry Ultralytics metadata (names, stride, imgsz) over so YOLO() can load the
    # INT8 file exactly like the FP32 one, and record how it was quantized.
    fp32 = onnx.load(str(args.model), load_external_data=False)
    int8 = onnx.load(str(output))
    settings = {
        "method": args.method, "percentile": args.percentile if args.method == "percentile" else None,
        "weights": "int8 symmetric " + ("per-tensor" if args.per_tensor else "per-channel"),
        "activations": "uint8 asymmetric", "format": "QDQ",
        "head_decode": "int8" if args.quantize_head else "float", "excluded_nodes": len(excluded),
        "calibration_images": len(ids), "calibration_ids_sha256": ids_sha256(ids),
        "calibration_chunk": stride,
        "quant_pre_process": preprocessed,
    }
    del int8.metadata_props[:]
    for prop in fp32.metadata_props:
        int8.metadata_props.add(key=prop.key, value=prop.value)
    int8.metadata_props.add(key="please_ye_quantization", value=json.dumps(settings))
    onnx.save(int8, str(output))
    onnx.checker.check_model(str(output))

    ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])  # loads = runnable
    record = {
        "int8_onnx": str(output.resolve()), "int8_sha256": file_sha256(output),
        "int8_bytes": output.stat().st_size,
        "fp32_onnx": str(args.model.resolve()), "fp32_sha256": file_sha256(args.model),
        "fp32_bytes": args.model.stat().st_size,
        "size_ratio": output.stat().st_size / args.model.stat().st_size,
        "input_shape": shape, "settings": settings, "excluded_node_names": excluded,
        "calibration_file": str(args.calibration.resolve()),
        "quantize_seconds": round(seconds, 1),
        "versions": {"python": platform.python_version(), "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__},
        "platform": platform.platform(),
    }
    record_path = output.with_suffix(".json")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nINT8 ONNX : {output}  ({output.stat().st_size / 1024 ** 2:.2f} MiB, "
          f"{record['size_ratio']:.2f}x of FP32, {seconds:.0f}s)")
    print(f"record    : {record_path}")
    print("next      : python scripts/predict_kitti.py --weights "
          f"{output} --output runs/predictions/{output.stem}_internal_val --device cpu --batch 1")


if __name__ == "__main__":
    main()
