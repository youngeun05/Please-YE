#!/usr/bin/env python3
"""Export the trained baseline to a static-shape FP32 ONNX model and verify it.

This ONNX file is the denominator of every comparison in the report: the FP32
baseline measured in ONNX Runtime, the INT8 model made from it by
``quantize_ort.py``, and the Hailo DFC input for the final round.

    python scripts/export_onnx.py ^
        --weights runs/baseline/yolo11n_exact3_1248/weights/best.pt ^
        --kitti-root D:/datasets/KITTI

What it guarantees before it reports success:
  1. The input shape is exactly [1, 3, 384, 1248] (the shape predict_kitti.py
     scores and the shape Hailo will compile). No dynamic axes.
  2. onnx.checker accepts the graph.
  3. ONNX Runtime and the PyTorch checkpoint give the same output on one real
     calibration image (or a seeded random tensor without --kitti-root).
     The check image comes from splits/calibration.txt, never from eval_val.
  4. A JSON record next to the .onnx holds hashes, shapes and versions, so the
     report can state exactly which file produced which number.

NMS is not inside the graph (Ultralytics default). Pure-tensor output
(1, 4 + nc, anchors) keeps the graph Hailo-friendly; NMS runs on the host,
exactly as Hailo Model Zoo does for yolov11n.

Extra dependencies (not in requirements-tested.txt): see requirements-onnx.txt.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from baseline_common import check_names, check_splits, read_ids
from onnx_common import ROOT, file_sha256, parse_wh, preprocess, static_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="Ultralytics .pt checkpoint")
    parser.add_argument("--imgsz", default="1248x384",
                        help="WIDTHxHEIGHT; must equal the shape predict_kitti.py uses")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true", help="skip onnxslim graph simplification")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <weights dir>/<stem>_<W>x<H>_fp32.onnx")
    parser.add_argument("--kitti-root", type=Path, default=None,
                        help="if given, the numeric check uses a real calibration image")
    parser.add_argument("--atol-box", type=float, default=0.5, help="max |diff| of box outputs, pixels")
    parser.add_argument("--atol-score", type=float, default=1e-3, help="max |diff| of class scores")
    return parser.parse_args()


def check_input(kitti_root: Path | None, width: int, height: int) -> tuple[np.ndarray, str]:
    if kitti_root is None:
        rng = np.random.default_rng(20260917)
        return rng.random((1, 3, height, width), dtype=np.float32), "seeded random tensor"
    calibration = read_ids(ROOT / "splits/calibration.txt")
    image_id = sorted(calibration)[0]
    path = kitti_root / "training" / "image_2" / f"{image_id}.png"
    return preprocess(path, width, height), f"calibration image {image_id}"


def torch_reference(weights: Path, sample: np.ndarray) -> np.ndarray | None:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights)).model.float().eval()
    with torch.inference_mode():
        output = model(torch.from_numpy(sample))
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        return None
    return output.cpu().numpy()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    width, height = parse_wh(args.imgsz)
    check_splits(ROOT)  # same leakage guard as training; also validates the split files

    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO

    output = args.output or args.weights.parent / f"{args.weights.stem}_{width}x{height}_fp32.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(str(args.weights))
    if yolo.task != "detect":
        raise SystemExit("Only detection checkpoints are supported")
    check_names(yolo.names)

    print(f"exporting {args.weights} at {width}x{height} (H,W passed as [{height}, {width}])")
    started = time.perf_counter()
    exported = yolo.export(
        format="onnx",
        imgsz=[height, width],
        opset=args.opset,
        simplify=not args.no_simplify,
        dynamic=False,
        batch=1,
        half=False,
        device="cpu",
    )
    exported = Path(exported)
    if exported.resolve() != output.resolve():
        shutil.move(str(exported), str(output))
    export_seconds = time.perf_counter() - started

    # 1. static shape
    input_name, input_shape = static_input(output)
    if input_shape != [1, 3, height, width]:
        raise SystemExit(f"Exported input shape {input_shape} != [1, 3, {height}, {width}]")
    # 2. structural check
    onnx.checker.check_model(str(output))
    model = onnx.load(str(output))
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    # 3. numeric check against PyTorch
    sample, sample_source = check_input(args.kitti_root, width, height)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {input_name: sample})[0]
    reference = torch_reference(args.weights, sample)
    check: dict[str, object] = {"sample": sample_source, "onnx_output_shape": list(onnx_out.shape)}
    if reference is None:
        check["status"] = "skipped: PyTorch model did not return a tensor"
        print("WARNING: numeric check skipped (unexpected PyTorch output type)")
    else:
        if reference.shape != onnx_out.shape:
            raise SystemExit(f"Output shape mismatch: torch {reference.shape} vs onnx {onnx_out.shape}")
        box_diff = float(np.abs(reference[:, :4] - onnx_out[:, :4]).max())
        score_diff = float(np.abs(reference[:, 4:] - onnx_out[:, 4:]).max())
        check.update(max_abs_box_diff_px=box_diff, max_abs_score_diff=score_diff,
                     atol_box=args.atol_box, atol_score=args.atol_score)
        print(f"torch vs onnx : box {box_diff:.2e} px, score {score_diff:.2e}  ({sample_source})")
        if box_diff > args.atol_box or score_diff > args.atol_score:
            raise SystemExit("ONNX output differs from PyTorch beyond tolerance; do not use this file")
        check["status"] = "pass"

    # 4. record
    record = {
        "onnx": str(output.resolve()),
        "onnx_sha256": file_sha256(output),
        "onnx_bytes": output.stat().st_size,
        "weights": str(args.weights.resolve()),
        "weights_sha256": file_sha256(args.weights),
        "input_name": input_name,
        "input_shape": input_shape,
        "output_shape": list(onnx_out.shape),
        "opset": args.opset,
        "simplified": not args.no_simplify,
        "nms_in_graph": False,
        "ultralytics_metadata": metadata,
        "numeric_check": check,
        "export_seconds": round(export_seconds, 1),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "platform": platform.platform(),
    }
    record_path = output.with_suffix(".json")
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nFP32 ONNX : {output}  ({output.stat().st_size / 1024 ** 2:.2f} MiB)")
    print(f"record    : {record_path}")
    print("next      : python scripts/predict_kitti.py --weights "
          f"{output} --output runs/predictions/onnx_fp32_internal_val --device cpu --batch 1")


if __name__ == "__main__":
    main()
