#!/usr/bin/env python3
"""Evaluate KITTI 2D detections with the pinned OpenMMLab R40 reference.

Each prediction file must use KITTI's 16-column result format:
type truncation occlusion alpha x1 y1 x2 y2 h w l x y z ry score
An empty file is a valid prediction for an image with no detections.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np


CLASSES = ("Car", "Pedestrian", "Cyclist")
REFERENCE_COMMIT = "fe25f7a51d36e3702f961e198894580d83c4387b"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--reference-evaluator",
        type=Path,
        default=repo_root
        / "third_party/mmdetection3d/mmdet3d/evaluation/functional/kitti_utils/eval.py",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _empty_annotation(detection: bool) -> dict[str, np.ndarray]:
    annotation = {
        "name": np.empty((0,), dtype="<U1"),
        "truncated": np.empty((0,), dtype=np.float64),
        "occluded": np.empty((0,), dtype=np.int64),
        "alpha": np.empty((0,), dtype=np.float64),
        "bbox": np.empty((0, 4), dtype=np.float64),
        "dimensions": np.empty((0, 3), dtype=np.float64),
        "location": np.empty((0, 3), dtype=np.float64),
        "rotation_y": np.empty((0,), dtype=np.float64),
    }
    if detection:
        annotation["score"] = np.empty((0,), dtype=np.float64)
    return annotation


def parse_label(path: Path, detection: bool) -> dict[str, np.ndarray]:
    rows: list[list[str]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        expected = 16 if detection else 15
        if len(fields) != expected:
            raise ValueError(f"{path}:{line_number}: expected {expected} fields, got {len(fields)}")
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric KITTI field") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite KITTI field")
        if fields[0] == "DontCare" and detection:
            raise ValueError(f"{path}:{line_number}: DontCare is not a prediction class")
        if values[5] <= values[3] or values[6] <= values[4]:
            raise ValueError(f"{path}:{line_number}: invalid bbox ordering")
        rows.append(fields)

    if not rows:
        return _empty_annotation(detection)

    numeric = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=np.float64)
    annotation = {
        "name": np.asarray([row[0] for row in rows]),
        "truncated": numeric[:, 0],
        "occluded": numeric[:, 1].astype(np.int64),
        "alpha": numeric[:, 2],
        "bbox": numeric[:, 3:7],
        "dimensions": numeric[:, 7:10],
        "location": numeric[:, 10:13],
        "rotation_y": numeric[:, 13],
    }
    if detection:
        annotation["score"] = numeric[:, 14]
    return annotation


def _install_numba_fallback() -> None:
    """Provide semantics-preserving decorators when numba is unavailable.

    This changes execution speed only: the decorated reference functions run as
    ordinary Python/Numpy functions. It lets the scorer remain usable on a clean
    Python installation and is tested against the same reference source.
    """

    def jit(function=None, **_kwargs):
        if function is None:
            return lambda wrapped: wrapped
        return function

    sys.modules["numba"] = types.SimpleNamespace(jit=jit, prange=range)


def load_reference(path: Path) -> tuple[Any, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference evaluator not found: {path}. Run scripts/setup_evaluator.ps1 first."
        )
    try:
        import numba  # noqa: F401

        acceleration = "numba"
    except ImportError:
        _install_numba_fallback()
        acceleration = "python_fallback"

    spec = importlib.util.spec_from_file_location("pinned_kitti_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, acceleration


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Split must be non-empty and contain unique IDs: {path}")
    if any(len(sample_id) != 6 or not sample_id.isdigit() for sample_id in ids):
        raise ValueError(f"Split contains a non-KITTI sample ID: {path}")
    return ids


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    sample_ids = read_ids(args.split)
    gt_dir = args.kitti_root / "training/label_2"
    missing_gt = [sample_id for sample_id in sample_ids if not (gt_dir / f"{sample_id}.txt").is_file()]
    missing_dt = [
        sample_id for sample_id in sample_ids if not (args.predictions / f"{sample_id}.txt").is_file()
    ]
    if missing_gt or missing_dt:
        raise FileNotFoundError(
            f"Missing files: ground_truth={len(missing_gt)}, predictions={len(missing_dt)}; "
            f"examples={missing_gt[:3] + missing_dt[:3]}"
        )

    gt_annos = [parse_label(gt_dir / f"{sample_id}.txt", False) for sample_id in sample_ids]
    dt_annos = [parse_label(args.predictions / f"{sample_id}.txt", True) for sample_id in sample_ids]
    reference, acceleration = load_reference(args.reference_evaluator)
    report_text, metrics = reference.kitti_eval(gt_annos, dt_annos, list(CLASSES), ["bbox"])

    class_ap = {
        class_name: float(metrics[f"KITTI/{class_name}_2D_AP40_moderate_strict"])
        for class_name in CLASSES
    }
    result: dict[str, Any] = {
        "metric": "KITTI 2D Moderate AP40, strict class IoU",
        "reference": {
            "project": "open-mmlab/mmdetection3d",
            "commit": REFERENCE_COMMIT,
            "file": str(args.reference_evaluator.resolve()),
            "execution": acceleration,
        },
        "split": str(args.split.resolve()),
        "samples": len(sample_ids),
        "class_ap40_moderate": class_ap,
        "mean_ap40_moderate": float(sum(class_ap.values()) / len(class_ap)),
        "reference_report": report_text,
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
