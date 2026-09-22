#!/usr/bin/env python3
"""Generate perfect and empty predictions to sanity-check the R40 evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_kitti import CLASSES, evaluate


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=repo_root / "splits/internal_val.txt")
    parser.add_argument(
        "--reference-evaluator",
        type=Path,
        default=repo_root
        / "third_party/mmdetection3d/mmdet3d/evaluation/functional/kitti_utils/eval.py",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def prediction_row(gt_fields: list[str], score: float) -> str:
    # Alpha is deliberately disabled because the competition metric is 2D AP, not AOS.
    fields = gt_fields.copy()
    fields[1] = "-1"
    fields[2] = "-1"
    fields[3] = "-10"
    return " ".join(fields + [f"{score:.9f}"])


def main() -> None:
    args = parse_args()
    sample_ids = [line.strip() for line in args.split.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    perfect_dir = args.work_dir / "perfect"
    empty_dir = args.work_dir / "empty"
    perfect_dir.mkdir(parents=True)
    empty_dir.mkdir(parents=True)

    score_index = 0
    for sample_id in sample_ids:
        rows: list[str] = []
        gt_path = args.kitti_root / "training/label_2" / f"{sample_id}.txt"
        for line in gt_path.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if fields and fields[0] in CLASSES:
                score = 1.0 - score_index * 1e-9
                rows.append(prediction_row(fields, score))
                score_index += 1
        (perfect_dir / f"{sample_id}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
        )
        (empty_dir / f"{sample_id}.txt").write_text("", encoding="utf-8")

    common = {
        "kitti_root": args.kitti_root,
        "split": args.split,
        "reference_evaluator": args.reference_evaluator,
        "output": None,
    }
    perfect = evaluate(argparse.Namespace(predictions=perfect_dir, **common))
    empty = evaluate(argparse.Namespace(predictions=empty_dir, **common))
    for class_name in CLASSES:
        if abs(perfect["class_ap40_moderate"][class_name] - 100.0) > 1e-8:
            raise AssertionError(f"Perfect {class_name} AP is not 100: {perfect}")
        if abs(empty["class_ap40_moderate"][class_name]) > 1e-8:
            raise AssertionError(f"Empty {class_name} AP is not 0: {empty}")
    print(json.dumps({"perfect": perfect["class_ap40_moderate"], "empty": empty["class_ap40_moderate"]}, indent=2))


if __name__ == "__main__":
    main()
