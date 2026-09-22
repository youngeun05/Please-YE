#!/usr/bin/env python3
"""Convert KITTI 2D labels to a local YOLO dataset without copying images."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image


CLASS_IDS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
NEIGHBOR_CLASSES = {"Van": "Car", "Person_sitting": "Pedestrian"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--official-eval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--neighbor-policy", choices=("ignore", "map"), default="ignore")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate ids in {path}")
    return values


def yaml_path(path: Path) -> str:
    return path.resolve().as_posix()


def write_image_list(path: Path, image_dir: Path, image_ids: list[str]) -> None:
    # Keep the junction path instead of resolving it to raw ``image_2``.
    # Ultralytics derives labels by replacing ``images`` with ``labels``;
    # dereferencing the junction here would silently make every label missing.
    lines = [f"{(image_dir / f'{image_id}.png').absolute()}\n" for image_id in sorted(image_ids)]
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    raw_image_dir = args.kitti_root / "training" / "image_2"
    raw_label_dir = args.kitti_root / "training" / "label_2"
    linked_image_dir = args.output_root / "images" / "all"
    yolo_label_dir = args.output_root / "labels" / "all"
    if not linked_image_dir.is_dir():
        raise FileNotFoundError(
            f"Expected an image directory junction at {linked_image_dir}; target it at {raw_image_dir}"
        )
    yolo_label_dir.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(path.stem for path in raw_image_dir.glob("*.png"))
    train_ids = read_ids(args.splits_dir / "train.txt")
    val_ids = read_ids(args.splits_dir / "internal_val.txt")
    calibration_ids = read_ids(args.splits_dir / "calibration.txt")
    official_ids = read_ids(args.official_eval)
    if set(train_ids) | set(val_ids) | set(official_ids) != set(all_ids):
        raise ValueError("Split ids do not cover the extracted labeled image set")
    if set(train_ids) & set(val_ids) or set(train_ids) & set(official_ids) or set(val_ids) & set(official_ids):
        raise ValueError("Split leakage detected")
    if not set(calibration_ids) <= set(train_ids):
        raise ValueError("Calibration ids must be a subset of train")

    source_counts: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    clipped_boxes = 0
    for index, image_id in enumerate(all_ids, start=1):
        image_path = raw_image_dir / f"{image_id}.png"
        label_path = raw_label_dir / f"{image_id}.txt"
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        converted: list[str] = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 15:
                raise ValueError(f"Malformed KITTI label: {label_path}:{line_number}")
            source_class = fields[0]
            source_counts[source_class] += 1
            target_class = source_class
            if args.neighbor_policy == "map":
                target_class = NEIGHBOR_CLASSES.get(source_class, source_class)
            if target_class not in CLASS_IDS:
                continue

            left, top, right, bottom = map(float, fields[4:8])
            clipped = (
                min(max(left, 0.0), float(image_width)),
                min(max(top, 0.0), float(image_height)),
                min(max(right, 0.0), float(image_width)),
                min(max(bottom, 0.0), float(image_height)),
            )
            if clipped != (left, top, right, bottom):
                clipped_boxes += 1
            left, top, right, bottom = clipped
            box_width, box_height = right - left, bottom - top
            if box_width <= 0 or box_height <= 0:
                raise ValueError(f"Non-positive target box: {label_path}:{line_number}")
            center_x = (left + right) / (2 * image_width)
            center_y = (top + bottom) / (2 * image_height)
            normalized_width = box_width / image_width
            normalized_height = box_height / image_height
            values = (center_x, center_y, normalized_width, normalized_height)
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"Invalid normalized box: {label_path}:{line_number} {values}")
            converted.append(
                f"{CLASS_IDS[target_class]} " + " ".join(f"{value:.8f}" for value in values)
            )
            output_counts[target_class] += 1
        (yolo_label_dir / f"{image_id}.txt").write_text(
            "".join(f"{line}\n" for line in converted), encoding="utf-8", newline="\n"
        )
        if index % 1_000 == 0:
            print(f"Converted {index}/{len(all_ids)} labels", flush=True)

    local_lists = args.output_root / "lists"
    local_lists.mkdir(parents=True, exist_ok=True)
    write_image_list(local_lists / "train.txt", linked_image_dir, train_ids)
    write_image_list(local_lists / "internal_val.txt", linked_image_dir, val_ids)
    write_image_list(local_lists / "official_eval.txt", linked_image_dir, official_ids)
    write_image_list(local_lists / "calibration.txt", linked_image_dir, calibration_ids)

    label_token = f"{os.sep}images{os.sep}"
    if label_token not in str(linked_image_dir.absolute()):
        raise ValueError("Image junction path must contain an 'images' directory for YOLO label mapping")
    for image_id in all_ids:
        if not (linked_image_dir / f"{image_id}.png").is_file():
            raise FileNotFoundError(f"Missing linked image: {image_id}")
        if not (yolo_label_dir / f"{image_id}.txt").is_file():
            raise FileNotFoundError(f"Missing converted label: {image_id}")

    dataset_yaml = (
        f"path: {yaml_path(args.output_root)}\n"
        "train: lists/train.txt\n"
        "val: lists/internal_val.txt\n"
        "test: lists/official_eval.txt\n"
        "names:\n"
        "  0: Car\n"
        "  1: Pedestrian\n"
        "  2: Cyclist\n"
    )
    (args.output_root / "kitti.local.yaml").write_text(dataset_yaml, encoding="utf-8", newline="\n")
    # Every selectable split points at calibration data, preventing an exporter default from opening eval images.
    calibration_yaml = (
        f"path: {yaml_path(args.output_root)}\n"
        "train: lists/calibration.txt\n"
        "val: lists/calibration.txt\n"
        "test: lists/calibration.txt\n"
        "names:\n"
        "  0: Car\n"
        "  1: Pedestrian\n"
        "  2: Cyclist\n"
    )
    (args.output_root / "calibration.local.yaml").write_text(
        calibration_yaml, encoding="utf-8", newline="\n"
    )
    report = {
        "neighbor_policy": args.neighbor_policy,
        "images": len(all_ids),
        "source_class_instances": dict(sorted(source_counts.items())),
        "output_class_instances": dict(sorted(output_counts.items())),
        "clipped_boxes": clipped_boxes,
        "splits": {
            "train": len(train_ids),
            "internal_val": len(val_ids),
            "official_eval": len(official_ids),
            "calibration": len(calibration_ids),
        },
    }
    (args.output_root / "conversion_report.local.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
