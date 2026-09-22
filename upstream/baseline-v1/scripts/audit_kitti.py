#!/usr/bin/env python3
"""Audit extracted KITTI images, labels, and official evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--eval-split", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="Output JSON path (defaults to <kitti-root>/audit_report.local.json)",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    image_dir = args.kitti_root / "training" / "image_2"
    label_dir = args.kitti_root / "training" / "label_2"
    images = {path.stem: path for path in image_dir.glob("*.png")}
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    eval_ids = [line.strip() for line in args.eval_split.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    if len(images) != 7_481 or len(labels) != 7_481 or images.keys() != labels.keys():
        raise ValueError(f"Invalid extracted dataset: images={len(images)}, labels={len(labels)}")
    if len(eval_ids) != 1_000 or len(set(eval_ids)) != 1_000 or not set(eval_ids) <= images.keys():
        raise ValueError("Invalid official evaluation split")

    classes: Counter[str] = Counter()
    dimensions: Counter[tuple[int, int]] = Counter()
    for index, image_id in enumerate(sorted(images), start=1):
        with Image.open(images[image_id]) as image:
            image.verify()
        with Image.open(images[image_id]) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(f"Unexpected image format/mode: {images[image_id]} {image.format}/{image.mode}")
            image_width, image_height = image.size
            dimensions[image.size] += 1

        for line_number, line in enumerate(labels[image_id].read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 15:
                raise ValueError(f"Malformed label: {labels[image_id]}:{line_number}")
            classes[fields[0]] += 1
            left, top, right, bottom = map(float, fields[4:8])
            if not (
                0 <= left <= right <= image_width
                and 0 <= top <= bottom <= image_height
            ):
                raise ValueError(f"Out-of-range bbox: {labels[image_id]}:{line_number}")
        if index % 500 == 0:
            print(f"Audited {index}/{len(images)} images", flush=True)

    report = {
        "kitti_root": str(args.kitti_root.resolve()),
        "training_images": len(images),
        "training_labels": len(labels),
        "official_eval_ids": len(eval_ids),
        "development_ids": len(images.keys() - set(eval_ids)),
        "eval_split_sha256": file_sha256(args.eval_split),
        "class_instances": dict(sorted(classes.items())),
        "image_dimensions": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in dimensions.most_common()
        ],
    }
    report_path = args.report or (args.kitti_root / "audit_report.local.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
