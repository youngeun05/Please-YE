#!/usr/bin/env python3
"""Write KITTI-format 2D predictions so evaluate_kitti.py can score a model.

evaluate_kitti.py expects one file per image in KITTI's 16-column result format:

    type truncation occlusion alpha x1 y1 x2 y2 h w l x y z ry score

Only the 2D box and the score matter for the Moderate AP40 metric, so the 3D
columns carry KITTI's conventional placeholders. An image with no detection gets
an empty file, which the scorer accepts.

The confidence threshold is deliberately very low: average precision integrates
the whole precision/recall curve, so discarding low-scoring boxes silently caps
recall and understates AP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
# truncation, occlusion, alpha and the 3D block are not part of the 2D metric.
PLACEHOLDER_HEAD = "-1 -1 -10"
PLACEHOLDER_3D = "-1 -1 -1 -1000 -1000 -1000 -10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--kitti-root", type=Path, default=Path(r"D:\datasets\KITTI"))
    parser.add_argument("--split", type=Path, default=REPO_ROOT / "splits/internal_val.txt",
                        help="file of 6-digit KITTI ids")
    parser.add_argument("--output", type=Path, required=True, help="directory for prediction files")
    parser.add_argument("--imgsz", default="1248x384",
                        help="WIDTHxHEIGHT, forced exactly so the measured shape equals the "
                             "shape that will be exported for Hailo; must match training")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise SystemExit(f"Split must be non-empty with unique ids: {path}")
    if any(len(value) != 6 or not value.isdecimal() for value in ids):
        raise SystemExit(f"Split contains a non-KITTI id: {path}")
    return ids


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    image_dir = args.kitti_root / "training" / "image_2"
    if not image_dir.is_dir():
        raise SystemExit(f"Extracted images not found: {image_dir}")

    sample_ids = read_ids(args.split)
    missing = [value for value in sample_ids if not (image_dir / f"{value}.png").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing, e.g. {missing[:3]}")

    width_text, height_text = args.imgsz.lower().split("x", 1)
    # Ultralytics takes (height, width) and skips its adaptive padding when the
    # size is given explicitly, so the tensor is exactly this shape.
    predict_imgsz = [int(height_text), int(width_text)]
    if any(value % 32 for value in predict_imgsz):
        raise SystemExit(f"--imgsz {args.imgsz}: both sides must be multiples of 32")
    print(f"predicting at {width_text}x{height_text} (H,W passed as {predict_imgsz})")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    written = 0
    detections = 0
    for start in range(0, len(sample_ids), args.batch):
        chunk = sample_ids[start:start + args.batch]
        sources = [str(image_dir / f"{value}.png") for value in chunk]
        results = model.predict(
            sources,
            imgsz=predict_imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )
        for sample_id, result in zip(chunk, results, strict=True):
            rows: list[str] = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confidence = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), score, class_id in zip(xyxy, confidence, class_ids, strict=True):
                    name = CLASS_NAMES.get(int(class_id))
                    if name is None:
                        raise SystemExit(f"Model produced class id {class_id}, outside the three eval classes")
                    if x2 <= x1 or y2 <= y1:
                        continue  # the scorer rejects degenerate boxes
                    rows.append(
                        f"{name} {PLACEHOLDER_HEAD} "
                        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                        f"{PLACEHOLDER_3D} {score:.6f}"
                    )
            (args.output / f"{sample_id}.txt").write_text(
                "".join(f"{row}\n" for row in rows), encoding="utf-8", newline="\n"
            )
            written += 1
            detections += len(rows)
        if written % 200 < args.batch:
            print(f"Wrote {written}/{len(sample_ids)} prediction files", flush=True)

    print(f"\n{written} files in {args.output}  ({detections} boxes, {detections / written:.1f} per image)")
    print("next: python scripts/evaluate_kitti.py --kitti-root "
          f"{args.kitti_root} --split {args.split} --predictions {args.output} "
          "--output runs/metrics.json")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    main()
