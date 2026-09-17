#!/usr/bin/env python3
"""Train a baseline detector on the fixed KITTI train split.

This is the real training entry point. ``smoke_train.py`` stays what it is: a
one-epoch CUDA path check whose numbers must not be reported.

Resolution follows the benchmark notice, which caps width at 1280, requires
multiples of 32 and strongly recommends a rectangle matching KITTI's ~3.31
aspect instead of a square that wastes compute on letterbox padding.

Measured on this dataset, rectangular batching rounds to stride 32 like this:

    --imgsz 1248  ->  train 1248x384, val 1280x416   (both within the width cap)
    --imgsz 1280  ->  train 1280x416, val 1312x416   (val exceeds the cap)
    --imgsz 640   ->  train  640x224, val  672x224

The notice's named presets (1280x384, 640x192) have aspect 3.333 while KITTI is
3.31, so they are not what rectangular rounding lands on. ``--imgsz 1248`` is the
closest fit that keeps every shape inside the cap, which is why it is the default
here. Whatever is chosen, export and prediction must use the same shape.

The official evaluation split is never read here. Ultralytics only touches the
``train`` and ``val`` entries of the dataset yaml during training, and this
script additionally refuses to start if the val list is the official split or if
the train list intersects it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "exact3": REPO_ROOT / "data/kitti_yolo_exact3/kitti.local.yaml",
    "neighbors3": REPO_ROOT / "data/kitti_yolo_neighbors3/kitti.local.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="exact3",
                        help="which converted dataset to train on")
    parser.add_argument("--data", type=Path, default=None,
                        help="dataset yaml, overriding --variant")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Ultralytics weights (.pt, COCO-pretrained) or a .yaml for scratch")
    parser.add_argument("--imgsz", type=int, default=1248,
                        help="long side; 1248 -> 1248x384 train / 1280x416 val, both within the cap")
    parser.add_argument("--batch", type=int, default=32,
                        help="yolo11n fits 48 at this size; 32 leaves room for the validation pass")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0,
                        help="0 by default: on this Windows host, 8 workers crashed training with "
                             "an access violation (0xC0000005) a few steps into the first epoch. "
                             "The repo's smoke_train.py already pinned workers=0 for the same "
                             "class of problem. Raise it only if you re-verify stability.")
    parser.add_argument("--cache", default="none", choices=("disk", "ram", "none"),
                        help="'disk' writes .npy into the raw KITTI image directory, so it is off "
                             "by default; with 31.7GB of RAM the OS page cache already holds the "
                             "12GB of PNGs after the first epoch")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--name", default=None, help="run name under runs/baseline")
    parser.add_argument("--patience", type=int, default=30)
    return parser.parse_args()


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}


def stems(list_path: Path) -> set[str]:
    return {Path(line).stem for line in read_ids(list_path)}


def guard_against_leakage(data_yaml: Path) -> None:
    """Refuse to train if the official evaluation split could be seen."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    official = read_ids(REPO_ROOT / "eval_val.txt")
    committed_train = read_ids(REPO_ROOT / "splits/train.txt")

    train_ids = stems(root / config["train"])
    val_ids = stems(root / config["val"])

    if train_ids & official:
        raise SystemExit(f"Leakage guard: {len(train_ids & official)} official eval ids are in the train list")
    if val_ids & official:
        raise SystemExit("Leakage guard: the val list overlaps the official evaluation split")
    if train_ids != committed_train:
        raise SystemExit("Leakage guard: the train list does not match splits/train.txt")
    print(f"leakage guard : train={len(train_ids)} val={len(val_ids)} official held out={len(official)}")


def main() -> None:
    args = parse_args()
    data_yaml = (args.data or VARIANTS[args.variant]).resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Dataset yaml not found: {data_yaml}. Run scripts/run_prepare.ps1 first.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to train a baseline on CPU")

    guard_against_leakage(data_yaml)

    cache = False if args.cache == "none" else args.cache
    workers = args.workers
    if cache == "ram" and workers:
        # Windows spawns dataloader workers, so a RAM cache would be copied per
        # worker and exhaust host memory.
        print(f"cache=ram forces workers=0 on Windows (was {workers})")
        workers = 0

    name = args.name or f"{Path(args.model).stem}_{args.variant}_{args.imgsz}"
    print(f"data          : {data_yaml}")
    print(f"model         : {args.model}")
    print(f"imgsz         : {args.imgsz} long side, rectangular batches")
    print(f"batch         : {args.batch}")
    print(f"gpu           : {torch.cuda.get_device_name(0)}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        rect=True,          # keeps KITTI's aspect instead of padding to a square
        batch=args.batch,
        device=args.device,
        workers=workers,
        cache=cache,
        amp=True,
        val=True,
        plots=True,
        deterministic=True,
        seed=args.seed,
        patience=args.patience,
        project=str((REPO_ROOT / "runs/baseline").resolve()),
        name=name,
        exist_ok=True,
    )
    save_dir = Path(results.save_dir)
    print(f"\nbaseline weights: {save_dir / 'weights/best.pt'}")
    print("next: python scripts/predict_kitti.py --weights "
          f"{save_dir / 'weights/best.pt'} --split splits/internal_val.txt")


if __name__ == "__main__":
    main()
