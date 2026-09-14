#!/usr/bin/env python3
"""Run a tiny CUDA training pass to verify the local KITTI pipeline end to end."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
import ultralytics.data.dataset as yolo_dataset


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}


class SerialPool:
    """Drop-in cache scanner for restricted Windows sessions without IPC pipes."""

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    @staticmethod
    def imap(function=None, iterable=None, func=None, **_kwargs):
        worker = function if function is not None else func
        return map(worker, iterable)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=repo_root / "data/kitti_yolo_exact3/kitti.local.yaml"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--fraction", type=float, default=0.01)
    args = parser.parse_args()

    train_ids = read_ids(repo_root / "splits/train.txt")
    val_ids = read_ids(repo_root / "splits/internal_val.txt")
    official_ids = read_ids(repo_root / "eval_val.txt")
    if train_ids & val_ids or train_ids & official_ids or val_ids & official_ids:
        raise RuntimeError("Leakage guard failed: train, internal validation, and official evaluation overlap")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to substitute a CPU smoke test")

    # Codex's restricted Windows host denies the anonymous pipe used by Python's
    # ThreadPool. Cache scanning is equivalent when serialized and training is
    # unaffected; normal terminals may omit this compatibility shim.
    yolo_dataset.ThreadPool = SerialPool
    model = YOLO("yolo11n.yaml")
    result = model.train(
        data=str(args.data.resolve()),
        epochs=1,
        fraction=args.fraction,
        imgsz=320,
        batch=8,
        device=args.device,
        workers=0,
        amp=False,
        val=False,
        plots=False,
        cache=False,
        deterministic=True,
        seed=20260911,
        project=str((repo_root / "runs/smoke").resolve()),
        name="yolo11n_exact3",
        exist_ok=True,
    )
    print(f"Smoke train completed on CUDA device {args.device}: {result.save_dir}")


if __name__ == "__main__":
    main()
