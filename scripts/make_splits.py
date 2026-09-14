#!/usr/bin/env python3
"""Create deterministic, leakage-safe KITTI train/validation/calibration splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TARGET_CLASSES = ("Car", "Pedestrian", "Cyclist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, required=True)
    parser.add_argument("--eval-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-count", type=int, default=1_000)
    parser.add_argument("--calibration-count", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--random-candidates", type=int, default=4_000)
    parser.add_argument("--swap-steps", type=int, default=40_000)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate ids in {path}")
    return values


def feature_names() -> list[str]:
    names = ["images"]
    for class_name in TARGET_CLASSES:
        names.extend(
            [
                f"{class_name}:images",
                f"{class_name}:objects",
                f"{class_name}:moderate_objects",
                f"{class_name}:height_le_25",
                f"{class_name}:height_25_40",
                f"{class_name}:height_gt_40",
            ]
        )
    names.extend([f"presence_pattern:{pattern:03b}" for pattern in range(8)])
    return names


def image_features(label_path: Path) -> np.ndarray:
    values: list[float] = [1.0]
    parsed: list[tuple[str, float, int, float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if len(fields) != 15:
            raise ValueError(f"Malformed label: {label_path}:{line_number}")
        parsed.append((fields[0], float(fields[1]), int(fields[2]), float(fields[7]) - float(fields[5])))

    present = []
    for class_name in TARGET_CLASSES:
        objects = [row for row in parsed if row[0] == class_name]
        present.append(bool(objects))
        moderate = sum(height > 25 and occlusion <= 1 and truncation <= 0.3 for _, truncation, occlusion, height in objects)
        values.extend(
            [
                float(bool(objects)),
                float(len(objects)),
                float(moderate),
                float(sum(height <= 25 for _, _, _, height in objects)),
                float(sum(25 < height <= 40 for _, _, _, height in objects)),
                float(sum(height > 40 for _, _, _, height in objects)),
            ]
        )
    pattern = sum((1 << index) for index, flag in enumerate(present) if flag)
    values.extend(float(pattern == index) for index in range(8))
    return np.asarray(values, dtype=np.float64)


def distribution_score(selected_sum: np.ndarray, total_sum: np.ndarray, count: int, total_count: int) -> float:
    target = total_sum * (count / total_count)
    # A difference of one unit is unavoidable for some rare integer features.
    denominator = np.maximum(target, 1.0)
    relative = np.abs(selected_sum - target) / denominator
    return float(relative.max() + np.mean(relative**2))


def representative_subset(
    matrix: np.ndarray,
    count: int,
    rng: np.random.Generator,
    random_candidates: int,
    swap_steps: int,
) -> np.ndarray:
    total_count = len(matrix)
    if not 0 < count < total_count:
        raise ValueError(f"Subset size must be between 1 and {total_count - 1}, got {count}")
    total_sum = matrix.sum(axis=0)
    best_indices: np.ndarray | None = None
    best_sum: np.ndarray | None = None
    best_score = float("inf")
    for _ in range(random_candidates):
        indices = rng.choice(total_count, size=count, replace=False)
        selected_sum = matrix[indices].sum(axis=0)
        score = distribution_score(selected_sum, total_sum, count, total_count)
        if score < best_score:
            best_indices, best_sum, best_score = indices, selected_sum, score
    assert best_indices is not None and best_sum is not None

    selected = np.zeros(total_count, dtype=bool)
    selected[best_indices] = True
    for _ in range(swap_steps):
        inside = int(rng.choice(np.flatnonzero(selected)))
        outside = int(rng.choice(np.flatnonzero(~selected)))
        candidate_sum = best_sum - matrix[inside] + matrix[outside]
        candidate_score = distribution_score(candidate_sum, total_sum, count, total_count)
        if candidate_score < best_score:
            selected[inside] = False
            selected[outside] = True
            best_sum, best_score = candidate_sum, candidate_score
    return np.flatnonzero(selected)


def write_ids(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in sorted(values)), encoding="utf-8", newline="\n")


def digest_ids(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def summarize(matrix: np.ndarray, names: list[str], ids: list[str]) -> dict[str, object]:
    sums = matrix.sum(axis=0)
    return {
        "count": len(ids),
        "sha256": digest_ids(ids),
        "features": {name: int(value) for name, value in zip(names, sums, strict=True)},
    }


def main() -> None:
    args = parse_args()
    label_dir = args.kitti_root / "training" / "label_2"
    all_ids = sorted(path.stem for path in label_dir.glob("*.txt"))
    official_eval = read_ids(args.eval_split)
    if len(all_ids) != 7_481 or len(official_eval) != 1_000:
        raise ValueError(f"Unexpected dataset size: labels={len(all_ids)}, eval={len(official_eval)}")
    if not set(official_eval) <= set(all_ids):
        raise ValueError("Official evaluation split is not a subset of KITTI training labels")

    development_ids = sorted(set(all_ids) - set(official_eval))
    names = feature_names()
    development_matrix = np.stack([image_features(label_dir / f"{image_id}.txt") for image_id in development_ids])
    if development_matrix.shape[1] != len(names):
        raise AssertionError("Feature vector and names differ")

    rng = np.random.default_rng(args.seed)
    val_indices = representative_subset(
        development_matrix,
        args.val_count,
        rng,
        args.random_candidates,
        args.swap_steps,
    )
    val_mask = np.zeros(len(development_ids), dtype=bool)
    val_mask[val_indices] = True
    internal_val = [development_ids[index] for index in np.flatnonzero(val_mask)]
    train = [development_ids[index] for index in np.flatnonzero(~val_mask)]
    train_matrix = development_matrix[~val_mask]

    calibration_indices = representative_subset(
        train_matrix,
        args.calibration_count,
        rng,
        args.random_candidates,
        args.swap_steps,
    )
    calibration = [train[index] for index in calibration_indices]

    if set(train) & set(internal_val) or set(train) & set(official_eval) or set(internal_val) & set(official_eval):
        raise AssertionError("Split leakage detected")
    if set(train) | set(internal_val) | set(official_eval) != set(all_ids):
        raise AssertionError("Split union does not cover all labeled images")
    if not set(calibration) <= set(train):
        raise AssertionError("Calibration set must be a subset of train")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_ids(args.output_dir / "train.txt", train)
    write_ids(args.output_dir / "internal_val.txt", internal_val)
    write_ids(args.output_dir / "calibration.txt", calibration)

    full_matrix = np.stack([image_features(label_dir / f"{image_id}.txt") for image_id in all_ids])
    eval_index = {image_id: index for index, image_id in enumerate(all_ids)}
    report = {
        "algorithm": "random candidate search plus deterministic improving swaps",
        "seed": args.seed,
        "feature_names": names,
        "all": summarize(full_matrix, names, all_ids),
        "development": summarize(development_matrix, names, development_ids),
        "train": summarize(train_matrix, names, train),
        "internal_val": summarize(development_matrix[val_mask], names, internal_val),
        "official_eval": summarize(full_matrix[[eval_index[value] for value in official_eval]], names, official_eval),
        "calibration": summarize(train_matrix[calibration_indices], names, calibration),
    }
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: report[key] for key in ("train", "internal_val", "official_eval", "calibration")}, indent=2))


if __name__ == "__main__":
    main()
