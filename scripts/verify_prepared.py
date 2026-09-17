#!/usr/bin/env python3
"""Verify the prepared KITTI/YOLO artifacts before any training run.

Checks, in order:
  1. split files      counts, uniqueness, pairwise disjointness, calibration subset of train
  2. extracted KITTI  7,481 images and labels, ids identical (optional, needs --kitti-root)
  3. YOLO output      label file per image, class ids in {0,1,2}, boxes normalized in [0,1]
  4. list files       every listed image path exists and maps to an existing label path
  5. dataset yaml     present and pointing at the list files

Exit code is 0 only when every enabled check passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"train": 5_481, "internal_val": 1_000, "official_eval": 1_000, "calibration": 1_024}
TOTAL_IMAGES = 7_481


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        if condition:
            print(f"  [ OK ] {label}")
        else:
            print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))
            self.failures.append(label)
        return condition

    def section(self, name: str) -> None:
        print(f"\n== {name}")


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def label_path_for(image_path: Path) -> Path:
    """Map .../images/all/000000.png to .../labels/all/000000.txt the way Ultralytics does."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def check_splits(report: Report) -> dict[str, set[str]]:
    report.section("splits")
    sources = {
        "train": REPO_ROOT / "splits" / "train.txt",
        "internal_val": REPO_ROOT / "splits" / "internal_val.txt",
        "calibration": REPO_ROOT / "splits" / "calibration.txt",
        "official_eval": REPO_ROOT / "eval_val.txt",
    }
    ids: dict[str, set[str]] = {}
    for name, path in sources.items():
        if not report.check(path.is_file(), f"{name} file exists", str(path)):
            ids[name] = set()
            continue
        values = read_ids(path)
        unique = set(values)
        report.check(len(values) == len(unique), f"{name} has no duplicates", f"{len(values)} lines / {len(unique)} unique")
        report.check(len(unique) == EXPECTED[name], f"{name} count == {EXPECTED[name]}", f"got {len(unique)}")
        report.check(
            all(len(v) == 6 and v.isdecimal() for v in unique),
            f"{name} ids are 6-digit",
        )
        ids[name] = unique

    for a, b in (("train", "internal_val"), ("train", "official_eval"), ("internal_val", "official_eval")):
        overlap = ids[a] & ids[b]
        report.check(not overlap, f"{a} and {b} are disjoint", f"{len(overlap)} shared ids")
    report.check(ids["calibration"] <= ids["train"], "calibration is a subset of train")
    union = ids["train"] | ids["internal_val"] | ids["official_eval"]
    report.check(len(union) == TOTAL_IMAGES, f"three splits cover {TOTAL_IMAGES} images", f"got {len(union)}")
    return ids


def check_kitti(report: Report, kitti_root: Path, ids: dict[str, set[str]]) -> None:
    report.section(f"extracted KITTI  ({kitti_root})")
    image_dir = kitti_root / "training" / "image_2"
    label_dir = kitti_root / "training" / "label_2"
    if not report.check(image_dir.is_dir(), "training/image_2 exists", str(image_dir)):
        return
    if not report.check(label_dir.is_dir(), "training/label_2 exists", str(label_dir)):
        return
    images = {p.stem for p in image_dir.glob("*.png")}
    labels = {p.stem for p in label_dir.glob("*.txt")}
    report.check(len(images) == TOTAL_IMAGES, f"{TOTAL_IMAGES} images", f"got {len(images)}")
    report.check(len(labels) == TOTAL_IMAGES, f"{TOTAL_IMAGES} labels", f"got {len(labels)}")
    report.check(images == labels, "image ids == label ids")
    union = ids["train"] | ids["internal_val"] | ids["official_eval"]
    report.check(union <= images, "every split id exists on disk", f"{len(union - images)} missing")


def check_yolo(report: Report, output_root: Path, ids: dict[str, set[str]]) -> None:
    report.section(f"YOLO dataset  ({output_root.name})")
    if not report.check(output_root.is_dir(), "output root exists", str(output_root)):
        return
    label_dir = output_root / "labels" / "all"
    image_dir = output_root / "images" / "all"
    if not report.check(label_dir.is_dir(), "labels/all exists", str(label_dir)):
        return
    report.check(image_dir.is_dir(), "images/all junction resolves", str(image_dir))

    label_files = sorted(label_dir.glob("*.txt"))
    report.check(len(label_files) == TOTAL_IMAGES, f"{TOTAL_IMAGES} label files", f"got {len(label_files)}")

    bad_class, bad_range, bad_fields, boxes, empty = [], [], [], 0, 0
    per_class = {0: 0, 1: 0, 2: 0}
    for path in label_files:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            empty += 1
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                bad_fields.append(path.name)
                continue
            try:
                class_id = int(fields[0])
                values = [float(v) for v in fields[1:]]
            except ValueError:
                bad_fields.append(path.name)
                continue
            if class_id not in (0, 1, 2):
                bad_class.append(path.name)
                continue
            if not all(0.0 <= v <= 1.0 for v in values) or values[2] <= 0 or values[3] <= 0:
                bad_range.append(path.name)
                continue
            per_class[class_id] += 1
            boxes += 1

    report.check(not bad_fields, "every line has 5 fields", f"{len(bad_fields)} bad files e.g. {bad_fields[:3]}")
    report.check(not bad_class, "class ids in {0,1,2}", f"{len(bad_class)} bad files e.g. {bad_class[:3]}")
    report.check(not bad_range, "boxes normalized in [0,1] with positive size", f"{len(bad_range)} bad files e.g. {bad_range[:3]}")
    print(f"  [info] boxes={boxes}  Car={per_class[0]}  Pedestrian={per_class[1]}  Cyclist={per_class[2]}  empty label files={empty}")

    report.section(f"list files  ({output_root.name})")
    for name, expected in (("train", "train"), ("internal_val", "internal_val"), ("official_eval", "official_eval"), ("calibration", "calibration")):
        list_path = output_root / "lists" / f"{name}.txt"
        if not report.check(list_path.is_file(), f"lists/{name}.txt exists"):
            continue
        entries = read_ids(list_path)
        report.check(len(entries) == EXPECTED[expected], f"lists/{name}.txt has {EXPECTED[expected]} entries", f"got {len(entries)}")
        listed_ids = {Path(e).stem for e in entries}
        report.check(listed_ids == ids[expected], f"lists/{name}.txt ids match {expected} split")
        missing_img = [e for e in entries[:200] if not Path(e).is_file()]
        report.check(not missing_img, f"lists/{name}.txt images exist (first 200 checked)", f"e.g. {missing_img[:2]}")
        missing_lbl = [e for e in entries[:200] if not label_path_for(Path(e)).is_file()]
        report.check(not missing_lbl, f"lists/{name}.txt labels resolve (first 200 checked)", f"e.g. {missing_lbl[:2]}")

    for yaml_name in ("kitti.local.yaml", "calibration.local.yaml"):
        report.check((output_root / yaml_name).is_file(), f"{yaml_name} exists")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti-root", type=Path, default=None, help="Extracted KITTI root; skipped when omitted")
    parser.add_argument(
        "--output-root",
        type=Path,
        action="append",
        default=None,
        help="YOLO dataset root; repeatable. Defaults to data/kitti_yolo_exact3 and data/kitti_yolo_neighbors3 when present.",
    )
    parser.add_argument("--splits-only", action="store_true", help="Check splits only")
    args = parser.parse_args()

    report = Report()
    ids = check_splits(report)

    if not args.splits_only:
        if args.kitti_root is not None:
            check_kitti(report, args.kitti_root, ids)
        outputs = args.output_root
        if outputs is None:
            outputs = [
                REPO_ROOT / "data" / "kitti_yolo_exact3",
                REPO_ROOT / "data" / "kitti_yolo_neighbors3",
            ]
            outputs = [p for p in outputs if p.is_dir()]
            if not outputs:
                print("\n== YOLO dataset\n  [skip] no data/kitti_yolo_* directory yet")
        for output_root in outputs:
            check_yolo(report, output_root, ids)

    print("\n" + "=" * 60)
    if report.failures:
        print(f"FAILED: {len(report.failures)} check(s)")
        for item in report.failures:
            print(f"  - {item}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
