#!/usr/bin/env python3
"""Verify KITTI archives and safely extract the labeled training split."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


EXPECTED_IMAGES = 7_481
EXPECTED_LABELS = 7_481
EXPECTED_EVAL_IDS = 1_000
IMAGE_PREFIX = PurePosixPath("training/image_2")
LABEL_PREFIX = PurePosixPath("training/label_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-zip", type=Path, required=True)
    parser.add_argument("--label-zip", type=Path, required=True)
    parser.add_argument("--eval-split", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--skip-full-crc",
        action="store_true",
        help="Skip ZipFile.testzip(). This is faster but does not verify every payload.",
    )
    return parser.parse_args()


def normalized_members(archive: zipfile.ZipFile, prefix: PurePosixPath, suffix: str) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        member = PurePosixPath(info.filename)
        if info.is_dir() or member.parent != prefix or member.suffix.lower() != suffix:
            continue
        image_id = member.stem
        if len(image_id) != 6 or not image_id.isdecimal():
            raise ValueError(f"Unexpected KITTI member name: {info.filename}")
        if image_id in members:
            raise ValueError(f"Duplicate KITTI id in archive: {image_id}")
        members[image_id] = info
    return members


def read_eval_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    invalid = [value for value in ids if len(value) != 6 or not value.isdecimal()]
    if invalid:
        raise ValueError(f"Invalid eval ids: {invalid[:10]}")
    if len(ids) != EXPECTED_EVAL_IDS or len(set(ids)) != EXPECTED_EVAL_IDS:
        raise ValueError(f"Expected {EXPECTED_EVAL_IDS} unique eval ids, got {len(ids)} lines/{len(set(ids))} unique")
    return ids


def full_crc_check(path: Path) -> None:
    print(f"Checking ZIP CRCs: {path}", flush=True)
    with zipfile.ZipFile(path) as archive:
        first_bad = archive.testzip()
    if first_bad is not None:
        raise zipfile.BadZipFile(f"CRC failure in {path}: {first_bad}")


def extract_members(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image_id in enumerate(sorted(members), start=1):
        info = members[image_id]
        destination = output_dir / PurePosixPath(info.filename).name
        if destination.exists():
            if destination.stat().st_size != info.file_size:
                raise FileExistsError(f"Existing file has unexpected size: {destination}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with archive.open(info) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size != info.file_size:
            raise IOError(f"Incomplete extraction: {temporary}")
        temporary.replace(destination)
        if index % 500 == 0:
            print(f"Extracted {index}/{len(members)} files into {output_dir}", flush=True)


def main() -> int:
    args = parse_args()
    for path in (args.image_zip, args.label_zip, args.eval_split):
        if not path.is_file():
            raise FileNotFoundError(path)

    eval_ids = read_eval_ids(args.eval_split)
    if not args.skip_full_crc:
        full_crc_check(args.image_zip)
        full_crc_check(args.label_zip)

    with zipfile.ZipFile(args.image_zip) as image_archive, zipfile.ZipFile(args.label_zip) as label_archive:
        image_members = normalized_members(image_archive, IMAGE_PREFIX, ".png")
        label_members = normalized_members(label_archive, LABEL_PREFIX, ".txt")
        if len(image_members) != EXPECTED_IMAGES:
            raise ValueError(f"Expected {EXPECTED_IMAGES} training images, got {len(image_members)}")
        if len(label_members) != EXPECTED_LABELS:
            raise ValueError(f"Expected {EXPECTED_LABELS} training labels, got {len(label_members)}")
        if image_members.keys() != label_members.keys():
            raise ValueError("Training image and label ids differ")
        missing_eval = sorted(set(eval_ids) - image_members.keys())
        if missing_eval:
            raise ValueError(f"Evaluation ids missing from archives: {missing_eval[:10]}")

        image_dir = args.output_root / "training" / "image_2"
        label_dir = args.output_root / "training" / "label_2"
        extract_members(image_archive, image_members, image_dir)
        extract_members(label_archive, label_members, label_dir)

    manifest = {
        "image_zip": str(args.image_zip.resolve()),
        "label_zip": str(args.label_zip.resolve()),
        "eval_split": str(args.eval_split.resolve()),
        "training_images": EXPECTED_IMAGES,
        "training_labels": EXPECTED_LABELS,
        "official_eval_ids": EXPECTED_EVAL_IDS,
    }
    manifest_path = args.output_root / "dataset_manifest.local.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared KITTI training data at {args.output_root}")
    print(f"Wrote local manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
