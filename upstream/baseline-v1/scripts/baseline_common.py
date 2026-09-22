"""Small shared helpers; no training dependencies needed for CLI help/tests."""
from pathlib import Path
import math
import re

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('Car', 'Pedestrian', 'Cyclist')


def read_ids(path):
    ids = Path(path).read_text(encoding='utf-8-sig').split()
    if not ids or len(ids) != len(set(ids)) or any(not re.fullmatch(r'[0-9]{6}', i) for i in ids):
        raise ValueError(f'Expected unique, non-empty six-digit KITTI IDs: {path}')
    return ids


def check_splits(root=ROOT):
    train = set(read_ids(root / 'splits/train.txt'))
    val = set(read_ids(root / 'splits/internal_val.txt'))
    official = set(read_ids(root / 'eval_val.txt'))
    if train & val or train & official or val & official:
        raise ValueError('Leakage guard: train/internal_val/official_eval overlap')
    calibration = set(read_ids(root / 'splits/calibration.txt'))
    if not calibration <= train:
        raise ValueError('Calibration must be a subset of train')
    return train, val, official


def check_names(names):
    actual = names if isinstance(names, list) else [names.get(i) for i in range(3)]
    if len(names) != 3 or list(actual) != list(CLASSES):
        raise ValueError(f'Expected class IDs 0/1/2 = {CLASSES}, got {names}')


def kitti_row(class_id, xyxy, score):
    if class_id not in range(3) or not all(math.isfinite(x) for x in [*xyxy, score]):
        raise ValueError('Invalid detection class or non-finite value')
    if not 0 <= score <= 1 or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
        raise ValueError('Invalid detection score or box')
    # Unknown orientation/3D fields: this output is for bbox evaluation only.
    coords = ' '.join(f'{x:.8f}' for x in xyxy)
    return f'{CLASSES[class_id]} -1 -1 -10 {coords} -1 -1 -1 -1000 -1000 -1000 -10 {score:.8f}\n'
