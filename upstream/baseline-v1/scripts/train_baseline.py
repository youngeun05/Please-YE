#!/usr/bin/env python3
"""Fine-tune pretrained YOLO11 on the fixed exact3 training split."""
import argparse
import json
from pathlib import Path
from baseline_common import ROOT, check_names, check_splits


def validate_data(data_path, root=ROOT):
    import yaml
    train, val, official = check_splits(root)
    data = yaml.safe_load(data_path.read_text(encoding='utf-8-sig'))
    check_names(data['names'])
    base = Path(data['path'])
    if not base.is_absolute():
        raise ValueError('Dataset path must be absolute; rerun convert_kitti_to_yolo.py')
    report = json.loads((base / 'conversion_report.local.json').read_text(encoding='utf-8'))
    if report.get('neighbor_policy') != 'ignore':
        raise ValueError('Baseline requires exact3 (neighbor_policy=ignore)')
    # Inspect the actual manifests consumed by YOLO, not just the split files.
    for key, expected in [('train', train), ('val', val)]:
        if not isinstance(data[key], str):
            raise ValueError('Expected a single converter-generated manifest per split')
        manifest = base / data[key]
        paths = [Path(s.strip()) for s in manifest.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        ids = [p.stem for p in paths]
        if len(ids) != len(set(ids)) or set(ids) != expected or set(ids) & official:
            raise ValueError(f'Leakage guard: actual {key} manifest differs from fixed split')
        for p in paths:
            # absolute() deliberately preserves the images/all junction.
            wanted = base / 'images/all' / f'{p.stem}.png'
            if not p.is_absolute() or p.absolute() != wanted.absolute():
                raise ValueError(f'Unexpected image path (preserve images/all junction): {p}')
            label = base / 'labels/all' / f'{p.stem}.txt'
            if not p.is_file() or not label.is_file():
                raise FileNotFoundError(f'Missing image or converted label: {p}')
    return {'train': len(train), 'internal_val': len(val), 'official_eval_excluded': len(official)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/kitti_yolo_exact3/kitti.local.yaml')
    p.add_argument('--model', default='yolo11n.pt')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--device', default='0')
    p.add_argument('--seed', type=int, default=20260911)
    p.add_argument('--project', type=Path, default=ROOT / 'runs/baseline')
    p.add_argument('--name', default='yolo11n_640_exact3')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--serial-cache', action='store_true', help='Restricted Windows cache scanner compatibility')
    p.add_argument('--check-only', action='store_true', help='Validate data without loading weights or training')
    a = p.parse_args()
    if a.imgsz <= 0 or a.imgsz % 32 or a.epochs <= 0 or a.batch <= 0 or a.workers < 0:
        p.error('Use positive epochs/batch, nonnegative workers and imgsz divisible by 32')
    if not a.model.endswith('.pt'):
        p.error('--model must be pretrained .pt weights, not a scratch .yaml model')
    print(json.dumps(validate_data(a.data), indent=2))
    if a.check_only:
        return
    from ultralytics import YOLO
    if a.serial_cache:
        import ultralytics.data.dataset as dataset
        from smoke_train import SerialPool
        dataset.ThreadPool = SerialPool
    model = YOLO(a.model)
    if model.task != 'detect':
        raise ValueError('Only detection models are supported')
    model.train(data=str(a.data.resolve()), imgsz=a.imgsz, epochs=a.epochs,
                batch=a.batch, device=a.device, seed=a.seed, project=str(a.project.absolute()),
                name=a.name, workers=a.workers, amp=a.amp, deterministic=True,
                pretrained=True, fraction=1.0, val=True, cache=False, exist_ok=False,
                patience=0, save=True)
    print(f'Checkpoint: {model.trainer.best}')


if __name__ == '__main__':
    main()
