#!/usr/bin/env python3
"""Write original-image-coordinate KITTI 16-column bbox detections."""
import argparse
import json
from pathlib import Path
from baseline_common import ROOT, read_ids, check_names, kitti_row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--kitti-root', type=Path, required=True)
    p.add_argument('--split', type=Path, default=ROOT / 'splits/internal_val.txt')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/predictions/yolo11n_640_exact3')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.7)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--allow-official-eval', action='store_true', help='Final evaluation only; never tune on official IDs')
    a = p.parse_args()
    if not 0 <= a.conf <= 1 or not 0 <= a.iou <= 1 or min(a.batch, a.max_det, a.imgsz) <= 0 or a.imgsz % 32:
        p.error('Invalid confidence/IoU/batch/max-det/imgsz')
    ids = read_ids(a.split)
    if set(ids) & set(read_ids(ROOT / 'eval_val.txt')) and not a.allow_official_eval:
        p.error('Official evaluation is held out; use --allow-official-eval only for final evaluation')
    if not a.weights.is_file():
        raise FileNotFoundError(a.weights)
    paths = [a.kitti_root / 'training/image_2' / f'{i}.png' for i in ids]
    missing = [str(x) for x in paths if not x.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} images: {missing[:3]}')
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError('Use a new or empty output directory to avoid stale predictions')
    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    if model.task != 'detect':
        raise ValueError('Only detection models are supported')
    check_names(model.names)
    a.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    for start in range(0, len(paths), a.batch):
        batch = paths[start:start + a.batch]
        results = model.predict(source=[str(x) for x in batch], imgsz=a.imgsz,
                                device=a.device, conf=a.conf, iou=a.iou, max_det=a.max_det,
                                rect=False, half=False, augment=False, save=False, verbose=False)
        if len(results) != len(batch):
            raise RuntimeError('Unexpected inference result count')
        for path, result in zip(batch, results):
            if Path(result.path).stem != path.stem:
                raise RuntimeError('Inference result order mismatch')
            # Ultralytics xyxy is already mapped back to original pixel coordinates.
            rows = [kitti_row(int(c), box, score) for box, c, score in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist())]
            (a.output / f'{path.stem}.txt').write_text(''.join(rows), encoding='utf-8')
            completed += 1
    report = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    report.update(samples=completed, coordinate_space='original image pixels', rect=False, half=False)
    (a.output / 'prediction_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {completed} KITTI detection files to {a.output}')


if __name__ == '__main__':
    main()
