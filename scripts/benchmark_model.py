#!/usr/bin/env python3
"""Measure eager PyTorch forward latency (not preprocessing/NMS or Hailo latency)."""
import argparse
import copy
import json
import math
import platform
import statistics
import time
from pathlib import Path
from baseline_common import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'runs/benchmark.json')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0', help='Single CUDA index or cpu')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--repeats', type=int, default=200)
    p.add_argument('--half', action='store_true')
    a = p.parse_args()
    if min(a.imgsz, a.batch, a.repeats) <= 0 or a.warmup < 0 or a.imgsz % 32:
        p.error('Positive sizes/repeats, nonnegative warmup and imgsz divisible by 32 required')
    if a.device != 'cpu' and not a.device.isdigit():
        p.error('--device must be cpu or one CUDA index')
    if not a.weights.is_file():
        raise FileNotFoundError(a.weights)
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device
    device = select_device(a.device, batch=a.batch)
    if a.half and device.type != 'cuda':
        p.error('--half requires CUDA')
    yolo = YOLO(str(a.weights))
    if yolo.task != 'detect':
        raise ValueError('Only detection models are supported')
    model = yolo.model.eval().float()
    params = sum(p.numel() for p in model.parameters())
    # Profile the actual requested square input on a disposable CPU model.
    # THOP covers supported operators only, so this is explicitly an estimate.
    flops = {'value': None, 'unit': 'FLOPs/image', 'method': '2 * THOP MACs, batch=1',
             'scope': 'supported forward operators; excludes preprocessing and NMS',
             'status': 'unavailable', 'input_shape': [1, 3, a.imgsz, a.imgsz]}
    try:
        import thop
        with torch.inference_mode():
            macs, _ = thop.profile(copy.deepcopy(model).cpu(),
                                   inputs=(torch.zeros(1, 3, a.imgsz, a.imgsz),), verbose=False)
        if not math.isfinite(macs) or macs <= 0:
            raise ValueError('Profiler returned no usable operation count')
        flops.update(value=float(2 * macs), status='estimate', version=getattr(thop, '__version__', 'unknown'))
    except Exception as exc:
        flops['reason'] = f'{type(exc).__name__}: {exc}'
    model = model.to(device)
    model = model.half() if a.half else model.float()
    sample = torch.zeros(a.batch, 3, a.imgsz, a.imgsz, device=device,
                         dtype=torch.float16 if a.half else torch.float32)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    durations = []
    with torch.inference_mode():
        for _ in range(a.warmup):
            model(sample)
        synchronize()
        for _ in range(a.repeats):
            synchronize()
            start = time.perf_counter()
            model(sample)
            synchronize()
            durations.append((time.perf_counter() - start) * 1000)
    mean = statistics.mean(durations)
    report = {
        'weights': str(a.weights.resolve()), 'parameters': params,
        'checkpoint_bytes': a.weights.stat().st_size,
        'checkpoint_mib': a.weights.stat().st_size / 1024 ** 2,
        'flops': flops, 'input_shape': list(sample.shape),
        'precision': 'fp16' if a.half else 'fp32', 'device': str(device),
        'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'cuda_runtime': torch.version.cuda, 'python': platform.python_version(),
        'torch_threads': torch.get_num_threads(), 'platform': platform.platform(),
        'warmup': a.warmup, 'repeats': a.repeats, 'fused': False,
        'latency_scope': 'eager forward, resident synthetic tensor; excludes IO/transfers/preprocess/NMS',
        'latency_ms_per_batch': {'mean': mean, 'median': statistics.median(durations),
                                 'p95': sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
                                 'min': min(durations), 'max': max(durations)},
        'amortized_ms_per_image': mean / a.batch,
        'images_per_second': a.batch * 1000 / mean, 'samples_ms': durations,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Benchmark saved to {a.output}; mean forward {mean:.3f} ms/batch')


if __name__ == '__main__':
    main()
