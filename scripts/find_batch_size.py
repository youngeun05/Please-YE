#!/usr/bin/env python3
"""Measure the real training VRAM footprint on this GPU to pick a batch size.

The original README was written on RTX A6000 48GB x2; this PC has a single
RTX 3080 10GB, so the batch size has to be re-derived here rather than copied.

This does not need the KITTI images. Each trial builds the actual Ultralytics
detection model, runs forward -> detection loss -> backward -> optimizer step on
synthetic batches shaped like KITTI (5 boxes per image on average), and reports
the peak CUDA memory and the seconds per step. A trial that raises CUDA OOM is
reported as OOM instead of crashing the sweep.

Numbers are a floor, not a ceiling: caching, plotting and validation add memory
on top, which is why --budget-gb defaults below the free VRAM.
"""

from __future__ import annotations

import argparse
import time

import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG

NUM_CLASSES = 3          # Car, Pedestrian, Cyclist
BOXES_PER_IMAGE = 5      # KITTI averages ~4.6 eval-class boxes per image
TRAIN_IMAGES = 5_481     # splits/train.txt, for the epoch-time estimate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="yolo11n,yolo11s",
                        help="comma-separated Ultralytics model yamls (without .yaml)")
    parser.add_argument("--imgsz", default="1280x384,640x192",
                        help="comma-separated WIDTHxHEIGHT, or a bare number for square. "
                             "The benchmark notice caps width at 1280, requires multiples of 32 "
                             "and strongly recommends rectangles matching KITTI's ~3.31 aspect "
                             "(1280x384 for detail, 640x192 for speed); square input wastes NPU "
                             "compute on letterbox padding.")
    parser.add_argument("--batches", default="8,16,32,48,64",
                        help="comma-separated batch sizes, ascending")
    parser.add_argument("--amp", default="on", choices=("on", "off", "both"),
                        help="fp16 autocast + GradScaler, as Ultralytics uses it")
    parser.add_argument("--budget-gb", type=float, default=8.0,
                        help="peak allocated ceiling treated as safe")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def build_model(cfg: str) -> DetectionModel:
    model = DetectionModel(cfg=f"{cfg}.yaml", nc=NUM_CLASSES, verbose=False)
    # v8DetectionLoss reads box/cls/dfl gains off model.args.
    model.args = get_cfg(DEFAULT_CFG)
    return model.cuda().train()


MAX_WIDTH = 1280  # benchmark notice: "최대 가로 해상도는 1280 이하로 제한"


def parse_size(token: str) -> tuple[int, int]:
    """Accept '1280x384' or a bare '640' (square) and validate against the notice."""
    token = token.strip().lower()
    if "x" in token:
        width_text, height_text = token.split("x", 1)
        width, height = int(width_text), int(height_text)
    else:
        width = height = int(token)
    if width % 32 or height % 32:
        raise SystemExit(f"{token}: both sides must be multiples of 32 for the Hailo/YOLO stride")
    if width > MAX_WIDTH:
        raise SystemExit(f"{token}: width exceeds the {MAX_WIDTH} cap set by the benchmark notice")
    return width, height


def make_batch(batch: int, width: int, height: int) -> dict[str, torch.Tensor]:
    total = batch * BOXES_PER_IMAGE
    generator = torch.Generator().manual_seed(20260917)
    # Centers kept away from the border so every synthetic box stays inside the image.
    centers = torch.rand((total, 2), generator=generator) * 0.6 + 0.2
    sizes = torch.rand((total, 2), generator=generator) * 0.15 + 0.03
    return {
        "img": torch.rand((batch, 3, height, width), generator=generator).cuda(non_blocking=True),
        "batch_idx": torch.arange(batch).repeat_interleave(BOXES_PER_IMAGE).float().cuda(),
        "cls": torch.randint(0, NUM_CLASSES, (total, 1), generator=generator).float().cuda(),
        "bboxes": torch.cat([centers, sizes], dim=1).cuda(),
    }


def trial(cfg: str, size: tuple[int, int], batch: int, amp: bool, warmup: int, steps: int) -> dict:
    width, height = size
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = {"model": cfg, "size": f"{width}x{height}", "batch": batch, "amp": amp}
    model = None
    try:
        model = build_model(cfg)
        # AdamW is Ultralytics' 'auto' pick at this dataset size and is the
        # memory-heavier option (two optimizer states per parameter).
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}  # Ultralytics keeps an EMA copy
        data = make_batch(batch, width, height)

        for index in range(warmup + steps):
            if index == warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, _ = model.loss(data)
            scaler.scale(loss.sum()).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / steps
        result["peak_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        result["reserved_gb"] = torch.cuda.max_memory_reserved() / 1024 ** 3
        result["sec_per_step"] = elapsed
        result["epoch_min"] = (TRAIN_IMAGES / batch) * elapsed / 60
        result["status"] = "ok"
        del ema
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except RuntimeError as error:
        result["status"] = "OOM" if "out of memory" in str(error).lower() else f"ERR: {error}"
    finally:
        del model
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this sweep is meaningless on CPU")

    free, total = torch.cuda.mem_get_info()
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
    print(f"vram total : {total / 1024 ** 3:.2f} GB")
    print(f"vram free  : {free / 1024 ** 3:.2f} GB   (the desktop session holds the rest)")
    print(f"budget     : {args.budget_gb:.2f} GB peak allocated treated as safe")
    print(f"synthetic  : {BOXES_PER_IMAGE} boxes/image, nc={NUM_CLASSES}, AdamW, EMA copy held")

    torch.backends.cudnn.benchmark = True
    amp_modes = {"on": [True], "off": [False], "both": [True, False]}[args.amp]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [parse_size(v) for v in args.imgsz.split(",")]
    batches = [int(v) for v in args.batches.split(",")]

    rows: list[dict] = []
    header = f"\n{'model':<9} {'size':>10} {'amp':>4} {'batch':>5} {'peak GB':>8} {'resv GB':>8} {'s/step':>7} {'epoch min':>9}  verdict"
    print(header)
    print("-" * len(header.strip()))

    for cfg in models:
        for size in sizes:
            label = f"{size[0]}x{size[1]}"
            for amp in amp_modes:
                for batch in batches:
                    row = trial(cfg, size, batch, amp, args.warmup, args.steps)
                    rows.append(row)
                    tag = "amp" if amp else "fp32"
                    if row["status"] == "ok":
                        safe = row["peak_gb"] <= args.budget_gb
                        verdict = "fits" if safe else "over budget"
                        print(f"{cfg:<9} {label:>10} {tag:>4} {batch:>5} "
                              f"{row['peak_gb']:>8.2f} {row['reserved_gb']:>8.2f} "
                              f"{row['sec_per_step']:>7.3f} {row['epoch_min']:>9.1f}  {verdict}")
                        if not safe:
                            break  # larger batches only get worse
                    else:
                        print(f"{cfg:<9} {label:>10} {tag:>4} {batch:>5} "
                              f"{'-':>8} {'-':>8} {'-':>7} {'-':>9}  {row['status']}")
                        break

    print("\n== largest batch within budget")
    best: dict[tuple[str, str, bool], dict] = {}
    for row in rows:
        if row["status"] != "ok" or row["peak_gb"] > args.budget_gb:
            continue
        key = (row["model"], row["size"], row["amp"])
        if key not in best or row["batch"] > best[key]["batch"]:
            best[key] = row
    if not best:
        print("  nothing fit the budget; lower --imgsz or use a smaller model")
        return 1
    for (cfg, size, amp), row in sorted(best.items()):
        print(f"  {cfg:<9} {size:<10} {'amp' if amp else 'fp32':<4} "
              f"batch={row['batch']:<3} peak={row['peak_gb']:.2f} GB  "
              f"epoch~{row['epoch_min']:.1f} min (GPU-bound, excludes disk I/O)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
