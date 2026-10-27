#!/usr/bin/env python3
"""Collect the submission model files into models/ with a manifest and one zip.

Run on the PC that trained and quantized the models (PC00):

    python scripts/package_models.py

It copies, from runs/baseline/<run>/weights/ (git-ignored, so only that PC has them):

    best.pt                              -> models/yolo11n_kitti_fp32.pt          (original PyTorch baseline)
    best_1248x384_fp32.onnx              -> models/yolo11n_kitti_1248x384_fp32.onnx  (baseline as measured)
    best_1248x384_int8_<method>.onnx     -> models/yolo11n_kitti_1248x384_int8.onnx  (optimized model)

plus the export/quantization records, then writes models/MODELS.md and
models/manifest.json (SHA-256, size, format, runtime versions, quantization
settings) and models/please-ye-models.zip for a single upload.

Checks before copying:
  * the FP32 ONNX record says it was exported from this best.pt (SHA-256 match);
  * the INT8 record says it was quantized from this FP32 ONNX (SHA-256 match);
  * the INT8 file keeps the Detect-head decoding in float. An INT8 file whose
    head was quantized too scores 0 AP (all class scores round to 0), so it is
    refused.
--include NAME=RUN adds another run (for example the YOLOv8n comparison).
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from onnx_common import ROOT, file_sha256, static_input

DEFAULT_RUN = "yolo11n_exact3_1248"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN, help="run folder under runs/baseline")
    parser.add_argument("--name", default="yolo11n", help="file name prefix in models/")
    parser.add_argument("--method", default="percentile", help="calibration method of the INT8 file to ship")
    parser.add_argument("--imgsz", default="1248x384")
    parser.add_argument("--include", action="append", default=[], metavar="NAME=RUN",
                        help="also package another run, e.g. yolov8n=yolov8n_exact3_1248")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Record not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def quantization_settings(path: Path) -> dict:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    meta = {prop.key: prop.value for prop in model.metadata_props}
    if "please_ye_quantization" not in meta:
        raise SystemExit(f"{path.name}: no please_ye_quantization metadata; was it made by quantize_ort.py?")
    return json.loads(meta["please_ye_quantization"])


def collect(name: str, run: str, method: str, imgsz: str, output: Path) -> list[dict]:
    weights = ROOT / "runs/baseline" / run / "weights"
    pt, fp32, int8 = weights / "best.pt", weights / f"best_{imgsz}_fp32.onnx", weights / f"best_{imgsz}_int8_{method}.onnx"
    for path in (pt, fp32, int8):
        if not path.is_file():
            raise SystemExit(f"Missing: {path}")

    pt_sha, fp32_sha = file_sha256(pt), file_sha256(fp32)
    fp32_record = read_json(fp32.with_suffix(".json"))
    if fp32_record.get("weights_sha256") != pt_sha:
        raise SystemExit(f"{fp32.name} was not exported from this {pt.name} (record SHA-256 differs)")
    int8_record = read_json(int8.with_suffix(".json"))
    if int8_record.get("fp32_sha256") != fp32_sha:
        raise SystemExit(f"{int8.name} was not quantized from this {fp32.name} (record SHA-256 differs)")
    settings = quantization_settings(int8)
    head = settings.get("head_decode", "float" if settings.get("keep_head_fp32") else "unknown")
    if head != "float":
        raise SystemExit(f"{int8.name}: Detect-head decoding is {head!r}, not float. Such a model scores 0 AP; "
                         "re-run scripts/quantize_ort.py without --quantize-head.")

    _, shape = static_input(fp32)
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    plan = [
        (pt, f"{name}_kitti_fp32.pt", "baseline (original)", "PyTorch checkpoint, Ultralytics", None),
        (fp32, f"{name}_kitti_{imgsz}_fp32.onnx", "baseline (measured)", "ONNX FP32, opset "
         f"{fp32_record.get('opset')}", fp32.with_suffix(".json")),
        (int8, f"{name}_kitti_{imgsz}_int8.onnx", "optimized", "ONNX INT8 QDQ (ONNX Runtime static PTQ)",
         int8.with_suffix(".json")),
    ]
    entries = []
    for source, target_name, role, fmt, record in plan:
        target = output / target_name
        shutil.copy2(source, target)
        if file_sha256(target) != file_sha256(source):
            raise SystemExit(f"Copy mismatch: {target}")
        if record is not None:
            shutil.copy2(record, records / f"{Path(target_name).stem}.json")
        entries.append({
            "file": target_name, "role": role, "model": name, "format": fmt,
            "sha256": file_sha256(target), "bytes": target.stat().st_size,
            "input_shape": shape, "source": str(source.relative_to(ROOT)),
        })
    entries[1]["versions"] = fp32_record.get("versions")
    entries[2]["versions"] = int8_record.get("versions")
    entries[2]["quantization"] = settings
    print(f"{name}: packaged {len(entries)} files from runs/baseline/{run}")
    return entries


def write_manifest(entries: list[dict], output: Path) -> None:
    manifest = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "packaged_on": platform.platform(), "models": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Please-YE 제출 모델 파일", "",
             "| 파일 | 역할 | 포맷 | 입력 | 크기 (MiB) | SHA-256 |", "|---|---|---|---|---|---|"]
    for e in entries:
        shape = "x".join(str(v) for v in e["input_shape"])
        lines.append(f"| `{e['file']}` | {e['role']} | {e['format']} | {shape} | "
                     f"{e['bytes'] / 1024 ** 2:.2f} | `{e['sha256']}` |")
    versions = next((e["versions"] for e in entries if e.get("versions")), {}) or {}
    lines += ["", "실행 환경 (export 기록): " + ", ".join(f"{k} {v}" for k, v in versions.items()), "",
              "INT8 설정: " + json.dumps(next(e["quantization"] for e in entries if "quantization" in e),
                                        ensure_ascii=False), "",
              "검증: `python scripts/run_benchmark.py --kitti-root <KITTI>` (README.md 참고).",
              "파일이 바뀌지 않았는지는 SHA-256으로 확인합니다 (Windows: `Get-FileHash <파일>`, Linux: `sha256sum <파일>`)."]
    (output / "MODELS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = collect(args.name, args.run, args.method, args.imgsz, args.output)
    for item in args.include:
        if "=" not in item:
            raise SystemExit(f"--include expects NAME=RUN, got {item!r}")
        name, run = item.split("=", 1)
        entries += collect(name, run, args.method, args.imgsz, args.output)
    write_manifest(entries, args.output)

    if not args.no_zip:
        archive = args.output / "please-ye-models.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for e in entries:
                bundle.write(args.output / e["file"], f"models/{e['file']}")
            for extra in ("manifest.json", "MODELS.md"):
                bundle.write(args.output / extra, f"models/{extra}")
            for record in sorted((args.output / "records").glob("*.json")):
                bundle.write(record, f"models/records/{record.name}")
        print(f"zip     : {archive} ({archive.stat().st_size / 1024 ** 2:.1f} MiB)")
    print(f"manifest: {args.output / 'MODELS.md'}")
    for e in entries:
        print(f"  {e['file']:<42} {e['bytes'] / 1024 ** 2:7.2f} MiB  {e['sha256'][:16]}…")


if __name__ == "__main__":
    main()
