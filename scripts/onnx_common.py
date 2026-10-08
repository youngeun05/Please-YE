"""Shared helpers for the ONNX export / INT8 quantization / benchmark scripts.

Preprocessing here must stay byte-for-byte equivalent to what Ultralytics does
when ``predict_kitti.py`` passes an explicit ``imgsz`` (no adaptive padding):

    BGR image -> LetterBox(new_shape=(H, W), auto=False, scaleup=True, center=True,
                           padding 114, INTER_LINEAR) -> RGB -> CHW -> float32 / 255

If calibration images were preprocessed differently from inference images, the
INT8 ranges would be measured on the wrong distribution and the Drop Rate would
be wrong for a reason unrelated to quantization.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_WIDTH = 1280  # benchmark notice: width <= 1280
PAD_VALUE = 114


def parse_wh(text: str) -> tuple[int, int]:
    """'1248x384' -> (1248, 384), validated against the benchmark notice."""
    try:
        width_text, height_text = text.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except ValueError as error:
        raise SystemExit(f"--imgsz must look like WIDTHxHEIGHT, got {text!r}") from error
    if width % 32 or height % 32:
        raise SystemExit(f"--imgsz {text}: both sides must be multiples of 32")
    if width > MAX_WIDTH:
        raise SystemExit(f"--imgsz {text}: width exceeds the {MAX_WIDTH} cap in the benchmark notice")
    return width, height


def letterbox(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Replicate ultralytics.data.augment.LetterBox(auto=False, center=True)."""
    import cv2

    shape = image_bgr.shape[:2]  # (h, w)
    ratio = min(height / shape[0], width / shape[1])
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))  # (w, h)
    dw = (width - new_unpad[0]) / 2
    dh = (height - new_unpad[1]) / 2
    if (shape[1], shape[0]) != new_unpad:
        image_bgr = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_bgr = cv2.copyMakeBorder(
        image_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    if image_bgr.shape[:2] != (height, width):
        raise AssertionError(f"letterbox produced {image_bgr.shape[:2]}, expected {(height, width)}")
    return image_bgr


def preprocess(image_path: Path, width: int, height: int) -> np.ndarray:
    """Read one KITTI PNG and return a (1, 3, H, W) float32 tensor in [0, 1]."""
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = letterbox(image, width, height)
    tensor = image[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    return tensor[None]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: list[str]) -> str:
    """Same digest convention as make_splits.py (sorted ids, one per line)."""
    payload = "".join(f"{value}\n" for value in sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def static_input(model_path: Path) -> tuple[str, list[int]]:
    """Return (input name, [N, C, H, W]) and refuse dynamic shapes."""
    import onnx

    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {init.name for init in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    if len(inputs) != 1:
        raise SystemExit(f"{model_path}: expected exactly one graph input, found {len(inputs)}")
    dims = inputs[0].type.tensor_type.shape.dim
    shape = [dim.dim_value if dim.HasField("dim_value") else -1 for dim in dims]
    if len(shape) != 4 or min(shape) <= 0:
        raise SystemExit(f"{model_path}: input shape {shape} is not a static NCHW shape; "
                         "re-export with export_onnx.py (dynamic=False)")
    return inputs[0].name, shape
