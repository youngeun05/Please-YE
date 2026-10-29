#!/usr/bin/env python3
"""Check whether this CPU's ONNX Runtime INT8 kernel saturates (u8 x s8).

    python scripts/int8_probe.py

Why: ONNX Runtime runs uint8-activation x int8-weight products with the CPU's
best instruction. With VNNI (AVX-VNNI on Intel 12th gen+, AVX512-VNNI on
Cascade Lake+ / Zen 4) products are accumulated in 32 bits and are exact.
Without VNNI (AVX2 / AVX-512 without VNNI, e.g. Broadwell, Skylake, Zen 2/3)
it uses VPMADDUBSW, which adds pairs of products into 16 bits with saturation,
so large activations x large weights are clipped. Same model file, different
INT8 results; FP32 is unaffected. ONNX Runtime documents this behaviour.

The probe multiplies 255 (uint8) by 127 (int8) over 64 terms with MatMulInteger
on the CPU provider. Exact answer: 255 * 127 * 64 = 2,072,640. A saturating
kernel clips each pair (255*127*2 = 64,770) to 32,767 and returns about half.

Exit code 0 always; the verdict is printed and returned by probe().
"""

from __future__ import annotations

import json
import platform
import sys


def probe() -> dict:
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper

    k, m, n = 64, 8, 8
    graph = helper.make_graph(
        [helper.make_node("MatMulInteger", ["a", "b"], ["y"])],
        "int8_probe",
        [helper.make_tensor_value_info("a", TensorProto.UINT8, [m, k]),
         helper.make_tensor_value_info("b", TensorProto.INT8, [k, n])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [m, n])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    a = np.full((m, k), 255, dtype=np.uint8)
    b = np.full((k, n), 127, dtype=np.int8)
    got = int(session.run(None, {"a": a, "b": b})[0].min())
    exact = 255 * 127 * k
    return {
        "saturates": got != exact,
        "int8_path": "saturating u8s8 (no VNNI)" if got != exact else "exact (VNNI)",
        "expected_exact": exact,
        "got": got,
        "onnxruntime": ort.__version__,
        "onnx": onnx.__version__,
        "machine": platform.machine(),
        "processor": platform.processor() or "",
    }


def main() -> None:
    result = probe()
    print(json.dumps(result, indent=2))
    if result["saturates"]:
        print("\nThis CPU uses the SATURATING INT8 path (no VNNI). INT8 accuracy differs from VNNI CPUs; "
              "see README 'CPU에 따른 INT8 결과'.")
    else:
        print("\nThis CPU computes INT8 products exactly (VNNI). INT8 results match the report's main values.")


if __name__ == "__main__":
    sys.exit(main())
