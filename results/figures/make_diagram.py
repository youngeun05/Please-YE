"""Report diagram for section 5.3: optimization pipeline (top) and precision map of the submitted model (bottom)."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

out = Path(sys.argv[1])
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})

INT8, FP32, HOST, STEP, EVAL = "#dbe8f7", "#fde3cc", "#e6e6e6", "#f4f4f4", "#e3f1e0"
EDGE = {"int8": "#2f6eb5", "fp32": "#d9822b", "host": "#7a7a7a", "step": "#555555", "eval": "#3d8b37"}

fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=220)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")
ax.text(50, 59.5, "최적화 흐름도 — 적용 순서와 제출 모델의 정밀도 구성", ha="center", va="top", fontsize=9.5, fontweight="bold")


def box(x, y, w, h, title, body, face, edge, title_size=8.2, body_size=6.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                facecolor=face, edgecolor=edge, linewidth=1.1))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color="#222222")
    ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top", fontsize=body_size, color="#333333", linespacing=1.35)


def arrow(x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=color, linewidth=1.1))


# --- top: pipeline ---
ax.text(0.5, 54, "A. 적용 흐름", fontsize=9, fontweight="bold", va="top")
steps = [
    ("FP32 학습", "PyTorch · 1248×384\nKITTI 학습 5,481장"),
    ("ONNX 변환", "opset 17 · 입력 고정\n박스 차이 0.0015 px"),
    ("정적 INT8 PTQ", "ORT quantize_static\nQDQ · 보정 1,024장"),
    ("보정 방식 선택", "internal_val에서\nminmax·entropy·\npercentile 비교"),
    ("측정·최종 채점", "P코어 고정 지연\neval_val 채점\n모델 해시 잠금"),
]
w, h, gap, y = 18.0, 15.0, 2.4, 34.5
for i, (title, body) in enumerate(steps):
    x = 0.5 + i * (w + gap)
    face, edge = (EVAL, EDGE["eval"]) if i >= 3 else (STEP, EDGE["step"])
    box(x, y, w, h, title, body, face, edge)
    if i:
        arrow(x - gap + 0.15, y + h / 2, x - 0.25, y + h / 2)

# --- bottom: precision map of the submitted INT8 model ---
ax.text(0.5, 29.5, "B. 제출 모델의 정밀도 구성 (합성곱은 모두 INT8, 디코딩만 FP32)", fontsize=9, fontweight="bold", va="top")
parts = [
    ("입력", "384×1248\nletterbox", HOST, EDGE["host"], 10.5),
    ("Backbone", "Conv · C3k2 · SPPF\nC2PSA(어텐션)", INT8, EDGE["int8"], 17.5),
    ("Neck", "FPN/PAN\nC3k2 · Concat", INT8, EDGE["int8"], 14.0),
    ("Head 합성곱", "박스 cv2 · 분류 cv3\n3개 스케일", INT8, EDGE["int8"], 16.0),
    ("디코딩 24노드", "DFL · Sigmoid\n좌표 계산 · Concat", FP32, EDGE["fp32"], 17.0),
    ("NMS", "호스트 CPU\n(그래프 밖)", HOST, EDGE["host"], 11.5),
]
x, y, h = 0.5, 9.0, 16.0
for i, (title, body, face, edge, w) in enumerate(parts):
    box(x, y, w, h, title, body, face, edge)
    if i < len(parts) - 1:
        arrow(x + w + 0.2, y + h / 2, x + w + 2.2, y + h / 2)
    x += w + 2.4

# legend and notes
for k, (face, edge, label) in enumerate([(INT8, EDGE["int8"], "INT8 (가중치 int8 채널별 · 활성값 uint8)"),
                                          (FP32, EDGE["fp32"], "FP32 (전부 INT8이면 검출 0개, 5.3 ①)"),
                                          (HOST, EDGE["host"], "그래프 밖 / 호스트")]):
    lx = [0.5, 39.0, 75.0][k]
    ax.add_patch(FancyBboxPatch((lx, 2.2), 2.6, 2.6, boxstyle="round,pad=0.1,rounding_size=0.5",
                                facecolor=face, edgecolor=edge, linewidth=1.0))
    ax.text(lx + 3.6, 3.5, label, fontsize=6.6, va="center", color="#333333")

fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
print("saved", out)
