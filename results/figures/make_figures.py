"""Report figures: accuracy-latency trade-off (6.5) and layer-wise INT8 SQNR (6.5 loss points)."""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo, layer_dir, out_dir = (Path(p) for p in sys.argv[1:4])
out_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})

# --- Figure 1: eval_val AP40 vs latency, same-session speed runs (results/640) ---
runs = [json.loads((repo / f"results/640/benchmark_640_pcores_run{r}.json").read_text(encoding="utf-8")) for r in (1, 2)]
p50 = [sum(run["models"][i]["runtime"]["latency_ms"]["median"] for run in runs) / 2 for i in range(4)]
summary = json.loads((repo / "results/640/eval_val_summary.json").read_text(encoding="utf-8"))
print("summary keys:", list(summary)[:12])


def mean_ap(label):
    return summary["accuracy"][label]["mean_ap40_moderate"]


points = [  # (name, speed-run index, summary label, GFLOPs, marker)
    ("1248 FP32 (Baseline)", 0, "baseline", 7.47, "o"),
    ("1248 INT8 (제출)", 1, "y1248_int8", 7.47, "*"),
    ("640 FP32", 2, "y640_fp32", 1.90, "o"),
    ("640 INT8", 3, "optimized", 1.90, "s"),
]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
for name, i, label, gflops, marker in points:
    ap = mean_ap(label)
    color = "#d62728" if "제출" in name else ("#1f77b4" if name.startswith("1248") else "#7f7f7f")
    ax.scatter(p50[i], ap, s=160 if marker == "*" else 60, marker=marker, color=color, zorder=3)
    ax.annotate(f"{name}\n{ap:.2f} / {p50[i]:.1f} ms / {gflops:.2f} GFLOPs", (p50[i], ap),
                textcoords="offset points", xytext={"640 INT8": (8, -22), "640 FP32": (8, 4)}.get(name, (8, -4)),
                fontsize=8)
    print(f"{name}: AP40 {ap:.2f}, p50 {p50[i]:.2f} ms")
ax.set_xlabel("추론 Latency p50 (ms, i9-12900 P코어 4스레드, 배치 1)")
ax.set_ylabel("eval_val Moderate AP40 (평균)")
ax.set_xlim(0, 36)
ax.set_ylim(87.5, 95.5)
ax.grid(alpha=0.3)
ax.set_title("그림 1. 해상도·INT8 운용점별 정확도와 지연 (같은 세션 측정)", fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / "fig1_tradeoff.png")
plt.close(fig)

# --- Figure 2: accumulated INT8 error (xmodel SQNR) at each block output, internal_val 16 images ---
rows = list(csv.DictReader((layer_dir / "block_sqnr.csv").open(encoding="utf-8")))
rows = [r for r in rows if int(r["block"]) <= 22]
x = [int(r["block"]) for r in rows]
y = [float(r["xmodel_err_out_db"]) for r in rows]
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
ax.axvspan(-0.5, 10.5, color="#1f77b4", alpha=0.06)
ax.axvspan(10.5, 22.5, color="#2ca02c", alpha=0.06)
ax.plot(x, y, marker="o", color="#333333", lw=1.2, ms=4)
ax.scatter([2, 10], [y[2], y[10]], color="#d62728", s=40, zorder=3)
ax.annotate("model.2 C3k2\n-6.0 dB", (2, y[2]), textcoords="offset points", xytext=(6, -22), fontsize=8, color="#d62728")
ax.annotate("model.10 C2PSA -4.0 dB\n(Softmax 자체 16.9 dB, 최저)", (10, y[10]), textcoords="offset points", xytext=(-10, -26), fontsize=8, color="#d62728")
ax.text(5, 31.2, "backbone", ha="center", fontsize=8, color="#1f77b4")
ax.text(16.5, 31.2, "neck (head 입력: 16·19·22)", ha="center", fontsize=8, color="#2ca02c")
ax.set_xticks(range(0, 23, 2))
ax.set_xlabel("블록 (Ultralytics model.N)")
ax.set_ylabel("누적 SQNR (dB, 높을수록 FP32와 가까움)")
ax.set_ylim(13, 32.5)
ax.grid(alpha=0.3)
ax.set_title("그림 2. 블록 출력의 INT8 누적 오차 (internal_val 16장, ORT qdq_loss_debug)", fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / "fig2_layer_sqnr.png")
plt.close(fig)
print("saved", sorted(p.name for p in out_dir.glob("*.png")))
