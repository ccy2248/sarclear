#!/usr/bin/env python3
"""LuojiaSET 评估指标可视化：折线图 + LaTeX 表格"""
import json, os
import matplotlib.pyplot as plt
import numpy as np

BASE = "/media/user/newMemery/worksapce/aaai_material/model_output/LuojiaSET_OSFCR_output/SarClear"
OUT  = os.path.join(BASE, "eval_vis.png")
TEX  = os.path.join(BASE, "eval_table.tex")

LEVELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
KEYS   = ["0_20", "20_40", "40_60", "60_80", "80_100"]
METRICS = {
    "PSNR":   ("psnr",  0, "↑"),
    "SSIM":   ("ssim",  1, "↑"),
    "LPIPS":  ("lpips", 2, "↓"),
    "MAE":    ("mae",   3, "↓"),
    "SAM":    ("sam",   4, "↓°"),
}

# 读取数据
data = {m: [] for m in METRICS}
for k in KEYS:
    path = os.path.join(BASE, f"eval_{k}.json")
    with open(path) as f:
        d = json.load(f)
    for name, (key, _, _) in METRICS.items():
        if key in d:
            val = sum(d[key]) / len(d[key])
            if key == "sam":
                val = val * 180 / np.pi  # 弧度 → 度
            data[name].append(val)

# ── 折线图 ──
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
x = np.arange(len(LEVELS))

for i, (name, (_, idx, arrow)) in enumerate(METRICS.items()):
    ax = axes[idx]
    ax.plot(x, data[name], "o-", linewidth=2, markersize=8, color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS, rotation=30)
    ax.set_title(f"{name} ({arrow})", fontsize=13)
    ax.grid(True, alpha=0.3)

axes[5].set_visible(False)  # 隐藏第6个空子图
fig.suptitle("SarClear on LuojiaSET-OSFCR — Metrics vs Cloud Coverage", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Chart saved: {OUT}")

# ── LaTeX 表格 ──
with open(TEX, "w") as f:
    f.write(r"\begin{table}[h]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{SarClear on LuojiaSET-OSFCR across cloud coverage levels.}" + "\n")
    f.write(r"\label{tab:luojia_eval}" + "\n")
    cols = "l" + "c" * len(METRICS)
    f.write(r"\begin{tabular}{" + cols + r"}" + "\n")
    f.write(r"\toprule" + "\n")
    header = "Cloud Coverage & " + " & ".join(f"{n} ({a})" for n, (_, _, a) in METRICS.items()) + r" \\" + "\n"
    f.write(header)
    f.write(r"\midrule" + "\n")
    for i, lv in enumerate(LEVELS):
        vals = " & ".join(f"{data[n][i]:.4f}" for n in METRICS)
        f.write(f"${lv}$ & {vals} \\\\" + "\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")
print(f"Table saved: {TEX}")

# ── 终端打印 ──
print(f"\n{'Level':<10}", end="")
for n in METRICS: print(f"{n:>10}", end="")
print()
for i, lv in enumerate(LEVELS):
    print(f"{lv:<10}", end="")
    for n in METRICS: print(f"{data[n][i]:>10.4f}", end="")
    print()
