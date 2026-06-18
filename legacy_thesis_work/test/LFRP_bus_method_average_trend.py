from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = [
    ("低难度", 0.718, "CEU", 0.545),
    ("低难度", 0.718, "GradDiff", 0.378),
    ("低难度", 0.718, "NPO", 0.464),
    ("低难度", 0.718, "RMU", 0.487),
    ("中等难度", 0.741, "CEU", 0.497),
    ("中等难度", 0.741, "GradDiff", 0.392),
    ("中等难度", 0.741, "NPO", 0.468),
    ("中等难度", 0.741, "RMU", 0.469),
    ("高难度", 0.785, "CEU", 0.421),
    ("高难度", 0.785, "GradDiff", 0.233),
    ("高难度", 0.785, "NPO", 0.334),
    ("高难度", 0.785, "RMU", 0.336),
    ("极高难度", 0.811, "CEU", 0.360),
    ("极高难度", 0.811, "GradDiff", 0.215),
    ("极高难度", 0.811, "NPO", 0.227),
    ("极高难度", 0.811, "RMU", 0.238),
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

df = pd.DataFrame(data, columns=["difficulty", "lfrp", "method", "bus"])
difficulty_order = ["低难度", "中等难度", "高难度", "极高难度"]
method_order = ["CEU", "GradDiff", "NPO", "RMU"]

colors = {
    "CEU": "#1F5A92",
    "GradDiff": "#72B7B2",
    "NPO": "#8E8E8E",
    "RMU": "#C5CDD7",
    "方法平均": "#222222",
}
markers = {"CEU": "o", "GradDiff": "s", "NPO": "^", "RMU": "D", "方法平均": "X"}

fig, ax = plt.subplots(figsize=(6.4, 4.8))

for method in method_order:
    subset = df[df["method"] == method].set_index("difficulty").loc[difficulty_order]
    ax.plot(
        subset["lfrp"],
        subset["bus"],
        marker=markers[method],
        markersize=7,
        linewidth=1.8,
        color=colors[method],
        label=method,
        zorder=3,
    )

avg = df.groupby(["difficulty", "lfrp"], as_index=False)["bus"].mean()
avg = avg.set_index("difficulty").loc[difficulty_order].reset_index()
ax.plot(
    avg["lfrp"],
    avg["bus"],
    marker=markers["方法平均"],
    markersize=8,
    linewidth=2.2,
    linestyle="--",
    color=colors["方法平均"],
    label="方法平均",
    zorder=4,
)

tick_positions = [df[df["difficulty"] == d]["lfrp"].iloc[0] for d in difficulty_order]
tick_labels = [f"{d}\n{pos:.3f}" for d, pos in zip(difficulty_order, tick_positions)]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_xlabel("局部纠缠度难度层")
ax.set_ylabel("平\n衡\n效\n用\n得\n分\n（BUS）", rotation=0, labelpad=42, ha="center", va="center")
ax.set_ylim(0.18, 0.57)
ax.grid(color="#D8DEE8", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    frameon=True,
    framealpha=0.96,
    edgecolor="#B8C0CC",
)

fig.tight_layout(rect=(0, 0, 1, 0.88))

out_dir = Path("figures_new")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "lfrp_bus_method_average_trend.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(out_dir / "lfrp_bus_method_average_trend.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

print("Saved to figures_new/lfrp_bus_method_average_trend.pdf")
print("Saved to figures_new/lfrp_bus_method_average_trend.png")
