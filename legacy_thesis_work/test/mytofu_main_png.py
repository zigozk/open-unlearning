from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


records = [
    ("CEU", 0.459, 0.543, 0.497, "梯度优化类"),
    ("DPO", 0.338, 0.674, 0.450, "偏好优化类"),
    ("GradAscent", 0.344, 0.648, 0.450, "梯度优化类"),
    ("GradDiff", 0.370, 0.624, 0.464, "梯度优化类"),
    ("GradDiff pcgrad", 0.364, 0.617, 0.458, "梯度优化类"),
    ("GradDiff sago", 0.188, 0.704, 0.297, "梯度优化类"),
    ("NPO", 0.353, 0.656, 0.459, "偏好优化类"),
    ("NPO pcgrad", 0.349, 0.662, 0.457, "偏好优化类"),
    ("NPO sago", 0.298, 0.654, 0.410, "偏好优化类"),
    ("PDU", 0.370, 0.624, 0.464, "框架对照类"),
    ("RMU", 0.333, 0.668, 0.445, "表示干预类"),
    ("SatImp", 0.370, 0.624, 0.464, "框架对照类"),
    ("SimNPO", 0.113, 0.705, 0.195, "偏好优化类"),
    ("SimNPO pcgrad", 0.125, 0.717, 0.213, "偏好优化类"),
    ("SimNPO sago", 0.033, 0.711, 0.064, "偏好优化类"),
    ("UNDIAL", 0.344, 0.648, 0.450, "框架对照类"),
    ("WGA", 0.370, 0.624, 0.464, "框架对照类"),
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

category_colors = {
    "梯度优化类": "#1F5A92",
    "偏好优化类": "#72B7B2",
    "表示干预类": "#8E8E8E",
    "框架对照类": "#C5CDD7",
}
category_markers = {
    "梯度优化类": "o",
    "偏好优化类": "s",
    "表示干预类": "^",
    "框架对照类": "D",
}

fig, ax = plt.subplots(figsize=(6.4, 4.9))

x_min, x_max = 0.02, 0.49
y_min, y_max = 0.52, 0.73
mem_mid = 0.30
utility_mid = 0.64

ax.axvspan(x_min, mem_mid, ymin=(utility_mid - y_min) / (y_max - y_min), ymax=1, color="#D8F1F3", alpha=0.85, zorder=0)
ax.axvspan(mem_mid, x_max, ymin=(y_min - y_min) / (y_max - y_min), ymax=(utility_mid - y_min) / (y_max - y_min), color="#DDE3EA", alpha=0.85, zorder=0)
ax.axvspan(mem_mid, x_max, ymin=(utility_mid - y_min) / (y_max - y_min), ymax=1, color="#E8EEF3", alpha=0.65, zorder=0)
ax.axvspan(x_min, mem_mid, ymin=0, ymax=(utility_mid - y_min) / (y_max - y_min), color="#F2F5F8", alpha=0.85, zorder=0)

ax.text(0.045, 0.650, "低遗忘 / 高效用", color="#225B65", fontsize=10, ha="left", va="bottom")
ax.text(0.365, 0.545, "高遗忘 / 低效用", color="#45505A", fontsize=10, ha="center", va="center")
ax.text(0.386, 0.705, "折中区域", color="#3B4A57", fontsize=10, ha="center", va="center")

for category in category_colors:
    subset = [r for r in records if r[4] == category]
    ax.scatter(
        [r[1] for r in subset],
        [r[2] for r in subset],
        s=86,
        marker=category_markers[category],
        label=category,
        color=category_colors[category],
        edgecolor="#1A1A1A",
        linewidth=0.7,
        alpha=0.95,
        zorder=3,
    )

ax.scatter([], [], s=120, marker="s", color="#E8EEF3", edgecolor="#AAB5C2", label="折中区域")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("遗忘侧综合得分（MYTOFU Mem）")
ax.set_ylabel(
    "保\n留\n侧\n综\n合\n得\n分\n（\nMYTOFU\nUtility\n）",
    rotation=0,
    labelpad=50,
    ha="center",
    va="center",
)
ax.grid(color="#D8DEE8", linewidth=0.75, alpha=0.85, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=True,
    framealpha=0.96,
    edgecolor="#B8C0CC",
)

fig.tight_layout(rect=(0, 0, 1, 0.87))

out_dir = Path("figures_new")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "mytofu_mem_utility_tradeoff_scatter.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(out_dir / "mytofu_mem_utility_tradeoff_scatter.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

print("Saved to figures_new/mytofu_mem_utility_tradeoff_scatter.pdf")
print("Saved to figures_new/mytofu_mem_utility_tradeoff_scatter.png")
