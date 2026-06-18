from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


metrics = [
    "标准答案偏好率",
    "归一化概率间隔",
    "自由生成包含率",
]

data = {
    "完整记忆模型-遗忘侧": [1.00, 1.00, 0.48],
    "保留参照模型-遗忘侧": [0.53, 0.07, 0.04],
    "完整记忆模型-保留侧": [1.00, 0.87, 0.78],
    "保留参照模型-保留侧": [1.00, 0.82, 0.84],
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

colors = ["#1F5A92", "#72B7B2", "#8E8E8E", "#D8F1F3"]
hatches = ["", "//", "", "\\\\"]

fig, ax = plt.subplots(figsize=(9.2, 4.8))
x = np.arange(len(metrics))
bar_width = 0.18
offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(data))

for i, (label, vals) in enumerate(data.items()):
    bars = ax.bar(
        x + offsets[i],
        vals,
        width=bar_width,
        label=label,
        color=colors[i],
        edgecolor="#1A1A1A",
        linewidth=0.75,
        hatch=hatches[i],
        zorder=3,
    )
    for bar, value in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.024,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

ax.set_ylabel("指\n标\n值", rotation=0, labelpad=32, ha="center", va="center")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.12)
ax.grid(axis="y", color="#D8DEE8", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#333333")
ax.spines["bottom"].set_color("#333333")
ax.legend(
    frameon=True,
    fancybox=True,
    framealpha=0.95,
    edgecolor="#B8C0CC",
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
)

fig.tight_layout(rect=(0, 0, 1, 0.88))

out_dir = Path("figures_new")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "full_retain_target_sanity.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(out_dir / "full_retain_target_sanity.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print("Saved to figures_new/full_retain_target_sanity.pdf")
print("Saved to figures_new/full_retain_target_sanity.png")
