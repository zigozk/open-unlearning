from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


labels = ["规范化文本", "指标计算文本", "多视图融合"]
raw = [0.722, 0.679, 0.691]
centered = [0.679, 0.607, 0.615]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

x = np.arange(len(labels))
width = 0.28

fig, ax = plt.subplots(figsize=(6.2, 4.4))
bars_raw = ax.bar(
    x - width / 2,
    raw,
    width,
    label="去偏前",
    color="#8E8E8E",
    edgecolor="#1A1A1A",
    linewidth=0.7,
    zorder=3,
)
bars_centered = ax.bar(
    x + width / 2,
    centered,
    width,
    label="去偏后",
    color="#1F5A92",
    edgecolor="#1A1A1A",
    linewidth=0.7,
    zorder=3,
)

for bars in [bars_raw, bars_centered]:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

ax.set_ylabel("局\n部\n纠\n缠\n度\n（LFRP@10）", rotation=0, labelpad=42, ha="center", va="center")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.55, 0.75)
ax.grid(axis="y", color="#D8DEE8", linewidth=0.8, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=True,
    framealpha=0.96,
    edgecolor="#B8C0CC",
)

fig.tight_layout(rect=(0, 0, 1, 0.88))

out_dir = Path("figures_new")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "lfrp10_bar.pdf", bbox_inches="tight", pad_inches=0.04)
fig.savefig(out_dir / "lfrp10_bar.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)

print("Saved to figures_new/lfrp10_bar.pdf")
print("Saved to figures_new/lfrp10_bar.png")
