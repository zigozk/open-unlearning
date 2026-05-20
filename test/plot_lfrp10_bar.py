from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


labels = ["canonical", "metric", "multi-view"]
group_a = [0.722, 0.679, 0.691]
group_b = [0.679, 0.607, 0.615]


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "SimHei",
            "Microsoft YaHei",
            "Arial Unicode MS",
            "DejaVu Sans",
            "Arial",
        ],
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


fig, ax = plt.subplots(figsize=(6.2, 4.4))

x = np.arange(len(labels))
width = 0.34

bars_a = ax.bar(
    x - width / 2,
    group_a,
    width,
    color="#BDBDBD",
    edgecolor="#333333",
    linewidth=0.8,
    label="Raw（去偏前）",
    zorder=3,
)
bars_b = ax.bar(
    x + width / 2,
    group_b,
    width,
    color="#4C78A8",
    edgecolor="#333333",
    linewidth=0.8,
    label="Centered（去偏后）",
    zorder=3,
)

for bars in [bars_a, bars_b]:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#111111",
        )

# ax.set_title("LFRP@10", pad=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.9)
ax.set_yticks(np.arange(0, 0.9, 0.2))
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
ax.legend(
    frameon=True,
    framealpha=0.94,
    edgecolor="#B8C0CC",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
)

for spine in ax.spines.values():
    spine.set_color("#444444")
    spine.set_linewidth(0.9)

fig.tight_layout(rect=(0, 0, 1, 0.92))

out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "lfrp10_bar.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
fig.savefig(out_dir / "lfrp10_bar.pdf", bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print("Saved to figures/lfrp10_bar.png")
print("Saved to figures/lfrp10_bar.pdf")
