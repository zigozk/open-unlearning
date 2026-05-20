from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# Data
# =========================
metrics = [
    "标准答案偏好率",
    "归一化概率间隔",
    "自由生成复现率",
]

data = {
    ("完整记忆模型", "forget"): [1.0, 1.0, 0.48],
    ("保留参照模型", "forget"): [0.53, 0.07, 0.04],
    ("完整记忆模型", "retain"): [1.0, 0.87, 0.78],
    ("保留参照模型", "retain"): [1.0, 0.82, 0.84],
}

groups = list(data.keys())


# =========================
# Style
# =========================
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "SimHei",
            "Microsoft YaHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

colors = [
    "#1F5A92",  # 完整记忆模型 / forget
    "#72B7B2",  # 保留参照模型 / forget
    "#8E8E8E",  # 完整记忆模型 / retain
    "#D8F1F3",  # 保留参照模型 / retain
]

hatches = [
    "",
    "//",
    "",
    "\\\\",
]

grid_gray = "#D8DEE8"
text_gray = "#222222"


# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(9.2, 4.8))

x = np.arange(len(metrics))
bar_width = 0.18
offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(groups))

for i, ((model, side), vals) in enumerate(data.items()):
    label = f"{model} · {side}"
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
            color=text_gray,
        )

ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.12)
ax.grid(axis="y", color=grid_gray, linewidth=0.8, linestyle="-", zorder=0)
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


# =========================
# Save
# =========================
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "full_retain_target_sanity.pdf"
png_path = out_dir / "full_retain_target_sanity.png"

fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print(f"Saved PDF to: {pdf_path}")
print(f"Saved PNG to: {png_path}")
