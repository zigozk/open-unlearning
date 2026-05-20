from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


# =========================
# 1. Data
# =========================
data = [
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

df = pd.DataFrame(
    data,
    columns=["方法", "MYTOFU_Mem", "MYTOFU_Utility", "BUS", "类别"],
)


# =========================
# 2. Style
# =========================
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

method_colors = {
    "梯度优化类": "#4C78A8",
    "偏好优化类": "#72B7B2",
    "表示干预类": "#F58518",
    "框架对照类": "#8E8E8E",
}

method_markers = {
    "梯度优化类": "o",
    "偏好优化类": "s",
    "表示干预类": "^",
    "框架对照类": "D",
}

region_styles = {
    "高遗忘 / 低效用": "#F4A261",
    "低遗忘 / 高效用": "#5DB7A8",
    "折中区域": "#C9CED6",
    "低遗忘 / 低效用": "#E6E8EC",
}


# =========================
# 3. Plot
# =========================
fig, ax = plt.subplots(figsize=(7.0, 5.0))

x_min, x_max = 0.00, 0.50
y_min, y_max = 0.52, 0.74
x_split = 0.23
y_split = 0.66

# Region backgrounds. Higher MYTOFU Mem means stronger forgetting in this plot.
ax.axvspan(
    x_split,
    x_max,
    ymin=0,
    ymax=(y_split - y_min) / (y_max - y_min),
    facecolor=region_styles["高遗忘 / 低效用"],
    alpha=0.28,
    zorder=0,
)
ax.axvspan(
    x_min,
    x_split,
    ymin=(y_split - y_min) / (y_max - y_min),
    ymax=1,
    facecolor=region_styles["低遗忘 / 高效用"],
    alpha=0.28,
    zorder=0,
)
ax.axvspan(
    x_split,
    x_max,
    ymin=(y_split - y_min) / (y_max - y_min),
    ymax=1,
    facecolor=region_styles["折中区域"],
    alpha=0.26,
    zorder=0,
)
ax.axvspan(
    x_min,
    x_split,
    ymin=0,
    ymax=(y_split - y_min) / (y_max - y_min),
    facecolor=region_styles["低遗忘 / 低效用"],
    alpha=0.24,
    zorder=0,
)

ax.axvline(x=x_split, color="#9A9A9A", linestyle="--", linewidth=1.1, alpha=0.75)
ax.axhline(y=y_split, color="#9A9A9A", linestyle="--", linewidth=1.1, alpha=0.75)

for category in ["梯度优化类", "偏好优化类", "表示干预类", "框架对照类"]:
    sub = df[df["类别"] == category]
    ax.scatter(
        sub["MYTOFU_Mem"],
        sub["MYTOFU_Utility"],
        s=150,
        c=method_colors[category],
        marker=method_markers[category],
        edgecolors="black",
        linewidths=0.65,
        alpha=0.92,
        label=category,
        zorder=3,
    )

ax.set_xlabel("MYTOFU Mem")
ax.set_ylabel("MYTOFU Utility")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.35, zorder=1)

region_handles = [
    Patch(facecolor=color, edgecolor="none", alpha=0.28, label=label)
    for label, color in region_styles.items()
]

method_handles = [
    plt.Line2D(
        [0],
        [0],
        marker=method_markers[category],
        color="w",
        markerfacecolor=method_colors[category],
        markeredgecolor="black",
        markeredgewidth=0.65,
        markersize=10,
        label=category,
    )
    for category in ["梯度优化类", "偏好优化类", "表示干预类", "框架对照类"]
]

legend = ax.legend(
    handles=method_handles + region_handles,
    frameon=True,
    framealpha=0.92,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    columnspacing=1.2,
    handletextpad=0.5,
)

fig.tight_layout(rect=(0, 0, 1, 0.88))


# =========================
# 4. Save
# =========================
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "mytofu_mem_utility_tradeoff_scatter.pdf"
png_path = out_dir / "mytofu_mem_utility_tradeoff_scatter.png"

fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)

plt.show()

print(f"Saved PDF to: {pdf_path}")
print(f"Saved PNG to: {png_path}")
