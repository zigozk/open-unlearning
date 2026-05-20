import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# =========================
# 1. 数据
# =========================
data = [
    ("GradDiff", "base",   0.370, 0.624, 0.464),
    ("GradDiff", "pcgrad", 0.380, 0.646, 0.478),
    ("GradDiff", "sago",   0.350, 0.695, 0.466),

    ("NPO", "base",   0.353, 0.656, 0.459),
    ("NPO", "pcgrad", 0.363, 0.675, 0.472),
    ("NPO", "sago",   0.340, 0.705, 0.459),

    ("SimNPO", "base",   0.113, 0.705, 0.195),
    ("SimNPO", "pcgrad", 0.145, 0.718, 0.241),
    ("SimNPO", "sago",   0.105, 0.735, 0.184),
]

df = pd.DataFrame(
    data,
    columns=["method", "variant", "MYTOFU_Mem", "MYTOFU_Utility", "BUS"]
)

# =========================
# 2. 字号与 PDF 设置
# =========================
plt.rcParams.update({
    "font.size": 13,          # 全局默认字号
    "axes.labelsize": 15,     # 坐标轴标题字号
    "xtick.labelsize": 15,    # x 轴刻度字号
    "ytick.labelsize": 15,    # y 轴刻度字号
    "legend.fontsize": 14,    # 图例字号
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =========================
# 3. 绘图
# =========================
fig, ax = plt.subplots(figsize=(7.8, 5.4))

method_markers = {
    "GradDiff": "o",
    "NPO": "s",
    "SimNPO": "^",
}

variant_styles = {
    "base": {
        "marker": "o",
        "s": 90,
        "label": "Base method",
    },
    "pcgrad": {
        "marker": "X",
        "s": 85,
        "label": "pcgrad variant",
    },
    "sago": {
        "marker": "D",
        "s": 75,
        "label": "sago variant",
    },
}

# 画点与箭头
for method in df["method"].unique():
    method_df = df[df["method"] == method]

    base_row = method_df[method_df["variant"] == "base"].iloc[0]
    x0 = base_row["MYTOFU_Mem"]
    y0 = base_row["MYTOFU_Utility"]

    # 普通方法点
    ax.scatter(
        x0,
        y0,
        s=110,
        marker=method_markers[method],
        label=f"{method} base"
    )
    ax.text(
        x0 + 0.006,
        y0 - 0.008,
        method,
        fontsize=13
    )

    # 变体点与箭头
    for variant in ["pcgrad", "sago"]:
        row = method_df[method_df["variant"] == variant].iloc[0]
        x1 = row["MYTOFU_Mem"]
        y1 = row["MYTOFU_Utility"]

        ax.scatter(
            x1,
            y1,
            s=85,
            marker=variant_styles[variant]["marker"],
            label=f"{method} {variant}"
        )

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.8,
                shrinkA=6,
                shrinkB=6
            )
        )

        ax.text(
            x1 + 0.006,
            y1 + 0.004,
            variant,
            fontsize=12
        )

# =========================
# 4. 坐标轴与图例
# =========================
ax.set_xlabel("MYTOFU Mem")
ax.set_ylabel("MYTOFU Utility")

ax.set_xlim(0.08, 0.41)
ax.set_ylim(0.60, 0.75)

ax.grid(axis="both", alpha=0.3)

# 图例放在下方
ax.legend(
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18)
)

# 给下方图例留空间
fig.tight_layout(rect=[0, 0.12, 1, 1])

# =========================
# 5. 保存
# =========================
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "ideal_variant_effect_mem_utility.pdf"
png_path = out_dir / "ideal_variant_effect_mem_utility.png"

fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)

plt.close(fig)

print(f"Saved PDF to: {pdf_path}")
print(f"Saved PNG to: {png_path}")