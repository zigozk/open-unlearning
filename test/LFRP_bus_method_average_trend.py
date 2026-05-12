import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# =========================
# 1. 数据
# =========================
data = [
    ("easy", 0.718, "CEU", 0.545),
    ("easy", 0.718, "GradDiff", 0.378),
    ("easy", 0.718, "NPO", 0.464),
    ("easy", 0.718, "RMU", 0.487),

    ("medium", 0.741, "CEU", 0.497),
    ("medium", 0.741, "GradDiff", 0.392),
    ("medium", 0.741, "NPO", 0.468),
    ("medium", 0.741, "RMU", 0.469),

    ("hard", 0.785, "CEU", 0.421),
    ("hard", 0.785, "GradDiff", 0.233),
    ("hard", 0.785, "NPO", 0.334),
    ("hard", 0.785, "RMU", 0.336),

    ("hard plus", 0.811, "CEU", 0.360),
    ("hard plus", 0.811, "GradDiff", 0.215),
    ("hard plus", 0.811, "NPO", 0.227),
    ("hard plus", 0.811, "RMU", 0.238),
]

df = pd.DataFrame(data, columns=["difficulty", "LFRP", "method", "BUS"])

difficulty_order = ["easy", "medium", "hard", "hard plus"]
df["difficulty"] = pd.Categorical(
    df["difficulty"],
    categories=difficulty_order,
    ordered=True
)

avg = (
    df.groupby(["difficulty", "LFRP"], observed=True)["BUS"]
    .mean()
    .reset_index()
)

# =========================
# 2. 字号与 PDF 设置
# =========================
plt.rcParams.update({
    "font.size": 15,          # 全局默认字号
    "axes.labelsize": 16,     # x 轴、y 轴标题字号
    "xtick.labelsize": 14,    # x 轴刻度字号
    "ytick.labelsize": 14,    # y 轴刻度字号
    "legend.fontsize": 13,    # 图例字号
    "pdf.fonttype": 42,       # 让 PDF 中字体更适合 LaTeX/AI 后期编辑
    "ps.fonttype": 42,
})

# =========================
# 3. 绘图
# =========================
fig, ax = plt.subplots(figsize=(7.8, 5.2))

# 各方法线
for method, sub in df.groupby("method"):
    sub = sub.sort_values("difficulty")
    ax.plot(
        sub["LFRP"],
        sub["BUS"],
        marker="o",
        linewidth=2.1,
        markersize=7,
        label=method
    )

# 平均线：加粗、虚线、方形 marker，使其明显区别于单个方法
ax.plot(
    avg["LFRP"],
    avg["BUS"],
    marker="s",
    linewidth=3.2,
    markersize=8,
    linestyle="--",
    label="Average"
)

# 坐标轴
ax.set_xlabel("LFRP")
ax.set_ylabel("BUS")
ax.set_ylim(0.18, 0.58)

# 网格
ax.grid(axis="y", alpha=0.3)

# 平均值文字标注
# 如果觉得图太挤，可以删除这一段
for _, row in avg.iterrows():
    ax.text(
        row["LFRP"],
        row["BUS"] + 0.014,
        f"{row['BUS']:.3f}",
        ha="center",
        va="bottom",
        fontsize=13       # 平均值数字标注字号
    )

# 图例放在下方
ax.legend(
    frameon=False,
    ncol=5,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18)
)

# 给下方图例留空间
fig.tight_layout(rect=[0, 0.10, 1, 1])

# =========================
# 4. 保存
# =========================
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_path = out_dir / "lfrp_bus_method_average_trend.pdf"
png_path = out_dir / "lfrp_bus_method_average_trend.png"

fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.03)

plt.close(fig)

print(f"Saved PDF to: {pdf_path}")
print(f"Saved PNG to: {png_path}")