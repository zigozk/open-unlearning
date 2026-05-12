#!/usr/bin/env python3
"""Plot full-vs-retain sanity check results for MYTOFU."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


HERE = Path(__file__).resolve().parent
SCORE_SUMMARY = HERE / "score_sanity_check_20260327_175031.summary.json"
GEN_SUMMARY = HERE / "sanity_check_20260327_175212.summary.json"

# Change this to True if you want the figure to be rebuilt from the two summary
# JSON files above. The default is False, so the displayed data can be edited
# directly in MANUAL_DATA.
USE_SUMMARY_DATA = False

# =========================
# Editable Figure Data
# =========================
# All values below are on a 0-1 display scale. The second metric should use a
# normalized log-prob margin instead of the raw margin, otherwise the bar may be
# greater than 1.
METRICS = [
    "标准答案偏好率",
    "归一化 log-prob 间隔",
    "自由生成包含率",
]

MANUAL_DATA = {
    ("完整记忆模型", "forget"): [1.00, 1.00, 0.30],
    ("保留参照模型", "forget"): [0.58, 0.18, 0.03],
    ("完整记忆模型", "retain"): [1.00, 0.87, 0.75],
    ("保留参照模型", "retain"): [1.00, 0.65, 0.68],
}

# If you prefer using exact match instead of contains-gold for the third metric
# in summary-data mode, change this to "exact_match".
GENERATION_METRIC = "contains_answer"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_chinese_font() -> str:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def build_data_from_summaries() -> tuple[list[str], dict[tuple[str, str], list[float]]]:
    score = load_json(SCORE_SUMMARY)["aggregate"]
    gen = load_json(GEN_SUMMARY)["aggregate"]

    margins = {
        ("完整记忆模型", "forget"): score["forget"]["full_margin_avg_mean"],
        ("保留参照模型", "forget"): score["forget"]["retain_margin_avg_mean"],
        ("完整记忆模型", "retain"): score["retain"]["full_margin_avg_mean"],
        ("保留参照模型", "retain"): score["retain"]["retain_margin_avg_mean"],
    }
    margin_scale = max(margins.values())

    if GENERATION_METRIC == "exact_match":
        full_gen_key = "full_exact_match_rate"
        retain_gen_key = "retain_exact_match_rate"
        gen_label = "自由生成复现率"
    else:
        full_gen_key = "full_contains_answer_rate"
        retain_gen_key = "retain_contains_answer_rate"
        gen_label = "自由生成包含率"

    metrics = [
        "标准答案偏好率",
        "归一化 log-prob 间隔",
        gen_label,
    ]

    data = {
        ("完整记忆模型", "forget"): [
            score["forget"]["full_prefer_gold_rate"],
            margins[("完整记忆模型", "forget")] / margin_scale,
            gen["forget"][full_gen_key],
        ],
        ("保留参照模型", "forget"): [
            score["forget"]["retain_prefer_gold_rate"],
            margins[("保留参照模型", "forget")] / margin_scale,
            gen["forget"][retain_gen_key],
        ],
        ("完整记忆模型", "retain"): [
            score["retain"]["full_prefer_gold_rate"],
            margins[("完整记忆模型", "retain")] / margin_scale,
            gen["retain"][full_gen_key],
        ],
        ("保留参照模型", "retain"): [
            score["retain"]["retain_prefer_gold_rate"],
            margins[("保留参照模型", "retain")] / margin_scale,
            gen["retain"][retain_gen_key],
        ],
    }
    return metrics, data


def validate_data(metrics: list[str], data: dict[tuple[str, str], list[float]]) -> None:
    if not metrics:
        raise ValueError("METRICS cannot be empty.")
    for group, values in data.items():
        if len(values) != len(metrics):
            raise ValueError(
                f"{group} has {len(values)} values, but METRICS has {len(metrics)}."
            )
        for value in values:
            if not 0 <= float(value) <= 1.12:
                raise ValueError(
                    f"{group} contains {value}; this plotting script expects display-scale values."
                )


def main() -> None:
    if USE_SUMMARY_DATA:
        metrics, data = build_data_from_summaries()
    else:
        metrics = METRICS
        data = MANUAL_DATA
    validate_data(metrics, data)

    font_name = choose_chinese_font()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                font_name,
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,

            # 论文图中字体应略大，避免缩放进 LaTeX 后过小
            "font.size": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 11,

            # PDF 中尽量保留可编辑/可嵌入字体
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = [
        "#1F3A5F",  # full / forget
        "#F4A261",  # retain-ref / forget
        "#6D819C",  # full / retain
        "#4FB3BF",  # retain-ref / retain
    ]
    hatches = ["", "//", "", "\\\\"]
    grid_gray = "#D9DEE7"
    text_gray = "#222222"

    # 更适合 LaTeX 单栏或双栏中的紧凑比例
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    x = np.arange(len(metrics))
    bar_width = 0.20
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, len(data))

    for i, ((model, side), vals) in enumerate(data.items()):
        label = f"{model} · {side}"
        bars = ax.bar(
            x + offsets[i],
            vals,
            width=bar_width,
            label=label,
            color=colors[i],
            edgecolor="#1A1A1A",
            linewidth=0.7,
            hatch=hatches[i],
            zorder=3,
        )

        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color=text_gray,
            )

    # 不要图内标题，标题交给 LaTeX caption
    ax.set_ylabel("Score")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)

    ax.set_ylim(0, 1.10)

    # 压缩左右空白
    ax.set_xlim(-0.45, len(metrics) - 0.55)

    ax.grid(
        axis="y",
        color=grid_gray,
        linewidth=0.8,
        linestyle="-",
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.tick_params(axis="both", length=3, width=0.8, color="#333333")

    # 图例放进图内，减少底部额外留白
    # 图例放在下方，但贴近坐标轴，避免底部空白过大
    ax.legend(
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#B8C0CC",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        columnspacing=1.2,
        handlelength=1.6,
        handletextpad=0.5,
        borderpad=0.45,
        labelspacing=0.45,
    )

    # 为底部图例留出少量空间；不要用太大的 bottom
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.985,
        bottom=0.29,
    )

    out_pdf = HERE / "full_retain_target_sanity.pdf"
    out_png = HERE / "full_retain_target_sanity.png"

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.015)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Chinese font: {font_name}")
    print(f"Data mode: {'summary JSON' if USE_SUMMARY_DATA else 'manual data'}")


if __name__ == "__main__":
    main()
