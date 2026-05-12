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


def main() -> None:
    score = load_json(SCORE_SUMMARY)["aggregate"]
    gen = load_json(GEN_SUMMARY)["aggregate"]

    margins = {
        ("完整记忆模型", "forget"): score["forget"]["full_margin_avg_mean"],
        ("保留参照模型", "forget"): score["forget"]["retain_margin_avg_mean"],
        ("完整记忆模型", "retain"): score["retain"]["full_margin_avg_mean"],
        ("保留参照模型", "retain"): score["retain"]["retain_margin_avg_mean"],
    }
    margin_scale = max(margins.values())

    metrics = [
        "标准答案偏好率",
        "归一化 log-prob 间隔",
        "自由生成包含率",
    ]

    data = {
        ("完整记忆模型", "forget"): [
            score["forget"]["full_prefer_gold_rate"],
            margins[("完整记忆模型", "forget")] / margin_scale,
            gen["forget"]["full_contains_answer_rate"],
        ],
        ("保留参照模型", "forget"): [
            score["forget"]["retain_prefer_gold_rate"],
            margins[("保留参照模型", "forget")] / margin_scale,
            gen["forget"]["retain_contains_answer_rate"],
        ],
        ("完整记忆模型", "retain"): [
            score["retain"]["full_prefer_gold_rate"],
            margins[("完整记忆模型", "retain")] / margin_scale,
            gen["retain"]["full_contains_answer_rate"],
        ],
        ("保留参照模型", "retain"): [
            score["retain"]["retain_prefer_gold_rate"],
            margins[("保留参照模型", "retain")] / margin_scale,
            gen["retain"]["retain_contains_answer_rate"],
        ],
    }

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
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = [
        "#1F3A5F",
        "#F4A261",
        "#6D819C",
        "#4FB3BF",
    ]
    hatches = ["", "//", "", "\\\\"]
    grid_gray = "#D9DEE7"
    text_gray = "#222222"

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    x = np.arange(len(metrics))
    bar_width = 0.18
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
            linewidth=0.8,
            hatch=hatches[i],
            zorder=3,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=text_gray,
            )

    ax.set_title("完整记忆模型与保留参照模型的前提验证结果", pad=14, fontweight="bold")
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
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )

    fig.tight_layout()

    out_pdf = HERE / "full_retain_target_sanity.pdf"
    out_png = HERE / "full_retain_target_sanity.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Chinese font: {font_name}")


if __name__ == "__main__":
    main()
