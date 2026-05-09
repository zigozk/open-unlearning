#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def _missing_dep(name: str, hint: str) -> RuntimeError:
    return RuntimeError(
        f"Missing dependency: {name}.\n"
        f"Hint: {hint}\n"
        "If you're in this repo, you likely want to activate the project env (conda/venv) first."
    )


def _extract_scores(obj: object) -> List[float]:
    """Extract forget_truth_ratio.value_by_index[*].score from a loaded JSON."""
    if not isinstance(obj, dict):
        raise ValueError("Top-level JSON must be an object")

    ftr = obj.get("forget_truth_ratio")
    if not isinstance(ftr, dict):
        raise KeyError("Missing forget_truth_ratio object")

    vbi = ftr.get("value_by_index")
    if vbi is None:
        raise KeyError("Missing forget_truth_ratio.value_by_index")

    items: Iterable[object]
    if isinstance(vbi, dict):
        items = vbi.values()
    elif isinstance(vbi, list):
        items = vbi
    else:
        raise TypeError("value_by_index must be dict or list")

    scores: List[float] = []
    for it in items:
        if isinstance(it, dict) and "score" in it:
            scores.append(float(it["score"]))
        else:
            # fallback: allow raw number
            try:
                scores.append(float(it))
            except Exception as e:
                raise ValueError(f"Cannot extract score from item: {it!r}") from e

    if not scores:
        raise ValueError("No scores found in value_by_index")

    return scores


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summary(scores: List[float], agg_value: float | None) -> str:
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise _missing_dep("numpy", "pip install numpy") from e

    arr = np.asarray(scores, dtype=float)
    parts = [
        f"n={arr.size}",
        f"mean={arr.mean():.4f}",
        f"median={np.median(arr):.4f}",
        f"std={arr.std(ddof=1) if arr.size > 1 else 0.0:.4f}",
        f"min={arr.min():.4f}",
        f"max={arr.max():.4f}",
    ]
    if agg_value is not None:
        parts.insert(1, f"agg_value={agg_value:.4f}")
    return ", ".join(parts)


def load_scores(path: Path) -> Tuple[List[float], float | None]:
    data = _read_json(path)
    ftr = data.get("forget_truth_ratio") if isinstance(data, dict) else None
    agg_value = None
    if isinstance(ftr, dict) and "agg_value" in ftr:
        try:
            agg_value = float(ftr["agg_value"])
        except Exception:
            agg_value = None
    scores = _extract_scores(data)
    return scores, agg_value


def _clip_by_percentile(x, low_pct: float | None, high_pct: float | None):
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise _missing_dep("numpy", "pip install numpy") from e

    if low_pct is None and high_pct is None:
        return x

    lo = None
    hi = None
    if low_pct is not None:
        lo = float(np.percentile(x, low_pct))
    if high_pct is not None:
        hi = float(np.percentile(x, high_pct))

    if lo is not None:
        x = np.maximum(x, lo)
    if hi is not None:
        x = np.minimum(x, hi)
    return x


def _auto_bins(x, max_bins: int = 80) -> int:
    """Freedman–Diaconis bin rule with a reasonable cap."""
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise _missing_dep("numpy", "pip install numpy") from e

    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 10
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return min(max_bins, 30)
    bin_width = 2 * iqr / (x.size ** (1 / 3))
    if bin_width <= 0:
        return min(max_bins, 30)
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return int(np.clip(bins, 10, max_bins))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot forget_truth_ratio distribution for two eval JSON files."
    )
    parser.add_argument(
        "--a",
        required=True,
        help="Path to first eval JSON (e.g., NPO_TOFU_EVAL.json)",
    )
    parser.add_argument(
        "--b",
        required=True,
        help="Path to second eval JSON (e.g., simNPO_TOFU_EVAL.json)",
    )
    parser.add_argument("--label-a", default="A", help="Legend label for --a")
    parser.add_argument("--label-b", default="B", help="Legend label for --b")
    parser.add_argument(
        "--out",
        default="forget_truth_ratio_dist.png",
        help="Output image path (png/pdf/svg)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=0,
        help="Histogram bins (0 = auto)",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=80,
        help="Max bins when --bins=0",
    )
    parser.add_argument(
        "--clip-low-pct",
        type=float,
        default=None,
        help="Clip lower tail by percentile (e.g., 1 for p1)",
    )
    parser.add_argument(
        "--clip-high-pct",
        type=float,
        default=None,
        help="Clip upper tail by percentile (e.g., 99 for p99)",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        help="Apply log1p(score) before histogram (useful for heavy tails)",
    )
    parser.add_argument(
        "--mode",
        choices=["side", "overlay"],
        default="side",
        help="Histogram style: side-by-side bars or overlay",
    )
    parser.add_argument(
        "--kde",
        action="store_true",
        help="Overlay KDE curve (requires scipy)",
    )

    args = parser.parse_args()

    path_a = Path(args.a)
    path_b = Path(args.b)

    scores_a, agg_a = load_scores(path_a)
    scores_b, agg_b = load_scores(path_b)

    try:
        import numpy as np
    except ModuleNotFoundError as e:
        raise _missing_dep("numpy", "pip install numpy") from e

    # Lazy import so the script can still parse JSON even if matplotlib is missing.
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise _missing_dep("matplotlib", "pip install matplotlib") from e

    # Try to configure a CJK-capable font to avoid "Glyph missing" warnings.
    try:
        from matplotlib import font_manager as _fm

        preferred_fonts = [
            "Noto Sans CJK SC",
            "Noto Sans CJK",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "SimHei",
            "Microsoft YaHei",
            "PingFang SC",
        ]
        available = {f.name for f in _fm.fontManager.ttflist}
        for name in preferred_fonts:
            if name in available:
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                break
    except Exception:
        # Font configuration is best-effort; ignore failures.
        pass

    x_a = np.asarray(scores_a, dtype=float)
    x_b = np.asarray(scores_b, dtype=float)

    # Optional preprocessing for readability with large / heavy-tail distributions.
    if args.log1p:
        x_a = np.log1p(np.maximum(x_a, 0.0))
        x_b = np.log1p(np.maximum(x_b, 0.0))
    x_a = _clip_by_percentile(x_a, args.clip_low_pct, args.clip_high_pct)
    x_b = _clip_by_percentile(x_b, args.clip_low_pct, args.clip_high_pct)

    # Shared x-range for visual comparability
    x_min = float(min(x_a.min(), x_b.min()))
    x_max = float(max(x_a.max(), x_b.max()))
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.5
    xlim = (x_min - pad, x_max + pad)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)

    # (1) Histogram + optional KDE
    ax = axes[0]
    bins = args.bins if args.bins and args.bins > 0 else _auto_bins(np.concatenate([x_a, x_b]), args.max_bins)
    edges = np.linspace(xlim[0], xlim[1], bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = (edges[1] - edges[0])

    hist_a, _ = np.histogram(x_a, bins=edges, density=True)
    hist_b, _ = np.histogram(x_b, bins=edges, density=True)

    if args.mode == "side":
        ax.bar(
            centers - 0.2 * width,
            hist_a,
            width=0.4 * width,
            alpha=0.75,
            label=f"{args.label_a} ({_summary(scores_a, agg_a)})",
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.4,
        )
        ax.bar(
            centers + 0.2 * width,
            hist_b,
            width=0.4 * width,
            alpha=0.75,
            label=f"{args.label_b} ({_summary(scores_b, agg_b)})",
            color="#ff7f0e",
            edgecolor="white",
            linewidth=0.4,
        )
    else:
        ax.hist(
            x_a,
            bins=edges,
            density=True,
            alpha=0.45,
            label=f"{args.label_a} ({_summary(scores_a, agg_a)})",
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hist(
            x_b,
            bins=edges,
            density=True,
            alpha=0.45,
            label=f"{args.label_b} ({_summary(scores_b, agg_b)})",
            color="#ff7f0e",
            edgecolor="white",
            linewidth=0.5,
        )

    if args.kde:
        try:
            from scipy.stats import gaussian_kde

            xs = np.linspace(xlim[0], xlim[1], 400)
            ax.plot(xs, gaussian_kde(x_a)(xs), color="#1f77b4", lw=2)
            ax.plot(xs, gaussian_kde(x_b)(xs), color="#ff7f0e", lw=2)
        except ModuleNotFoundError as e:
            ax.text(
                0.02,
                0.98,
                "KDE skipped: missing scipy (pip install scipy)",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )
        except Exception as e:
            ax.text(
                0.02,
                0.98,
                f"KDE failed: {e}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )

    ax.set_title("forget_truth_ratio 分布（直方图对比）")
    ax.set_xlabel("log1p(score)" if args.log1p else "score")
    ax.set_ylabel("density")
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, frameon=True)

    # (2) Boxplot + strip-like scatter (lightweight)
    ax = axes[1]
    ax.boxplot(
        [x_a, x_b],
        tick_labels=[args.label_a, args.label_b],
        showfliers=False,
        widths=0.5,
    )
    rng = np.random.default_rng(0)
    jitter_a = (rng.random(x_a.size) - 0.5) * 0.15
    jitter_b = (rng.random(x_b.size) - 0.5) * 0.15
    ax.scatter(1 + jitter_a, x_a, s=6, alpha=0.15, color="#1f77b4", linewidths=0)
    ax.scatter(2 + jitter_b, x_b, s=6, alpha=0.15, color="#ff7f0e", linewidths=0)

    ax.set_title("forget_truth_ratio 分布（箱线图）")
    ax.set_ylabel("score")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("forget_truth_ratio: NPO vs simNPO", fontsize=12)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved plot to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
