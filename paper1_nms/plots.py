"""Paper 1 figures.

All figures saved to P1_REPORTS as PDF.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import P1_REPORTS, LABELS, OVERLAP_PCTS, BLOCK_SIZES
from core.blocks import ALL_BLOCK_SIZES
from core.compositional import ilr_inv
from core.plots_common import save_fig, alpha_ci_band, add_reference_lines, LABEL_COLOURS

logger = logging.getLogger(__name__)

_BLOCK_LABEL = {0: "full", 1: "1×1", 2: "2×2", 3: "3×3", 4: "4×4", 5: "5×5"}


def fig1_scale_dependence(curve_df: pd.DataFrame, out_dir: Path = P1_REPORTS) -> None:
    """α vs block size with 95% CI band, faceted by overlap.  Headline figure."""
    overlaps = sorted(curve_df["overlap_pct"].unique())
    n_cols = min(3, len(overlaps))
    n_rows = int(np.ceil(len(overlaps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    x_ticks = sorted(curve_df["block_size"].unique())
    x_labels = [_BLOCK_LABEL.get(bs, str(bs)) for bs in x_ticks]

    for ax, op in zip(axes.flat, overlaps):
        sub = curve_df[curve_df["overlap_pct"] == op].sort_values("block_size")
        alpha_ci_band(
            ax,
            sub["block_size"].to_numpy(),
            sub["alpha"].to_numpy(),
            sub["lo95"].to_numpy(),
            sub["hi95"].to_numpy(),
            label=f"overlap={op:.0%}",
        )
        add_reference_lines(ax)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_xlabel("Block size")
        ax.set_ylabel("Krippendorff α (Aitchison)")
        ax.set_title(f"overlap = {op:.0%}")
        ax.set_ylim(-0.1, 1.05)

    for ax in axes.flat[len(overlaps):]:
        ax.set_visible(False)

    fig.suptitle("Scale-dependence of inter-method agreement (α)", fontsize=11, y=1.02)
    save_fig(fig, out_dir / "fig1_scale_dependence.pdf")
    logger.info("Saved fig1_scale_dependence.pdf")


def fig2_method_consensus_scatter(
    weighted_means_df: pd.DataFrame,
    out_dir: Path = P1_REPORTS,
) -> None:
    """Pairwise ILR scatter of weighted method means; outliers highlighted."""
    if weighted_means_df.empty:
        return

    ilr_cols = [c for c in weighted_means_df.columns if c.startswith("ilr_")]
    if len(ilr_cols) < 2:
        return

    n = len(ilr_cols)
    fig, axes = plt.subplots(n - 1, n - 1, figsize=(3 * (n - 1), 3 * (n - 1)))
    if n == 2:
        axes = np.array([[axes]])

    outlier_flag = "flagged_outlier" in weighted_means_df.columns

    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            xi = ilr_cols[j]
            yi = ilr_cols[i + 1]
            colours = (
                ["#CC3311" if f else "#4477AA" for f in weighted_means_df["flagged_outlier"]]
                if outlier_flag else "#4477AA"
            )
            ax.scatter(weighted_means_df[xi], weighted_means_df[yi],
                       c=colours, s=20, alpha=0.8)
            ax.set_xlabel(xi)
            ax.set_ylabel(yi)

    fig.suptitle("Weighted method means in ILR space", fontsize=11)
    save_fig(fig, out_dir / "fig2_method_consensus_scatter.pdf")
    logger.info("Saved fig2_method_consensus_scatter.pdf")


def fig3_outlier_heatmap(
    outlier_df: pd.DataFrame,
    out_dir: Path = P1_REPORTS,
) -> None:
    """Methods × (block_size × overlap) heatmap of upper-tail fraction."""
    if outlier_df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(outlier_df) * 0.4), 5))
    methods = outlier_df["nms_method"].tolist()
    fracs = outlier_df["frac_settings_flagged"].to_numpy()
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, fracs, color=["#CC3311" if f > 0.05 else "#4477AA"
                                  for f in outlier_df["frac_settings_flagged"]])
    ax.axvline(0.05, color="gray", linestyle="--", linewidth=0.8, label="5% threshold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_xlabel("Fraction of (block_size, overlap) settings flagged as outlier")
    ax.set_title("Method outlier rate across all analysis scales")
    ax.legend()
    save_fig(fig, out_dir / "fig3_outlier_heatmap.pdf")
    logger.info("Saved fig3_outlier_heatmap.pdf")


def fig4_aitchison_distance_boxplot(out_dir: Path = P1_REPORTS) -> None:
    """Per-method Aitchison-to-consensus distance distribution."""
    from core.db import connect
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT nms_method, aitchison_distance FROM method_deviation"
        ).fetchdf()

    if df.empty:
        return

    methods = sorted(df["nms_method"].unique())
    data = [df.loc[df["nms_method"] == m, "aitchison_distance"].dropna().to_numpy()
            for m in methods]

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 0.4), 5))
    ax.boxplot(data, tick_labels=methods, vert=True, patch_artist=True,
               medianprops={"color": "black"})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Aitchison distance to consensus")
    ax.set_title("Per-method deviation from inter-method consensus")
    save_fig(fig, out_dir / "fig4_aitchison_distance_boxplot.pdf")
    logger.info("Saved fig4_aitchison_distance_boxplot.pdf")


def fig5_timing_boxplot(out_dir: Path = P1_REPORTS) -> None:
    """CPU time per method, log scale."""
    from core.db import connect
    with connect(read_only=True) as con:
        df = con.execute("SELECT nms_method, elapsed_cpu FROM timing").fetchdf()

    if df.empty:
        return

    methods = sorted(df["nms_method"].unique())
    data = [df.loc[df["nms_method"] == m, "elapsed_cpu"].dropna().to_numpy() for m in methods]

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 0.4), 5))
    ax.boxplot(data, tick_labels=methods, vert=True, patch_artist=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("CPU time (s, log scale)")
    ax.set_title("NMS method computational cost")
    save_fig(fig, out_dir / "fig5_timing_boxplot.pdf")
    logger.info("Saved fig5_timing_boxplot.pdf")


def make_all_plots(
    curve_df: pd.DataFrame,
    weighted_means_df: pd.DataFrame,
    outlier_df: pd.DataFrame,
    out_dir: Path = P1_REPORTS,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig1_scale_dependence(curve_df, out_dir)
    fig2_method_consensus_scatter(weighted_means_df, out_dir)
    fig3_outlier_heatmap(outlier_df, out_dir)
    fig4_aitchison_distance_boxplot(out_dir)
    fig5_timing_boxplot(out_dir)
