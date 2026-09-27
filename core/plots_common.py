"""Shared matplotlib style and helper functions for all papers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

# ── Colour palette (colour-blind-safe) ───────────────────────────────────────
LABEL_COLOURS = {
    "Makrophage":              "#4477AA",
    "Lymphozyt":               "#EE6677",
    "NeutrophilerGranulozyt":  "#228833",
    "EosinophilerGranulozyt":  "#CCBB44",
}

METHOD_CMAP = "tab20"


# ── Common helpers ─────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, path: Path, *, close: bool = True) -> None:
    """Save figure to path (creates parent dirs) then optionally close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)


def alpha_ci_band(
    ax: plt.Axes,
    x: np.ndarray,
    alpha: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    label: str = "",
    colour: str = "#4477AA",
) -> None:
    """Plot α curve with 95% CI shading."""
    ax.plot(x, alpha, marker="o", color=colour, label=label)
    ax.fill_between(x, lo, hi, alpha=0.2, color=colour)


def add_reference_lines(ax: plt.Axes, levels: list[float] = (0.667, 0.8)) -> None:
    """Draw horizontal Krippendorff α threshold lines."""
    styles = ["--", ":"]
    labels_ = ["α ≥ 0.667 (tentative)", "α ≥ 0.800 (strong)"]
    for level, ls, lbl in zip(levels, styles, labels_):
        ax.axhline(level, linestyle=ls, color="gray", linewidth=0.8, label=lbl)
