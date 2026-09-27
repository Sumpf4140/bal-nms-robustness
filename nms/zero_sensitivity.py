"""Zero-replacement sensitivity for the inter-method α.

The headline metric (Krippendorff's α on ILR compositions) is computed after
Bayesian-multiplicative zero replacement, which imputes a zero count as the
pseudo-proportion ``0.5/n`` (n = block total). For a label that is zero across
methods (eosinophils are routinely 0 even in ≥500-cell blocks; neutrophils
sometimes), the imputed *effective count* is pinned at 0.5 while the real parts
scale with a method's absolute detection count. Methods that deduplicate little
(`none`, `iou_grid_n1`) carry far larger counts at high overlap than global
methods, so the **balance involving a structural-zero label differs between
methods purely from count magnitude** — a confound that deflates α, and does so
*more at higher overlap*, which could be misread as "methods disagree more when
tiles overlap more".

This module quantifies and neutralises that confound by comparing α under three
zero-handling regimes per (block_size, overlap):

  * ``as_is``        — the analysis as run (totals differ across methods);
  * ``common_scale`` — each method's block counts rescaled to a common total
                       before zero replacement, so the imputed-zero proportion
                       is method-invariant (count-scale-invariant α);
  * ``no_eos``       — α on the {Mac, Lym, Neu} subcomposition (drops the
                       usually-zero Eosinophil balance entirely).

``gap = alpha_common_scale − alpha_as_is`` is the size of the artefact. Per-ILR-
coordinate α (on the as-is panel) localises which balance is responsible. If the
gap is small the headline is robust; if it is material, report ``common_scale``
(or ``no_eos``) as the primary number and keep the others as sensitivity.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from config import LABELS, OVERLAP_PCTS
from core.blocks import ALL_BLOCK_SIZES
from core.consensus import (
    krippendorff_alpha_aitchison,
    krippendorff_alpha_per_coordinate,
)
from nms.consensus_analysis import _blocks_to_ilr_panel, filter_blocks

logger = logging.getLogger(__name__)

# Eosinophils: the routinely-zero label most exposed to the count-scale leak.
_EOS_LABEL = "EosinophilerGranulozyt"
_NO_EOS_LABELS = [lbl for lbl in LABELS if lbl != _EOS_LABEL]


def _clean_alpha(block_df: pd.DataFrame, *, common_scale: bool = False,
                 labels: list[str] | None = None) -> tuple[float, np.ndarray, int]:
    """α (and per-coordinate α, n_blocks) for one zero-handling regime."""
    _, panel, _ = _blocks_to_ilr_panel(block_df, common_scale=common_scale,
                                       labels=labels)
    valid = ~np.isnan(panel).any(axis=(0, 2))
    panel = panel[:, valid, :]
    if panel.shape[1] < 2:
        return float("nan"), np.full(panel.shape[2], np.nan), int(panel.shape[1])
    return (krippendorff_alpha_aitchison(panel),
            krippendorff_alpha_per_coordinate(panel),
            int(panel.shape[1]))


def alpha_zero_sensitivity(block_size: int, overlap_pct: float) -> dict:
    """Compare α across zero-handling regimes for one (block_size, overlap).

    Returns dict: block_size, overlap_pct, n_blocks, alpha_as_is,
    alpha_common_scale, alpha_no_eos, gap (= common_scale − as_is), and
    alpha_coord_1..(D-1) (per-coordinate α on the as-is panel).
    """
    out = {
        "block_size": block_size, "overlap_pct": float(overlap_pct),
        "n_blocks": 0, "alpha_as_is": float("nan"),
        "alpha_common_scale": float("nan"), "alpha_no_eos": float("nan"),
        "gap": float("nan"),
    }
    block_df = filter_blocks(block_size, overlap_pct)
    if block_df.empty:
        return out

    a_asis, percoord, n_blocks = _clean_alpha(block_df)
    a_cs, _, _ = _clean_alpha(block_df, common_scale=True)
    a_noeos, _, _ = _clean_alpha(block_df, labels=_NO_EOS_LABELS)

    out.update({
        "n_blocks": n_blocks,
        "alpha_as_is": a_asis,
        "alpha_common_scale": a_cs,
        "alpha_no_eos": a_noeos,
        "gap": a_cs - a_asis,
    })
    for k, v in enumerate(percoord, start=1):
        out[f"alpha_coord_{k}"] = float(v)
    return out


def zero_sensitivity_curve(tick: Callable[[], None] | None = None) -> pd.DataFrame:
    """Run `alpha_zero_sensitivity` over all (block_size, overlap) settings.

    The companion to `scale_dependence_curve`: it shows whether the α-vs-scale
    and α-vs-overlap story survives a count-scale-invariant zero treatment.
    `tick`, if given, is called once per (block_size, overlap) setting.
    """
    rows = []
    for bs in ALL_BLOCK_SIZES:
        for op in OVERLAP_PCTS:
            logger.info("Zero-sensitivity α: block_size=%d overlap=%.3f", bs, op)
            rows.append(alpha_zero_sensitivity(bs, op))
            if tick:
                tick()
    return pd.DataFrame(rows)
