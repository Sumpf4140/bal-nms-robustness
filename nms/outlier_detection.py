"""Method-level Mahalanobis outlier identification.

A method is flagged as a statistical outlier when it lands in the upper tail of
the within-block Mahalanobis-distance distribution **more often than chance**.
The naive "> 5% of blocks" rule is miscalibrated: within a block of M methods,
the expected per-method exceedance rate of the (1−q)-quantile is ≈ (1−q), not a
fixed 5%, and — crucially — exceedances are *dependent* across methods (a
roughly fixed number exceeds per block), so per-method binomial tests are
anti-conservative for the most-extreme method. We instead use a permutation
null that respects the per-block structure: under exchangeability the upper-tail
slots in each block are assigned to methods at random. Family-wise error is
controlled by the Westfall–Young max-statistic (compare each method's observed
exceedance count to the permutation distribution of the *maximum* count across
methods). flagged_outlier = p_fwer < α.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from config import ALPHA, OVERLAP_PCTS, BLOCK_MIN_CELLS, N_BOOTSTRAP, RNG_SEED
from core.blocks import ALL_BLOCK_SIZES
from core.db import connect

logger = logging.getLogger(__name__)

_OUTLIER_TAIL_Q = 0.95       # upper-tail percentile
_BLOCK_KEY = ["slide_id", "overlap_pct", "block_size", "block_x", "block_y"]


def _fwer_permutation_pvalues(
    method_counts: np.ndarray,
    block_tail_counts: np.ndarray,
    n_methods: int,
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Westfall–Young max-statistic FWER p-values for per-method exceedance counts.

    method_counts      — (M,) observed #blocks each method is in the upper tail.
    block_tail_counts  — (N_blocks,) #methods in the upper tail per block (fixed).
    Under H0 each block's tail slots are placed uniformly at random among methods.
    p_fwer[m] = (1 + #{perm: max_j null_count_j >= method_counts[m]}) / (n_perm+1).
    """
    t_b = block_tail_counts[block_tail_counts > 0].astype(int)
    B, M = len(t_b), n_methods
    rows = np.arange(B)
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        keys = rng.random((B, M))
        # tail members of a block = the t_b columns with the largest keys.
        # threshold = the t_b-th largest key in each row (vectorised, no block loop).
        thr = -np.sort(-keys, axis=1)[rows, t_b - 1]
        member = keys >= thr[:, None]            # exactly t_b True per row (ties P=0)
        null_max[p] = member.sum(axis=0).max()
    return np.array([
        (1 + int(np.sum(null_max >= c))) / (n_perm + 1) for c in method_counts
    ])


def compute_upper_tail_flags(
    df: pd.DataFrame,
    q: float = _OUTLIER_TAIL_Q,
    alpha: float = ALPHA,
    n_perm: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """Per-method upper-tail exceedance + FWER-calibrated outlier flag (pure fn).

    `df` is a method_deviation-like frame with columns _BLOCK_KEY + nms_method +
    mahalanobis_distance. For each block the threshold is the q-quantile of that
    block's method distances; a (method, block) cell is in the upper tail if its
    distance exceeds the threshold. Per-method exceedance counts are compared to
    a permutation null (random per-block tail assignment) with family-wise error
    controlled by the max-statistic.

    Returns columns: nms_method, frac_upper_tail, expected_frac, n_blocks,
                     p_value (= p_fwer), flagged_outlier.
    """
    cols = ["nms_method", "frac_upper_tail", "expected_frac", "n_blocks",
            "p_value", "flagged_outlier"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    thresholds = (
        df.groupby(_BLOCK_KEY)["mahalanobis_distance"]
        .quantile(q)
        .reset_index()
        .rename(columns={"mahalanobis_distance": "threshold"})
    )
    merged = df.merge(thresholds, on=_BLOCK_KEY)
    merged["in_upper_tail"] = merged["mahalanobis_distance"] > merged["threshold"]

    p0 = float(merged["in_upper_tail"].mean())
    summary = (
        merged.groupby("nms_method")
        .agg(n_upper=("in_upper_tail", "sum"), n_blocks=("in_upper_tail", "count"))
        .reset_index()
    )
    summary["frac_upper_tail"] = summary["n_upper"] / summary["n_blocks"]
    summary["expected_frac"] = p0

    block_tail_counts = (
        merged.groupby(_BLOCK_KEY)["in_upper_tail"].sum().to_numpy()
    )
    rng = np.random.default_rng(seed)
    summary["p_value"] = _fwer_permutation_pvalues(
        summary["n_upper"].to_numpy(), block_tail_counts,
        n_methods=len(summary), n_perm=n_perm, rng=rng,
    )
    summary["flagged_outlier"] = summary["p_value"] < alpha
    return (summary[cols]
            .sort_values(["flagged_outlier", "frac_upper_tail"], ascending=False)
            .reset_index(drop=True))


def per_block_method_ranking(block_size: int, overlap_pct: float) -> pd.DataFrame:
    """For each block, rank methods by Mahalanobis distance to consensus.

    Returns DataFrame: slide_id, overlap_pct, block_size, block_x, block_y,
                       nms_method, mahalanobis_distance, rank (1 = closest).
    """
    with connect(read_only=True) as con:
        df = con.execute(
            """SELECT * FROM method_deviation
               WHERE block_size=? AND overlap_pct=?""",
            [block_size, float(overlap_pct)],
        ).fetchdf()

    if df.empty:
        return df

    df["rank"] = df.groupby(_BLOCK_KEY)["mahalanobis_distance"].rank(method="average")
    return df.sort_values(_BLOCK_KEY + ["rank"])


def methods_in_upper_tail_fraction(
    block_size: int,
    overlap_pct: float,
    q: float = _OUTLIER_TAIL_Q,
) -> pd.DataFrame:
    """Per-method upper-tail exceedance + calibrated outlier flag for one setting.

    Thin DB wrapper around `compute_upper_tail_flags`. Returns columns:
    nms_method, frac_upper_tail, expected_frac, n_blocks, p_value, p_holm,
    flagged_outlier.
    """
    with connect(read_only=True) as con:
        df = con.execute(
            """SELECT * FROM method_deviation
               WHERE block_size=? AND overlap_pct=?""",
            [block_size, float(overlap_pct)],
        ).fetchdf()

    return compute_upper_tail_flags(df, q=q)


def outlier_summary(tick: Callable[[], None] | None = None) -> pd.DataFrame:
    """Aggregate flagged-outlier rate per method across all (block_size, overlap_pct).

    Returns DataFrame: nms_method, n_settings_flagged, n_settings_total,
                       frac_settings_flagged, mean_frac_upper_tail.
    `tick`, if given, is called once per (block_size, overlap) setting.
    """
    rows = []
    for bs in ALL_BLOCK_SIZES:
        for op in OVERLAP_PCTS:
            df = methods_in_upper_tail_fraction(bs, op)
            if not df.empty:
                df["block_size"] = bs
                df["overlap_pct"] = op
                rows.append(df)
            if tick:
                tick()

    if not rows:
        return pd.DataFrame()

    all_df = pd.concat(rows, ignore_index=True)
    n_settings = all_df.groupby("nms_method")["flagged_outlier"].count()
    n_flagged = all_df.groupby("nms_method")["flagged_outlier"].sum()
    mean_frac = all_df.groupby("nms_method")["frac_upper_tail"].mean()

    result = pd.DataFrame({
        "nms_method": n_settings.index,
        "n_settings_flagged": n_flagged.values.astype(int),
        "n_settings_total": n_settings.values.astype(int),
        "frac_settings_flagged": (n_flagged / n_settings).values,
        "mean_frac_upper_tail": mean_frac.values,
    }).sort_values("frac_settings_flagged", ascending=False)

    return result.reset_index(drop=True)
