"""Statistical utilities shared across all three papers.

All tests operate on ILR-transformed compositions unless noted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import N_BOOTSTRAP, RNG_SEED


# ── Friedman / Nemenyi ───────────────────────────────────────────────────────

def friedman_test(values_by_method: dict[str, np.ndarray]) -> dict:
    """Friedman test across methods.

    values_by_method: {method_name: 1-D array of length N_subjects}
    Subjects (rows) = blocks or slides; treatments (columns) = methods.

    Returns dict with keys: statistic, df, p_value, kendall_w.
    """
    methods = list(values_by_method.keys())
    if len(methods) < 3:
        raise ValueError("Need at least 3 methods for Friedman test")

    data = [np.asarray(values_by_method[m], dtype=float) for m in methods]
    n = len(data[0])
    k = len(methods)

    # On tied/degenerate input scipy's ties-correction denominator hits 0, raising a
    # benign numpy divide/invalid RuntimeWarning before returning nan. Silence it here
    # (it floods the console during `nms all`, which runs Friedman per ILR coordinate);
    # the nan is handled just below.
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic, p_value = scipy_stats.friedmanchisquare(*data)
    # Identical data → statistic=0, ties-correction denominator=0 → nan; treat as p=1
    if np.isnan(p_value):
        p_value = 1.0
    # Kendall's W = χ² / (n·(k-1))
    kendall_w = statistic / (n * (k - 1)) if n * (k - 1) > 0 else float("nan")

    return {
        "statistic": float(statistic),
        "df": k - 1,
        "p_value": float(p_value),
        "kendall_w": float(kendall_w),
        "n_subjects": n,
        "n_methods": k,
    }


def nemenyi_post_hoc(values_by_method: dict[str, np.ndarray]) -> pd.DataFrame:
    """Nemenyi post-hoc test after Friedman, with Holm correction.

    Returns symmetric (k×k) DataFrame of p-values (Holm-corrected).
    """
    import scikit_posthocs as sp

    methods = list(values_by_method.keys())
    data_matrix = np.column_stack([values_by_method[m] for m in methods])
    df_data = pd.DataFrame(data_matrix, columns=methods)
    result = sp.posthoc_nemenyi_friedman(df_data)
    return result


def hierarchical_correction(p_values: pd.Series, family_size: int) -> pd.Series:
    """Bonferroni-Holm correction at outer family level.

    p_values: Series of p-values from individual tests.
    family_size: number of tests in the outer family.

    Returns corrected p-values (Series, same index).
    """
    from statsmodels.stats.multitest import multipletests
    rejected, p_corrected, _, _ = multipletests(
        p_values.fillna(1.0).values,
        alpha=0.05,
        method="holm",
    )
    return pd.Series(p_corrected, index=p_values.index)


# ── Equivalence (TOST-style, non-parametric) ──────────────────────────────────

def bootstrap_equivalence(
    diffs: np.ndarray,
    margin: float,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> dict:
    """Non-parametric equivalence test on paired differences.

    Bootstraps the median paired difference and declares equivalence when the
    90% percentile CI (the 1−2α interval matching a TOST at α=0.05) lies entirely
    within ±margin. Positive evidence of "no meaningful difference" — which a
    non-significant Friedman cannot provide. Returns observed median, 90% CI, n,
    and an `equivalent` flag.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 3:
        return {"n": n, "median": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "equivalent": False}
    rng = np.random.default_rng(seed)
    boot = np.median(diffs[rng.integers(0, n, size=(n_boot, n))], axis=1)
    ci_low, ci_high = np.percentile(boot, [5, 95])
    return {
        "n": n,
        "median": float(np.median(diffs)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "equivalent": bool(ci_low > -margin and ci_high < margin),
    }


# ── Weighted mean ─────────────────────────────────────────────────────────────

def weighted_ilr_mean(
    ilr_per_slide: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted mean of ILR compositions across slides.

    Input:  ilr_per_slide — (S, D-1)
            weights       — (S,) positive weights.
    Output: (D-1,) weighted mean ILR vector.
    """
    ilr_arr = np.asarray(ilr_per_slide, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    # All-zero weights would give a nan vector; fall back to a uniform mean.
    w = w / total if total > 0 else np.full_like(w, 1.0 / len(w))
    return (w[:, None] * ilr_arr).sum(axis=0)
