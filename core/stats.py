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
    # (it floods the console during `p1 all`, which runs Friedman per ILR coordinate);
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


# ── Bland-Altman ─────────────────────────────────────────────────────────────

def bland_altman_stats(
    method1: np.ndarray,
    method2: np.ndarray,
) -> dict:
    """Bland-Altman agreement statistics.

    Returns: mean_diff, std_diff, loa_lower, loa_upper (95% limits of agreement).
    """
    diff = np.asarray(method1, dtype=float) - np.asarray(method2, dtype=float)
    mean_diff = float(diff.mean()) if diff.size else float("nan")
    # ddof=1 std is undefined for <2 points; return nan without the numpy warning.
    std_diff = float(diff.std(ddof=1)) if diff.size >= 2 else float("nan")
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_lower": mean_diff - 1.96 * std_diff,
        "loa_upper": mean_diff + 1.96 * std_diff,
    }


# ── Inverse-variance weighting ────────────────────────────────────────────────

def inverse_variance_weights(
    n_per_slide: np.ndarray,
    moran_i_per_slide: np.ndarray | None = None,
    cluster_size_per_slide: np.ndarray | None = None,
) -> np.ndarray:
    """Per-slide inverse-variance weights for the weighted ILR mean.

    Without Moran's I:   w_s = N_s              (multinomial precision ∝ cells)
    With Moran's I:      w_s = N_s / DEFF_s,    DEFF_s = 1 + (m̄_s − 1)·ρ_s

    where DEFF_s is the **cluster-sampling design effect** (Kish): tiles are the
    clusters, cells the elements, m̄_s = N_s / T_s is the mean cluster size
    (cells per occupied tile), and ρ_s ∈ [0, 1] is the intra-cluster correlation
    proxied by the per-slide Moran's I (negative values clipped to 0).

    The cluster size, **not** the raw cell count, multiplies ρ_s. Using N_s
    there (the previous behaviour) made DEFF ≈ N_s·ρ_s, so w_s ≈ 1/ρ_s collapsed
    to a near-constant across slides and the weighting did nothing. With the
    correct m̄_s, N_eff = N_s / DEFF_s ≈ T_s / ρ_s scales with the number of
    *independent* spatial units — the intended behaviour. If
    cluster_size_per_slide is None, m̄_s falls back to N_s (documented legacy
    behaviour; pass cluster sizes for the spatially-honest weight).

    Input:  n_per_slide             — (S,) cell counts per slide.
            moran_i_per_slide       — (S,) Moran's I values or None.
            cluster_size_per_slide  — (S,) mean cells-per-tile, or None.
    Output: (S,) positive weights (not normalised to sum to 1).
    """
    n = np.asarray(n_per_slide, dtype=float)
    if moran_i_per_slide is None:
        return n.copy()

    rho = np.clip(np.asarray(moran_i_per_slide, dtype=float), 0.0, 1.0)
    m_bar = (np.asarray(cluster_size_per_slide, dtype=float)
             if cluster_size_per_slide is not None else n)
    denom = 1.0 + (m_bar - 1.0) * rho
    denom = np.where(denom > 0, denom, 1.0)  # guard against m̄=1, rho=1 edge
    return n / denom


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
