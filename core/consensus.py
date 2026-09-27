"""Inter-method consensus on compositional data.

Geometric median in Aitchison geometry: L1-optimal central tendency,
solved iteratively via Weiszfeld's algorithm (Weiszfeld 1937).
Robust to outlier methods (up to 50% breakdown).

Krippendorff's α with Aitchison distance generalises agreement coefficients
to M raters with a real-valued distance metric (Hayes & Krippendorff 2007).

  α = 1 − D_observed / D_expected

  D_observed = mean pairwise squared Aitchison distance within blocks
               averaged over all blocks
  D_expected = mean pairwise squared distance over all (unit1, unit2)
               pairs with unit1 ≠ unit2, pooled from the full panel
"""
from __future__ import annotations

import numpy as np

from config import N_BOOTSTRAP, RNG_SEED
from core.compositional import ilr, ilr_inv


# ── Geometric median (Weiszfeld) ──────────────────────────────────────────────

def geometric_median_ilr(
    ilr_array: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> np.ndarray:
    """Weiszfeld iteration in ILR space.

    Input:  ilr_array — (M, D-1)  ILR-transformed compositions for M methods.
    Output: (D-1,) geometric median.

    The geometric median minimises Σ_m ‖x − ilr_m‖₂ and is robust to
    outlier methods (50% breakdown point).
    """
    ilr_array = np.asarray(ilr_array, dtype=float)
    M = ilr_array.shape[0]

    if M == 0:
        raise ValueError("geometric_median_ilr: empty input")
    if M == 1:
        return ilr_array[0].copy()

    # Initialise at the coordinate-wise median (robust starting point)
    y = np.median(ilr_array, axis=0)

    for _ in range(max_iter):
        dists = np.linalg.norm(ilr_array - y, axis=1)  # (M,)
        nonzero = dists > 1e-12

        if not nonzero.any():
            # All points coincide with current estimate
            break

        weights = np.where(nonzero, 1.0 / dists, 0.0)
        y_new = (weights[:, None] * ilr_array).sum(axis=0) / weights.sum()

        if np.linalg.norm(y_new - y) < tol:
            y = y_new
            break
        y = y_new

    return y


# ── Per-method distances to consensus ────────────────────────────────────────

def aitchison_distances_to_consensus(
    method_ilrs: np.ndarray,
    consensus: np.ndarray,
) -> np.ndarray:
    """Per-method Aitchison distance to the consensus point.

    Input:  method_ilrs — (M, D-1)
            consensus   — (D-1,)
    Output: (M,) distances.
    """
    method_ilrs = np.asarray(method_ilrs, dtype=float)
    consensus = np.asarray(consensus, dtype=float)
    return np.linalg.norm(method_ilrs - consensus, axis=1)


def _safe_inv(cov: np.ndarray, ridge: float = 1e-3, abs_floor: float = 1e-6) -> np.ndarray:
    """Invert a covariance matrix with regularisation, bounded in every regime.

    The NMS variants are near-identical (α≈0.98), so the leave-one-out
    covariance of the other M−1 methods is *near-singular* (their ILR vectors are
    nearly collinear). A plain `inv` then returns a numerically explosive inverse
    — Mahalanobis distances reached ~1e23 — *without* raising LinAlgError, so the
    pseudo-inverse fallback never fired. We add `(λ·tr(cov)/d + ε)·I`:
      • the scale-aware ridge `λ·tr(cov)/d` (λ=1e-3) floors the smallest eigenvalue
        at a fraction of the mean variance — handles the *near*-singular case at
        any ILR magnitude; and
      • a small absolute floor `ε` (1e-6) additionally bounds the *fully* degenerate
        case where all other methods are identical (tr(cov)≈0): a purely relative
        ridge would vanish there and the inverse would still explode.
    Together they keep distances finite (≲1e4) while the regularisation is small
    enough that the within-block *ordering* — and hence the rank/quantile-based
    outlier flags — is preserved.
    """
    d = cov.shape[0]
    scale = np.trace(cov) / d
    cov = cov + (ridge * max(scale, 0.0) + abs_floor) * np.eye(d)
    try:
        cov_inv = np.linalg.inv(cov)
        np.linalg.cholesky(cov_inv)  # verify positive-definiteness
        return cov_inv
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov)


def mahalanobis_distances_to_consensus(
    method_ilrs: np.ndarray,
    consensus: np.ndarray,
) -> np.ndarray:
    """Per-method **leave-one-out** Mahalanobis distance to the consensus.

    For each method m, the within-block ILR covariance is estimated from the
    *other* M−1 methods (README §10.5). This avoids the masking effect: with a
    pooled covariance that includes m, a genuine outlier inflates the covariance
    and shrinks its own distance, hiding exactly the methods we want to detect.
    Falls back to pseudo-inverse when a covariance matrix is singular, and to
    plain Euclidean distance when M−1 ≤ D−1 (covariance under-determined).

    Input:  method_ilrs — (M, D-1)
            consensus   — (D-1,)
    Output: (M,) Mahalanobis distances.
    """
    method_ilrs = np.asarray(method_ilrs, dtype=float)
    consensus = np.asarray(consensus, dtype=float)
    M, d = method_ilrs.shape
    diffs = method_ilrs - consensus  # (M, D-1)

    if M - 1 <= d:
        # Not enough *other* methods to estimate full-rank covariance.
        return np.linalg.norm(diffs, axis=1)

    distances = np.empty(M)
    for m in range(M):
        others = np.delete(method_ilrs, m, axis=0)  # (M-1, D-1)
        cov_inv = _safe_inv(np.cov(others.T))
        dv = diffs[m]
        distances[m] = float(np.sqrt(np.maximum(0.0, dv @ cov_inv @ dv)))
    return distances


# ── Krippendorff's α with Aitchison distance ──────────────────────────────────

def krippendorff_alpha_aitchison(ilr_panel: np.ndarray) -> float:
    """Krippendorff's α with Aitchison distance metric.

    ilr_panel: (M_methods, N_blocks, D-1)

    α = 1 - D_observed / D_expected

    D_observed: mean within-unit squared ILR distance (over ordered method
                pairs, averaged over blocks).
    D_expected: canonical Krippendorff expected disagreement — the mean squared
                ILR distance over *all* ordered pairs of values in the panel
                (n = M·N values, n(n−1) ordered pairs, same-unit pairs included).

    This is the textbook interval/metric Krippendorff α (squared-distance δ),
    so the 1-D case matches the reference `krippendorff` package exactly
    (validated in tests). With the squared Aitchison metric and an orthonormal
    ILR basis it reduces to the multivariate interval α (ratio of summed within-
    to total variance), which is what makes it scale- and basis-invariant.

    Returns NaN if N_blocks < 2 or M_methods < 2 or D_expected ≈ 0.
    """
    ilr_panel = np.asarray(ilr_panel, dtype=float)
    M, N, D1 = ilr_panel.shape

    if M < 2 or N < 2:
        return float("nan")

    # ── Vectorised computation ────────────────────────────────────────────────
    # sq_norms[m, u] = ‖ilr_panel[m, u]‖²
    sq_norms = (ilr_panel ** 2).sum(axis=2)  # (M, N)

    # For each block u: Σ_{m<m'} ‖x_{mu} - x_{m'u}‖²
    # = M·Σ_m‖x_{mu}‖² - ‖Σ_m x_{mu}‖²
    block_sum_sq_norms = sq_norms.sum(axis=0)                    # (N,)  Σ_m ‖x_{mu}‖²
    block_sum = ilr_panel.sum(axis=0)                            # (N, D1)
    block_sum_norm_sq = (block_sum ** 2).sum(axis=1)             # (N,)  ‖Σ_m x_{mu}‖²

    # Σ_{m<m'} ‖x_{mu} - x_{m'u}‖² = (M·Σ_m‖x‖² - ‖Σ_m x‖²)  [half of ordered sum]
    within_sq = M * block_sum_sq_norms - block_sum_norm_sq       # (N,)

    n_obs_pairs_per_block = M * (M - 1) / 2
    D_o = (within_sq / n_obs_pairs_per_block).mean()

    # ── D_expected (canonical Krippendorff: ALL pairs over the value pool) ─────
    # Σ_{all (m,u),(m',u')} ‖x_{mu} - x_{m'u'}‖²
    # = 2·NM·Σ‖x‖² - 2·‖Σx‖²   (self-pairs contribute 0, so this == Σ_{a≠b})
    total_sq = sq_norms.sum()                                    # scalar
    total_sum = ilr_panel.sum(axis=(0, 1))                       # (D1,)
    total_sum_norm_sq = (total_sum ** 2).sum()                   # scalar

    all_pairs_sq = 2 * N * M * total_sq - 2 * total_sum_norm_sq

    n_vals = M * N                                               # total values
    n_pairs = n_vals * (n_vals - 1)                              # ordered, a ≠ b
    if n_pairs == 0:
        return float("nan")

    D_e = all_pairs_sq / n_pairs

    if D_e <= 0:
        return 1.0 if D_o <= 0 else float("nan")

    return float(1.0 - D_o / D_e)


def krippendorff_alpha_per_coordinate(ilr_panel: np.ndarray) -> np.ndarray:
    """Per-ILR-coordinate Krippendorff α.

    Runs the scalar α independently on each ILR coordinate. Because the
    multivariate (squared-distance) α is the ratio of summed within- to total
    variance, the per-coordinate breakdown localises *which* balance drives
    (dis)agreement — e.g. isolating the often-zero Eosinophil balance, whose
    apparent disagreement is sensitive to the count-scale of the zero
    replacement (see nms.zero_sensitivity).

    ilr_panel: (M, N, D-1). Returns (D-1,) array of per-coordinate α.
    """
    panel = np.asarray(ilr_panel, dtype=float)
    D1 = panel.shape[2]
    return np.array([
        krippendorff_alpha_aitchison(panel[:, :, [k]]) for k in range(D1)
    ])


def krippendorff_alpha_bootstrap_ci(
    ilr_panel: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for Krippendorff's α by resampling blocks (columns).

    For cluster-bootstrap over slides, the caller should supply a panel
    already resampled at slide level; this function does iid block resampling.

    Returns: (alpha, lo95, hi95).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    ilr_panel = np.asarray(ilr_panel, dtype=float)
    M, N, D1 = ilr_panel.shape

    alpha = krippendorff_alpha_aitchison(ilr_panel)

    boot_alphas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot_alphas[b] = krippendorff_alpha_aitchison(ilr_panel[:, idx, :])

    lo = float(np.nanpercentile(boot_alphas, 2.5))
    hi = float(np.nanpercentile(boot_alphas, 97.5))
    return alpha, lo, hi
