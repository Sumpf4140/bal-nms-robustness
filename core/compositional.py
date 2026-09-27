"""Compositional data transforms for 4-part BAL compositions.

All hypothesis tests operate on ILR coordinates; ALR is for plotting only.
Zero replacement uses the Bayesian-multiplicative (CZM) approach from
Martín-Fernández et al. (2015), Stat Modelling 15(2):134–158.

ILR basis: Helmert contrast (sequential binary partition), which is
orthonormal with respect to the Aitchison inner product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABELS, ALR_REF_LABEL


# ── Zero replacement ──────────────────────────────────────────────────────────

def cmult_repl(counts: np.ndarray) -> np.ndarray:
    """Bayesian-multiplicative zero replacement (CZM prior, t = 0.5/n per zero).

    For each zero count x_j in row:
        p_j_new = 0.5 / n            (pseudoproportion)
        p_i_new = p_i * (1 - nz*0.5/n)  for non-zero i  (multiplicative rescale)
    where n = row total count, nz = number of zero components.

    Input:  counts — (D,) or (N, D)  integer or float counts, non-negative.
    Output: strictly-positive proportions summing to 1, same shape.
    """
    counts = np.asarray(counts, dtype=float)
    squeeze = counts.ndim == 1
    if squeeze:
        counts = counts[None, :]

    n_samples, D = counts.shape
    result = np.empty_like(counts)

    for i in range(n_samples):
        row = counts[i]
        n = row.sum()

        if n == 0:
            # Undefined composition: return uniform
            result[i] = np.ones(D) / D
            continue

        zeros = row == 0
        nz = int(zeros.sum())

        if nz == 0:
            result[i] = row / n
            continue

        # CZM replacement
        t_j = 0.5 / n                    # pseudoproportion per zero component
        zero_mass = nz * t_j             # total mass ceded to zeros

        if zero_mass >= 1.0:
            # Degenerate row: so few counts (e.g. a single detection, n=1, nz=3)
            # that the pseudo-mass for the zeros meets or exceeds the whole
            # composition, driving the multiplicative rescale (1 - zero_mass) ≤ 0
            # and producing a NEGATIVE "proportion" — which then poisons ilr()'s
            # np.log with NaN. Fall back to the exact estimator the multiplicative
            # form approximates: the Dirichlet(0.5) posterior mean
            # p_j = (x_j + 0.5)/(n + 0.5·D), strictly positive for any non-negative
            # counts. Only fires for near-empty rows (e.g. rarefaction at k=1);
            # every well-sampled row keeps the original multiplicative result.
            result[i] = (row + 0.5) / (n + 0.5 * D)
            continue

        p = row / n
        new_p = np.where(zeros, t_j, p * (1.0 - zero_mass))
        # Normalise to absorb floating-point drift (should be ~1 already)
        result[i] = new_p / new_p.sum()

    if squeeze:
        result = result[0]
    return result


# ── Log-ratio transforms ──────────────────────────────────────────────────────

def alr(p: np.ndarray, ref_idx: int | None = None) -> np.ndarray:
    """Additive log-ratio transform.

    Returns D-1 coordinates: log(p_i / p_ref) for i ≠ ref_idx.
    Default ref_idx corresponds to ALR_REF_LABEL in LABELS.

    Input:  p — (D,) or (N, D) strictly-positive proportions summing to 1.
    Output: (D-1,) or (N, D-1).
    """
    if ref_idx is None:
        ref_idx = LABELS.index(ALR_REF_LABEL)

    p = np.asarray(p, dtype=float)
    squeeze = p.ndim == 1
    if squeeze:
        p = p[None, :]

    non_ref = [i for i in range(p.shape[1]) if i != ref_idx]
    result = np.log(p[:, non_ref] / p[:, [ref_idx]])

    return result[0] if squeeze else result


def ilr(p: np.ndarray) -> np.ndarray:
    """Isometric log-ratio transform using the Helmert (sequential binary) basis.

    For D=4 (and generalised to any D):
      z_k = sqrt(k/(k+1)) * (mean(log p_1..p_k) - log p_{k+1})   k = 1..D-1

    This yields D-1 orthonormal coordinates in R^{D-1}.

    Input:  p — (D,) or (N, D) strictly-positive proportions.
    Output: (D-1,) or (N, D-1).
    """
    p = np.asarray(p, dtype=float)
    squeeze = p.ndim == 1
    if squeeze:
        p = p[None, :]

    n, D = p.shape
    log_p = np.log(p)
    result = np.empty((n, D - 1))

    for k in range(1, D):
        # z_k = sqrt(k/(k+1)) * (mean(log p_0..p_{k-1}) - log p_k)
        result[:, k - 1] = np.sqrt(k / (k + 1)) * (
            log_p[:, :k].mean(axis=1) - log_p[:, k]
        )

    return result[0] if squeeze else result


def ilr_inv(z: np.ndarray) -> np.ndarray:
    """Inverse ILR: map R^{D-1} back to the simplex.

    Uses the same Helmert basis as ilr().

    Input:  z — (D-1,) or (N, D-1).
    Output: (D,) or (N, D) strictly-positive proportions summing to 1.
    """
    z = np.asarray(z, dtype=float)
    squeeze = z.ndim == 1
    if squeeze:
        z = z[None, :]

    n, Dm1 = z.shape
    D = Dm1 + 1

    # Build Helmert basis matrix V: shape (D-1, D)
    # V[k-1, :] = coefficient vector for z_k
    V = np.zeros((D - 1, D))
    for k in range(1, D):
        coeff = np.sqrt(k / (k + 1))
        V[k - 1, :k] = coeff / k
        V[k - 1, k] = -coeff

    # log(p) = V^T z  (up to an additive constant for normalisation)
    log_p = z @ V           # (N, D)
    p = np.exp(log_p)
    p /= p.sum(axis=1, keepdims=True)

    return p[0] if squeeze else p


def aitchison_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Aitchison distance between two compositions: ‖ilr(p) − ilr(q)‖₂.

    Input: p, q — (D,) strictly-positive proportions (or counts, will be normalised).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() != 1.0:
        p = p / p.sum()
    if q.sum() != 1.0:
        q = q / q.sum()
    return float(np.linalg.norm(ilr(p) - ilr(q)))


# ── DataFrame-level helper ────────────────────────────────────────────────────

def prepare_compositional(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform a count DataFrame into ALR and ILR DataFrames.

    df must contain one column per label in LABELS plus arbitrary index columns.
    Rows with zero total are dropped with a warning.

    Returns:
        alr_df — (N, D-1) ALR coordinates, same index as df (after dropping zeros).
        ilr_df — (N, D-1) ILR coordinates, same index as df (after dropping zeros).
    """
    counts = df[LABELS].to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    valid = totals > 0
    if not valid.all():
        import warnings
        warnings.warn(f"{(~valid).sum()} rows with zero total dropped in prepare_compositional")

    idx = df.index[valid]
    counts = counts[valid]
    props = cmult_repl(counts)

    ref_idx = LABELS.index(ALR_REF_LABEL)
    non_ref = [l for l in LABELS if l != ALR_REF_LABEL]

    alr_arr = alr(props, ref_idx=ref_idx)
    ilr_arr = ilr(props)

    alr_df = pd.DataFrame(alr_arr, index=idx, columns=non_ref)
    ilr_cols = [f"ilr_{k}" for k in range(1, len(LABELS))]
    ilr_df = pd.DataFrame(ilr_arr, index=idx, columns=ilr_cols)

    return alr_df, ilr_df
