"""Inverse-variance-weighted slide-level summary for Paper 1.

Computes w_s = N_s / (1 + (N_s - 1)·ρ_s) where ρ_s is the per-slide
effective autocorrelation from Moran's I (optional table
morans_i, not part of this repository).  With ρ_s = 0 falls
back to w_s = N_s (vanilla inverse-variance weighting).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import LABELS, OVERLAP_PCTS
from core.compositional import cmult_repl, ilr, alr, ilr_inv
from core.db import connect
from core.stats import inverse_variance_weights, weighted_ilr_mean

logger = logging.getLogger(__name__)


def per_slide_method_ilrs(overlap_pct: float) -> pd.DataFrame:
    """One row per (slide_id, nms_method): ILR-transformed slide-total composition.

    Returns DataFrame: slide_id, nms_method, ilr_1, ilr_2, ilr_3, n_total.
    """
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT slide_id, nms_method, label, count "
            "FROM cell_counts WHERE overlap_pct=?",
            [float(overlap_pct)],
        ).fetchdf()

    if df.empty:
        return pd.DataFrame()

    pivot = (
        df.pivot_table(
            index=["slide_id", "nms_method"],
            columns="label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=LABELS, fill_value=0)
        .reset_index()
    )

    counts = pivot[LABELS].to_numpy(dtype=float)
    props = cmult_repl(counts)
    ilr_coords = ilr(props)  # (N, 3)

    result = pivot[["slide_id", "nms_method"]].copy()
    result["n_total"] = counts.sum(axis=1).astype(int)
    for k in range(ilr_coords.shape[1]):
        result[f"ilr_{k + 1}"] = ilr_coords[:, k]

    # Occupied-tile count per (slide, method) → mean cluster size for the
    # cluster-sampling design effect used in the Moran-corrected weighting.
    with connect(read_only=True) as con:
        t_df = con.execute(
            "SELECT slide_id, nms_method, COUNT(DISTINCT (tile_x, tile_y)) AS n_tiles "
            "FROM tile_counts WHERE overlap_pct=? GROUP BY slide_id, nms_method",
            [float(overlap_pct)],
        ).fetchdf()
    result = result.merge(t_df, on=["slide_id", "nms_method"], how="left")
    result["n_tiles"] = result["n_tiles"].fillna(1).clip(lower=1).astype(int)

    return result


def weighted_method_means(
    overlap_pct: float,
    use_moran_correction: bool = True,
) -> pd.DataFrame:
    """Inverse-variance-weighted mean ILR composition per NMS method.

    Weights: w_s = N_s / (1 + (N_s - 1)·ρ_s)  (or w_s = N_s if Moran unavailable).

    Returns DataFrame: nms_method, ilr_1..3, alr_Mac, alr_Neu, alr_Eos, n_slides.
    """
    ilr_df = per_slide_method_ilrs(overlap_pct)
    if ilr_df.empty:
        return pd.DataFrame()

    # Try to load Moran's I per slide (averaged over labels and methods)
    moran_per_slide: dict[str, float] = {}
    if use_moran_correction:
        try:
            with connect(read_only=True) as con:
                mi_df = con.execute(
                    "SELECT slide_id, AVG(moran_i) as mean_moran "
                    "FROM morans_i WHERE overlap_pct=? GROUP BY slide_id",
                    [float(overlap_pct)],
                ).fetchdf()
            if not mi_df.empty:
                moran_per_slide = dict(zip(mi_df["slide_id"], mi_df["mean_moran"]))
        except Exception:
            pass  # morans_i table may not exist yet

    ilr_cols = [f"ilr_{k}" for k in range(1, len(LABELS))]
    rows = []
    for method, grp in ilr_df.groupby("nms_method"):
        n_per_slide = grp["n_total"].to_numpy(dtype=float)
        moran_arr = np.array([
            moran_per_slide.get(sid, 0.0) for sid in grp["slide_id"]
        ]) if moran_per_slide else None
        # Mean cluster size m̄ = cells / occupied tiles (cluster-sampling DEFF).
        n_tiles = grp["n_tiles"].to_numpy(dtype=float) if "n_tiles" in grp else None
        cluster_size = (np.divide(n_per_slide, n_tiles,
                                  out=np.ones_like(n_per_slide), where=n_tiles > 0)
                        if (moran_arr is not None and n_tiles is not None) else None)

        weights = inverse_variance_weights(n_per_slide, moran_arr, cluster_size)
        ilr_arr = grp[ilr_cols].to_numpy(dtype=float)
        wmean_ilr = weighted_ilr_mean(ilr_arr, weights)

        # Convert to composition then ALR for interpretability
        comp = ilr_inv(wmean_ilr)
        from config import ALR_REF_LABEL, LABEL_ABBR
        ref_idx = LABELS.index(ALR_REF_LABEL)
        non_ref = [l for l in LABELS if l != ALR_REF_LABEL]
        alr_vals = np.log(comp[[i for i in range(len(LABELS)) if i != ref_idx]]
                          / comp[ref_idx])

        row = {"nms_method": method, "n_slides": len(grp)}
        for k, v in enumerate(wmean_ilr):
            row[f"ilr_{k + 1}"] = float(v)
        for lbl, v in zip(non_ref, alr_vals):
            row[f"alr_{LABEL_ABBR[lbl]}"] = float(v)  # Mac/Neu/Eos (matches block_consensus)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("nms_method").reset_index(drop=True)
