"""Count-weighted slide-level summary.

Per variant, the mean ilr composition across slides weighted by each slide's
number of retained detections (w_s = N_s; multinomial precision is
proportional to the number of counted cells).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import LABELS, OVERLAP_PCTS
from core.compositional import cmult_repl, ilr, alr, ilr_inv
from core.db import connect
from core.stats import weighted_ilr_mean

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

    return result


def weighted_method_means(
    overlap_pct: float,
) -> pd.DataFrame:
    """Count-weighted mean ILR composition per NMS method (w_s = N_s).

    Returns DataFrame: nms_method, ilr_1..3, alr_Mac, alr_Neu, alr_Eos, n_slides.
    """
    ilr_df = per_slide_method_ilrs(overlap_pct)
    if ilr_df.empty:
        return pd.DataFrame()

    ilr_cols = [f"ilr_{k}" for k in range(1, len(LABELS))]
    rows = []
    for method, grp in ilr_df.groupby("nms_method"):
        weights = grp["n_total"].to_numpy(dtype=float)
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
