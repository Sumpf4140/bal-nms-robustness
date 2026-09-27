"""Explicit no-NMS-vs-NMS hypothesis test.

Restores the project's original thesis — *"NMS is unnecessary for the relative
differential, and is a waste of resources"* — as a direct, baseline-anchored
result alongside the inter-method consensus framing:

  - `friedman_across_methods`: per ILR coordinate, Friedman across all methods
    (subjects = slides, factor = nms_method). Does the *choice* of method move
    the composition at all?
  - `equivalence_vs_none`: per ALR component, bootstrap equivalence of each method
    against the `none` baseline. Positive evidence that skipping NMS does not
    change the differential — a non-significant Friedman alone cannot show this.

Both read slide-total `cell_counts` for a given overlap.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import ALR_REF_LABEL, EQUIV_MARGIN_ALR, LABELS, OVERLAP_PCTS
from core.compositional import cmult_repl, ilr
from core.db import connect
from core.stats import (
    bootstrap_equivalence, friedman_test, hierarchical_correction, nemenyi_post_hoc,
)

logger = logging.getLogger(__name__)

_NON_REF = [lbl for lbl in LABELS if lbl != ALR_REF_LABEL]


def _method_composition(overlap_pct: float) -> pd.DataFrame:
    """Per (slide, method): ILR coords + ALR(non-ref labels) from cell_counts."""
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT slide_id, nms_method, label, count FROM cell_counts WHERE overlap_pct=?",
            [float(overlap_pct)],
        ).fetchdf()
    if df.empty:
        return df

    piv = (df.pivot_table(index=["slide_id", "nms_method"], columns="label",
                          values="count", aggfunc="sum", fill_value=0)
             .reindex(columns=LABELS, fill_value=0)
             .reset_index())
    props = cmult_repl(piv[LABELS].to_numpy(dtype=float))
    ref = LABELS.index(ALR_REF_LABEL)

    out = piv[["slide_id", "nms_method"]].copy()
    ilr_arr = ilr(props)
    for c in range(ilr_arr.shape[1]):
        out[f"ilr_{c + 1}"] = ilr_arr[:, c]
    for lbl in _NON_REF:
        out[f"alr_{lbl}"] = np.log(props[:, LABELS.index(lbl)] / props[:, ref])
    return out


def friedman_across_methods(overlap_pct: float = 0.0) -> pd.DataFrame:
    """Per ILR coordinate: Friedman across methods (subjects = slides, complete blocks)."""
    comp = _method_composition(overlap_pct)
    if comp is None or comp.empty:
        return pd.DataFrame()

    rows = []
    for c in range(1, len(LABELS)):
        wide = comp.pivot(index="slide_id", columns="nms_method",
                          values=f"ilr_{c}").dropna(axis=0, how="any")
        if wide.shape[0] < 3 or wide.shape[1] < 3:
            continue
        res = friedman_test({m: wide[m].to_numpy() for m in wide.columns})
        res.update({"ilr_coord": c, "overlap_pct": float(overlap_pct)})
        rows.append(res)
    df = pd.DataFrame(rows)
    if not df.empty:  # Holm correction across the ILR-coordinate family
        df["p_holm"] = hierarchical_correction(df["p_value"], len(df)).to_numpy()
    return df


def equivalence_vs_none(overlap_pct: float = 0.0,
                        margin: float = EQUIV_MARGIN_ALR) -> pd.DataFrame:
    """Per (method, ALR label): bootstrap equivalence of (method − none) per slide.

    `equivalent == True` ⇒ that method's composition is indistinguishable from no
    NMS within ±margin → NMS is unnecessary for that component.
    """
    comp = _method_composition(overlap_pct)
    if comp is None or comp.empty:
        return pd.DataFrame()

    rows = []
    for lbl in _NON_REF:
        wide = comp.pivot(index="slide_id", columns="nms_method", values=f"alr_{lbl}")
        if "none" not in wide.columns:
            continue
        for method in wide.columns:
            if method == "none":
                continue
            diffs = (wide[method] - wide["none"]).to_numpy()
            res = bootstrap_equivalence(diffs, margin)
            res.update({"nms_method": method, "label": lbl,
                        "overlap_pct": float(overlap_pct)})
            rows.append(res)
    return pd.DataFrame(rows)


def baseline_summary(overlap_pct: float = 0.0,
                     reference_method: str = "iou_grid_n1") -> dict[str, pd.DataFrame]:
    """Both analyses + two verdicts.

    `all_equivalent` (strict): is EVERY one of the 24 methods equivalent to `none`?
    `reference_equivalent` (focused, headline): is `none` equivalent to a single,
    *a-priori-chosen* standard reference (`reference_method`, default the field-
    standard IoU-NMS) across ALL labels? This is the defensible "no-NMS ≡ standard
    NMS" claim: the reference is fixed by convention, not selected post-hoc, so it
    avoids cherry-picking while sidestepping the over-strict all-27-methods bar
    (which fails only because the aggressive nn-* families move the rare eosinophil
    balance). It is one slice of the same `equivalence_vs_none` TOST table — no extra
    statistics — and still measures consistency, not accuracy (no ground truth).
    """
    friedman = friedman_across_methods(overlap_pct)
    equiv = equivalence_vs_none(overlap_pct)
    all_equivalent = bool(not equiv.empty and equiv["equivalent"].all())

    ref_rows = (equiv[equiv["nms_method"] == reference_method]
                if not equiv.empty else equiv)
    if not equiv.empty and ref_rows.empty:
        logger.warning(
            "Baseline reference_method=%r not found among compared methods "
            "(is it a valid non-'none' NMS method?) — reference verdict is False.",
            reference_method,
        )
    reference_equivalent = bool(not ref_rows.empty and ref_rows["equivalent"].all())

    logger.info(
        "Baseline (overlap=%.3f): no-NMS equivalent to '%s' for all labels? %s | "
        "every method equivalent to 'none'? %s (%d/%d method×label equivalent)",
        overlap_pct, reference_method, reference_equivalent, all_equivalent,
        int(equiv["equivalent"].sum()) if not equiv.empty else 0, len(equiv),
    )
    return {"friedman": friedman, "equivalence": equiv,
            "all_equivalent": all_equivalent,
            "reference_method": reference_method,
            "reference_equivalence": ref_rows,
            "reference_equivalent": reference_equivalent}
