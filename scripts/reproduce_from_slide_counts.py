"""Reproduce the slide-level results from results/slide_counts.csv alone.

No database, images or detector output are needed. Recomputes, with the same
library functions as the full pipeline,

  * the equivalence of every post-processing variant versus No-NMS at every
    tile overlap (Table 1, Table S4)  -> compared with results/equivalence_all_overlaps.csv
  * the Friedman tests across the 25 variants at 0% overlap
                                      -> compared with results/revision/J_friedman_25methods.csv
  * the cohort proportions used in Table S1 (count-weighted mean composition
    of the No-NMS differential at 0% overlap)

and reports the largest absolute deviation from the published values.

Run:  python scripts/reproduce_from_slide_counts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import ALR_REF_LABEL, EQUIV_MARGIN_ALR, LABELS, OVERLAP_PCTS, RESULTS_DIR
from core.compositional import cmult_repl, ilr, ilr_inv
from core.stats import bootstrap_equivalence, friedman_test, weighted_ilr_mean

NON_REF = [lbl for lbl in LABELS if lbl != ALR_REF_LABEL]
DEDUP_24 = [f"{fam}_{scope}" for fam in ("iou", "nn_dist", "nn_center", "nn_cluster")
            for scope in ("grid_n1", "grid_n2", "grid_n3", "grid_n4", "grid_n5", "global")]


def compositions(counts: pd.DataFrame, overlap: float) -> pd.DataFrame:
    """Per (slide, variant): ilr coordinates and alr log-ratios of the slide total."""
    sub = counts[counts.overlap_pct == overlap]
    piv = (sub.pivot_table(index=["slide_id", "nms_method"], columns="label",
                           values="count", aggfunc="sum", fill_value=0)
              .reindex(columns=LABELS, fill_value=0).reset_index())
    props = cmult_repl(piv[LABELS].to_numpy(dtype=float))
    out = piv[["slide_id", "nms_method"]].copy()
    out["n_total"] = piv[LABELS].to_numpy().sum(axis=1)
    coords = ilr(props)
    for c in range(coords.shape[1]):
        out[f"ilr_{c + 1}"] = coords[:, c]
    ref = LABELS.index(ALR_REF_LABEL)
    for lbl in NON_REF:
        out[f"alr_{lbl}"] = np.log(props[:, LABELS.index(lbl)] / props[:, ref])
    return out


def equivalence(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ov in OVERLAP_PCTS:
        comp = compositions(counts, ov)
        for lbl in NON_REF:
            wide = comp.pivot(index="slide_id", columns="nms_method", values=f"alr_{lbl}")
            for method in wide.columns:
                if method == "none":
                    continue
                res = bootstrap_equivalence((wide[method] - wide["none"]).to_numpy(),
                                            EQUIV_MARGIN_ALR)
                rows.append({**res, "nms_method": method, "label": lbl,
                             "overlap_pct": float(ov)})
    return pd.DataFrame(rows)


def friedman(counts: pd.DataFrame) -> pd.DataFrame:
    comp = compositions(counts, 0.0)
    comp = comp[comp.nms_method.isin(["none", *DEDUP_24])]
    rows = []
    for c in (1, 2, 3):
        wide = comp.pivot(index="slide_id", columns="nms_method", values=f"ilr_{c}")
        rows.append({**friedman_test({m: wide[m].to_numpy() for m in wide.columns}),
                     "ilr_coord": c})
    return pd.DataFrame(rows)


def table_s1_proportions(counts: pd.DataFrame) -> np.ndarray:
    comp = compositions(counts, 0.0)
    none = comp[comp.nms_method == "none"]
    mean_ilr = weighted_ilr_mean(none[["ilr_1", "ilr_2", "ilr_3"]].to_numpy(),
                                 none["n_total"].to_numpy(dtype=float))
    return ilr_inv(mean_ilr) * 100


def main() -> None:
    counts = pd.read_csv(RESULTS_DIR / "slide_counts.csv")
    print(f"loaded {len(counts):,} rows, {counts.slide_id.nunique()} slides, "
          f"{counts.nms_method.nunique()} variants")

    eq = equivalence(counts)
    pub = pd.read_csv(RESULTS_DIR / "equivalence_all_overlaps.csv")
    m = eq.merge(pub, on=["nms_method", "label", "overlap_pct"], suffixes=("", "_pub"))
    assert len(m) == len(pub) == 810, (len(m), len(pub))
    dev = max((m[c] - m[f"{c}_pub"]).abs().max() for c in ("median", "ci_low", "ci_high"))
    agree = (m["equivalent"] == m["equivalent_pub"]).all()
    print(f"equivalence: {len(m)} comparisons, max |deviation| = {dev:.2e}, "
          f"equivalence decisions identical: {agree}")
    prim = eq[(eq.nms_method == "iou_grid_n1") & (eq.overlap_pct == 0.0)]
    for r in prim.itertuples():
        print(f"  primary, {r.label:<24} Δ = {r.median:+.3f} "
              f"(90% CI {r.ci_low:+.3f} to {r.ci_high:+.3f})  equivalent: {r.equivalent}")

    fr = friedman(counts)
    pub_fr = pd.read_csv(RESULTS_DIR / "revision" / "J_friedman_25methods.csv")
    dev_fr = (fr["statistic"] - pub_fr["statistic"]).abs().max()
    print(f"friedman: chi2({int(fr['df'].iloc[0])}) = "
          f"{fr['statistic'].min():.1f}-{fr['statistic'].max():.1f}, "
          f"W = {fr['kendall_w'].min():.2f}-{fr['kendall_w'].max():.2f}, "
          f"n = {int(fr['n_subjects'].iloc[0])}; max |deviation| = {dev_fr:.2e}")

    p = table_s1_proportions(counts)
    print("Table S1 proportions (%): " + ", ".join(
        f"{lbl} {v:.1f}" for lbl, v in zip(LABELS, p)))


if __name__ == "__main__":
    main()
