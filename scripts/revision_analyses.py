"""Additional analyses requested during peer review (first revision).

Computes the numbers that are not produced by scripts/reproduce_analysis.py.
Read-only against the results DB. Writes CSVs to results/revision/ and prints a
summary. Labels refer to reviewer (R1/R2) comments; M = major comment.

  A  (R1.5)      stricter equivalence margin ±0.10 re-read from existing CIs
  C  (R2-M10)    zero-count frequencies by analysis scale x variant (eligible blocks)
  D  (R2-M10)    eosinophil equivalence under alternative zero-replacement settings
  F  (R2-M11)    edge-crop vs matching IoU-only variant (isolates edge removal)
  G  (R2-M6)     slide-level distribution of paired alr differences
  H  (R2-M5)     Table 3 column 2: expected exceedance + true largest-deviation props
  I  (R2-M8)     class-specific retention (method count / No-NMS count) per class
  J  (R2-M4)     Friedman across the 25 variants: statistic, df, exact p, W

Run:  python scripts/revision_analyses.py [A J G F I D H C]   (default: all)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import LABELS, ALR_REF_LABEL, OVERLAP_PCTS, N_BOOTSTRAP, RNG_SEED, RESULTS_DIR
from core.db import connect
from core.compositional import ilr
from core.stats import bootstrap_equivalence, friedman_test

OUT = RESULTS_DIR / "revision"
OUT.mkdir(parents=True, exist_ok=True)

EN = {"Makrophage": "Mac", "Lymphozyt": "Lym",
      "NeutrophilerGranulozyt": "Neu", "EosinophilerGranulozyt": "Eos"}
NON_REF = [l for l in LABELS if l != ALR_REF_LABEL]
SCOPES = ["grid_n1", "grid_n2", "grid_n3", "grid_n4", "grid_n5", "global"]
DEDUP_24 = [f"{fam}_{s}" for fam in ("iou", "nn_dist", "nn_center", "nn_cluster")
            for s in SCOPES]
METHODS_25 = ["none"] + DEDUP_24

t0 = time.time()
def log(msg: str) -> None:
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


# ── shared: slide-level counts wide (per overlap) ─────────────────────────────
def slide_counts(overlap: float) -> pd.DataFrame:
    """slide_id x nms_method -> 4 count columns (English short labels)."""
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT slide_id, nms_method, label, count FROM cell_counts "
            "WHERE overlap_pct=?", [float(overlap)]).fetchdf()
    piv = (df.pivot_table(index=["slide_id", "nms_method"], columns="label",
                          values="count", aggfunc="sum", fill_value=0)
             .reindex(columns=LABELS, fill_value=0).reset_index())
    return piv


def repl_props(counts: np.ndarray, s: float = 0.5, common_total: float | None = None
               ) -> np.ndarray:
    """CZM Bayesian-multiplicative replacement with adjustable prior strength.

    p_zero = s/n per zero part (default s=0.5 = the study setting); non-zero
    parts multiplicatively rescaled. common_total rescales every row to that
    total first (count-scale-invariant variant)."""
    counts = np.asarray(counts, dtype=float)
    out = np.empty_like(counts)
    for i, row in enumerate(counts):
        n = row.sum()
        if common_total is not None and n > 0:
            row = row * (common_total / n)
            n = common_total
        if n == 0:
            out[i] = np.ones(len(row)) / len(row)
            continue
        zeros = row == 0
        nz = int(zeros.sum())
        p = row / n
        p[zeros] = s / n
        p[~zeros] *= 1 - nz * s / n
        out[i] = p
    return out


def alr_table(piv: pd.DataFrame, s: float = 0.5, common_total: float | None = None
              ) -> pd.DataFrame:
    props = repl_props(piv[LABELS].to_numpy(), s=s, common_total=common_total)
    ref = LABELS.index(ALR_REF_LABEL)
    out = piv[["slide_id", "nms_method"]].copy()
    for lbl in NON_REF:
        out[f"alr_{EN[lbl]}"] = np.log(props[:, LABELS.index(lbl)] / props[:, ref])
    return out


def paired_diffs(alr_df: pd.DataFrame, method: str, ref: str, coord: str) -> np.ndarray:
    wide = alr_df.pivot(index="slide_id", columns="nms_method", values=coord)
    return (wide[method] - wide[ref]).dropna().to_numpy()


# ── A: stricter margin from existing CIs ─────────────────────────────────────
def analysis_A() -> None:
    eq = pd.read_csv(RESULTS_DIR / "equivalence_all_overlaps.csv")
    eq["label_en"] = eq["label"].map(EN)
    for m in (0.10, 0.15, 0.20):
        eq[f"equiv_{m:.2f}"] = (eq["ci_low"] > -m) & (eq["ci_high"] < m)
    eq.to_csv(OUT / "A_margin_sensitivity_full.csv", index=False)

    dedup = eq[~eq.nms_method.str.startswith("iou_edgecrop")]
    summ = (dedup.groupby(["nms_method", "label_en"])
            [["equiv_0.10", "equiv_0.15", "equiv_0.20"]]
            .sum().astype(int).reset_index())  # of 9 overlaps
    summ.to_csv(OUT / "A_margin_sensitivity_summary.csv", index=False)
    prim = dedup[dedup.nms_method == "iou_grid_n1"]
    log("A: primary comparison (iou_grid_n1 vs none), n overlaps equivalent of 9: "
        + ", ".join(f"{r['label_en']}: m.20={int(r['equiv_0.20'])} "
                    f"m.15={int(r['equiv_0.15'])} m.10={int(r['equiv_0.10'])}"
                    for _, r in summ[summ.nms_method == 'iou_grid_n1'].iterrows()))
    log("A: max |ci bound| for iou_grid_n1 across overlaps = "
        f"{max(prim.ci_low.abs().max(), prim.ci_high.abs().max()):.3f}")
    iou_all = dedup[dedup.nms_method.str.startswith("iou_")]
    log(f"A: IoU family rows equivalent at ±0.10: {int(iou_all['equiv_0.10'].sum())}"
        f"/{len(iou_all)}")


# ── J: Friedman across the 25 variants (slides = subjects) ───────────────────
def analysis_J() -> None:
    rows = []
    for ov in (0.0,):
        piv = slide_counts(ov)
        piv = piv[piv.nms_method.isin(METHODS_25)]
        props = repl_props(piv[LABELS].to_numpy())
        ilr_arr = ilr(props)
        d = piv[["slide_id", "nms_method"]].copy()
        for c in range(ilr_arr.shape[1]):
            d[f"ilr_{c+1}"] = ilr_arr[:, c]
        for c in (1, 2, 3):
            wide = d.pivot(index="slide_id", columns="nms_method",
                           values=f"ilr_{c}").dropna()
            res = friedman_test({m: wide[m].to_numpy() for m in wide.columns})
            res["log10_p"] = sps.chi2.logsf(res["statistic"], res["df"]) / np.log(10)
            res.update({"ilr_coord": c, "overlap_pct": ov})
            rows.append(res)
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "J_friedman_25methods.csv", index=False)
    for r in fr.itertuples():
        log(f"J: ilr_{r.ilr_coord} @ {r.overlap_pct:.0%}: chi2({r.df}) = {r.statistic:.1f}, "
            f"W = {r.kendall_w:.3f}, n = {r.n_subjects}, log10 p = {r.log10_p:.1f}")


# ── G: slide-level distribution of paired differences ────────────────────────
def analysis_G() -> None:
    rows = []
    piv0 = slide_counts(0.0)
    alr0 = alr_table(piv0)
    for coord in [f"alr_{EN[l]}" for l in NON_REF]:
        d = paired_diffs(alr0, "iou_grid_n1", "none", coord)
        rows.append(dict(comparison="iou_grid_n1 - none", overlap=0.0, coord=coord,
                         n=len(d), median=np.median(d),
                         q05=np.quantile(d, .05), q95=np.quantile(d, .95),
                         min=d.min(), max=d.max(),
                         n_gt_020=int((np.abs(d) > 0.20).sum()),
                         n_gt_010=int((np.abs(d) > 0.10).sum())))
    piv25 = slide_counts(0.25)
    alr25 = alr_table(piv25)
    d = paired_diffs(alr25, "nn_center_grid_n2", "none", "alr_Mac")
    rows.append(dict(comparison="nn_center_grid_n2 - none", overlap=0.25,
                     coord="alr_Mac", n=len(d), median=np.median(d),
                     q05=np.quantile(d, .05), q95=np.quantile(d, .95),
                     min=d.min(), max=d.max(),
                     n_gt_020=int((np.abs(d) > 0.20).sum()),
                     n_gt_010=int((np.abs(d) > 0.10).sum())))
    # nn eos at 0%: how many slides beyond margin
    for m in ("nn_dist_grid_n1", "nn_center_grid_n1", "nn_cluster_grid_n1"):
        d = paired_diffs(alr0, m, "none", "alr_Eos")
        rows.append(dict(comparison=f"{m} - none", overlap=0.0, coord="alr_Eos",
                         n=len(d), median=np.median(d),
                         q05=np.quantile(d, .05), q95=np.quantile(d, .95),
                         min=d.min(), max=d.max(),
                         n_gt_020=int((np.abs(d) > 0.20).sum()),
                         n_gt_010=int((np.abs(d) > 0.10).sum())))
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "G_slide_level_distribution.csv", index=False)
    for r in g.itertuples():
        log(f"G: {r.comparison} {r.coord} @ {r.overlap:.0%}: median {r.median:+.3f}, "
            f"5-95% [{r.q05:+.3f}, {r.q95:+.3f}], |d|>0.20 in {r.n_gt_020}/{r.n}")


# ── F: edge-crop vs matching IoU-only variant ─────────────────────────────────
def analysis_F() -> None:
    rows = []
    for ov in (0.0, 0.25):
        alr_df = alr_table(slide_counts(ov))
        for scope in ("grid_n1", "grid_n5", "global"):
            for coord in [f"alr_{EN[l]}" for l in NON_REF]:
                d = paired_diffs(alr_df, f"iou_edgecrop_{scope}", f"iou_{scope}", coord)
                res = bootstrap_equivalence(d, 0.20)
                rows.append(dict(scope=scope, overlap=ov, coord=coord, **res))
    f = pd.DataFrame(rows)
    f.to_csv(OUT / "F_edgecrop_vs_iou.csv", index=False)
    for r in f[f.coord == "alr_Mac"].itertuples():
        log(f"F: edgecrop-{r.scope} minus iou-{r.scope} Mac @ {r.overlap:.0%}: "
            f"{r.median:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] equiv={r.equivalent}")


# ── I: class-specific retention ───────────────────────────────────────────────
def analysis_I() -> None:
    rows = []
    for ov in (0.0, 0.25, 0.5):
        piv = slide_counts(ov)
        tot = piv.groupby("nms_method")[LABELS].sum()
        none = tot.loc["none"]
        keep = ["iou_grid_n1", "iou_grid_n5", "iou_global",
                "nn_dist_grid_n1", "nn_center_grid_n1", "nn_cluster_grid_n1"]
        for m in keep:
            if m not in tot.index:
                continue
            r = {"nms_method": m, "overlap": ov}
            for lbl in LABELS:
                r[f"ret_{EN[lbl]}"] = tot.loc[m, lbl] / none[lbl]
            r["ret_total"] = tot.loc[m].sum() / none.sum()
            rows.append(r)
    i = pd.DataFrame(rows)
    i.to_csv(OUT / "I_class_retention.csv", index=False)
    for r in i.itertuples():
        log(f"I: {r.nms_method} @ {r.overlap:.0%}: retention Mac {r.ret_Mac:.3f} "
            f"Lym {r.ret_Lym:.3f} Neu {r.ret_Neu:.3f} Eos {r.ret_Eos:.3f} "
            f"total {r.ret_total:.3f}")


# ── D: alternative zero-replacement settings ─────────────────────────────────
def analysis_D() -> None:
    piv = slide_counts(0.0)
    n_eos_zero = int((piv[piv.nms_method == "none"]["EosinophilerGranulozyt"] == 0).sum())
    log(f"D: slides with zero eosinophils under No-NMS @ 0%: {n_eos_zero}/76")
    settings = [("s=0.5 (study)", dict(s=0.5)),
                ("s=0.1", dict(s=0.1)),
                ("s=1.0", dict(s=1.0)),
                ("s=0.5, common total 1000", dict(s=0.5, common_total=1000.0))]
    methods = ["iou_grid_n1"] + [m for m in DEDUP_24 if m.startswith("nn_")]
    rows = []
    for name, kw in settings:
        alr_df = alr_table(piv, **kw)
        for m in methods:
            for coord in ("alr_Eos", "alr_Mac", "alr_Neu"):
                d = paired_diffs(alr_df, m, "none", coord)
                res = bootstrap_equivalence(d, 0.20)
                rows.append(dict(setting=name, nms_method=m, coord=coord, **res))
    dd = pd.DataFrame(rows)
    dd.to_csv(OUT / "D_zero_replacement_sensitivity.csv", index=False)
    eos = dd[dd.coord == "alr_Eos"]
    for name, _ in settings:
        sub = eos[eos.setting == name]
        nn = sub[sub.nms_method.str.startswith("nn_")]
        iou = sub[sub.nms_method == "iou_grid_n1"].iloc[0]
        log(f"D: [{name}] iou_grid_n1 Eos {iou['median']:+.3f} "
            f"[{iou['ci_low']:+.3f},{iou['ci_high']:+.3f}] equiv={iou['equivalent']}; "
            f"nn equivalent {int(nn.equivalent.sum())}/{len(nn)}, "
            f"median range {nn['median'].min():+.3f}..{nn['median'].max():+.3f}")


# ── H: Table 3 quantities ─────────────────────────────────────────────────────
def analysis_H() -> None:
    with connect(read_only=True) as con:
        # expected exceedance p0 per setting + true largest-deviation proportions
        h = con.execute("""
            WITH thr AS (
                SELECT slide_id, overlap_pct, block_size, block_x, block_y,
                       quantile_cont(mahalanobis_distance, 0.95) AS q95,
                       arg_max(nms_method, mahalanobis_distance) AS largest_method,
                       COUNT(*) AS n_methods
                FROM method_deviation
                GROUP BY 1,2,3,4,5
            ),
            exceed AS (
                SELECT md.block_size, md.overlap_pct, md.nms_method,
                       AVG(CASE WHEN md.mahalanobis_distance > t.q95 THEN 1 ELSE 0 END)
                           AS frac_upper_tail
                FROM method_deviation md
                JOIN thr t USING (slide_id, overlap_pct, block_size, block_x, block_y)
                GROUP BY 1,2,3
            ),
            largest AS (
                SELECT block_size, overlap_pct, largest_method AS nms_method,
                       COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER
                           (PARTITION BY block_size, overlap_pct) AS frac_largest
                FROM thr GROUP BY 1,2,3
            )
            SELECT e.block_size, e.overlap_pct, e.nms_method,
                   e.frac_upper_tail, COALESCE(l.frac_largest, 0) AS frac_largest
            FROM exceed e LEFT JOIN largest l
              USING (block_size, overlap_pct, nms_method)
        """).fetchdf()
    h.to_csv(OUT / "H_table3_per_setting.csv", index=False)
    summ = (h.groupby("nms_method")[["frac_upper_tail", "frac_largest"]]
            .mean().sort_values("frac_upper_tail", ascending=False).reset_index())
    summ.to_csv(OUT / "H_table3_summary.csv", index=False)
    p0 = h.groupby(["block_size", "overlap_pct"])["frac_upper_tail"].sum().mean()
    log(f"H: mean per-block #methods above q95 threshold = {p0:.3f} "
        f"(expected exceedance per method = {p0/25:.4f})")
    log("H: top rows (mean frac_upper_tail | mean frac_largest):")
    for r in summ.head(6).itertuples():
        log(f"H:   {r.nms_method}: {r.frac_upper_tail:.3f} | {r.frac_largest:.3f}")
    log(f"H: sum of mean frac_largest over methods = {summ.frac_largest.sum():.3f}")


# ── C: zero-count frequencies by scale x variant (eligible blocks) ────────────
def analysis_C() -> None:
    with connect(read_only=True) as con:
        c = con.execute("""
            WITH keep AS (
                SELECT slide_id, overlap_pct, block_size, block_x, block_y
                FROM block_counts
                WHERE nms_method='none' AND n_block_total >= 500
                  AND label='Makrophage'
            )
            SELECT bc.block_size, bc.overlap_pct, bc.nms_method, bc.label,
                   COUNT(*) AS n_blocks,
                   SUM(CASE WHEN bc.count = 0 THEN 1 ELSE 0 END) AS n_zero
            FROM block_counts bc
            JOIN keep USING (slide_id, overlap_pct, block_size, block_x, block_y)
            WHERE bc.label IN ('EosinophilerGranulozyt', 'Lymphozyt')
            GROUP BY 1,2,3,4
        """).fetchdf()
    c["frac_zero"] = c.n_zero / c.n_blocks
    c["label"] = c["label"].map(EN)
    c.to_csv(OUT / "C_zero_frequencies.csv", index=False)
    eos = c[(c.label == "Eos") & (c.nms_method.isin(["none", "iou_grid_n1",
                                                     "nn_dist_grid_n1"]))]
    for bs in sorted(eos.block_size.unique()):
        sub = eos[(eos.block_size == bs) & (eos.overlap_pct == 0.0)]
        parts = ", ".join(f"{r.nms_method}={r.frac_zero:.1%}" for r in sub.itertuples())
        log(f"C: eos zero-fraction @ 0% overlap, block_size {bs}: {parts}")
    lym = c[(c.label == "Lym")]
    log(f"C: max lymphocyte zero-fraction anywhere: {lym.frac_zero.max():.2%}")


# ── sanity: reproduce Table 1 numbers from the existing CSV ───────────────────
def sanity() -> None:
    eq = pd.read_csv(RESULTS_DIR / "equivalence_all_overlaps.csv")
    r = eq[(eq.nms_method == "iou_grid_n1") & (eq.overlap_pct == 0.0)]
    for row in r.itertuples():
        log(f"sanity: iou_grid_n1 @0% {EN[row.label]}: {row.median:+.3f} "
            f"[{row.ci_low:+.3f}, {row.ci_high:+.3f}]  "
            "(manuscript: Mac -0.065 [-0.097,-0.056], Neu +0.063 [0.042,0.083], "
            "Eos +0.116 [0.095,0.139])")


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"sanity", "A", "J", "G", "F", "I", "D", "H", "C"}
    steps = {"sanity": sanity, "A": analysis_A, "J": analysis_J, "G": analysis_G,
             "F": analysis_F, "I": analysis_I, "D": analysis_D, "H": analysis_H,
             "C": analysis_C}
    for name, fn in steps.items():
        if name in which:
            fn()
    log("ALL DONE")
