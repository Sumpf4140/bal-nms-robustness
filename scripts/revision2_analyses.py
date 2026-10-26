"""Additional analyses for the second revision (Reviewer 1, round 2).

Read-only against the results DB (raw detections, 0% tile overlap). Writes CSVs to
results/revision2/.

  K  sensitivity to the confidence threshold (0.37 = lowest exported score, 0.44 =
     study setting, 0.50, 0.60, 0.70) for No-NMS and the four single-tile rules
  L  sensitivity to the suppression parameters at the study threshold:
     IoU threshold (0.30, 0.45, 0.60), nn-distance radius and nn-cluster linking
     radius (10, 15, 20, 30, 40 px), nn-centre radius multiplier (1, 2, 3 SD)
  N  class-specific IoU suppression (IoU applied within each predicted class, single tile),
     to test whether the eosinophil shift of the nearest-neighbor rules comes from their
     class-specific suppression rather than from their distance criterion
  M  clinical classification at the BAL decision thresholds (lymphocytes > 15%,
     neutrophils > 3%, eosinophils > 1%) for every variant and tile overlap,
     compared with No-NMS (whole-slide percentages, all four cell types)

Run:  python scripts/revision2_analyses.py [K L M]   (K and L share one pass)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import LABELS, ALR_REF_LABEL, RESULTS_DIR, EQUIV_MARGIN_ALR  # noqa: E402
from core.db import connect, DB_PATH  # noqa: E402
from core import nms as N  # noqa: E402
from core.stats import bootstrap_equivalence  # noqa: E402

OUT = RESULTS_DIR / "revision2"
OUT.mkdir(parents=True, exist_ok=True)
EN = {"Makrophage": "Mac", "Lymphozyt": "Lym", "NeutrophilerGranulozyt": "Neu", "EosinophilerGranulozyt": "Eos"}
THRESHOLDS = [0.37, 0.44, 0.50, 0.60, 0.70]
RADII = [10, 15, 20, 30, 40]
IOU_T = [0.30, 0.45, 0.60]
K_SD = [1, 2, 3]
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def per_tile(df: pd.DataFrame, fn) -> np.ndarray:
    return N._apply_scope(df, "per_tile", fn)


def counts(df: pd.DataFrame, keep: np.ndarray) -> dict:
    vc = df.loc[keep, "label"].value_counts()
    return {EN[l]: int(vc.get(l, 0)) for l in LABELS}


def center_fn(df: pd.DataFrame, k: float):
    radii = N._compute_center_radii(df)
    radii = {l: r / 2.0 * k for l, r in radii.items()}          # module uses 2·SD
    return lambda g: N._nn_center_suppress_core(g, radii)


def one_slide(slide_id: str) -> list[dict]:
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM raw_detections WHERE slide_id=? AND overlap_pct=0.0",
                     [slide_id]).fetchdf()
    con.close()
    df = df.reset_index(drop=True)
    rows = []

    def add(analysis, setting, method, sub, keep):
        rows.append({"slide_id": slide_id, "analysis": analysis, "setting": setting, "method": method,
                     **counts(sub, keep)})

    # K: confidence threshold. Greedy IoU and nn-distance are monotone in the score order,
    # so one run on the lowest threshold gives the survivors for every higher threshold.
    base = df[df.confidence >= THRESHOLDS[0]].reset_index(drop=True)
    keep_iou = per_tile(base, N.greedy_iou_suppress)
    keep_nnd = per_tile(base, N._nn_dist_suppress_core)
    for t in THRESHOLDS:
        m = (base.confidence >= t).to_numpy()
        sub = base[m].reset_index(drop=True)
        add("K", t, "none", sub, np.ones(len(sub), bool))
        add("K", t, "iou", sub, keep_iou[m])
        add("K", t, "nn_dist", sub, keep_nnd[m])
        add("K", t, "nn_center", sub, per_tile(sub, center_fn(sub, 2)))
        add("K", t, "nn_cluster", sub, per_tile(sub, N._nn_cluster_suppress_core))

    # L: suppression parameters at the study threshold
    sub = df[df.confidence >= 0.44].reset_index(drop=True)
    add("L", "none", "none", sub, np.ones(len(sub), bool))
    for it in IOU_T:
        add("L", it, "iou", sub, per_tile(sub, lambda g, it=it: N.greedy_iou_suppress(g, it)))
    for r in RADII:
        add("L", r, "nn_dist", sub, per_tile(sub, lambda g, r=r: N._nn_dist_suppress_core(g, r)))
        add("L", r, "nn_cluster", sub, per_tile(sub, lambda g, r=r: N._nn_cluster_suppress_core(g, r)))
    for k in K_SD:
        add("L", k, "nn_center", sub, per_tile(sub, center_fn(sub, k)))
    return rows


def one_slide_N(slide_id: str) -> list[dict]:
    import duckdb
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute("SELECT * FROM raw_detections WHERE slide_id=? AND overlap_pct=0.0 AND confidence>=0.44",
                     [slide_id]).fetchdf().reset_index(drop=True)
    con.close()
    keep = N._groupby_suppress(df, ["tile_x", "tile_y", "label"], N.greedy_iou_suppress)
    return [{"slide_id": slide_id, "method": "none", **counts(df, np.ones(len(df), bool))},
            {"slide_id": slide_id, "method": "iou_classwise", **counts(df, keep)},
            {"slide_id": slide_id, "method": "iou", **counts(df, N._apply_scope(df, "per_tile", N.greedy_iou_suppress))}]


def analysis_N(n_workers: int = 6) -> None:
    with connect(read_only=True) as con:
        slides = [s for (s,) in con.execute(
            "SELECT DISTINCT slide_id FROM raw_detections WHERE overlap_pct=0.0 ORDER BY 1").fetchall()]
    rows = []
    with ProcessPoolExecutor(n_workers) as ex:
        for r in ex.map(one_slide_N, slides):
            rows += r
    rows = pd.DataFrame(rows)
    rows["slide_id"] = rows.slide_id.map({sid: f"S{i:02d}" for i, sid in enumerate(sorted(slides), 1)})
    rows.to_csv(OUT / "N_classwise_iou_counts.csv", index=False)
    d = alr(rows)
    ref = d[d.method == "none"].set_index("slide_id")
    res = []
    for meth in ("iou", "iou_classwise"):
        v = d[d.method == meth].set_index("slide_id").loc[ref.index]
        for c in ("Mac", "Neu", "Eos"):
            diff = (v[f"alr_{c}"] - ref[f"alr_{c}"]).to_numpy()
            res.append({"method": meth, "coord": c, **bootstrap_equivalence(diff, EQUIV_MARGIN_ALR),
                        "n_outside_020": int((np.abs(diff) > EQUIV_MARGIN_ALR).sum())})
    pd.DataFrame(res).to_csv(OUT / "N_classwise_iou.csv", index=False)
    log("N done")


def alr(df: pd.DataFrame) -> pd.DataFrame:
    """alr relative to lymphocytes after CZM zero replacement, as in the main analysis
    (zero counts occur only at the strictest settings, e.g. eosinophils at a threshold of 0.70)."""
    from core.compositional import cmult_repl
    out = df.copy()
    props = cmult_repl(out[["Mac", "Lym", "Neu", "Eos"]].to_numpy(dtype=float))
    for i, c in ((0, "Mac"), (2, "Neu"), (3, "Eos")):
        out[f"alr_{c}"] = np.log(props[:, i] / props[:, 1])
    return out


def equivalence(rows: pd.DataFrame, analysis: str) -> pd.DataFrame:
    d = alr(rows[rows.analysis == analysis])
    res = []
    if analysis == "K":
        for t, g in d.groupby("setting"):
            ref = g[g.method == "none"].set_index("slide_id")
            for meth in ("iou", "nn_dist", "nn_center", "nn_cluster"):
                v = g[g.method == meth].set_index("slide_id").loc[ref.index]
                for c in ("Mac", "Neu", "Eos"):
                    diff = (v[f"alr_{c}"] - ref[f"alr_{c}"]).to_numpy()
                    r = bootstrap_equivalence(diff, EQUIV_MARGIN_ALR)
                    res.append({"threshold": t, "method": meth, "coord": c, **r,
                                "n_outside_020": int((np.abs(diff) > EQUIV_MARGIN_ALR).sum())})
    else:
        ref = d[d.method == "none"].set_index("slide_id")
        for (meth, s), g in d[d.method != "none"].groupby(["method", "setting"]):
            v = g.set_index("slide_id").loc[ref.index]
            for c in ("Mac", "Neu", "Eos"):
                diff = (v[f"alr_{c}"] - ref[f"alr_{c}"]).to_numpy()
                r = bootstrap_equivalence(diff, EQUIV_MARGIN_ALR)
                res.append({"method": meth, "parameter": s, "coord": c, **r,
                            "n_outside_020": int((np.abs(diff) > EQUIV_MARGIN_ALR).sum())})
    return pd.DataFrame(res)


def analysis_KL(n_workers: int = 6, slides: list[str] | None = None) -> None:
    if slides is None:
        with connect(read_only=True) as con:
            slides = [s for (s,) in con.execute(
                "SELECT DISTINCT slide_id FROM raw_detections WHERE overlap_pct=0.0 ORDER BY 1").fetchall()]
    rows = []
    with ProcessPoolExecutor(n_workers) as ex:
        for i, r in enumerate(ex.map(one_slide, slides), 1):
            rows += r
            if i % 10 == 0:
                log(f"KL: {i}/{len(slides)} slides")
    rows = pd.DataFrame(rows)
    pseudo = {sid: f"S{i:02d}" for i, sid in enumerate(sorted(slides), 1)}   # same scheme as slide_counts.csv
    # sanity: the study setting must reproduce the pipeline counts
    with connect(read_only=True) as con:
        cc = con.execute("SELECT slide_id, nms_method, label, count FROM cell_counts WHERE overlap_pct=0.0 "
                         "AND nms_method IN ('none','iou_grid_n1','nn_dist_grid_n1','nn_center_grid_n1',"
                         "'nn_cluster_grid_n1')").fetchdf()
    cc["label"] = cc.label.map(EN)
    cc["method"] = cc.nms_method.str.replace("_grid_n1", "", regex=False)
    piv = cc.pivot_table(index=["slide_id", "method"], columns="label", values="count").reset_index()
    k44 = rows[(rows.analysis == "K") & (rows.setting == 0.44)]
    m = k44.merge(piv, on=["slide_id", "method"], suffixes=("", "_db"))
    dev = max(int((m[c] - m[f"{c}_db"]).abs().max()) for c in ("Mac", "Lym", "Neu", "Eos"))
    log(f"KL: check at 0.44 against the pipeline: {len(m)} slide-method pairs, max count difference {dev}")
    rows["slide_id"] = rows.slide_id.map(pseudo)
    rows.to_csv(OUT / "KL_slide_counts.csv", index=False)
    equivalence(rows, "K").to_csv(OUT / "K_confidence_threshold.csv", index=False)
    equivalence(rows, "L").to_csv(OUT / "L_parameters.csv", index=False)
    log("KL done")


def analysis_M() -> None:
    sc = pd.read_csv(RESULTS_DIR / "slide_counts.csv")
    sc["label"] = sc.label.map(EN)
    piv = sc.pivot_table(index=["slide_id", "overlap_pct", "nms_method"], columns="label",
                         values="count").reset_index()
    tot = piv[["Mac", "Lym", "Neu", "Eos"]].sum(axis=1)
    for c, thr, name in (("Lym", 15, "lymphocytosis"), ("Neu", 3, "neutrophilia"), ("Eos", 1, "eosinophilia")):
        piv[f"pct_{c}"] = 100 * piv[c] / tot
        piv[name] = piv[f"pct_{c}"] > thr
    piv.to_csv(OUT / "M_slide_percentages.csv", index=False)
    ref = piv[piv.nms_method == "none"].set_index(["slide_id", "overlap_pct"])
    rows = []
    for meth, g in piv[piv.nms_method != "none"].groupby("nms_method"):
        g = g.set_index(["slide_id", "overlap_pct"])
        for name, c, thr in (("lymphocytosis", "Lym", 15), ("neutrophilia", "Neu", 3), ("eosinophilia", "Eos", 1)):
            for ov, gg in g.groupby(level="overlap_pct"):
                r = ref.loc[gg.index]
                changed = gg[name] != r[name]
                rows.append({"nms_method": meth, "overlap_pct": ov, "criterion": name,
                             "n_slides": len(gg), "n_positive_none": int(r[name].sum()),
                             "n_positive_variant": int(gg[name].sum()), "n_changed": int(changed.sum()),
                             "changed_slides": ";".join(gg.index[changed].get_level_values(0))})
    pd.DataFrame(rows).to_csv(OUT / "M_classification_changes.csv", index=False)
    log("M done")


def summaries() -> None:
    """Recompute K/L summaries from the stored per-slide counts."""
    rows = pd.read_csv(OUT / "KL_slide_counts.csv", dtype={"setting": str})
    rows["setting"] = rows.setting.map(lambda v: v if v == "none" else float(v))
    zeros = rows[["Mac", "Lym", "Neu", "Eos"]].eq(0).sum()
    log(f"S: zero counts in KL rows: {zeros.to_dict()}")
    equivalence(rows, "K").to_csv(OUT / "K_confidence_threshold.csv", index=False)
    equivalence(rows, "L").to_csv(OUT / "L_parameters.csv", index=False)
    log("S done")


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"K", "M"}
    if "S" in which:
        summaries()
    if "N" in which:
        analysis_N()
    if "M" in which:
        analysis_M()
    if "K" in which or "L" in which:
        analysis_KL()
    log("ALL DONE")
