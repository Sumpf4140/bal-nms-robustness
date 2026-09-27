"""Integration tests for the analysis pipeline.

These tests use synthetic in-memory data and a temporary DuckDB to verify
the end-to-end flow: NMS run → block aggregation → consensus → deviations.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from config import LABELS, OVERLAP_PCTS, TILE_WIDTH_PX, TILE_HEIGHT_PX
from core.db import init_db, connect
from core.nms import NMS_REGISTRY
from tests.conftest import make_detection_df, make_tile_counts_df


# ── Helpers ───────────────────────────────────────────────────────────────────

def populate_db(db_path, n_detections: int = 200, n_tiles_x: int = 4, n_tiles_y: int = 4):
    """Insert synthetic slide + raw_detections + cell_counts + tile_counts into DB."""
    init_db(db_path=db_path)
    slide_id = "synth_slide"
    img_w = n_tiles_x * TILE_WIDTH_PX
    img_h = n_tiles_y * TILE_HEIGHT_PX

    with connect(db_path=db_path) as con:
        con.execute(
            "INSERT OR REPLACE INTO slides VALUES (?, ?, ?, ?, ?, ?, ?)",
            [slide_id, "/fake/path.png", img_w, img_h, n_tiles_x, n_tiles_y,
             datetime.datetime.utcnow()],
        )

        # Insert raw detections for overlap 0.0
        df_raw = make_detection_df(n=n_detections, n_tiles_x=n_tiles_x, n_tiles_y=n_tiles_y)
        df_raw["slide_id"] = slide_id
        df_raw["overlap_pct"] = 0.0
        cols = ["slide_id", "overlap_pct", "label", "confidence",
                "x1", "y1", "x2", "y2", "cx", "cy", "x_correct", "y_correct",
                "tile_x", "tile_y"]
        con.register("_raw", df_raw[cols])
        con.execute("INSERT INTO raw_detections SELECT * FROM _raw")
        con.unregister("_raw")

    # Generate NMS outputs (using all 28 methods, one overlap)
    with connect(read_only=True, db_path=db_path) as con:
        df_raw2 = con.execute(
            "SELECT * FROM raw_detections WHERE slide_id=? AND overlap_pct=?",
            [slide_id, 0.0]
        ).fetchdf()

    cell_rows, tile_rows, timing_rows = [], [], []
    for method_name, method_fn in NMS_REGISTRY.items():
        keep = method_fn(df_raw2)
        survivors = df_raw2.loc[keep]
        total = len(survivors)
        for lbl in LABELS:
            count = int((survivors["label"] == lbl).sum())
            cell_rows.append({
                "slide_id": slide_id, "overlap_pct": 0.0,
                "nms_method": method_name, "label": lbl,
                "count": count,
                "rel_count": count / total if total > 0 else 0.0,
            })
        # tile counts
        for (tx, ty), grp in survivors.groupby(["tile_x", "tile_y"]):
            for lbl in LABELS:
                tile_rows.append({
                    "slide_id": slide_id, "overlap_pct": 0.0,
                    "nms_method": method_name,
                    "tile_x": int(tx), "tile_y": int(ty),
                    "label": lbl,
                    "count": int((grp["label"] == lbl).sum()),
                })
        timing_rows.append({
            "slide_id": slide_id, "overlap_pct": 0.0,
            "nms_method": method_name,
            "elapsed_cpu": 0.01, "n_input": len(df_raw2), "n_output": int(keep.sum()),
        })

    with connect(db_path=db_path) as con:
        cc_df = pd.DataFrame(cell_rows)
        tc_df = pd.DataFrame(tile_rows)
        ti_df = pd.DataFrame(timing_rows)
        con.register("_cc", cc_df); con.execute("INSERT OR REPLACE INTO cell_counts SELECT * FROM _cc"); con.unregister("_cc")
        con.register("_tc", tc_df); con.execute("INSERT OR REPLACE INTO tile_counts SELECT * FROM _tc"); con.unregister("_tc")
        con.register("_ti", ti_df); con.execute("INSERT OR REPLACE INTO timing SELECT * FROM _ti"); con.unregister("_ti")

    return slide_id


# ── cell_counts correctness ───────────────────────────────────────────────────

class TestCellCounts:
    def test_cell_counts_has_35x4_rows_per_slide_overlap(self, tmp_path):
        db = tmp_path / "test.duckdb"
        populate_db(db)
        with connect(read_only=True, db_path=db) as con:
            n = con.execute(
                "SELECT COUNT(*) FROM cell_counts WHERE slide_id='synth_slide' AND overlap_pct=0.0"
            ).fetchone()[0]
        assert n == 31 * 4, f"Expected {31*4} rows, got {n}"

    def test_rel_count_sums_to_one_per_method(self, tmp_path):
        db = tmp_path / "test.duckdb"
        populate_db(db)
        with connect(read_only=True, db_path=db) as con:
            df = con.execute(
                "SELECT nms_method, SUM(rel_count) as total "
                "FROM cell_counts WHERE slide_id='synth_slide' AND overlap_pct=0.0 "
                "GROUP BY nms_method"
            ).fetchdf()
        np.testing.assert_allclose(df["total"].values, 1.0, atol=1e-10,
                                   err_msg="rel_count must sum to 1 per method")

    def test_none_method_has_highest_count(self, tmp_path):
        db = tmp_path / "test.duckdb"
        populate_db(db)
        with connect(read_only=True, db_path=db) as con:
            df = con.execute(
                "SELECT nms_method, SUM(count) as total_count "
                "FROM cell_counts WHERE slide_id='synth_slide' AND overlap_pct=0.0 "
                "GROUP BY nms_method"
            ).fetchdf()
        none_count = df.loc[df["nms_method"] == "none", "total_count"].iloc[0]
        max_count = df["total_count"].max()
        assert none_count == max_count, \
            f"'none' should have the highest count ({none_count} vs max {max_count})"

    def test_all_labels_present_for_each_method(self, tmp_path):
        db = tmp_path / "test.duckdb"
        populate_db(db)
        with connect(read_only=True, db_path=db) as con:
            df = con.execute(
                "SELECT DISTINCT nms_method, label FROM cell_counts "
                "WHERE slide_id='synth_slide' AND overlap_pct=0.0"
            ).fetchdf()
        for method in NMS_REGISTRY:
            method_labels = set(df.loc[df["nms_method"] == method, "label"].values)
            assert set(LABELS) == method_labels, \
                f"Method {method} missing labels: {set(LABELS) - method_labels}"


# ── block aggregation ─────────────────────────────────────────────────────────

class TestBlockAggregation:
    def test_build_block_counts_populates_table(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        populate_db(db, n_tiles_x=4, n_tiles_y=4)
        import core.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", db)
        import nms.consensus_analysis as ca
        monkeypatch.setattr(ca, "connect", lambda **kw: db_mod.connect(**kw, db_path=db))

        # Direct call using internal functions
        from core.db import connect as _connect
        from core.blocks import aggregate_all_block_sizes

        with _connect(read_only=True, db_path=db) as con:
            tc_df = con.execute("SELECT * FROM tile_counts").fetchdf()

        bc_df = aggregate_all_block_sizes(tc_df)
        assert len(bc_df) > 0

        # All expected block sizes present
        from core.blocks import ALL_BLOCK_SIZES
        present = set(bc_df["block_size"].unique())
        assert set(ALL_BLOCK_SIZES) == present

    def test_n_block_total_matches_sum(self, tmp_path):
        tc = make_tile_counts_df(n_tiles_x=3, n_tiles_y=3, n_methods=3)
        from core.blocks import aggregate_to_blocks
        bc = aggregate_to_blocks(tc, block_size=2)
        key_cols = ["slide_id", "overlap_pct", "nms_method", "block_size", "block_x", "block_y"]
        for _, grp in bc.groupby(key_cols):
            assert grp["count"].sum() == grp["n_block_total"].iloc[0]


# ── consensus and deviations (synthetic) ─────────────────────────────────────

class TestConsensus:
    def test_geometric_median_convergence(self):
        """28 synthetic method ILRs → geometric median converges, distances finite."""
        from core.compositional import cmult_repl, ilr
        from core.consensus import geometric_median_ilr, aitchison_distances_to_consensus
        rng = np.random.default_rng(0)
        counts = rng.integers(10, 1000, (28, 4))
        props = cmult_repl(counts)
        ilrs = ilr(props)  # (28, 3)
        median = geometric_median_ilr(ilrs)
        assert median.shape == (3,)
        assert np.isfinite(median).all()
        dists = aitchison_distances_to_consensus(ilrs, median)
        assert np.isfinite(dists).all()
        assert (dists >= 0).all()

    def test_consensus_is_closer_than_mean_to_outlier_case(self):
        """Geometric median should not be pulled by a single extreme outlier."""
        from core.consensus import geometric_median_ilr
        rng = np.random.default_rng(1)
        pts = rng.standard_normal((27, 3)) * 0.1   # tight cluster near origin
        outlier = np.array([[50.0, 50.0, 50.0]])
        all_pts = np.vstack([pts, outlier])

        median = geometric_median_ilr(all_pts)
        mean = all_pts.mean(axis=0)

        # Median should be closer to cluster centre (origin) than the mean
        dist_median = np.linalg.norm(median)
        dist_mean = np.linalg.norm(mean)
        assert dist_median < dist_mean, \
            f"Median ({dist_median:.2f}) not closer to cluster than mean ({dist_mean:.2f})"

    def test_krippendorff_alpha_on_synthetic_panel(self):
        """α computed on a panel where methods agree within label → α near 1."""
        from core.compositional import cmult_repl, ilr
        from core.consensus import krippendorff_alpha_aitchison
        from tests.conftest import make_ilr_panel

        # Low noise → high α
        panel = make_ilr_panel(n_methods=28, n_blocks=40, noise=0.01, seed=99)
        alpha = krippendorff_alpha_aitchison(panel)
        assert alpha > 0.9, f"Expected high α for low-noise panel, got {alpha:.3f}"

    def test_krippendorff_alpha_bootstrap_ci_width(self):
        """CI width should be positive and finite."""
        from core.consensus import krippendorff_alpha_bootstrap_ci
        from tests.conftest import make_ilr_panel
        panel = make_ilr_panel(n_methods=10, n_blocks=30, noise=0.05, seed=10)
        alpha, lo, hi = krippendorff_alpha_bootstrap_ci(panel, n_boot=100)
        assert np.isfinite(alpha) and np.isfinite(lo) and np.isfinite(hi)
        assert hi >= lo


# ── weighted_summary ──────────────────────────────────────────────────────────

class TestWeightedSummary:
    def test_per_slide_method_ilrs_shape(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        populate_db(db)
        import core.db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = db
        try:
            from nms.weighted_summary import per_slide_method_ilrs
            df = per_slide_method_ilrs(overlap_pct=0.0)
            assert not df.empty
            assert "nms_method" in df.columns
            assert "n_total" in df.columns
            for k in range(1, len(LABELS)):
                assert f"ilr_{k}" in df.columns
        finally:
            db_mod.DB_PATH = orig

    def test_weighted_means_one_row_per_method(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        populate_db(db)
        import core.db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = db
        try:
            from nms.weighted_summary import weighted_method_means
            df = weighted_method_means(overlap_pct=0.0)
            assert not df.empty
            assert len(df) == 31, f"Expected 31 methods, got {len(df)}"
            assert df["nms_method"].nunique() == 31
        finally:
            db_mod.DB_PATH = orig


# ── outlier detection (no DB, synthetic) ─────────────────────────────────────

class TestOutlierDetection:
    def test_methods_in_upper_tail_returns_all_methods(self, tmp_path, monkeypatch):
        db = tmp_path / "test.duckdb"
        populate_db(db, n_tiles_x=6, n_tiles_y=6)

        # Write block_counts and method_deviation synthetically
        from core.blocks import aggregate_all_block_sizes
        from core.compositional import cmult_repl, ilr
        from core.consensus import (
            geometric_median_ilr,
            aitchison_distances_to_consensus,
            mahalanobis_distances_to_consensus,
        )

        with connect(read_only=True, db_path=db) as con:
            tc_df = con.execute("SELECT * FROM tile_counts WHERE overlap_pct=0.0").fetchdf()

        bc_df = aggregate_all_block_sizes(tc_df)
        with connect(db_path=db) as con:
            con.register("_bc", bc_df)
            con.execute("INSERT OR REPLACE INTO block_counts SELECT * FROM _bc")
            con.unregister("_bc")

        # Compute minimal method_deviation for block_size=1
        from config import BLOCK_MIN_CELLS
        bs = 0  # full-slide
        bc_sub = bc_df[
            (bc_df["block_size"] == bs) & (bc_df["n_block_total"] >= BLOCK_MIN_CELLS)
        ]
        if bc_sub.empty:
            pytest.skip("No blocks pass min-cells filter for this synthetic slide")

        dev_rows = []
        for (sid, op, bx, by), grp in bc_sub.groupby(["slide_id", "overlap_pct", "block_x", "block_y"]):
            pivot = (
                grp.pivot_table(index="nms_method", columns="label", values="count",
                                aggfunc="sum", fill_value=0)
                .reindex(columns=LABELS, fill_value=0)
            )
            if len(pivot) < 2:
                continue
            props = cmult_repl(pivot[LABELS].values)
            ilr_arr = ilr(props)
            med = geometric_median_ilr(ilr_arr)
            ait_d = aitchison_distances_to_consensus(ilr_arr, med)
            mah_d = mahalanobis_distances_to_consensus(ilr_arr, med)
            for mi, method in enumerate(pivot.index):
                dev_rows.append({
                    "slide_id": sid, "overlap_pct": op, "nms_method": method,
                    "block_size": bs, "block_x": int(bx), "block_y": int(by),
                    "aitchison_distance": float(ait_d[mi]),
                    "mahalanobis_distance": float(mah_d[mi]),
                })

        if not dev_rows:
            pytest.skip("No deviation rows computed")

        with connect(db_path=db) as con:
            dev_df = pd.DataFrame(dev_rows)
            con.register("_md", dev_df)
            con.execute("INSERT OR REPLACE INTO method_deviation SELECT * FROM _md")
            con.unregister("_md")

        import core.db as db_mod
        orig = db_mod.DB_PATH
        db_mod.DB_PATH = db
        try:
            from nms.outlier_detection import methods_in_upper_tail_fraction
            result = methods_in_upper_tail_fraction(block_size=bs, overlap_pct=0.0)
            assert not result.empty
            assert "nms_method" in result.columns
            assert "frac_upper_tail" in result.columns
            assert "flagged_outlier" in result.columns
            assert (result["frac_upper_tail"] >= 0).all()
            assert (result["frac_upper_tail"] <= 1.0 + 1e-9).all()
        finally:
            db_mod.DB_PATH = orig
