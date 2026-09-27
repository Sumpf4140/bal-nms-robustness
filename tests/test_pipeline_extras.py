"""Tests for controlled timing, method-similarity, and multiplicity correction."""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from config import LABELS, TILE_WIDTH_PX, TILE_HEIGHT_PX
from core.db import init_db, connect
from tests.conftest import make_detection_df

_UTC = datetime.timezone.utc
_METHODS = ["none", "iou_grid_n1", "nn_dist_global", "nn_center_global"]


def _seed_cellcounts(db, shift_method=None, shift=0.0, n_slides=10):
    init_db(db_path=db)
    rng = np.random.default_rng(0)
    rows = []
    for s in range(n_slides):
        base = rng.dirichlet([6, 6, 2, 1])
        for m in _METHODS:
            comp = base.copy()
            if m == shift_method:
                comp = np.clip(comp + np.array([shift, -shift, 0.0, 0.0]), 1e-6, None)
                comp = comp / comp.sum()
            for li, lbl in enumerate(LABELS):
                rows.append((f"s{s}", 0.0, m, lbl, int(round(comp[li] * 1000)), float(comp[li])))
    with connect(db_path=db) as con:
        con.executemany("INSERT OR REPLACE INTO cell_counts VALUES (?,?,?,?,?,?)", rows)


def _seed_raw(db, slide="r1", overlaps=(0.0, 0.025), n=4):
    init_db(db_path=db)
    cols = ["slide_id", "overlap_pct", "label", "confidence", "x1", "y1", "x2", "y2",
            "cx", "cy", "x_correct", "y_correct", "tile_x", "tile_y"]
    with connect(db_path=db) as con:
        con.execute("INSERT OR REPLACE INTO slides VALUES (?,?,?,?,?,?,?)",
                    [slide, "/f.png", n * TILE_WIDTH_PX, n * TILE_HEIGHT_PX, n, n,
                     datetime.datetime.now(_UTC)])
        for ov in overlaps:
            df = make_detection_df(n=200, n_tiles_x=n, n_tiles_y=n, seed=int(ov * 1000))
            df["slide_id"] = slide
            df["overlap_pct"] = float(ov)
            con.register("_r", df[cols])
            con.execute("INSERT INTO raw_detections SELECT * FROM _r")
            con.unregister("_r")


class TestTiming:
    def test_time_methods_pure(self):
        from nms.timing import time_methods
        df = make_detection_df(n=60)
        out = time_methods(df, ["none", "iou_grid_n1", "nn_cluster_global"],
                           repeats=2, warmup=True)
        assert set(out) == {"none", "iou_grid_n1", "nn_cluster_global"}
        for reps in out.values():
            assert len(reps) == 2
            assert all(cpu >= 0 and wall >= 0 for cpu, wall in reps)

    def test_run_timing_writes_clean(self, tmp_path, monkeypatch):
        db = tmp_path / "t.duckdb"
        _seed_raw(db)
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)
        from nms.timing import run_timing, timing_summary
        n = run_timing(n_slides=1, repeats=2, warmup=False)
        assert n == 1 * 2 * 31 * 2          # slides × overlaps × methods × repeats
        summary = timing_summary()
        assert len(summary) == 31
        assert {"cpu", "wall"} <= set(summary.columns)


class TestMethodSimilarity:
    def test_identical_methods_zero_distance(self, tmp_path, monkeypatch):
        db = tmp_path / "ms.duckdb"
        _seed_cellcounts(db)  # all methods identical per slide
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)
        from nms.method_similarity import method_distance_matrix
        dm = method_distance_matrix(0.0)
        assert set(dm.index) == set(_METHODS)
        assert np.nanmax(dm.to_numpy()) < 1e-9

    def test_shifted_method_is_farther(self, tmp_path, monkeypatch):
        db = tmp_path / "ms2.duckdb"
        _seed_cellcounts(db, shift_method="iou_grid_n1", shift=0.4)
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)
        from nms.method_similarity import method_distance_matrix
        dm = method_distance_matrix(0.0)
        assert dm.loc["none", "iou_grid_n1"] > dm.loc["none", "nn_dist_global"]


class TestMultiplicity:
    def test_friedman_has_holm_column(self, tmp_path, monkeypatch):
        db = tmp_path / "mc.duckdb"
        _seed_cellcounts(db, shift_method="iou_grid_n1", shift=0.3)
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)
        from nms.baseline_comparison import friedman_across_methods
        fr = friedman_across_methods(0.0)
        assert "p_holm" in fr.columns
        assert (fr["p_holm"] >= fr["p_value"] - 1e-12).all()


def _deviation_df(n_blocks, methods, outlier=None, seed=0):
    """Synthetic method_deviation frame; `outlier` method sits far in every block."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        for m in methods:
            d = abs(rng.standard_normal()) * 0.1
            if m == outlier:
                d += 10.0
            rows.append(("s0", 0.0, 1, b, 0, m, d))
    return pd.DataFrame(rows, columns=[
        "slide_id", "overlap_pct", "block_size", "block_x", "block_y",
        "nms_method", "mahalanobis_distance"])


class TestOutlierCalibration:
    def test_flags_true_outlier(self):
        from nms.outlier_detection import compute_upper_tail_flags
        df = _deviation_df(150, _METHODS, outlier="none", seed=1)
        res = compute_upper_tail_flags(df, n_perm=500)
        flagged = set(res.loc[res["flagged_outlier"], "nms_method"])
        assert flagged == {"none"}, f"Expected only 'none' flagged, got {flagged}"
        assert "p_value" in res.columns and "expected_frac" in res.columns

    def test_fwer_controlled_under_exchangeability(self):
        """Family-wise false-positive rate ≈ α over replicates (calibration).

        A single calibrated 5%-FWER test false-positives ~5% of the time, so we
        check the *rate* over independent exchangeable datasets is bounded — this
        passes the permutation test and would fail the old anti-conservative
        > 5% / binomial rule (whose FWER is far above α).
        """
        from nms.outlier_detection import compute_upper_tail_flags
        n_rep, any_flag = 30, 0
        for r in range(n_rep):
            df = _deviation_df(120, _METHODS, outlier=None, seed=100 + r)
            res = compute_upper_tail_flags(df, n_perm=200, seed=r)
            any_flag += int(res["flagged_outlier"].any())
        # True FWER ≈ 0.05 → ~1-2 of 30; allow generous slack, still ≪ a broken rule.
        assert any_flag <= 6, f"FWER too high: {any_flag}/{n_rep} replicates flagged"


class TestBlockFilter:
    def _seed_blocks(self, db):
        """Two full-slide positions: (0,0) dense under `none`, (0,1) sparse."""
        init_db(db_path=db)
        methods = sorted(__import__("core.nms", fromlist=["NMS_REGISTRY"]).NMS_REGISTRY)
        rows = []
        for bx, none_tot in [(0, 600), (1, 142)]:        # (1,*) sparse even for none
            for m in methods:
                tot = none_tot if m in ("none", "iou_grid_n1") else none_tot // 2
                comp = [int(tot * 0.7), int(tot * 0.2), int(tot * 0.08), tot - int(tot*0.7) - int(tot*0.2) - int(tot*0.08)]
                for lbl, c in zip(LABELS, comp):
                    rows.append(("s0", 0.0, m, 0, bx, 0, lbl, c, tot))
        with connect(db_path=db) as con:
            con.register("_b", pd.DataFrame(rows, columns=[
                "slide_id", "overlap_pct", "nms_method", "block_size",
                "block_x", "block_y", "label", "count", "n_block_total"]))
            con.execute("INSERT OR REPLACE INTO block_counts SELECT * FROM _b")
            con.unregister("_b")

    def test_filter_is_method_independent_and_balanced(self, tmp_path, monkeypatch):
        import numpy as np
        db = tmp_path / "bf.duckdb"
        self._seed_blocks(db)
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)
        from nms.consensus_analysis import filter_blocks, _blocks_to_ilr_panel
        fb = filter_blocks(0, 0.0)
        # Only the dense position survives, regardless of method aggressiveness.
        positions = set(map(tuple, fb[["block_x", "block_y"]].drop_duplicates().to_numpy()))
        assert positions == {(0, 0)}
        # Balanced panel: every method present at the retained position (no NaN).
        _, panel, _ = _blocks_to_ilr_panel(fb)
        present = (~np.isnan(panel).any(axis=2)).sum(axis=0)
        assert present.tolist() == [31]


class TestZeroScaleSensitivity:
    def _block_df(self):
        # Same composition, 3x scale, Eosinophils zero in both → only the
        # zero balance should differ under as-is, and vanish under common-scale.
        rows = []
        counts = {"none": [300, 60, 15, 0], "nn_dist_global": [100, 20, 5, 0]}
        for method, c in counts.items():
            for lbl, ct in zip(LABELS, c):
                rows.append(("s0", 0.0, 1, 0, 0, method, lbl, ct))
        return pd.DataFrame(rows, columns=[
            "slide_id", "overlap_pct", "block_size", "block_x", "block_y",
            "nms_method", "label", "count"])

    def test_common_scale_removes_zero_balance_gap(self):
        from nms.consensus_analysis import _blocks_to_ilr_panel
        bdf = self._block_df()
        _, panel_asis, methods = _blocks_to_ilr_panel(bdf)
        _, panel_cs, _ = _blocks_to_ilr_panel(bdf, common_scale=True)
        i_none = methods.index("none")
        i_nn = methods.index("nn_dist_global")
        # Eosinophil balance is the last ILR coordinate.
        gap_asis = abs(panel_asis[i_none, 0, -1] - panel_asis[i_nn, 0, -1])
        gap_cs = abs(panel_cs[i_none, 0, -1] - panel_cs[i_nn, 0, -1])
        assert gap_asis > 0.5, f"Expected a real zero-scale gap, got {gap_asis:.3f}"
        assert gap_cs < 1e-9, f"Common-scale should remove the gap, got {gap_cs:.3e}"
        # Non-zero balances (Mac/Lym/Neu) already agree under as-is.
        assert abs(panel_asis[i_none, 0, 0] - panel_asis[i_nn, 0, 0]) < 1e-9

    def test_alpha_zero_sensitivity_keys_on_empty(self):
        """Graceful structure when no blocks pass the filter."""
        from nms.zero_sensitivity import alpha_zero_sensitivity
        db = None
        import core.db as dbm
        # Point at an initialised but empty DB so filter_blocks returns empty.
        import tempfile, pathlib
        d = pathlib.Path(tempfile.mkdtemp()) / "empty.duckdb"
        init_db(db_path=d)
        orig = dbm.DB_PATH
        dbm.DB_PATH = d
        try:
            out = alpha_zero_sensitivity(1, 0.0)
        finally:
            dbm.DB_PATH = orig
        for k in ("alpha_as_is", "alpha_common_scale", "alpha_no_eos", "gap"):
            assert k in out
