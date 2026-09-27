"""Tests for the explicit no-NMS-vs-NMS baseline thesis (nms.baseline_comparison)."""
from __future__ import annotations

import numpy as np

from config import LABELS
from core.db import init_db, connect

_METHODS = ["none", "iou_grid_n1", "nn_dist_global", "nn_center_global"]


def _seed(db, n_slides=12, shift_method=None, shift=0.0, seed=0):
    """cell_counts where every method equals a slide-specific base composition,
    optionally shifting one method (mass moved Lymphozyt→Makrophage)."""
    init_db(db_path=db)
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_slides):
        base = rng.dirichlet([6, 6, 2, 1])  # Mak, Lym, Neu, Eos
        for m in _METHODS:
            comp = base.copy()
            if m == shift_method:
                comp = comp + np.array([shift, -shift, 0.0, 0.0])
                comp = np.clip(comp, 1e-6, None)
                comp = comp / comp.sum()
            total = 1000
            for li, lbl in enumerate(LABELS):
                cnt = int(round(comp[li] * total))
                rows.append((f"s{s}", 0.0, m, lbl, cnt, float(comp[li])))
    with connect(db_path=db) as con:
        con.executemany("INSERT OR REPLACE INTO cell_counts VALUES (?,?,?,?,?,?)", rows)


class TestBaselineEquivalence:
    def test_all_equivalent_when_methods_identical(self, tmp_path, monkeypatch):
        db = tmp_path / "b.duckdb"
        _seed(db)  # no shift → every method == none per slide
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)

        from nms.baseline_comparison import equivalence_vs_none, friedman_across_methods
        eq = equivalence_vs_none(0.0, margin=0.2)
        assert not eq.empty
        assert eq["equivalent"].all(), "identical methods must all be equivalent to none"

        fr = friedman_across_methods(0.0)
        assert not fr.empty
        assert (fr["p_value"] > 0.05).all(), "no method effect → Friedman non-significant"

    def test_shifted_method_flagged_not_equivalent(self, tmp_path, monkeypatch):
        db = tmp_path / "b2.duckdb"
        _seed(db, shift_method="iou_grid_n1", shift=0.5)  # large, consistent shift
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)

        from nms.baseline_comparison import equivalence_vs_none, friedman_across_methods
        eq = equivalence_vs_none(0.0, margin=0.2)
        # Makrophage moved for iou_per_tile → that method×label is NOT equivalent
        row = eq[(eq["nms_method"] == "iou_grid_n1") & (eq["label"] == "Makrophage")]
        assert len(row) == 1 and not bool(row["equivalent"].iloc[0])
        # an unshifted method stays equivalent
        nn = eq[(eq["nms_method"] == "nn_dist_global") & (eq["label"] == "Makrophage")]
        assert bool(nn["equivalent"].iloc[0])

        fr = friedman_across_methods(0.0)
        assert (fr["p_value"] < 0.05).any(), "a real method shift → Friedman significant"


class TestReferenceVerdict:
    def test_reference_equivalent_when_identical(self, tmp_path, monkeypatch):
        db = tmp_path / "r.duckdb"
        _seed(db)  # every method == none
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)

        from nms.baseline_comparison import baseline_summary
        res = baseline_summary(0.0, reference_method="iou_grid_n1")
        assert res["reference_equivalent"] is True
        assert not res["reference_equivalence"].empty
        assert (res["reference_equivalence"]["nms_method"] == "iou_grid_n1").all()

    def test_reference_verdict_isolated_from_other_methods(self, tmp_path, monkeypatch):
        """Shifting iou_per_tile breaks ITS verdict; choosing an unshifted reference
        (nn_dist_global) still passes — the focused verdict ignores other methods."""
        db = tmp_path / "r2.duckdb"
        _seed(db, shift_method="iou_grid_n1", shift=0.5)
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)

        from nms.baseline_comparison import baseline_summary
        res_iou = baseline_summary(0.0, reference_method="iou_grid_n1")
        assert res_iou["reference_equivalent"] is False
        bad = res_iou["reference_equivalence"]
        assert not bool(bad[bad["label"] == "Makrophage"]["equivalent"].iloc[0])

        res_nn = baseline_summary(0.0, reference_method="nn_dist_global")
        assert res_nn["reference_equivalent"] is True

    def test_unknown_reference_is_false_and_empty(self, tmp_path, monkeypatch):
        db = tmp_path / "r3.duckdb"
        _seed(db)
        import core.db as dbm
        monkeypatch.setattr(dbm, "DB_PATH", db)

        from nms.baseline_comparison import baseline_summary
        res = baseline_summary(0.0, reference_method="does_not_exist")
        assert res["reference_equivalent"] is False
        assert res["reference_equivalence"].empty
