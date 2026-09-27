"""Regression tests for the overlap_pct DOUBLE fix and the run_all engine.

These guard two bugs found in the audit:
  1. overlap_pct stored as FLOAT made `WHERE overlap_pct = 0.025` match 0 rows.
  2. run_all(n_workers>1) wrote (RW) while workers held read-only connections.
"""
from __future__ import annotations

import datetime

import pandas as pd

from config import TILE_WIDTH_PX, TILE_HEIGHT_PX, OVERLAP_PCTS
from core.db import init_db, connect
from tests.conftest import make_detection_df


def _seed(db_path, slide_id: str, overlaps: list[float]) -> None:
    init_db(db_path=db_path)
    with connect(db_path=db_path) as con:
        con.execute(
            "INSERT OR REPLACE INTO slides VALUES (?,?,?,?,?,?,?)",
            [slide_id, "/fake.png", 4 * TILE_WIDTH_PX, 4 * TILE_HEIGHT_PX, 4, 4,
             datetime.datetime.now(datetime.timezone.utc)],
        )
        cols = ["slide_id", "overlap_pct", "label", "confidence", "x1", "y1", "x2", "y2",
                "cx", "cy", "x_correct", "y_correct", "tile_x", "tile_y"]
        for ov in overlaps:
            df = make_detection_df(n=300, n_tiles_x=4, n_tiles_y=4, seed=int(ov * 1000))
            df["slide_id"] = slide_id
            df["overlap_pct"] = float(ov)
            con.register("_r", df[cols])
            con.execute("INSERT INTO raw_detections SELECT * FROM _r")
            con.unregister("_r")


class TestFractionalOverlap:
    def test_equality_filter_matches_fractional_overlap(self, tmp_path):
        db = tmp_path / "f.duckdb"
        _seed(db, "s1", [0.025, 0.15, 0.5])
        with connect(read_only=True, db_path=db) as con:
            dtype = con.execute(
                "SELECT overlap_pct FROM raw_detections LIMIT 1"
            ).fetchdf()["overlap_pct"].dtype
            n = con.execute(
                "SELECT COUNT(*) FROM raw_detections WHERE overlap_pct = ?", [0.025]
            ).fetchone()[0]
        assert "float64" in str(dtype), "overlap_pct must be DOUBLE, not float32"
        assert n > 0, "fractional-overlap equality filter must match rows (DOUBLE storage)"


class TestRunEngine:
    def test_serial_run_completes(self, tmp_path, monkeypatch):
        db = tmp_path / "r.duckdb"
        _seed(db, "s1", [0.0, 0.025])
        import core.db as db_mod
        monkeypatch.setattr(db_mod, "DB_PATH", db)

        from paper1_nms.process import run_all
        run_all(n_workers=1, slide_filter="s1")

        with connect(read_only=True, db_path=db) as con:
            done = con.execute(
                "SELECT COUNT(*) FROM checkpoint WHERE status='done'").fetchone()[0]
            errors = con.execute(
                "SELECT COUNT(*) FROM checkpoint WHERE status='error'").fetchone()[0]
            # The fractional overlap was read+processed end-to-end (proves the
            # DOUBLE fix: run_pair's `WHERE overlap_pct=0.025` matched real rows).
            frac = con.execute(
                "SELECT SUM(count) FROM cell_counts WHERE overlap_pct = ?", [0.025]
            ).fetchone()[0]
        assert errors == 0
        assert done == len(OVERLAP_PCTS) * 31   # run_all covers all config overlaps
        assert frac is not None and frac > 0, "fractional overlap produced no counts"

    def test_parallel_matches_serial(self, tmp_path, monkeypatch):
        import core.db as db_mod
        from paper1_nms.process import run_all

        dbs = tmp_path / "serial.duckdb"
        _seed(dbs, "s1", [0.0, 0.025])
        monkeypatch.setattr(db_mod, "DB_PATH", dbs)
        run_all(n_workers=1, slide_filter="s1")
        with connect(read_only=True, db_path=dbs) as con:
            serial = con.execute(
                "SELECT slide_id, overlap_pct, nms_method, label, count "
                "FROM cell_counts ORDER BY 1,2,3,4").fetchdf()

        dbp = tmp_path / "parallel.duckdb"
        _seed(dbp, "s1", [0.0, 0.025])
        monkeypatch.setattr(db_mod, "DB_PATH", dbp)
        run_all(n_workers=2, slide_filter="s1")
        with connect(read_only=True, db_path=dbp) as con:
            parallel = con.execute(
                "SELECT slide_id, overlap_pct, nms_method, label, count "
                "FROM cell_counts ORDER BY 1,2,3,4").fetchdf()

        pd.testing.assert_frame_equal(serial, parallel)
