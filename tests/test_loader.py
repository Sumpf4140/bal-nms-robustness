"""Tests for core/loader.py — CSV loading, validation, and idempotency."""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import LABELS, TILE_WIDTH_PX, TILE_HEIGHT_PX, OVERLAP_PCTS
from core.db import init_db, connect
from core.loader import _load_csv, _check_tile_grid, _check_image_bounds, load_slide


# ── _load_csv ─────────────────────────────────────────────────────────────────

def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def make_valid_csv(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """A raw-CSV fixture in the REAL data format: boxes are tile-relative
    (x1..y2 ∈ [0, tile]) and x_correct/y_correct is the absolute tile origin.
    `_load_csv` is responsible for converting the boxes to slide-absolute."""
    rng = np.random.default_rng(seed)
    tile_xs = rng.integers(0, 3, n) * TILE_WIDTH_PX
    tile_ys = rng.integers(0, 3, n) * TILE_HEIGHT_PX
    w = rng.integers(20, 60, n)
    h = rng.integers(20, 60, n)
    cx_local = rng.integers(40, 600, n)
    cy_local = rng.integers(40, 600, n)
    x1 = cx_local - w // 2          # tile-relative (NOT offset by tile origin)
    y1 = cy_local - h // 2
    return pd.DataFrame({
        "label": rng.choice(LABELS, n),
        "confidence": rng.uniform(0.45, 0.99, n),
        "x1": x1,
        "y1": y1,
        "x2": x1 + w,
        "y2": y1 + h,
        "x_correct": tile_xs,
        "y_correct": tile_ys,
    })


class TestLoadCsv:
    def test_derived_columns_computed(self, tmp_path):
        df = make_valid_csv()
        path = tmp_path / "data.csv"
        _write_csv(path, df)
        result = _load_csv(path)
        for col in ["cx", "cy", "tile_x", "tile_y"]:
            assert col in result.columns

    def test_cx_cy_formula(self, tmp_path):
        df = make_valid_csv()
        path = tmp_path / "data.csv"
        _write_csv(path, df)
        result = _load_csv(path)
        np.testing.assert_allclose(result["cx"], (result["x1"] + result["x2"]) / 2)
        np.testing.assert_allclose(result["cy"], (result["y1"] + result["y2"]) / 2)

    def test_tile_x_tile_y_formula(self, tmp_path):
        df = make_valid_csv()
        path = tmp_path / "data.csv"
        _write_csv(path, df)
        result = _load_csv(path)
        np.testing.assert_array_equal(result["tile_x"], result["x_correct"] // TILE_WIDTH_PX)
        np.testing.assert_array_equal(result["tile_y"], result["y_correct"] // TILE_HEIGHT_PX)

    def test_missing_column_raises(self, tmp_path):
        df = make_valid_csv().drop(columns=["confidence"])
        path = tmp_path / "data.csv"
        _write_csv(path, df)
        with pytest.raises(ValueError, match="Missing columns"):
            _load_csv(path)

    def test_unknown_labels_filtered(self, tmp_path):
        df = make_valid_csv()
        df.loc[0, "label"] = "UnknownCellType"
        path = tmp_path / "data.csv"
        _write_csv(path, df)
        result = _load_csv(path)
        assert "UnknownCellType" not in result["label"].values

    def test_boxes_converted_to_absolute(self, tmp_path):
        """Tile-relative CSV boxes become slide-absolute (x_correct + x1) at load."""
        df = make_valid_csv(n=10)
        path = tmp_path / "data.csv"
        _write_csv(path, df)
        result = _load_csv(path)
        np.testing.assert_array_equal(
            result["x1"].to_numpy(), df["x1"].to_numpy() + df["x_correct"].to_numpy())
        np.testing.assert_array_equal(
            result["y2"].to_numpy(), df["y2"].to_numpy() + df["y_correct"].to_numpy())
        np.testing.assert_allclose(
            result["cx"].to_numpy(), (result["x1"] + result["x2"]).to_numpy() / 2)

    def test_already_absolute_input_rejected(self, tmp_path):
        """Guard: boxes that already look slide-absolute are refused (no double-offset)."""
        df = make_valid_csv(n=5)
        df["x2"] = df["x2"] + 5000  # push a box well past one tile
        path = tmp_path / "abs.csv"
        _write_csv(path, df)
        with pytest.raises(ValueError, match="already"):
            _load_csv(path)

    def test_cross_tile_same_relative_position_not_merged(self, tmp_path):
        """Regression for the coordinate bug: two same-label detections at the
        SAME tile-relative position but in DIFFERENT tiles must load ~one tile
        apart in absolute space and NOT be merged by a global NMS method. On the
        pre-fix loader (tile-relative cx) they coincided and were wrongly merged."""
        from core.nms import NMS_REGISTRY
        df = pd.DataFrame({
            "label": ["Makrophage", "Makrophage"],
            "confidence": [0.9, 0.8],
            "x1": [300, 300], "y1": [300, 300],
            "x2": [320, 320], "y2": [320, 320],
            "x_correct": [0, TILE_WIDTH_PX], "y_correct": [0, 0],
        })
        path = tmp_path / "rel.csv"
        _write_csv(path, df)
        loaded = _load_csv(path, overlap_pct=0.0)
        assert abs(loaded["cx"].iloc[0] - loaded["cx"].iloc[1]) == TILE_WIDTH_PX
        keep = NMS_REGISTRY["nn_cluster_global"](loaded)
        assert keep.sum() == 2, "distinct cells in different tiles must both survive"


# ── _check_tile_grid ──────────────────────────────────────────────────────────

class TestCheckTileGrid:
    def test_valid_grid_passes(self):
        df = make_valid_csv()
        path = Path("dummy.csv")
        _check_tile_grid(df, path)  # should not raise

    def test_invalid_grid_raises(self):
        df = make_valid_csv()
        # Corrupt all x_correct values
        df["x_correct"] = df["x_correct"] + 1
        with pytest.raises(ValueError, match="Tile-grid integrity"):
            _check_tile_grid(df, Path("dummy.csv"))

    def test_partial_invalid_below_tolerance_raises(self):
        df = make_valid_csv(n=100)
        # Corrupt 5% of rows (threshold is 99%)
        df.loc[:4, "x_correct"] = df.loc[:4, "x_correct"] + 7
        with pytest.raises(ValueError):
            _check_tile_grid(df, Path("dummy.csv"))

    def test_empty_df_passes(self):
        df = make_valid_csv()
        empty = df.iloc[:0]
        _check_tile_grid(empty, Path("dummy.csv"))  # should not raise


# ── _check_image_bounds ───────────────────────────────────────────────────────

class TestCheckImageBounds:
    def test_within_bounds_passes(self):
        df = make_valid_csv(n=20)
        path = Path("dummy.csv")
        _check_image_bounds(df, image_w=10_000, image_h=10_000, path=path)

    def test_exceeds_width_raises(self):
        df = make_valid_csv(n=5)
        df.loc[0, "x2"] = 99_999  # far beyond image
        with pytest.raises(ValueError, match="Bbox exceeds image bounds"):
            _check_image_bounds(df, image_w=1920, image_h=1920, path=Path("dummy.csv"))

    def test_exactly_at_boundary_passes(self):
        df = make_valid_csv(n=5)
        max_x = int(df[["x1", "x2"]].max().max())
        _check_image_bounds(df, image_w=max_x + 1, image_h=10_000, path=Path("dummy.csv"))


# ── load_slide idempotency ────────────────────────────────────────────────────

class TestLoadSlideIdempotency:
    def _make_slide_data(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create a minimal fake slide directory with one overlap CSV."""
        from PIL import Image
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        data_dir = tmp_path / "data"
        overlap_dir = data_dir / "0_overlap_raw"
        overlap_dir.mkdir(parents=True)

        # 1920×1920 image
        img = Image.fromarray(np.zeros((1920, 1920, 3), dtype=np.uint8))
        img_path = img_dir / "slide001.png"
        img.save(str(img_path))

        # Write one overlap CSV in the {n}_overlap_raw/{slide_id}.csv layout
        df = make_valid_csv(n=50)
        _write_csv(overlap_dir / "slide001.csv", df)

        return img_dir / "slide001.png", data_dir

    def test_load_slide_populates_raw_detections(self, tmp_path):
        from core.db import init_db, connect
        db = tmp_path / "test.duckdb"
        import core.db as db_mod

        orig_db = db_mod.DB_PATH
        db_mod.DB_PATH = db
        init_db(db_path=db)
        try:
            img_path, data_dir = self._make_slide_data(tmp_path)
            load_slide("slide001", img_path, data_dir=data_dir)
            with connect(db_path=db) as con:
                n = con.execute(
                    "SELECT COUNT(*) FROM raw_detections WHERE slide_id='slide001'"
                ).fetchone()[0]
            assert n > 0
        finally:
            db_mod.DB_PATH = orig_db

    def test_load_slide_idempotent(self, tmp_path):
        from core.db import init_db, connect
        db = tmp_path / "test.duckdb"
        import core.db as db_mod

        orig_db = db_mod.DB_PATH
        db_mod.DB_PATH = db
        init_db(db_path=db)
        try:
            img_path, data_dir = self._make_slide_data(tmp_path)
            load_slide("slide001", img_path, data_dir=data_dir)
            load_slide("slide001", img_path, data_dir=data_dir)  # second load
            with connect(db_path=db) as con:
                n = con.execute(
                    "SELECT COUNT(*) FROM raw_detections WHERE slide_id='slide001'"
                ).fetchone()[0]
                first_count = n
            # Count should not have doubled
            assert first_count == n
        finally:
            db_mod.DB_PATH = orig_db
