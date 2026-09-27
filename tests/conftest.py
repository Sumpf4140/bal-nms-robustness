"""Shared fixtures and synthetic data for the test suite."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import LABELS, TILE_WIDTH_PX, TILE_HEIGHT_PX


# ── Synthetic detection DataFrames ───────────────────────────────────────────

def make_detection_df(
    n: int = 20,
    n_tiles_x: int = 3,
    n_tiles_y: int = 3,
    seed: int = 0,
    labels: list[str] | None = None,
    img_w: int = 1920,
    img_h: int = 1920,
) -> pd.DataFrame:
    """Generate a synthetic detection DataFrame matching the expected schema."""
    rng = np.random.default_rng(seed)
    if labels is None:
        labels = LABELS

    tile_xs = rng.integers(0, n_tiles_x, size=n) * TILE_WIDTH_PX
    tile_ys = rng.integers(0, n_tiles_y, size=n) * TILE_HEIGHT_PX

    # Bounding boxes within each tile
    w = rng.integers(20, 80, size=n)
    h = rng.integers(20, 80, size=n)
    local_cx = rng.integers(40, TILE_WIDTH_PX - 40, size=n)
    local_cy = rng.integers(40, TILE_HEIGHT_PX - 40, size=n)
    x1 = tile_xs + local_cx - w // 2
    y1 = tile_ys + local_cy - h // 2
    x2 = x1 + w
    y2 = y1 + h

    # Clip to image bounds
    x1 = np.clip(x1, 0, img_w - 1)
    x2 = np.clip(x2, x1 + 1, img_w)
    y1 = np.clip(y1, 0, img_h - 1)
    y2 = np.clip(y2, y1 + 1, img_h)

    chosen_labels = rng.choice(labels, size=n)
    confidence = rng.uniform(0.44, 1.0, size=n)

    return pd.DataFrame({
        "slide_id": "test_slide",
        "overlap_pct": 0.0,
        "label": chosen_labels,
        "confidence": confidence,
        "x1": x1.astype(int),
        "y1": y1.astype(int),
        "x2": x2.astype(int),
        "y2": y2.astype(int),
        "cx": ((x1 + x2) / 2.0),
        "cy": ((y1 + y2) / 2.0),
        "x_correct": tile_xs.astype(int),
        "y_correct": tile_ys.astype(int),
        "tile_x": (tile_xs // TILE_WIDTH_PX).astype(int),
        "tile_y": (tile_ys // TILE_HEIGHT_PX).astype(int),
    })


def make_tile_counts_df(
    slide_id: str = "s1",
    n_tiles_x: int = 3,
    n_tiles_y: int = 3,
    n_methods: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic tile_counts for block aggregation tests."""
    rng = np.random.default_rng(seed)
    methods = [f"method_{i}" for i in range(n_methods)]
    rows = []
    for method in methods:
        for tx in range(n_tiles_x):
            for ty in range(n_tiles_y):
                for lbl in LABELS:
                    rows.append({
                        "slide_id": slide_id,
                        "overlap_pct": 0.0,
                        "nms_method": method,
                        "tile_x": tx,
                        "tile_y": ty,
                        "label": lbl,
                        "count": int(rng.integers(50, 300)),
                    })
    return pd.DataFrame(rows)


def make_ilr_panel(
    n_methods: int = 4,
    n_blocks: int = 10,
    d1: int = 3,
    noise: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Random (M, N, D-1) ILR panel for consensus / α tests."""
    rng = np.random.default_rng(seed)
    # Shared "true" composition + per-method noise
    true_ilr = rng.standard_normal((n_blocks, d1))
    panel = true_ilr[None, :, :] + rng.standard_normal((n_methods, n_blocks, d1)) * noise
    return panel


# ── Temp DB fixture ───────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Provide a temporary DuckDB database, monkeypatched into core.db."""
    db_file = tmp_path / "test.duckdb"
    import core.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", db_file)
    from core.db import init_db
    init_db(db_path=db_file)
    return db_file


@pytest.fixture()
def detection_df():
    return make_detection_df(n=50)


@pytest.fixture()
def tile_counts_df():
    return make_tile_counts_df()
