"""Block-grid family for spatial aggregation.

A block of size n×n aggregates detections from n² adjacent tiles.
Block index (block_x, block_y) for tile (tx, ty) at size n: (tx//n, ty//n).
Full-slide block: block_size=0, (block_x, block_y) = (0, 0).

All block_size values used in the analysis:
  BLOCK_SIZES = [1, 2, 3, 4, 5]  →  stored as-is
  full-slide  → block_size = 0
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import BLOCK_SIZES, BLOCK_MIN_CELLS, INCLUDE_FULL_SLIDE, LABELS

logger = logging.getLogger(__name__)

# Canonical list of block_size values written to DB (0 = full-slide)
ALL_BLOCK_SIZES: list[int] = BLOCK_SIZES + ([0] if INCLUDE_FULL_SLIDE else [])


def block_indices(
    tile_x: np.ndarray,
    tile_y: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map tile coordinates to block coordinates.

    block_size = 0 → full-slide (all tiles mapped to block (0, 0)).
    """
    tile_x = np.asarray(tile_x)
    tile_y = np.asarray(tile_y)
    if block_size == 0:
        return np.zeros_like(tile_x), np.zeros_like(tile_y)
    return tile_x // block_size, tile_y // block_size


def aggregate_to_blocks(tile_counts_df: pd.DataFrame, block_size: int) -> pd.DataFrame:
    """Sum tile_counts into block_counts at the requested block_size.

    tile_counts_df must have columns:
        slide_id, overlap_pct, nms_method, tile_x, tile_y, label, count

    Returns DataFrame with:
        slide_id, overlap_pct, nms_method, block_size, block_x, block_y,
        label, count, n_block_total
    """
    df = tile_counts_df.copy()
    bx, by = block_indices(df["tile_x"].to_numpy(), df["tile_y"].to_numpy(), block_size)
    df["block_x"] = bx
    df["block_y"] = by
    df["block_size"] = block_size

    group_keys = ["slide_id", "overlap_pct", "nms_method", "block_size", "block_x", "block_y", "label"]
    agg = (
        df.groupby(group_keys, sort=False)["count"]
        .sum()
        .reset_index()
    )

    # n_block_total = total cells in block across all labels
    tot_keys = ["slide_id", "overlap_pct", "nms_method", "block_size", "block_x", "block_y"]
    totals = (
        agg.groupby(tot_keys, sort=False)["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "n_block_total"})
    )
    return agg.merge(totals, on=tot_keys)


def aggregate_all_block_sizes(tile_counts_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tile_counts to all block sizes (BLOCK_SIZES + full-slide).

    Returns concatenated DataFrame for all block sizes.
    """
    parts = [aggregate_to_blocks(tile_counts_df, bs) for bs in ALL_BLOCK_SIZES]
    return pd.concat(parts, ignore_index=True)


def filter_by_min_cells(
    block_counts: pd.DataFrame,
    min_cells: int = BLOCK_MIN_CELLS,
) -> pd.DataFrame:
    """Drop blocks where n_block_total < min_cells.

    Logs retained-block fraction per (block_size, slide_id).
    """
    before = len(block_counts[["slide_id", "overlap_pct", "nms_method",
                                "block_size", "block_x", "block_y"]].drop_duplicates())
    kept = block_counts[block_counts["n_block_total"] >= min_cells].copy()
    after = len(kept[["slide_id", "overlap_pct", "nms_method",
                       "block_size", "block_x", "block_y"]].drop_duplicates())
    frac = after / before if before > 0 else 0.0
    logger.info(
        "filter_by_min_cells: retained %d / %d unique blocks (%.1f%%)",
        after, before, 100 * frac,
    )
    return kept


def pivot_block_counts_to_matrix(
    block_counts: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Pivot block_counts to a composition matrix.

    Returns:
        meta_df  — one row per (slide_id, overlap_pct, nms_method, block_size, block_x, block_y)
        counts   — (N_blocks, 4) count matrix in LABELS order
        labels   — column names (= LABELS)
    """
    # Ensure all labels are present for every block (fill missing with 0)
    key_cols = ["slide_id", "overlap_pct", "nms_method", "block_size", "block_x", "block_y"]
    pivot = (
        block_counts.pivot_table(
            index=key_cols,
            columns="label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=LABELS, fill_value=0)
        .reset_index()
    )
    meta_df = pivot[key_cols + ["n_block_total"]].copy() if "n_block_total" in pivot.columns else pivot[key_cols].copy()
    counts = pivot[LABELS].to_numpy(dtype=float)
    return pivot, counts, LABELS
