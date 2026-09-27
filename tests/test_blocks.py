"""Correctness tests for core/blocks.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import BLOCK_MIN_CELLS, LABELS
from core.blocks import (
    block_indices,
    aggregate_to_blocks,
    aggregate_all_block_sizes,
    filter_by_min_cells,
    ALL_BLOCK_SIZES,
)
from tests.conftest import make_tile_counts_df


# ── block_indices ─────────────────────────────────────────────────────────────

class TestBlockIndices:
    def test_full_slide_all_zero(self):
        tx = np.array([0, 1, 2, 3, 10])
        ty = np.array([0, 2, 1, 4, 7])
        bx, by = block_indices(tx, ty, block_size=0)
        np.testing.assert_array_equal(bx, 0)
        np.testing.assert_array_equal(by, 0)

    def test_block_size_1_equals_tile(self):
        tx = np.array([0, 1, 2, 3])
        ty = np.array([0, 1, 2, 3])
        bx, by = block_indices(tx, ty, block_size=1)
        np.testing.assert_array_equal(bx, tx)
        np.testing.assert_array_equal(by, ty)

    def test_block_size_2_groups_correctly(self):
        tx = np.array([0, 1, 2, 3, 4, 5])
        ty = np.array([0, 0, 0, 0, 0, 0])
        bx, by = block_indices(tx, ty, block_size=2)
        expected_bx = np.array([0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(bx, expected_bx)
        np.testing.assert_array_equal(by, 0)

    def test_block_size_3_groups_correctly(self):
        tx = np.arange(9)
        ty = np.zeros(9, dtype=int)
        bx, _ = block_indices(tx, ty, block_size=3)
        expected = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(bx, expected)

    def test_2d_block_mapping(self):
        tx = np.array([0, 1, 2, 3])
        ty = np.array([0, 1, 2, 3])
        bx, by = block_indices(tx, ty, block_size=2)
        np.testing.assert_array_equal(bx, [0, 0, 1, 1])
        np.testing.assert_array_equal(by, [0, 0, 1, 1])

    def test_input_types(self):
        """Works with Python lists and different int types."""
        tx = [0, 1, 2]
        ty = [0, 0, 0]
        bx, by = block_indices(tx, ty, block_size=2)
        assert bx.dtype in (np.int32, np.int64, np.intp)


# ── aggregate_to_blocks ───────────────────────────────────────────────────────

class TestAggregateToBlocks:
    def setup_method(self):
        self.tc = make_tile_counts_df(n_tiles_x=4, n_tiles_y=4, n_methods=2)

    def test_output_columns_present(self):
        result = aggregate_to_blocks(self.tc, block_size=2)
        for col in ["slide_id", "overlap_pct", "nms_method", "block_size",
                    "block_x", "block_y", "label", "count", "n_block_total"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_block_size_stored_correctly(self):
        for bs in [1, 2, 3, 0]:
            result = aggregate_to_blocks(self.tc, block_size=bs)
            assert (result["block_size"] == bs).all()

    def test_full_slide_one_block(self):
        """block_size=0 → all tiles in one block (0, 0)."""
        result = aggregate_to_blocks(self.tc, block_size=0)
        assert (result["block_x"] == 0).all()
        assert (result["block_y"] == 0).all()

    def test_sum_preserved_per_method_label(self):
        """Sum of tile counts equals sum of block counts for each (method, label)."""
        for method in self.tc["nms_method"].unique():
            for lbl in LABELS:
                tile_sum = self.tc.loc[
                    (self.tc["nms_method"] == method) & (self.tc["label"] == lbl), "count"
                ].sum()
                block_result = aggregate_to_blocks(self.tc, block_size=2)
                block_sum = block_result.loc[
                    (block_result["nms_method"] == method) & (block_result["label"] == lbl), "count"
                ].sum()
                assert tile_sum == block_sum, \
                    f"Sum mismatch for method={method} label={lbl}: {tile_sum} vs {block_sum}"

    def test_n_block_total_correct(self):
        """n_block_total = sum of all label counts in that block."""
        result = aggregate_to_blocks(self.tc, block_size=2)
        for _, grp in result.groupby(
            ["slide_id", "overlap_pct", "nms_method", "block_size", "block_x", "block_y"]
        ):
            assert grp["n_block_total"].nunique() == 1, "n_block_total should be constant per block"
            assert grp["count"].sum() == grp["n_block_total"].iloc[0], \
                "n_block_total must equal sum of label counts"

    def test_block_size_1_matches_tile_level(self):
        """block_size=1 should produce same totals as the input tile counts."""
        result = aggregate_to_blocks(self.tc, block_size=1)
        # Each tile becomes one block
        for method in self.tc["nms_method"].unique():
            for tx in self.tc["tile_x"].unique():
                for ty in self.tc["tile_y"].unique():
                    for lbl in LABELS:
                        tile_count = self.tc.loc[
                            (self.tc["nms_method"] == method) &
                            (self.tc["tile_x"] == tx) & (self.tc["tile_y"] == ty) &
                            (self.tc["label"] == lbl), "count"
                        ].sum()
                        block_count = result.loc[
                            (result["nms_method"] == method) &
                            (result["block_x"] == tx) & (result["block_y"] == ty) &
                            (result["label"] == lbl), "count"
                        ].sum()
                        assert tile_count == block_count


class TestAggregateAllBlockSizes:
    def test_all_block_sizes_present(self):
        tc = make_tile_counts_df(n_tiles_x=3, n_tiles_y=3, n_methods=2)
        result = aggregate_all_block_sizes(tc)
        present = set(result["block_size"].unique())
        expected = set(ALL_BLOCK_SIZES)
        assert present == expected, f"Missing block sizes: {expected - present}"


# ── filter_by_min_cells ───────────────────────────────────────────────────────

class TestFilterByMinCells:
    def test_all_blocks_above_threshold_kept(self):
        tc = make_tile_counts_df(n_tiles_x=3, n_tiles_y=3, n_methods=2, seed=10)
        bc = aggregate_to_blocks(tc, block_size=1)
        # All blocks have n_block_total from random counts 50-300 → sum per block ≥ 200
        # Set threshold to 0 → keep all
        result = filter_by_min_cells(bc, min_cells=0)
        assert len(result) == len(bc)

    def test_blocks_below_threshold_dropped(self):
        """With a very high min_cells threshold, all blocks are dropped."""
        tc = make_tile_counts_df(n_tiles_x=2, n_tiles_y=2, n_methods=1, seed=5)
        bc = aggregate_to_blocks(tc, block_size=1)
        result = filter_by_min_cells(bc, min_cells=10_000_000)
        assert len(result) == 0

    def test_full_slide_block_always_above_threshold(self):
        """Full-slide block aggregates all tiles → n_block_total is always large."""
        tc = make_tile_counts_df(n_tiles_x=5, n_tiles_y=5, n_methods=3, seed=0)
        bc = aggregate_to_blocks(tc, block_size=0)
        result = filter_by_min_cells(bc, min_cells=BLOCK_MIN_CELLS)
        assert len(result) == len(bc), "Full-slide blocks should always pass the filter"

    def test_retained_fraction_monotone_in_threshold(self):
        """Higher threshold → fewer blocks retained."""
        tc = make_tile_counts_df(n_tiles_x=4, n_tiles_y=4, n_methods=2, seed=3)
        bc = aggregate_to_blocks(tc, block_size=1)
        n_prev = len(bc)
        for threshold in [0, 100, 500, 1000, 10_000]:
            result = filter_by_min_cells(bc, min_cells=threshold)
            assert len(result) <= n_prev, \
                f"Retained blocks not monotonically non-increasing with threshold"
            n_prev = len(result)
