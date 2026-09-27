"""Correctness tests for core/nms.py — all 31 NMS methods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import (
    LABELS, IOU_THRESH, NN_DIST_PX, DBSCAN_EPS_PX,
    TILE_WIDTH_PX, TILE_HEIGHT_PX, EDGE_MARGIN_PX,
)
from core.nms import (
    NMS_REGISTRY,
    pairwise_iou,
    greedy_iou_suppress,
    _nn_dist_suppress_core,
    _compute_center_radii,
    _nn_center_suppress_core,
    _nn_cluster_suppress_core,
)
from tests.conftest import make_detection_df


# ── Registry sanity ───────────────────────────────────────────────────────────

def test_registry_count():
    assert len(NMS_REGISTRY) == 31, f"Expected 31, got {len(NMS_REGISTRY)}"


def test_registry_contains_expected_names():
    expected_prefixes = {"none", "iou_", "nn_dist_", "nn_center_", "nn_cluster_"}
    for name in NMS_REGISTRY:
        assert any(name == "none" or name.startswith(p) for p in expected_prefixes), \
            f"Unexpected method name: {name}"


def test_all_methods_return_bool_mask_correct_length():
    df = make_detection_df(n=30)
    for name, fn in NMS_REGISTRY.items():
        mask = fn(df)
        assert mask.dtype == bool, f"{name}: mask dtype={mask.dtype}, expected bool"
        assert len(mask) == len(df), f"{name}: mask length={len(mask)}, expected {len(df)}"


def test_all_methods_empty_df():
    """All methods must handle empty input without error."""
    df = make_detection_df(n=0)
    for name, fn in NMS_REGISTRY.items():
        mask = fn(df)
        assert len(mask) == 0, f"{name}: non-empty mask for empty input"


def test_all_methods_single_detection():
    """All methods must keep the single detection."""
    df = make_detection_df(n=1)
    for name, fn in NMS_REGISTRY.items():
        mask = fn(df)
        assert mask.all(), f"{name}: single detection suppressed"


def test_none_is_passthrough():
    """none must return all True."""
    df = make_detection_df(n=40)
    mask = NMS_REGISTRY["none"](df)
    assert mask.all(), "none should return all True"


def test_none_highest_count():
    """none always produces the most (or equal) survivors compared to any other method."""
    df = make_detection_df(n=80, seed=7)
    n_none = NMS_REGISTRY["none"](df).sum()
    for name, fn in NMS_REGISTRY.items():
        if name == "none":
            continue
        n_method = fn(df).sum()
        assert n_method <= n_none, \
            f"{name} produced {n_method} survivors > {n_none} (none)"


# ── pairwise_iou ──────────────────────────────────────────────────────────────

def test_pairwise_iou_identical_boxes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    iou = pairwise_iou(boxes)
    assert iou.shape == (2, 2)
    np.testing.assert_allclose(iou[0, 1], 1.0, atol=1e-6)
    np.testing.assert_allclose(iou[1, 0], 1.0, atol=1e-6)


def test_pairwise_iou_non_overlapping():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    iou = pairwise_iou(boxes)
    np.testing.assert_allclose(iou[0, 1], 0.0, atol=1e-6)


def test_pairwise_iou_partial_overlap():
    # Two 10×10 boxes offset by 5 in x → 5×10 overlap = 50, union = 150
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=float)
    iou = pairwise_iou(boxes)
    expected = 50.0 / 150.0
    np.testing.assert_allclose(iou[0, 1], expected, atol=1e-6)


def test_pairwise_iou_symmetry():
    rng = np.random.default_rng(0)
    boxes = rng.integers(0, 100, (8, 4)).astype(float)
    boxes[:, 2] = boxes[:, 0] + rng.integers(5, 30, 8)
    boxes[:, 3] = boxes[:, 1] + rng.integers(5, 30, 8)
    iou = pairwise_iou(boxes)
    np.testing.assert_allclose(iou, iou.T, atol=1e-8)


def test_pairwise_iou_diagonal_is_one():
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    iou = pairwise_iou(boxes)
    np.testing.assert_allclose(np.diag(iou), [1.0, 1.0], atol=1e-6)


# ── greedy_iou_suppress ───────────────────────────────────────────────────────

def test_greedy_iou_suppress_two_identical_boxes():
    """Two identical boxes at the same location — lower confidence suppressed."""
    df = pd.DataFrame({
        "confidence": [0.9, 0.6],
        "x1": [0, 0], "y1": [0, 0], "x2": [10, 10], "y2": [10, 10],
        "cx": [5., 5.], "cy": [5., 5.],
        "label": ["Makrophage"] * 2,
        "tile_x": [0, 0], "tile_y": [0, 0],
    })
    mask = greedy_iou_suppress(df)
    assert mask.sum() == 1
    assert mask[0]   # higher confidence kept
    assert not mask[1]


def test_greedy_iou_suppress_non_overlapping_kept():
    """Non-overlapping boxes must all survive."""
    df = pd.DataFrame({
        "confidence": [0.9, 0.8, 0.7],
        "x1": [0, 20, 40], "y1": [0, 0, 0],
        "x2": [10, 30, 50], "y2": [10, 10, 10],
        "cx": [5., 25., 45.], "cy": [5., 5., 5.],
        "label": ["Makrophage"] * 3,
        "tile_x": [0, 0, 0], "tile_y": [0, 0, 0],
    })
    mask = greedy_iou_suppress(df)
    assert mask.all()


def test_greedy_iou_suppress_chain():
    """Box A overlaps B, B overlaps C, A does not overlap C → A and C kept."""
    # A=[0,0,10,10], B=[5,0,15,10], C=[12,0,22,10]
    # IoU(A,B)=50/150≈0.33, IoU(B,C)=30/170≈0.18, IoU(A,C)=0
    # With IOU_THRESH=0.45: nothing suppressed (IoU < thresh for all pairs)
    df = pd.DataFrame({
        "confidence": [0.9, 0.85, 0.8],
        "x1": [0, 5, 12], "y1": [0, 0, 0],
        "x2": [10, 15, 22], "y2": [10, 10, 10],
        "cx": [5., 10., 17.], "cy": [5., 5., 5.],
        "label": ["Makrophage"] * 3,
        "tile_x": [0, 0, 0], "tile_y": [0, 0, 0],
    })
    mask = greedy_iou_suppress(df, iou_thresh=0.45)
    assert mask.all(), "No pair exceeds 0.45 IoU, all should survive"


def test_greedy_iou_suppress_high_overlap():
    """Boxes with IoU > thresh → only highest confidence kept."""
    # Two 100×100 boxes at [0,0] and [1,1] → IoU very high
    df = pd.DataFrame({
        "confidence": [0.5, 0.9, 0.7],
        "x1": [0, 1, 0], "y1": [0, 1, 0],
        "x2": [100, 101, 100], "y2": [100, 101, 100],
        "cx": [50., 51., 50.], "cy": [50., 51., 50.],
        "label": ["Makrophage"] * 3,
        "tile_x": [0, 0, 0], "tile_y": [0, 0, 0],
    })
    mask = greedy_iou_suppress(df, iou_thresh=0.45)
    # Highest conf = index 1 (0.9); both 0 and 2 overlap strongly with 1
    assert mask[1]
    assert mask.sum() == 1


# ── nn_dist ───────────────────────────────────────────────────────────────────

def test_nn_dist_far_apart_all_survive():
    """Detections > NN_DIST_PX apart all survive."""
    df = pd.DataFrame({
        "label": LABELS[:2] * 5,
        "confidence": np.linspace(0.9, 0.5, 10),
        "cx": np.arange(10) * 100.0,
        "cy": np.zeros(10),
        "x1": np.arange(10) * 100, "y1": np.zeros(10, dtype=int),
        "x2": np.arange(10) * 100 + 10, "y2": np.full(10, 10, dtype=int),
        "tile_x": np.zeros(10, dtype=int),
        "tile_y": np.zeros(10, dtype=int),
    })
    mask = _nn_dist_suppress_core(df, dist_px=NN_DIST_PX)
    assert mask.all()


def test_nn_dist_duplicate_same_label_suppressed():
    """Two same-label detections within dist_px → keep higher confidence."""
    df = pd.DataFrame({
        "label": ["Makrophage", "Makrophage"],
        "confidence": [0.8, 0.6],
        "cx": [100.0, 105.0],   # 5 px apart < NN_DIST_PX=20
        "cy": [100.0, 100.0],
        "x1": [90, 95], "y1": [90, 90], "x2": [110, 115], "y2": [110, 110],
        "tile_x": [0, 0], "tile_y": [0, 0],
    })
    mask = _nn_dist_suppress_core(df, dist_px=20)
    assert mask.sum() == 1
    assert mask[0]


def test_nn_dist_different_labels_both_kept():
    """Two different-label detections within dist_px → both kept."""
    df = pd.DataFrame({
        "label": ["Makrophage", "Lymphozyt"],
        "confidence": [0.8, 0.6],
        "cx": [100.0, 102.0],
        "cy": [100.0, 100.0],
        "x1": [90, 92], "y1": [90, 90], "x2": [110, 112], "y2": [110, 110],
        "tile_x": [0, 0], "tile_y": [0, 0],
    })
    mask = _nn_dist_suppress_core(df, dist_px=20)
    assert mask.all()


# ── nn_center radius computation ──────────────────────────────────────────────

def test_compute_center_radii_returns_all_labels():
    df = make_detection_df(n=100, seed=5)
    radii = _compute_center_radii(df)
    for lbl in LABELS:
        assert lbl in radii
        assert radii[lbl] >= 0.0


def test_compute_center_radii_missing_label_returns_zero():
    df = make_detection_df(n=20, labels=["Makrophage"], seed=3)
    radii = _compute_center_radii(df)
    assert radii["Lymphozyt"] == 0.0
    assert radii["NeutrophilerGranulozyt"] == 0.0
    assert radii["EosinophilerGranulozyt"] == 0.0


def test_compute_center_radii_formula():
    """Radius = 2 * std(bbox_size) per label."""
    # Create detections with known sizes
    sizes = np.array([20.0, 30.0, 40.0, 50.0, 60.0])  # mean bbox_size
    df = pd.DataFrame({
        "label": ["Makrophage"] * 5,
        "confidence": [0.9] * 5,
        "x1": [0] * 5,
        "y1": [0] * 5,
        "x2": sizes.astype(int).tolist(),   # width = sizes
        "y2": sizes.astype(int).tolist(),   # height = sizes → mean(w,h) = sizes
        "cx": (sizes / 2).tolist(),
        "cy": (sizes / 2).tolist(),
        "tile_x": [0] * 5,
        "tile_y": [0] * 5,
    })
    radii = _compute_center_radii(df)
    expected = 2.0 * sizes.std(ddof=1)
    np.testing.assert_allclose(radii["Makrophage"], expected, rtol=1e-6)


def test_nn_center_suppress_within_radius():
    """Two same-label detections within 2·std radius → keep higher confidence."""
    # Force a known radius: 5 detections with sizes [10,20,30,40,50]
    # std = 15.81..., radius = 2 * std ≈ 31.62
    # Place two detections 20px apart (< radius) → one suppressed
    sizes = [10.0, 20.0, 30.0, 40.0, 50.0, 30.0, 30.0]
    df = pd.DataFrame({
        "label": ["Makrophage"] * 7,
        "confidence": [0.9, 0.8, 0.7, 0.6, 0.5, 0.95, 0.4],
        "x1": [0] * 7,
        "y1": [0] * 7,
        "x2": [int(s) for s in sizes],
        "y2": [int(s) for s in sizes],
        "cx": [500.0, 600.0, 700.0, 800.0, 900.0, 510.0, 1000.0],
        "cy": [500.0] * 7,
        "tile_x": [0] * 7,
        "tile_y": [0] * 7,
    })
    label_radius = _compute_center_radii(df)
    mask = _nn_center_suppress_core(df, label_radius)

    # Index 0 and 5 are 10px apart, both Makrophage → index 5 (conf 0.95) wins, index 0 suppressed
    assert mask[5], "Higher confidence in close pair must survive"
    assert not mask[0], "Lower confidence in close pair must be suppressed"


# ── nn_cluster ────────────────────────────────────────────────────────────────

def test_nn_cluster_one_cluster_keeps_highest_conf():
    """All detections of same label within eps → keep highest confidence."""
    df = pd.DataFrame({
        "label": ["Makrophage"] * 4,
        "confidence": [0.3, 0.9, 0.6, 0.7],
        "cx": [10.0, 12.0, 11.0, 9.0],
        "cy": [10.0, 11.0, 10.0, 9.0],
        "x1": [5] * 4, "y1": [5] * 4, "x2": [15] * 4, "y2": [15] * 4,
        "tile_x": [0] * 4, "tile_y": [0] * 4,
    })
    mask = _nn_cluster_suppress_core(df, eps=DBSCAN_EPS_PX)
    assert mask.sum() == 1
    assert mask[1]  # index 1 has highest confidence 0.9


def test_nn_cluster_separate_clusters_each_keeps_best():
    """Two clusters far apart → one survivor per cluster."""
    df = pd.DataFrame({
        "label": ["Makrophage"] * 4,
        "confidence": [0.9, 0.6, 0.5, 0.8],
        "cx": [10.0, 12.0, 100.0, 102.0],
        "cy": [10.0, 11.0, 100.0, 101.0],
        "x1": [5, 7, 95, 97], "y1": [5, 6, 95, 96],
        "x2": [15, 17, 105, 107], "y2": [15, 16, 105, 106],
        "tile_x": [0, 0, 0, 0], "tile_y": [0, 0, 0, 0],
    })
    mask = _nn_cluster_suppress_core(df, eps=20)
    assert mask.sum() == 2
    assert mask[0]  # best in cluster 1 (conf 0.9)
    assert mask[3]  # best in cluster 2 (conf 0.8)


def test_nn_cluster_different_labels_independent():
    """Clusters are computed per label; different labels don't merge."""
    df = pd.DataFrame({
        "label": ["Makrophage", "Lymphozyt"],
        "confidence": [0.8, 0.7],
        "cx": [10.0, 12.0],  # close enough to merge if same label
        "cy": [10.0, 10.0],
        "x1": [5, 7], "y1": [5, 5], "x2": [15, 17], "y2": [15, 15],
        "tile_x": [0, 0], "tile_y": [0, 0],
    })
    mask = _nn_cluster_suppress_core(df, eps=20)
    assert mask.all()  # different labels → both survive


def _cluster_df(cx, cy, conf, label="Makrophage"):
    cx = np.asarray(cx, float)
    return pd.DataFrame({
        "label": [label] * len(cx), "confidence": np.asarray(conf, float),
        "cx": cx, "cy": np.asarray(cy, float),
        "x1": cx - 1, "y1": 0, "x2": cx + 1, "y2": 1, "tile_x": 0, "tile_y": 0,
    })


def test_nn_cluster_chain_is_one_component():
    """A–B–C with A–B ≤ eps and B–C ≤ eps but A–C > eps is ONE cluster.

    The highest-confidence point sits at an end (A), so connected-components keeps
    exactly one survivor — whereas greedy radius suppression (nn_dist) would keep
    both ends (A suppresses B; C is outside A's radius). This locks the clustering
    semantics that nn_cluster must preserve.
    """
    df = _cluster_df(cx=[0.0, 15.0, 30.0], cy=[0.0, 0.0, 0.0], conf=[0.9, 0.5, 0.8])
    mask = _nn_cluster_suppress_core(df, eps=20)
    assert mask.sum() == 1, "chain A-B-C is a single connected component"
    assert mask[0], "the highest-confidence member (A) is kept"


def test_nn_cluster_all_isolated_keeps_all():
    """k ≥ 2 points all > eps apart → empty pair set → every point survives."""
    df = _cluster_df(cx=[0.0, 100.0, 200.0], cy=[0.0, 0.0, 0.0], conf=[0.5, 0.6, 0.7])
    mask = _nn_cluster_suppress_core(df, eps=20)
    assert mask.all()


def test_nn_cluster_matches_dbscan_reference():
    """Byte-identical to the previous DBSCAN(min_samples=1) implementation."""
    from sklearn.cluster import DBSCAN
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(30, 250))
        cx = rng.uniform(0, 180, n)
        cy = rng.uniform(0, 180, n)
        labels = rng.choice(LABELS, n)
        conf = rng.uniform(0.45, 0.99, n)
        df = pd.DataFrame({
            "label": labels, "confidence": conf, "cx": cx, "cy": cy,
            "x1": cx - 1, "y1": cy - 1, "x2": cx + 1, "y2": cy + 1,
            "tile_x": 0, "tile_y": 0,
        })
        new_mask = _nn_cluster_suppress_core(df, eps=DBSCAN_EPS_PX)

        ref = np.zeros(n, dtype=bool)
        for lbl in df["label"].unique():
            idx = np.where((df["label"] == lbl).to_numpy())[0]
            coords = df.iloc[idx][["cx", "cy"]].to_numpy(float)
            confs = df.iloc[idx]["confidence"].to_numpy(float)
            if len(idx) == 1:
                ref[idx[0]] = True
                continue
            cl = DBSCAN(eps=DBSCAN_EPS_PX, min_samples=1).fit(coords).labels_
            for c in np.unique(cl):
                m = cl == c
                ref[idx[m][confs[m].argmax()]] = True
        np.testing.assert_array_equal(new_mask, ref, err_msg=f"seed={seed}")


def test_nn_cluster_scale_runs_without_oom():
    """Large dense same-label cloud runs fast and returns a valid mask.

    Exercises the KD-tree/connected-components path that replaced DBSCAN (which
    OOM'd at high overlap); no 32 GB needed to guard the shape/behaviour.
    """
    rng = np.random.default_rng(1)
    n = 100_000
    cx = rng.uniform(0, 8000, n)
    cy = rng.uniform(0, 8000, n)
    df = _cluster_df(cx=cx, cy=cy, conf=rng.uniform(0.45, 0.99, n))
    mask = _nn_cluster_suppress_core(df)
    assert mask.shape == (n,)
    assert mask.dtype == bool
    assert 1 <= mask.sum() <= n


# ── Scope variants: per_tile isolation ───────────────────────────────────────

def test_iou_per_tile_isolated_suppression():
    """Boxes in different tiles are never suppressed against each other (IoU)."""
    # Two identical boxes in tile (0,0) and tile (1,0) — different tiles
    df = pd.DataFrame({
        "confidence": [0.9, 0.9],
        "x1": [0, TILE_WIDTH_PX], "y1": [0, 0],
        "x2": [10, TILE_WIDTH_PX + 10], "y2": [10, 10],
        "cx": [5.0, TILE_WIDTH_PX + 5.0], "cy": [5.0, 5.0],
        "label": ["Makrophage"] * 2,
        "tile_x": [0, 1], "tile_y": [0, 0],
    })
    mask = NMS_REGISTRY["iou_grid_n1"](df)
    assert mask.all(), "Boxes in different tiles must not suppress each other"


def test_iou_grid_n2_groups_tiles():
    """iou_grid_n2 groups tiles into 2×2 blocks; boxes in same group can suppress."""
    # Two nearly-identical boxes in tiles (0,0) and (1,0) → same 2×2 group
    df = pd.DataFrame({
        "confidence": [0.9, 0.4],
        "x1": [10, 10], "y1": [10, 10],
        "x2": [110, 110], "y2": [110, 110],
        "cx": [60.0, 60.0], "cy": [60.0, 60.0],
        "label": ["Makrophage"] * 2,
        "tile_x": [0, 1], "tile_y": [0, 0],
    })
    mask = NMS_REGISTRY["iou_grid_n2"](df)
    # Both in group (0,0) → lower conf suppressed
    assert mask.sum() == 1
    assert mask[0]


def test_grid_n1_equals_per_tile():
    """grid_n1 is a 1×1 block with no cross-tile overlap → identical to the
    per_tile grouping. per_tile is no longer a registered method (it duplicated
    grid_n1); this documents the equivalence at the _apply_scope level, which is
    exactly why grid_n1 is the single-tile scope."""
    from core.nms import _apply_scope, greedy_iou_suppress
    df = make_detection_df(n=400, n_tiles_x=4, n_tiles_y=4, seed=3)
    np.testing.assert_array_equal(
        _apply_scope(df, "grid_n1", greedy_iou_suppress),
        _apply_scope(df, "per_tile", greedy_iou_suppress),
        err_msg="grid_n1 must equal the per_tile grouping",
    )


def test_grid_overlap_dedups_cross_block_duplicate():
    """A duplicate in ADJACENT tiles is kept by per_tile (separate tiles) but
    deduped by grid_n2: the one-tile overlap puts both tiles in a common block.
    This is exactly the boundary case the old non-overlapping grid missed."""
    df = pd.DataFrame({
        "label": ["Makrophage", "Makrophage"],
        "confidence": [0.9, 0.5],
        "cx": [1000.0, 1000.0], "cy": [100.0, 100.0],     # coincident (same cell)
        "x1": [990, 990], "y1": [90, 90], "x2": [1010, 1010], "y2": [110, 110],
        "tile_x": [1, 2], "tile_y": [0, 0],               # adjacent tiles
    })
    assert NMS_REGISTRY["nn_cluster_grid_n1"](df).sum() == 2   # different tiles → both kept
    assert NMS_REGISTRY["nn_cluster_grid_n2"](df).sum() == 1    # overlap block {1,2} dedups
    assert NMS_REGISTRY["iou_grid_n2"](df).sum() == 1


def test_grid_overlap_no_partition_boundary_leak():
    """grid_n2 must merge a duplicate regardless of where it sits in tile space —
    the old tile//2 partition leaked duplicates that straddled an even boundary."""
    for tx_pair in [(0, 1), (1, 2), (2, 3)]:   # (1,2) straddles the old tile//2 cut
        df = pd.DataFrame({
            "label": ["Makrophage", "Makrophage"],
            "confidence": [0.9, 0.5],
            "cx": [5000.0, 5000.0], "cy": [50.0, 50.0],
            "x1": [4990, 4990], "y1": [40, 40], "x2": [5010, 5010], "y2": [60, 60],
            "tile_x": list(tx_pair), "tile_y": [0, 0],
        })
        assert NMS_REGISTRY["nn_cluster_grid_n2"](df).sum() == 1, f"leak at tiles {tx_pair}"


# ── iou_edgecrop: tile-edge filter + per-tile IoU ─────────────────────────────

def test_iou_edgecrop_drops_edge_keeps_interior_and_dedups():
    """Boxes within EDGE_MARGIN_PX of a tile edge are dropped; the interior
    survivors then undergo per-tile IoU NMS."""
    df = pd.DataFrame({                       # single tile (0,0): x_correct=y_correct=0
        "label": ["Makrophage"] * 7,
        "confidence": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.5],
        "x1": [100,   2, 300, 600, 400, 300, 302],   # row1 left-edge, row3 right-edge
        "y1": [100, 200,   3, 400, 600, 300, 302],   # row2 top-edge,  row4 bottom-edge
        "x2": [140,  42, 340, 638, 440, 340, 342],   # rows5,6 interior IoU-duplicates
        "y2": [140, 240,  43, 440, 638, 340, 342],
        "x_correct": [0] * 7, "y_correct": [0] * 7,
        "tile_x": [0] * 7, "tile_y": [0] * 7,
    })
    df["cx"] = (df.x1 + df.x2) / 2.0
    df["cy"] = (df.y1 + df.y2) / 2.0
    mask = NMS_REGISTRY["iou_edgecrop_grid_n1"](df)
    # rows 1–4 edge-dropped; row 6 IoU-suppressed by row 5; rows 0 & 5 survive
    assert mask.tolist() == [True, False, False, False, False, True, False]


def test_iou_edgecrop_survivors_all_interior():
    """Invariant: no survivor lies within EDGE_MARGIN_PX of its tile edge."""
    df = make_detection_df(n=400, n_tiles_x=4, n_tiles_y=4, seed=5)
    surv = df[NMS_REGISTRY["iou_edgecrop_grid_n1"](df)]
    rx1 = surv.x1 - surv.x_correct; ry1 = surv.y1 - surv.y_correct
    rx2 = surv.x2 - surv.x_correct; ry2 = surv.y2 - surv.y_correct
    m = EDGE_MARGIN_PX
    assert ((rx1 > m) & (ry1 > m)
            & (rx2 < TILE_WIDTH_PX - m) & (ry2 < TILE_HEIGHT_PX - m)).all()


def test_iou_edgecrop_is_per_tile():
    """Edge test is per tile and IoU does not cross tiles: two interior boxes at the
    same tile-relative position in adjacent tiles both survive."""
    df = pd.DataFrame({
        "label": ["Makrophage"] * 2,
        "confidence": [0.9, 0.9],
        "x1": [100, TILE_WIDTH_PX + 100], "y1": [100, 100],
        "x2": [140, TILE_WIDTH_PX + 140], "y2": [140, 140],
        "x_correct": [0, TILE_WIDTH_PX], "y_correct": [0, 0],
        "tile_x": [0, 1], "tile_y": [0, 0],
    })
    df["cx"] = (df.x1 + df.x2) / 2.0
    df["cy"] = (df.y1 + df.y2) / 2.0
    assert NMS_REGISTRY["iou_edgecrop_grid_n1"](df).sum() == 2


def test_nn_dist_global_cross_tile_suppression():
    """nn_dist_global suppresses same-label duplicates across tile boundaries."""
    df = pd.DataFrame({
        "label": ["Makrophage", "Makrophage"],
        "confidence": [0.9, 0.6],
        "cx": [TILE_WIDTH_PX - 1.0, TILE_WIDTH_PX + 1.0],  # 2px apart across tile boundary
        "cy": [100.0, 100.0],
        "x1": [TILE_WIDTH_PX - 6, TILE_WIDTH_PX - 4],
        "y1": [95, 95], "x2": [TILE_WIDTH_PX + 4, TILE_WIDTH_PX + 6], "y2": [105, 105],
        "tile_x": [0, 1], "tile_y": [0, 0],
    })
    mask = NMS_REGISTRY["nn_dist_global"](df)
    assert mask.sum() == 1
    assert mask[0]  # higher confidence survives


def test_nn_dist_per_tile_no_cross_tile_suppression():
    """nn_dist_per_tile does NOT suppress same-label detections across tiles."""
    df = pd.DataFrame({
        "label": ["Makrophage", "Makrophage"],
        "confidence": [0.9, 0.6],
        "cx": [TILE_WIDTH_PX - 1.0, TILE_WIDTH_PX + 1.0],  # 2px apart
        "cy": [100.0, 100.0],
        "x1": [TILE_WIDTH_PX - 6, TILE_WIDTH_PX - 4],
        "y1": [95, 95], "x2": [TILE_WIDTH_PX + 4, TILE_WIDTH_PX + 6], "y2": [105, 105],
        "tile_x": [0, 1], "tile_y": [0, 0],
    })
    mask = NMS_REGISTRY["nn_dist_grid_n1"](df)
    assert mask.all(), "Per-tile: different tiles → no cross-tile suppression"


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_all_methods_deterministic():
    """Calling a method twice on the same df produces identical masks."""
    df = make_detection_df(n=60, seed=99)
    for name, fn in NMS_REGISTRY.items():
        mask1 = fn(df)
        mask2 = fn(df)
        np.testing.assert_array_equal(mask1, mask2, err_msg=f"{name} is not deterministic")
