"""31 NMS implementations.

Registry count: 1 (none) + 6 (iou) + 6 (nn_dist) + 6 (nn_center) + 6 (nn_cluster)
+ 6 (iou_edgecrop) = 31. Every family spans the same 6 scopes
{grid_n1, grid_n2, grid_n3, grid_n4, grid_n5, global}. (`per_tile` is not a
separate entry: it is identical to grid_n1 by construction, so it was dropped to
avoid double-counting a duplicate rater; grid_n1 IS the single-tile scope.)

Common signature: NMSFunc = Callable[[pd.DataFrame], np.ndarray]
  Input:  DataFrame with columns label, confidence, x1, y1, x2, y2, cx, cy,
          x_correct, y_correct, tile_x, tile_y
          (confidence already filtered by CONF_THRESH upstream; boxes are
          slide-absolute, x_correct/y_correct is the tile origin)
  Output: boolean survivor mask of length len(df)

Scope variants:
  global    — suppression over entire slide at once
  grid_nN   — sliding N×N-tile blocks with a one-tile overlap (stride N−1),
              processed row-major with sequential carry-over (N = 1..5). N=1 is a
              single tile, so grid_n1 is the single-tile ("per-tile") scope.
  per_tile  — internal alias for grid_n1's grouping (used by _apply_scope); not a
              registered method of its own.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from config import (
    IOU_THRESH, NN_DIST_PX, DBSCAN_EPS_PX, LABELS,
    EDGE_MARGIN_PX, TILE_WIDTH_PX, TILE_HEIGHT_PX,
)

NMSFunc = Callable[[pd.DataFrame], np.ndarray]


# ── Low-level primitives ──────────────────────────────────────────────────────

def pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """Compute all-pairs IoU.  boxes: (N, 4) as [x1, y1, x2, y2].  Returns (N, N)."""
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
    union = areas[:, None] + areas[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _iou_one_vs_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one box [x1,y1,x2,y2] vs an (M,4) array — O(M) memory."""
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
    area_box = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    areas = (others[:, 2] - others[:, 0]).clip(0) * (others[:, 3] - others[:, 1]).clip(0)
    union = area_box + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_iou_suppress(df: pd.DataFrame, iou_thresh: float = IOU_THRESH) -> np.ndarray:
    """Standard greedy IoU NMS (highest confidence first).  Returns bool survivor mask.

    Memory-safe: candidate overlaps are found with a KD-tree on box centres
    (two boxes can only overlap if their centres are within the largest box
    diagonal), so this never materialises a dense N×N IoU matrix — essential on
    high-overlap grid blocks that can hold >10⁵ detections. Results are identical
    to the dense version (the radius is a conservative superset of all overlaps).
    """
    from scipy.spatial import cKDTree

    n = len(df)
    if n <= 1:
        return np.ones(n, dtype=bool)

    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    scores = df["confidence"].to_numpy(dtype=float)
    centres = np.column_stack([
        (boxes[:, 0] + boxes[:, 2]) / 2.0,
        (boxes[:, 1] + boxes[:, 3]) / 2.0,
    ])
    diag = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    radius = float(diag.max())
    tree = cKDTree(centres)

    order = scores.argsort()[::-1]
    suppress = np.zeros(n, dtype=bool)
    for i in order:
        if suppress[i]:
            continue
        cand = np.fromiter(
            (j for j in tree.query_ball_point(centres[i], radius)
             if j != i and not suppress[j]),
            dtype=int,
        )
        if cand.size == 0:
            continue
        ious = _iou_one_vs_many(boxes[i], boxes[cand])
        suppress[cand[ious >= iou_thresh]] = True

    return ~suppress


def _iou_edgecrop_suppress_core(
    df: pd.DataFrame,
    margin: float = EDGE_MARGIN_PX,
    iou_thresh: float = IOU_THRESH,
) -> np.ndarray:
    """Edge-crop + greedy IoU NMS (applied per tile).

    Two stages: (1) drop every detection whose box lies within `margin` px of a
    tile edge — the clipped "half" cells at a tile border; (2) run standard
    greedy IoU NMS on the interior survivors. With overlapping tiles a cell
    clipped at one tile's edge is captured whole (interior) in the neighbouring
    tile, so it re-enters the dataset there; only genuine edge artefacts are lost.

    The edge test is in tile-relative coordinates (`x1 − x_correct`, …), so it is
    correct per row regardless of how `df` is grouped; the IoU step is per-tile
    via `_apply_scope(df, "per_tile", …)`. Returns a bool survivor mask.
    """
    n = len(df)
    if n == 0:
        return np.ones(0, dtype=bool)

    x_org = df["x_correct"].to_numpy(dtype=float)
    y_org = df["y_correct"].to_numpy(dtype=float)
    rx1 = df["x1"].to_numpy(dtype=float) - x_org
    ry1 = df["y1"].to_numpy(dtype=float) - y_org
    rx2 = df["x2"].to_numpy(dtype=float) - x_org
    ry2 = df["y2"].to_numpy(dtype=float) - y_org
    interior = (
        (rx1 > margin) & (ry1 > margin)
        & (rx2 < TILE_WIDTH_PX - margin) & (ry2 < TILE_HEIGHT_PX - margin)
    )

    out = np.zeros(n, dtype=bool)
    int_pos = np.where(interior)[0]
    if int_pos.size:
        iou_keep = greedy_iou_suppress(df.iloc[int_pos].reset_index(drop=True), iou_thresh)
        out[int_pos[iou_keep]] = True
    return out


def _nn_dist_suppress_core(df: pd.DataFrame, dist_px: float = NN_DIST_PX) -> np.ndarray:
    """NN-distance NMS: same-label pairs within dist_px → keep higher confidence.

    Uses cKDTree for O(n log n + n·k) instead of O(n²) Python loops.
    """
    from scipy.spatial import cKDTree
    n = len(df)
    if n <= 1:
        return np.ones(n, dtype=bool)

    cx = df["cx"].to_numpy(dtype=float)
    cy = df["cy"].to_numpy(dtype=float)
    labels = df["label"].to_numpy()
    conf = df["confidence"].to_numpy(dtype=float)
    suppress = np.zeros(n, dtype=bool)

    for lbl in np.unique(labels):
        lbl_idx = np.where(labels == lbl)[0]
        if len(lbl_idx) <= 1:
            continue
        pts = np.column_stack([cx[lbl_idx], cy[lbl_idx]])
        lbl_conf = conf[lbl_idx]
        tree = cKDTree(pts)
        lbl_suppress = np.zeros(len(lbl_idx), dtype=bool)
        for local_i in lbl_conf.argsort()[::-1]:
            if lbl_suppress[local_i]:
                continue
            for j_local in tree.query_ball_point(pts[local_i], dist_px):
                if j_local != local_i:
                    lbl_suppress[j_local] = True
        suppress[lbl_idx[lbl_suppress]] = True

    return ~suppress


def _compute_center_radii(df: pd.DataFrame) -> dict[str, float]:
    """Per-label suppression radius = 2·std(bbox_size).  bbox_size = mean(w, h)."""
    radii: dict[str, float] = {}
    for lbl in LABELS:
        sub = df[df["label"] == lbl]
        if len(sub) == 0:
            radii[lbl] = 0.0
        else:
            sizes = ((sub["x2"] - sub["x1"]) + (sub["y2"] - sub["y1"])) / 2.0
            std = float(sizes.std(ddof=1)) if len(sizes) > 1 else 0.0
            radii[lbl] = 2.0 * std
    return radii


def _nn_center_suppress_core(df: pd.DataFrame, label_radius: dict[str, float]) -> np.ndarray:
    """NN-centre NMS: same-label pairs within data-driven radius → keep higher confidence.

    Uses cKDTree for O(n log n + n·k) instead of O(n²) Python loops.
    """
    from scipy.spatial import cKDTree
    n = len(df)
    if n <= 1:
        return np.ones(n, dtype=bool)

    cx = df["cx"].to_numpy(dtype=float)
    cy = df["cy"].to_numpy(dtype=float)
    labels = df["label"].to_numpy()
    conf = df["confidence"].to_numpy(dtype=float)
    suppress = np.zeros(n, dtype=bool)

    for lbl in np.unique(labels):
        radius = label_radius.get(lbl, 0.0)
        if radius <= 0.0:
            continue
        lbl_idx = np.where(labels == lbl)[0]
        if len(lbl_idx) <= 1:
            continue
        pts = np.column_stack([cx[lbl_idx], cy[lbl_idx]])
        lbl_conf = conf[lbl_idx]
        tree = cKDTree(pts)
        lbl_suppress = np.zeros(len(lbl_idx), dtype=bool)
        for local_i in lbl_conf.argsort()[::-1]:
            if lbl_suppress[local_i]:
                continue
            for j_local in tree.query_ball_point(pts[local_i], radius):
                if j_local != local_i:
                    lbl_suppress[j_local] = True
        suppress[lbl_idx[lbl_suppress]] = True

    return ~suppress


def _nn_cluster_suppress_core(df: pd.DataFrame, eps: float = DBSCAN_EPS_PX) -> np.ndarray:
    """Cluster per label (same-label detections within `eps`); keep the
    highest-confidence detection per cluster.

    The clusters are the connected components of the ε-radius graph, which is
    *exactly* DBSCAN with min_samples=1 (every point is a core point ⇒ no noise).
    Computing them with a KD-tree + scipy connected_components avoids sklearn
    DBSCAN's bulk-materialised neighbourhoods, which OOM (>32 GB) on the dense,
    cross-tile-duplicate point clouds at high overlap (cf. the KD-tree hardening
    of greedy_iou_suppress). Output is identical to the DBSCAN path.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(df)
    if n == 0:
        return np.ones(0, dtype=bool)

    keep = np.zeros(n, dtype=bool)
    df_reset = df.reset_index(drop=True)

    for lbl in df_reset["label"].unique():
        mask = (df_reset["label"] == lbl).to_numpy()
        orig_idx = np.where(mask)[0]
        sub = df_reset.iloc[orig_idx]

        coords = sub[["cx", "cy"]].to_numpy(dtype=float)
        confs = sub["confidence"].to_numpy(dtype=float)

        if len(coords) == 1:
            keep[orig_idx[0]] = True
            continue

        # Connected components of the ε-radius graph (≡ DBSCAN min_samples=1).
        pairs = cKDTree(coords).query_pairs(eps, output_type="ndarray")  # (m, 2), dist ≤ eps
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
            shape=(len(coords), len(coords)),
        )
        _, cluster_ids = connected_components(graph, directed=False)

        # Keep one survivor per cluster — the max-confidence member, ties broken
        # by lowest original index (identical to a per-cluster argmax). Vectorised:
        # sort by (cluster, −confidence, position) and take the first row of each
        # cluster group — O(k log k) instead of an O(n_clusters × k) Python loop.
        order = np.lexsort((np.arange(len(confs)), -confs, cluster_ids))
        sorted_cids = cluster_ids[order]
        first_in_group = np.concatenate(([True], sorted_cids[1:] != sorted_cids[:-1]))
        keep[orig_idx[order[first_in_group]]] = True

    return keep


# ── Scope dispatcher ─────────────────────────────────────────────────────────

def _groupby_suppress(
    df: pd.DataFrame,
    group_cols: list[str],
    suppress_fn: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    """Apply suppress_fn within each group defined by group_cols.

    df must have a clean RangeIndex (call reset_index(drop=True) before passing).
    """
    result = np.zeros(len(df), dtype=bool)
    if len(df) == 0:
        return result
    for _, grp in df.groupby(group_cols, sort=False):
        mask = suppress_fn(grp)
        result[grp.index.to_numpy()] = mask
    return result


def _apply_scope(
    df: pd.DataFrame,
    scope: str,
    suppress_fn: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    """Route to global / per_tile / grid_nN application."""
    df_r = df.reset_index(drop=True)

    if scope == "global":
        return suppress_fn(df_r)

    if scope == "per_tile":
        return _groupby_suppress(df_r, ["tile_x", "tile_y"], suppress_fn)

    # grid_nN — sliding n×n-tile blocks with a one-tile overlap.
    n = int(scope.split("_n")[1])
    if n <= 1:
        # A 1×1 block has no overlap → identical to per_tile.
        return _groupby_suppress(df_r, ["tile_x", "tile_y"], suppress_fn)
    return _grid_overlap_suppress(df_r, n, suppress_fn)


def _grid_overlap_suppress(
    df: pd.DataFrame,
    n: int,
    suppress_fn: Callable[[pd.DataFrame], np.ndarray],
) -> np.ndarray:
    """Sliding n×n-tile blocks with a one-tile overlap (stride n−1).

    Block size is n×n tiles (n·640 px square); adjacent blocks overlap by exactly
    one tile in x and y, so the step is (n−1) tiles. Blocks are anchored to the
    tile-grid origin and processed in **row-major order with sequential
    carry-over**: NMS runs within each block on the detections still alive, and
    whatever it suppresses is removed before the next block runs. Edge blocks use
    whatever tiles exist (clipped). Every adjacent tile pair shares a block, so
    cross-tile duplicates are deduplicated regardless of where the grid falls.

    `df` must have a clean RangeIndex.
    """
    keep = np.ones(len(df), dtype=bool)
    if len(df) == 0:
        return keep

    tx = df["tile_x"].to_numpy()
    ty = df["tile_y"].to_numpy()
    pos = np.arange(len(df))
    step = n - 1  # n ≥ 2 here, so step ≥ 1

    for by in range(0, int(ty.max()) + 1, step):
        for bx in range(0, int(tx.max()) + 1, step):
            sel = keep & (tx >= bx) & (tx < bx + n) & (ty >= by) & (ty < by + n)
            if not sel.any():
                continue
            idx = pos[sel]
            local = np.asarray(suppress_fn(df.iloc[idx].reset_index(drop=True)), dtype=bool)
            keep[idx[~local]] = False  # carry the suppression forward
    return keep


# ── Factory functions ────────────────────────────────────────────────────────

def _make_iou(scope: str) -> NMSFunc:
    def fn(df: pd.DataFrame) -> np.ndarray:
        return _apply_scope(df, scope, greedy_iou_suppress)
    fn.__name__ = f"iou_{scope}"
    return fn


def _make_iou_edgecrop(scope: str) -> NMSFunc:
    def fn(df: pd.DataFrame) -> np.ndarray:
        return _apply_scope(df, scope, _iou_edgecrop_suppress_core)
    fn.__name__ = f"iou_edgecrop_{scope}"
    return fn


def _make_nn_dist(scope: str) -> NMSFunc:
    def fn(df: pd.DataFrame) -> np.ndarray:
        return _apply_scope(df, scope, _nn_dist_suppress_core)
    fn.__name__ = f"nn_dist_{scope}"
    return fn


def _make_nn_center(scope: str) -> NMSFunc:
    def fn(df: pd.DataFrame) -> np.ndarray:
        # Compute global size statistics from the full input, then apply per scope.
        df_r = df.reset_index(drop=True)
        label_radius = _compute_center_radii(df_r)

        def _suppress(grp: pd.DataFrame) -> np.ndarray:
            return _nn_center_suppress_core(grp, label_radius)

        return _apply_scope(df_r, scope, _suppress)
    fn.__name__ = f"nn_center_{scope}"
    return fn


def _make_nn_cluster(scope: str) -> NMSFunc:
    def fn(df: pd.DataFrame) -> np.ndarray:
        return _apply_scope(df, scope, _nn_cluster_suppress_core)
    fn.__name__ = f"nn_cluster_{scope}"
    return fn


# ── Registry ─────────────────────────────────────────────────────────────────

NMS_REGISTRY: dict[str, NMSFunc] = {}

# None (1)
NMS_REGISTRY["none"] = lambda df: np.ones(len(df), dtype=bool)
NMS_REGISTRY["none"].__name__ = "none"  # type: ignore[attr-defined]

# Every family spans the same 6 scopes: grid_n1 (single-tile) + grid_n2..n5 + global.
# `per_tile` is NOT registered separately — it is identical to grid_n1 by construction.
_SCOPES = ["grid_n1", "grid_n2", "grid_n3", "grid_n4", "grid_n5", "global"]

# IoU (6)
for _scope in _SCOPES:
    NMS_REGISTRY[f"iou_{_scope}"] = _make_iou(_scope)

# IoU edge-crop (6): drop tile-edge detections, then IoU at each scope. The edge
# filter is per-tile at every scope; the IoU step follows the scope, so cross-tile
# scopes also dedupe the recovered overlap copies (expected: the single-tile size
# bias shrinks as the scope widens).
for _scope in _SCOPES:
    NMS_REGISTRY[f"iou_edgecrop_{_scope}"] = _make_iou_edgecrop(_scope)

# NN distance / centre / cluster (6 each)
for _scope in _SCOPES:
    NMS_REGISTRY[f"nn_dist_{_scope}"] = _make_nn_dist(_scope)
    NMS_REGISTRY[f"nn_center_{_scope}"] = _make_nn_center(_scope)
    NMS_REGISTRY[f"nn_cluster_{_scope}"] = _make_nn_cluster(_scope)

assert len(NMS_REGISTRY) == 31, f"Expected 31 NMS methods, got {len(NMS_REGISTRY)}"
