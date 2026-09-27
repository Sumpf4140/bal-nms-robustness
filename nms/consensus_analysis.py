"""Block aggregation, geometric-median consensus, per-method deviations, and α.

This module reads tile_counts from DuckDB, builds block_counts at all block
sizes, computes the inter-method consensus per block, and measures how far
each method deviates from that consensus.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from config import (
    BLOCK_FILTER_REFERENCE, BLOCK_MIN_CELLS, BLOCK_SIZES, INCLUDE_FULL_SLIDE,
    LABELS, N_BOOTSTRAP, OVERLAP_PCTS, RNG_SEED,
)
from core.blocks import aggregate_to_blocks, filter_by_min_cells, ALL_BLOCK_SIZES
from core.compositional import cmult_repl, ilr, alr, ilr_inv
from core.consensus import (
    geometric_median_ilr,
    aitchison_distances_to_consensus,
    mahalanobis_distances_to_consensus,
    krippendorff_alpha_aitchison,
    krippendorff_alpha_bootstrap_ci,
)
from core.db import connect, replace_into
from core.nms import NMS_REGISTRY

logger = logging.getLogger(__name__)

_BLOCK_KEY = ["slide_id", "overlap_pct", "block_size", "block_x", "block_y"]
_METHOD_BLOCK_KEY = ["slide_id", "overlap_pct", "nms_method", "block_size", "block_x", "block_y"]


# ── Block aggregation ─────────────────────────────────────────────────────────

def build_block_counts(
    overlap_pct: float | None = None,
    tick: Callable[[], None] | None = None,
) -> None:
    """Aggregate tile_counts → block_counts for all block sizes.

    Writes to block_counts table (INSERT OR REPLACE).
    If overlap_pct is None, runs for all OVERLAP_PCTS.
    `tick`, if given, is called once per processed overlap (progress reporting).
    """
    overlaps = [overlap_pct] if overlap_pct is not None else OVERLAP_PCTS

    # Process one overlap at a time: a single overlap's tile_counts is already
    # ~1 GB at 50% overlap, so reading every overlap at once would OOM at full
    # scale. Parameterised query (no str()-formatted IN-list → no float-repr risk).
    total = 0
    for ov in overlaps:
        with connect(read_only=True) as con:
            tc_df = con.execute(
                "SELECT slide_id, overlap_pct, nms_method, tile_x, tile_y, label, count "
                "FROM tile_counts WHERE overlap_pct=?",
                [float(ov)],
            ).fetchdf()
        if not tc_df.empty:
            combined = pd.concat(
                [aggregate_to_blocks(tc_df, bs) for bs in ALL_BLOCK_SIZES],
                ignore_index=True,
            )
            with connect() as con:
                con.register("_bc", combined)
                replace_into(con, "block_counts", "_bc")
                con.unregister("_bc")
            total += len(combined)
            logger.info("Wrote %d block_count rows for overlap=%.3f", len(combined), ov)
        if tick:
            tick()

    if total == 0:
        logger.warning("tile_counts is empty — run `nms run` first")
    else:
        logger.info("Wrote %d block_count rows total", total)


def filter_blocks(
    block_size: int,
    overlap_pct: float,
    reference_method: str = BLOCK_FILTER_REFERENCE,
) -> pd.DataFrame:
    """Return block_counts for retained block *positions* at (block_size, overlap).

    A block position is retained iff its cell count under `reference_method`
    (method-independent reference, default `none`) is ≥ BLOCK_MIN_CELLS, and then
    **all** methods' rows for that position are returned. This keeps the method
    panel balanced: the previous per-row `n_block_total >= MIN` admitted only the
    high-count methods at a position (none/per-tile) and dropped the rest to NaN,
    so α/consensus silently used method-dependent, aggression-biased block sets.
    """
    with connect(read_only=True) as con:
        df = con.execute(
            """
            SELECT bc.* FROM block_counts bc
            JOIN (
                SELECT slide_id, overlap_pct, block_size, block_x, block_y
                FROM block_counts
                WHERE block_size=? AND overlap_pct=? AND nms_method=?
                  AND n_block_total >= ?
            ) keep
              ON bc.slide_id=keep.slide_id AND bc.overlap_pct=keep.overlap_pct
             AND bc.block_size=keep.block_size
             AND bc.block_x=keep.block_x AND bc.block_y=keep.block_y
            WHERE bc.block_size=? AND bc.overlap_pct=?
            """,
            [block_size, float(overlap_pct), reference_method, BLOCK_MIN_CELLS,
             block_size, float(overlap_pct)],
        ).fetchdf()
    return df


# ── Consensus computation ─────────────────────────────────────────────────────

# Common total for the count-scale-invariant zero treatment (see
# nms.zero_sensitivity). The value is immaterial: rescaling every
# method's block counts to the same total leaves all non-zero log-ratios
# unchanged and only pins the imputed-zero pseudo-proportion (0.5/total) so it
# no longer depends on a method's absolute detection count.
_COMMON_TOTAL = 1000.0


def _blocks_to_ilr_panel(
    block_df: pd.DataFrame,
    common_scale: bool = False,
    labels: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Convert filtered block_counts to an ILR panel.

    common_scale — if True, rescale each method's per-block counts to a common
                   total before zero replacement (removes the count-scale
                   leakage that the 0.5/n zero replacement otherwise introduces
                   into balances involving frequently-zero labels).
    labels       — label subset to build the (sub)composition from (default
                   LABELS). Used for the drop-Eosinophil subcomposition check.

    Returns:
        block_meta — unique blocks (slide_id, overlap_pct, block_size, block_x, block_y)
        ilr_panel  — (M, N_blocks, D-1) float array, methods × blocks × ILR coords
        methods    — list of method names (length M)
    """
    labels = list(labels) if labels is not None else LABELS
    methods = sorted(NMS_REGISTRY.keys())
    # Unique blocks
    block_meta = (
        block_df[_BLOCK_KEY]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    N = len(block_meta)
    D1 = len(labels) - 1
    M = len(methods)

    ilr_panel = np.full((M, N, D1), np.nan)

    pivot = (
        block_df.pivot_table(
            index=_METHOD_BLOCK_KEY,
            columns="label",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=labels, fill_value=0)
        .reset_index()
    )

    # Build lookup: (slide_id, overlap_pct, block_size, block_x, block_y) → block index
    key_to_idx = {
        tuple(row[k] for k in _BLOCK_KEY): i
        for i, row in block_meta.iterrows()
    }

    for mi, method in enumerate(methods):
        sub = pivot[pivot["nms_method"] == method]
        if sub.empty:
            continue
        counts = sub[labels].to_numpy(dtype=float)
        if common_scale:
            tot = counts.sum(axis=1, keepdims=True)
            counts = np.divide(counts, tot, out=np.zeros_like(counts),
                               where=tot > 0) * _COMMON_TOTAL
        props = cmult_repl(counts)
        ilr_coords = ilr(props)  # (N_method_blocks, D-1)

        for pos, (_, row) in enumerate(sub.iterrows()):
            key = tuple(row[k] for k in _BLOCK_KEY)
            bi = key_to_idx.get(key)
            if bi is not None:
                ilr_panel[mi, bi, :] = ilr_coords[pos]

    return block_meta, ilr_panel, methods


def compute_consensus_per_block(block_size: int, overlap_pct: float) -> None:
    """Compute geometric-median consensus for each retained block.

    Writes to block_consensus table.
    """
    block_df = filter_blocks(block_size, overlap_pct)
    if block_df.empty:
        logger.warning("No retained blocks for block_size=%d overlap=%.2f", block_size, overlap_pct)
        return

    block_meta, ilr_panel, methods = _blocks_to_ilr_panel(block_df)
    M, N, D1 = ilr_panel.shape

    # ALR reference index: Lymphozyt (index 1 in LABELS)
    from config import ALR_REF_LABEL
    ref_idx = LABELS.index(ALR_REF_LABEL)
    non_ref_labels = [l for l in LABELS if l != ALR_REF_LABEL]

    # Pre-compute n_block_total lookup: key → total cells in block
    n_total_lookup: dict[tuple, int] = {}
    for _, row in (
        block_df[_BLOCK_KEY + ["n_block_total"]]
        .drop_duplicates(subset=_BLOCK_KEY)
        .iterrows()
    ):
        key = tuple(row[k] for k in _BLOCK_KEY)
        n_total_lookup[key] = int(row["n_block_total"])

    rows = []
    for bi in range(N):
        block_ilrs = ilr_panel[:, bi, :]
        valid_mask = ~np.isnan(block_ilrs).any(axis=1)
        valid_ilrs = block_ilrs[valid_mask]

        if valid_ilrs.shape[0] < 2:
            continue

        consensus_ilr = geometric_median_ilr(valid_ilrs)
        # Convert back to composition for ALR
        consensus_props = ilr_inv(consensus_ilr)
        consensus_alr = np.log(consensus_props[[i for i in range(len(LABELS)) if i != ref_idx]]
                               / consensus_props[ref_idx])

        meta = block_meta.iloc[bi]
        n_total = n_total_lookup[tuple(meta[k] for k in _BLOCK_KEY)]

        rows.append({
            "slide_id": meta["slide_id"],
            "overlap_pct": float(meta["overlap_pct"]),
            "block_size": int(meta["block_size"]),
            "block_x": int(meta["block_x"]),
            "block_y": int(meta["block_y"]),
            "consensus_alr_Mac": float(consensus_alr[0]) if len(non_ref_labels) > 0 else float("nan"),
            "consensus_alr_Neu": float(consensus_alr[1]) if len(non_ref_labels) > 1 else float("nan"),
            "consensus_alr_Eos": float(consensus_alr[2]) if len(non_ref_labels) > 2 else float("nan"),
            "n_methods_in_consensus": int(valid_mask.sum()),
            "n_block_total": n_total,
        })

    if not rows:
        return

    with connect() as con:
        df_out = pd.DataFrame(rows)
        con.register("_bc_cons", df_out)
        replace_into(con, "block_consensus", "_bc_cons")
        con.unregister("_bc_cons")

    logger.info("Wrote %d block_consensus rows (block_size=%d, overlap=%.2f)",
                len(rows), block_size, overlap_pct)


def compute_method_deviations(block_size: int, overlap_pct: float) -> None:
    """Compute per-method Aitchison and Mahalanobis distance to consensus.

    Writes to method_deviation table.
    """
    block_df = filter_blocks(block_size, overlap_pct)
    if block_df.empty:
        return

    block_meta, ilr_panel, methods = _blocks_to_ilr_panel(block_df)
    M, N, D1 = ilr_panel.shape

    rows = []
    for bi in range(N):
        block_ilrs = ilr_panel[:, bi, :]
        valid_mask = ~np.isnan(block_ilrs).any(axis=1)
        valid_ilrs = block_ilrs[valid_mask]

        if valid_ilrs.shape[0] < 2:
            continue

        consensus_ilr = geometric_median_ilr(valid_ilrs)
        ait_dists = aitchison_distances_to_consensus(valid_ilrs, consensus_ilr)
        mah_dists = mahalanobis_distances_to_consensus(valid_ilrs, consensus_ilr)

        meta = block_meta.iloc[bi]
        valid_methods = [methods[i] for i in range(M) if valid_mask[i]]

        for mi2, method in enumerate(valid_methods):
            rows.append({
                "slide_id": meta["slide_id"],
                "overlap_pct": float(meta["overlap_pct"]),
                "nms_method": method,
                "block_size": int(meta["block_size"]),
                "block_x": int(meta["block_x"]),
                "block_y": int(meta["block_y"]),
                "aitchison_distance": float(ait_dists[mi2]),
                "mahalanobis_distance": float(mah_dists[mi2]),
            })

    if not rows:
        return

    with connect() as con:
        df_out = pd.DataFrame(rows)
        con.register("_md", df_out)
        replace_into(con, "method_deviation", "_md")
        con.unregister("_md")

    logger.info("Wrote %d method_deviation rows (block_size=%d, overlap=%.2f)",
                len(rows), block_size, overlap_pct)


# ── Krippendorff's α at scale ─────────────────────────────────────────────────

def krippendorff_alpha_at_scale(
    block_size: int,
    overlap_pct: float,
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute Krippendorff's α with Aitchison distance for one (block_size, overlap).

    Bootstrap CI over slides (cluster bootstrap: resample slides with replacement).

    Returns dict with keys: alpha, lo95, hi95, n_blocks_used, n_blocks_filtered.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    # All block positions before filter (tuple DISTINCT — no string-concat collisions)
    with connect(read_only=True) as con:
        all_df = con.execute(
            "SELECT COUNT(DISTINCT (slide_id, block_x, block_y)) as n "
            "FROM block_counts WHERE block_size=? AND overlap_pct=?",
            [block_size, float(overlap_pct)],
        ).fetchone()
    n_all = int(all_df[0]) if all_df else 0

    block_df = filter_blocks(block_size, overlap_pct)
    n_retained = len(block_df[_BLOCK_KEY].drop_duplicates())
    n_filtered = n_all - n_retained

    if block_df.empty or n_retained < 2:
        return {
            "alpha": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
            "n_blocks_used": n_retained, "n_blocks_filtered": n_filtered,
        }

    block_meta, ilr_panel, methods = _blocks_to_ilr_panel(block_df)
    # Remove blocks with any NaN (method failed)
    valid_blocks = ~np.isnan(ilr_panel).any(axis=(0, 2))
    ilr_panel_clean = ilr_panel[:, valid_blocks, :]

    if ilr_panel_clean.shape[1] < 2:
        return {
            "alpha": float("nan"), "lo95": float("nan"), "hi95": float("nan"),
            "n_blocks_used": int(valid_blocks.sum()), "n_blocks_filtered": n_filtered,
        }

    # Slide-cluster bootstrap
    slide_ids = block_meta.loc[valid_blocks, "slide_id"].to_numpy()
    unique_slides = np.unique(slide_ids)
    slide_to_block_idx: dict[str, list[int]] = {
        s: np.where(slide_ids == s)[0].tolist() for s in unique_slides
    }

    alpha_val = krippendorff_alpha_aitchison(ilr_panel_clean)

    boot_alphas = np.empty(n_boot)
    for b in range(n_boot):
        boot_slides = rng.choice(unique_slides, size=len(unique_slides), replace=True)
        boot_idx = [i for s in boot_slides for i in slide_to_block_idx[s]]
        boot_panel = ilr_panel_clean[:, boot_idx, :]
        boot_alphas[b] = krippendorff_alpha_aitchison(boot_panel)

    return {
        "alpha": float(alpha_val),
        "lo95": float(np.nanpercentile(boot_alphas, 2.5)),
        "hi95": float(np.nanpercentile(boot_alphas, 97.5)),
        "n_blocks_used": int(valid_blocks.sum()),
        "n_blocks_filtered": n_filtered,
    }


def scale_dependence_curve(tick: Callable[[], None] | None = None) -> pd.DataFrame:
    """Compute α with CI for all (block_size, overlap_pct) combinations.

    Returns DataFrame with columns:
        block_size, overlap_pct, alpha, lo95, hi95, n_blocks_used, n_blocks_filtered
    `tick`, if given, is called once per (block_size, overlap) setting.
    """
    rows = []
    for bs in ALL_BLOCK_SIZES:
        for op in OVERLAP_PCTS:
            logger.info("Computing α: block_size=%d overlap=%.2f", bs, op)
            result = krippendorff_alpha_at_scale(bs, op)
            result["block_size"] = bs
            result["overlap_pct"] = op
            rows.append(result)
            if tick:
                tick()
    return pd.DataFrame(rows)
