"""Controlled, single-process timing of the 31 NMS variants → timing_clean.

The `timing` collected during `run_all` is distorted by --workers CPU contention
(frequency scaling + memory-bandwidth pressure, unevenly across methods), so it
must not be used for the reported time-save (P1) or cost (P3). This module times
each method in a single process over a representative batch and stores CPU + wall
time per repeat. For a pristine run also pin BLAS/OpenMP threads to 1 in the
environment before launching (the methods are mostly single-threaded anyway):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 balc p1 timing
"""
from __future__ import annotations

import logging
import random
import time

import pandas as pd

from config import CONF_THRESH, RNG_SEED
from core.db import connect
from core.nms import NMS_REGISTRY

logger = logging.getLogger(__name__)


def representative_batch(n_slides: int, seed: int = RNG_SEED,
                         db_path: str | None = None) -> list[tuple[str, float]]:
    """`n_slides` slides present at every overlap, paired with all overlaps —
    a balanced panel spanning the full detection-density range."""
    with connect(read_only=True, db_path=db_path) as con:
        overlaps = [r[0] for r in con.execute(
            "SELECT DISTINCT overlap_pct FROM raw_detections ORDER BY 1").fetchall()]
        slides = [r[0] for r in con.execute(
            "SELECT slide_id FROM raw_detections GROUP BY slide_id "
            "HAVING COUNT(DISTINCT overlap_pct) = ? ORDER BY slide_id",
            [len(overlaps)]).fetchall()]
    if not slides or not overlaps:
        return []
    random.Random(seed).shuffle(slides)
    chosen = slides[:n_slides]
    return [(s, ov) for s in chosen for ov in overlaps]


def time_methods(df, methods: list[str], repeats: int, warmup: bool) -> dict[str, list[tuple[float, float]]]:
    """Time each method on a fixed DataFrame. Pure (no DB) → unit-testable.
    Returns {method: [(cpu_seconds, wall_seconds), ... per repeat]}."""
    out: dict[str, list[tuple[float, float]]] = {}
    for method in methods:
        fn = NMS_REGISTRY[method]
        if warmup:
            fn(df)
        reps = []
        for _ in range(repeats):
            t_cpu = time.process_time()
            t_wall = time.perf_counter()
            fn(df)
            reps.append((time.process_time() - t_cpu, time.perf_counter() - t_wall))
        out[method] = reps
    return out


def run_timing(n_slides: int = 5, repeats: int = 3, warmup: bool = True,
               seed: int = RNG_SEED, db_path: str | None = None) -> int:
    """Measure controlled timing for all 31 variants over a representative batch
    and store every repeat in timing_clean. Returns the number of rows written."""
    pairs = representative_batch(n_slides, seed, db_path)
    if not pairs:
        logger.warning("No fully-populated slides found — load data first.")
        return 0

    methods = list(NMS_REGISTRY)
    with connect(db_path=db_path) as con:
        con.execute("DELETE FROM timing_clean")

    rows = []
    for slide_id, overlap_pct in pairs:
        with connect(read_only=True, db_path=db_path) as con:
            df = con.execute(
                "SELECT * FROM raw_detections WHERE slide_id=? AND overlap_pct=? AND confidence>=?",
                [slide_id, float(overlap_pct), CONF_THRESH],
            ).fetchdf()
        n_det = len(df)
        timed = time_methods(df, methods, repeats, warmup)
        for method, reps in timed.items():
            for rep, (cpu, wall) in enumerate(reps):
                rows.append((slide_id, float(overlap_pct), method, rep, n_det, cpu, wall))

    with connect(db_path=db_path) as con:
        con.executemany("INSERT OR REPLACE INTO timing_clean VALUES (?,?,?,?,?,?,?)", rows)
    logger.info("Wrote %d timing_clean rows (%d slides × %d overlaps × %d methods × %d repeats)",
                len(rows), n_slides, len({o for _, o in pairs}), len(methods), repeats)
    return len(rows)


def timing_summary(db_path: str | None = None) -> pd.DataFrame:
    """Per-method median CPU/wall seconds from timing_clean (repeats collapsed)."""
    with connect(read_only=True, db_path=db_path) as con:
        df = con.execute(
            "SELECT slide_id, overlap_pct, nms_method, "
            "median(elapsed_cpu) AS cpu, median(elapsed_wall) AS wall "
            "FROM timing_clean GROUP BY slide_id, overlap_pct, nms_method").fetchdf()
    if df.empty:
        return df
    return (df.groupby("nms_method")[["cpu", "wall"]]
              .median().reset_index().sort_values("cpu").reset_index(drop=True))
