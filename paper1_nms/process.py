"""Resumable 76 × 9 × 31 NMS run.

Single-writer rule: workers read DB read-only; main process writes all results
inside one transaction batch after workers return.

Usage:
    from paper1_nms.process import run_all
    run_all(n_workers=4)
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

from config import CONF_THRESH, OVERLAP_PCTS, LABELS
from core.db import connect, replace_into
from core.nms import NMS_REGISTRY

logger = logging.getLogger(__name__)


# ── Single-run logic (executed in worker process, reads DB read-only) ─────────

def run_one(slide_id: str, overlap_pct: float, method: str) -> dict[str, Any]:
    """Run one (slide, overlap, method) NMS combination.

    Returns a dict with keys: cell_counts, tile_counts, timing.
    Raises on any error (caller writes to checkpoint).
    """
    with connect(read_only=True) as con:
        df = con.execute(
            """SELECT * FROM raw_detections
               WHERE slide_id=? AND overlap_pct=? AND confidence>=?""",
            [slide_id, float(overlap_pct), CONF_THRESH],
        ).fetchdf()

    if len(df) == 0:
        logger.warning("No detections for slide=%s overlap=%s", slide_id, overlap_pct)

    t0 = time.process_time()
    keep = NMS_REGISTRY[method](df)
    elapsed = time.process_time() - t0

    survivors = df.loc[keep].copy()

    cell_counts = _agg_total(survivors, slide_id, overlap_pct, method)
    tile_counts = _agg_per_tile(survivors, slide_id, overlap_pct, method)
    timing = {
        "slide_id": slide_id,
        "overlap_pct": float(overlap_pct),
        "nms_method": method,
        "elapsed_cpu": float(elapsed),
        "n_input": int(len(df)),
        "n_output": int(keep.sum()),
    }
    return {"cell_counts": cell_counts, "tile_counts": tile_counts, "timing": timing}


def _agg_total(
    survivors: pd.DataFrame,
    slide_id: str,
    overlap_pct: float,
    method: str,
) -> list[dict]:
    """Slide-total counts and relative counts per label."""
    total = len(survivors)
    rows = []
    for lbl in LABELS:
        count = int((survivors["label"] == lbl).sum())
        rel = count / total if total > 0 else 0.0
        rows.append({
            "slide_id": slide_id,
            "overlap_pct": float(overlap_pct),
            "nms_method": method,
            "label": lbl,
            "count": count,
            "rel_count": rel,
        })
    return rows


def _agg_per_tile(
    survivors: pd.DataFrame,
    slide_id: str,
    overlap_pct: float,
    method: str,
) -> list[dict]:
    """Per-tile counts per label (zero-filled for missing label×tile combos)."""
    if len(survivors) == 0:
        return []

    rows = []
    tile_groups = survivors.groupby(["tile_x", "tile_y"])
    for (tx, ty), grp in tile_groups:
        for lbl in LABELS:
            count = int((grp["label"] == lbl).sum())
            rows.append({
                "slide_id": slide_id,
                "overlap_pct": float(overlap_pct),
                "nms_method": method,
                "tile_x": int(tx),
                "tile_y": int(ty),
                "label": lbl,
                "count": count,
            })
    return rows


# ── Per-pair worker (reads one slice, runs all methods) ───────────────────────

def run_pair(slide_id: str, overlap_pct: float, methods: list[str],
             db_path: str | None = None,
             on_method: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Read one (slide, overlap) slice read-only, run every method in `methods`.

    The DB is opened read-only and closed *before* computing, so many workers
    can read concurrently and the main process can take the write lock between
    rounds. Only the small aggregated result travels back — never the
    (multi-million-row) input. Reads the slice once for all 31 variants instead
    of once per method. `db_path` is passed explicitly so spawned workers (which
    don't inherit a monkeypatched module global) open the right database.

    `on_method(method)` is called after each method finishes (success or error) —
    used by the single-worker path to advance the progress bar per method. It is
    only ever passed in-process; parallel workers leave it None (not picklable).
    """
    with connect(read_only=True, db_path=db_path) as con:
        df = con.execute(
            """SELECT * FROM raw_detections
               WHERE slide_id=? AND overlap_pct=? AND confidence>=?""",
            [slide_id, float(overlap_pct), CONF_THRESH],
        ).fetchdf()

    cell_rows: list[dict] = []
    tile_rows: list[dict] = []
    timing_rows: list[dict] = []
    done: list[str] = []
    errors: list[tuple[str, str]] = []

    for method in methods:
        try:
            t0 = time.process_time()
            keep = NMS_REGISTRY[method](df)
            elapsed = time.process_time() - t0
            survivors = df.loc[keep]
            cell_rows.extend(_agg_total(survivors, slide_id, overlap_pct, method))
            tile_rows.extend(_agg_per_tile(survivors, slide_id, overlap_pct, method))
            timing_rows.append({
                "slide_id": slide_id, "overlap_pct": float(overlap_pct),
                "nms_method": method, "elapsed_cpu": float(elapsed),
                "n_input": int(len(df)), "n_output": int(keep.sum()),
            })
            done.append(method)
        except Exception as exc:  # noqa: BLE001 — record and continue
            errors.append((method, str(exc)))
        if on_method is not None:
            on_method(method)

    return {
        "slide_id": slide_id, "overlap_pct": float(overlap_pct),
        "cell": cell_rows, "tile": tile_rows, "timing": timing_rows,
        "done": done, "errors": errors,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _get_pending_pairs(
    con, slide_filter: str | None = None
) -> list[tuple[str, float, list[str]]]:
    """Pending work grouped per (slide, overlap): (slide_id, overlap_pct, [methods])."""
    slides_df = con.execute("SELECT DISTINCT slide_id FROM slides").fetchdf()
    if slides_df.empty:
        return []
    slides = slides_df["slide_id"].tolist()
    if slide_filter:
        slides = [s for s in slides if s == slide_filter]

    done_df = con.execute(
        "SELECT slide_id, overlap_pct, nms_method FROM checkpoint WHERE status='done'"
    ).fetchdf()
    done_set = set(
        zip(done_df["slide_id"], done_df["overlap_pct"].astype(float), done_df["nms_method"])
    ) if not done_df.empty else set()

    methods_all = list(NMS_REGISTRY)
    pairs: list[tuple[str, float, list[str]]] = []
    for slide_id in slides:
        for overlap_pct in OVERLAP_PCTS:
            pending = [m for m in methods_all
                       if (slide_id, float(overlap_pct), m) not in done_set]
            if pending:
                pairs.append((slide_id, float(overlap_pct), pending))
    return pairs


def _write_round(results: list[dict[str, Any]]) -> None:
    """Write a completed round's results + checkpoints in ONE write connection.

    Called only after every worker in the round has returned (all read-only
    connections closed), so taking the write lock here can't conflict."""
    import datetime
    results = [r for r in results if r is not None]
    if not results:
        return

    cell_rows = [r for res in results for r in res["cell"]]
    tile_rows = [r for res in results for r in res["tile"]]
    timing_rows = [r for res in results for r in res["timing"]]
    now = datetime.datetime.now(datetime.timezone.utc)
    cp_rows = []
    for res in results:
        for m in res["done"]:
            cp_rows.append((res["slide_id"], float(res["overlap_pct"]), m, "done", "", now))
        for m, err in res["errors"]:
            cp_rows.append((res["slide_id"], float(res["overlap_pct"]), m, "error", err[:500], now))

    with connect() as con:
        if cell_rows:
            con.register("_cc", pd.DataFrame(cell_rows))
            replace_into(con, "cell_counts", "_cc")
            con.unregister("_cc")
        if tile_rows:
            con.register("_tc", pd.DataFrame(tile_rows))
            replace_into(con, "tile_counts", "_tc")
            con.unregister("_tc")
        if timing_rows:
            con.register("_ti", pd.DataFrame(timing_rows))
            replace_into(con, "timing", "_ti")
            con.unregister("_ti")
        if cp_rows:
            con.executemany("INSERT OR REPLACE INTO checkpoint VALUES (?,?,?,?,?,?)", cp_rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all(
    n_workers: int = 1,
    slide_filter: str | None = None,
    batch_size: int = 200,  # accepted for CLI compatibility; rounds size = n_workers
) -> None:
    """Resumable full 76×9×31 NMS run.

    DuckDB allows EITHER one writer OR many readers across processes, never both.
    So workers read each (slide, overlap) slice read-only and return results; the
    main process writes between rounds, when no reader is open. Work is processed
    in rounds of `n_workers` (slide, overlap) pairs.

    n_workers=1  → single-process (also the most memory-frugal).
    n_workers>1  → ProcessPoolExecutor, round-based.

    Memory scales with n_workers: each worker loads its whole slice (a 50%-overlap
    tile can be ~1 GB). Keep n_workers moderate (≈4–6).
    """
    from rich.panel import Panel

    from core.db import _default_db
    from core.progress import console, pipeline_progress
    db_path = str(_default_db())

    with connect(read_only=True) as con:
        pairs = _get_pending_pairs(con, slide_filter)
        # Work universe mirrors _get_pending_pairs: slides × every config overlap
        # × every method (overlaps with no detections still run, fast, and are
        # checkpointed) — so `total` is comparable to `remaining`.
        if slide_filter:
            n_slides = 1
            done_count = con.execute(
                "SELECT COUNT(*) FROM checkpoint WHERE status='done' AND slide_id=?",
                [slide_filter]).fetchone()[0]
        else:
            n_slides = con.execute(
                "SELECT COUNT(DISTINCT slide_id) FROM slides").fetchone()[0]
            done_count = con.execute(
                "SELECT COUNT(*) FROM checkpoint WHERE status='done'").fetchone()[0]

    total_pairs = len(pairs)
    remaining = sum(len(m) for _, _, m in pairs)
    grand_total = n_slides * len(OVERLAP_PCTS) * len(NMS_REGISTRY)

    if not pairs:
        console.print("[bold green]✓ All combinations already processed.[/bold green]")
        return

    console.print(Panel.fit(
        f"[bold cyan]NMS Comparison Pipeline — Paper 1[/bold cyan]\n"
        f"Remaining: [bold]{remaining:,}[/bold] method-runs   "
        f"done: {done_count:,}   total: {grand_total:,}\n"
        f"Pairs to process: {total_pairs:,}   Workers: {n_workers}   "
        f"Methods/pair: {len(NMS_REGISTRY)}",
        border_style="blue",
    ))

    errors: list[tuple] = []

    def _record_errors(res: dict, progress) -> None:
        for method, err in res.get("errors", []):
            errors.append((res["slide_id"], res["overlap_pct"], method, err))
            progress.console.print(
                f"[red]  ✗ {res['slide_id']} · {res['overlap_pct']:.1%} · {method}[/red]")

    with pipeline_progress() as progress:
        task = progress.add_task("[cyan]Starting…[/cyan]", total=remaining)

        if n_workers == 1:
            for slide_id, overlap_pct, methods in pairs:
                progress.update(
                    task,
                    description=f"[dim]{slide_id}[/dim] [yellow]{overlap_pct:.1%}[/yellow]")

                # Advance one tick per method (smooth bar even on a heavy
                # 50%-overlap slice), showing the method currently finishing.
                def _tick(method: str, _sid=slide_id, _op=overlap_pct) -> None:
                    progress.update(
                        task, advance=1,
                        description=f"[dim]{_sid}[/dim] [yellow]{_op:.1%}[/yellow] "
                                    f"[green]{method}[/green]")

                res = run_pair(slide_id, overlap_pct, methods, db_path, on_method=_tick)
                _write_round([res])
                _record_errors(res, progress)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                for start in range(0, total_pairs, n_workers):
                    round_pairs = pairs[start:start + n_workers]
                    futs = {
                        pool.submit(run_pair, s, o, m, db_path): (s, o, len(m))
                        for s, o, m in round_pairs
                    }
                    results: list[dict] = []
                    for fut in as_completed(futs):
                        s, o, n_m = futs[fut]
                        progress.update(
                            task, description=f"[dim]{s}[/dim] [yellow]{o:.1%}[/yellow]")
                        try:
                            results.append(fut.result())
                        except Exception as exc:  # whole-pair failure → retried next run
                            errors.append((s, o, "<pair>", repr(exc)))
                            progress.console.print(
                                f"[red]  ✗ {s} · {o:.1%} (pair failed)[/red]")
                            progress.advance(task, n_m)
                    # Round complete → all read-only connections closed → safe to write.
                    _write_round(results)
                    for res in results:
                        _record_errors(res, progress)
                        progress.advance(task, len(res["done"]) + len(res["errors"]))

    if errors:
        console.print(f"\n[bold red]✗ Completed with {len(errors)} error(s):[/bold red]")
        for sid, op, m, err in errors[:20]:
            console.print(f"  [red]{sid} {op:.1%} {m}: {str(err)[:160]}[/red]")
        if len(errors) > 20:
            console.print(f"  [red]… and {len(errors) - 20} more[/red]")
    else:
        console.print("\n[bold green]✓ Pipeline complete — no errors.[/bold green]")
