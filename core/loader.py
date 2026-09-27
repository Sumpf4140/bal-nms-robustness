"""Load raw detection CSVs into DuckDB.

Enforces:
- Required column presence
- Tile-grid integrity (≥99 % of x_correct, y_correct divisible by TILE_WIDTH_PX)
- Image-bounds check (max bbox coordinate fits inside slide image)
- Idempotent: skips (slide_id, overlap_pct) already in raw_detections
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, IMAGES_DIR, LABELS, OVERLAP_PCTS,
    TILE_WIDTH_PX, TILE_HEIGHT_PX, TILE_GRID_TOLERANCE,
)
from core.db import connect, init_db
from core.slide_image import find_slide_images, get_image_dimensions, tile_grid_shape

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"label", "confidence", "x1", "y1", "x2", "y2", "x_correct", "y_correct"}

# Column order written to raw_detections (must match the table schema in core/db.py)
_RAW_COLS = [
    "slide_id", "overlap_pct", "label", "confidence",
    "x1", "y1", "x2", "y2", "cx", "cy",
    "x_correct", "y_correct", "tile_x", "tile_y",
]

def _csv_path(data_dir: Path, slide_id: str, overlap_pct: float) -> Path:
    return data_dir / f"{_overlap_dirname(overlap_pct)}_overlap_raw" / f"{slide_id}.csv"


def _tile_stride(overlap_pct: float) -> tuple[int, int]:
    """Pixel stride between tile origins for a given overlap fraction."""
    sx = round(TILE_WIDTH_PX * (1.0 - overlap_pct))
    sy = round(TILE_HEIGHT_PX * (1.0 - overlap_pct))
    return max(sx, 1), max(sy, 1)


def _load_csv(path: Path, overlap_pct: float = 0.0) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    df = df[df["label"].isin(LABELS)].copy()

    # CSV boxes are TILE-RELATIVE (x1..y2 ∈ [0, tile]); x_correct/y_correct is the
    # tile's absolute origin. Convert boxes to slide-absolute so that cx/cy AND the
    # IoU boxes are absolute. Without this, every cross-tile NMS scope (global,
    # grid n≥2, iou grid) deduplicates on tile-relative coordinates and is wrong —
    # detections in different tiles get compared by their within-tile position.
    # (per_tile / grid_n1 scopes are unaffected: the offset is uniform per tile.)
    if len(df) and float(df[["x1", "y1", "x2", "y2"]].to_numpy().max()) > TILE_WIDTH_PX * 1.5:
        raise ValueError(
            f"{path}: bounding boxes exceed {TILE_WIDTH_PX}px — input looks already "
            "slide-absolute, not tile-relative. Refusing to add x_correct/y_correct "
            "(would double-offset)."
        )
    df["x1"] = df["x1"] + df["x_correct"]
    df["x2"] = df["x2"] + df["x_correct"]
    df["y1"] = df["y1"] + df["y_correct"]
    df["y2"] = df["y2"] + df["y_correct"]

    # Derived columns (now slide-absolute)
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    sx, sy = _tile_stride(overlap_pct)
    df["tile_x"] = df["x_correct"] // sx
    df["tile_y"] = df["y_correct"] // sy
    return df


def _check_tile_grid(df: pd.DataFrame, path: Path, overlap_pct: float = 0.0) -> None:
    if len(df) == 0:
        return
    sx, sy = _tile_stride(overlap_pct)
    ok_x = (df["x_correct"] % sx == 0).mean()
    ok_y = (df["y_correct"] % sy == 0).mean()
    if ok_x < TILE_GRID_TOLERANCE or ok_y < TILE_GRID_TOLERANCE:
        raise ValueError(
            f"Tile-grid integrity failed for {path}: "
            f"x_correct divisible={ok_x:.3f}, y_correct divisible={ok_y:.3f} "
            f"(threshold {TILE_GRID_TOLERANCE})"
        )


def _check_image_bounds(df: pd.DataFrame, image_w: int, image_h: int, path: Path) -> None:
    if len(df) == 0:
        return
    max_x = df[["x1", "x2"]].to_numpy().max()
    max_y = df[["y1", "y2"]].to_numpy().max()
    # Absolute boxes in edge/partial tiles can legitimately extend up to one tile
    # beyond the image (slide dims are not multiples of the tile size), so allow a
    # one-tile tolerance before flagging a box as out of bounds.
    if max_x > image_w + TILE_WIDTH_PX or max_y > image_h + TILE_HEIGHT_PX:
        raise ValueError(
            f"Bbox exceeds image bounds in {path}: "
            f"max_x={max_x} > image_w={image_w} or max_y={max_y} > image_h={image_h}"
        )


def _already_loaded(con, slide_id: str, overlap_pct: float) -> bool:
    n = con.execute(
        "SELECT COUNT(*) FROM raw_detections WHERE slide_id=? AND overlap_pct=?",
        [slide_id, overlap_pct],
    ).fetchone()[0]
    return n > 0


def _overlap_dirname(overlap_pct: float) -> str:
    """Convert fraction to data directory prefix: 0.05 → '5', 0.025 → '2.5'."""
    pct = round(overlap_pct * 100, 6)
    return str(int(pct)) if pct == int(pct) else str(pct)


def _upsert_slide(con, slide_id: str, image_path: Path) -> tuple[int, int]:
    """Upsert slide metadata; returns (image_w, image_h)."""
    image_w, image_h = get_image_dimensions(image_path)
    n_tiles_x, n_tiles_y = tile_grid_shape(image_w, image_h)
    con.execute(
        "INSERT OR REPLACE INTO slides VALUES (?, ?, ?, ?, ?, ?, ?)",
        [slide_id, str(image_path), image_w, image_h, n_tiles_x, n_tiles_y,
         datetime.datetime.now(datetime.timezone.utc)],
    )
    return image_w, image_h


def _load_one_overlap(con, slide_id: str, overlap_pct: float, image_w: int,
                      image_h: int, data_dir: Path) -> int:
    """Load one (slide, overlap) CSV. Returns rows inserted, or -1 if already loaded."""
    if _already_loaded(con, slide_id, overlap_pct):
        return -1
    csv_path = _csv_path(data_dir, slide_id, overlap_pct)
    df = _load_csv(csv_path, overlap_pct)
    _check_tile_grid(df, csv_path, overlap_pct)
    _check_image_bounds(df, image_w, image_h, csv_path)
    df["slide_id"] = slide_id
    df["overlap_pct"] = float(overlap_pct)
    con.register("_tmp_df", df[_RAW_COLS])
    con.execute("INSERT INTO raw_detections SELECT * FROM _tmp_df")
    con.unregister("_tmp_df")
    return len(df)


def load_slide(slide_id: str, image_path: Path, data_dir: Path = DATA_DIR) -> dict:
    """Load all overlap CSVs for one slide into DuckDB (no progress bar).

    Returns a per-slide summary {rows, loaded, skipped, missing}.
    """
    init_db()
    summary = {"rows": 0, "loaded": 0, "skipped": 0, "missing": 0}
    with connect() as con:
        image_w, image_h = _upsert_slide(con, slide_id, image_path)
        for overlap_pct in OVERLAP_PCTS:
            if not _csv_path(data_dir, slide_id, overlap_pct).exists():
                summary["missing"] += 1
                continue
            n = _load_one_overlap(con, slide_id, overlap_pct, image_w, image_h, data_dir)
            if n < 0:
                summary["skipped"] += 1
            else:
                summary["rows"] += n
                summary["loaded"] += 1
    return summary


def load_all(slide_filter: str | None = None, images_dir: Path = IMAGES_DIR,
             data_dir: Path = DATA_DIR, *, progress: bool = True) -> dict:
    """Load all slides found in images_dir (idempotent). Returns run totals.

    Drives a single rich progress bar over every (slide, overlap) CSV file, so
    the bar advances smoothly per file with an accurate ETA.
    """
    from rich.panel import Panel

    from core.progress import console, pipeline_progress

    init_db()
    slides = find_slide_images(images_dir)
    if slide_filter:
        slides = {k: v for k, v in slides.items() if k == slide_filter}
        if not slides:
            raise FileNotFoundError(f"Slide '{slide_filter}' not found in {images_dir}")
    if not slides:
        console.print(f"[yellow]No slide images found in {images_dir} — nothing to load.[/yellow]")
        return {"slides": 0, "files": 0, "rows": 0, "ok": 0, "skipped": 0, "failed": 0}

    # One job per existing (slide, overlap) CSV → a smooth file-level bar.
    jobs = [
        (sid, img, ov)
        for sid, img in slides.items()
        for ov in OVERLAP_PCTS
        if _csv_path(data_dir, sid, ov).exists()
    ]
    if not jobs:
        console.print(f"[yellow]Found {len(slides)} slides but no CSVs under {data_dir}.[/yellow]")
        return {"slides": len(slides), "files": 0, "rows": 0, "ok": 0, "skipped": 0, "failed": 0}

    console.print(Panel.fit(
        f"[bold cyan]Loading detections — BAL[/bold cyan]\n"
        f"Slides: [bold]{len(slides):,}[/bold]   CSV files: [bold]{len(jobs):,}[/bold]   "
        f"Overlaps: {len(OVERLAP_PCTS)}",
        border_style="blue",
    ))

    totals = {"rows": 0, "ok": 0, "skipped": 0, "failed": 0}
    seen: dict[str, tuple[int, int]] = {}

    with connect() as con, pipeline_progress(disable=not progress) as progress:
        task = progress.add_task("[cyan]Loading…[/cyan]", total=len(jobs))
        for slide_id, image_path, overlap_pct in jobs:
            progress.update(
                task,
                description=f"[dim]{slide_id}[/dim] [yellow]{overlap_pct:.1%}[/yellow]")
            try:
                if slide_id not in seen:
                    seen[slide_id] = _upsert_slide(con, slide_id, image_path)
                image_w, image_h = seen[slide_id]
                n = _load_one_overlap(con, slide_id, overlap_pct, image_w, image_h, data_dir)
                if n < 0:
                    totals["skipped"] += 1
                else:
                    totals["rows"] += n
                    totals["ok"] += 1
            except Exception as exc:
                totals["failed"] += 1
                progress.console.print(f"[red]  ✗ {slide_id} · {overlap_pct:.1%}: {exc}[/red]")
            progress.advance(task)

    console.print(
        f"\n[bold green]✓ Loaded {totals['rows']:,} detections[/bold green] "
        f"— {totals['ok']} files, {totals['skipped']} already present, "
        f"{totals['failed']} failed across {len(seen)} slides."
    )
    totals.update(slides=len(slides), files=len(jobs))
    return totals
