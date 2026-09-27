"""DuckDB connection management and schema initialisation.

Single-writer rule: only the main process writes; workers open read-only.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Generator

import duckdb

import config as _config

# Expose as module attribute so tests can monkeypatch it
DB_PATH = _config.DB_PATH

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS slides (
    slide_id   VARCHAR PRIMARY KEY,
    image_path VARCHAR NOT NULL,
    image_w    INT NOT NULL,
    image_h    INT NOT NULL,
    n_tiles_x  INT NOT NULL,
    n_tiles_y  INT NOT NULL,
    loaded_at  TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_detections (
    slide_id    VARCHAR NOT NULL,
    overlap_pct DOUBLE  NOT NULL,
    label       VARCHAR NOT NULL,
    confidence  DOUBLE  NOT NULL,
    x1 INT, y1 INT, x2 INT, y2 INT,
    cx DOUBLE, cy DOUBLE,
    x_correct INT, y_correct INT,
    tile_x INT, tile_y INT
);
CREATE INDEX IF NOT EXISTS idx_raw_slide_overlap
    ON raw_detections(slide_id, overlap_pct);

-- Paper 1: NMS output
CREATE TABLE IF NOT EXISTS cell_counts (
    slide_id VARCHAR, overlap_pct DOUBLE, nms_method VARCHAR,
    label VARCHAR, count INT, rel_count DOUBLE,
    PRIMARY KEY (slide_id, overlap_pct, nms_method, label)
);
CREATE TABLE IF NOT EXISTS tile_counts (
    slide_id VARCHAR, overlap_pct DOUBLE, nms_method VARCHAR,
    tile_x INT, tile_y INT, label VARCHAR, count INT,
    PRIMARY KEY (slide_id, overlap_pct, nms_method, tile_x, tile_y, label)
);
CREATE TABLE IF NOT EXISTS timing (
    slide_id VARCHAR, overlap_pct DOUBLE, nms_method VARCHAR,
    elapsed_cpu DOUBLE, n_input INT, n_output INT,
    PRIMARY KEY (slide_id, overlap_pct, nms_method)
);
-- Controlled single-process timing on a representative batch (paper1_nms/timing.py).
-- Separate from `timing`, which is collected during the parallel run and is
-- distorted by --workers CPU contention. Use this for any reported time-save/cost.
CREATE TABLE IF NOT EXISTS timing_clean (
    slide_id VARCHAR, overlap_pct DOUBLE, nms_method VARCHAR, repeat INT,
    n_detections INT, elapsed_cpu DOUBLE, elapsed_wall DOUBLE,
    PRIMARY KEY (slide_id, overlap_pct, nms_method, repeat)
);
CREATE TABLE IF NOT EXISTS checkpoint (
    slide_id VARCHAR, overlap_pct DOUBLE, nms_method VARCHAR,
    status VARCHAR, error_msg VARCHAR, finished_at TIMESTAMP,
    PRIMARY KEY (slide_id, overlap_pct, nms_method)
);

-- Paper 1: Block-level aggregates and consensus
CREATE TABLE IF NOT EXISTS block_counts (
    slide_id VARCHAR, overlap_pct DOUBLE, nms_method VARCHAR,
    block_size INT,
    block_x INT, block_y INT,
    label VARCHAR,
    count INT,
    n_block_total INT,
    PRIMARY KEY (slide_id, overlap_pct, nms_method, block_size, block_x, block_y, label)
);
CREATE TABLE IF NOT EXISTS block_consensus (
    slide_id VARCHAR, overlap_pct DOUBLE, block_size INT,
    block_x INT, block_y INT,
    consensus_alr_Mac  DOUBLE,
    consensus_alr_Neu  DOUBLE,
    consensus_alr_Eos  DOUBLE,
    n_methods_in_consensus INT,
    n_block_total INT,
    PRIMARY KEY (slide_id, overlap_pct, block_size, block_x, block_y)
);
CREATE TABLE IF NOT EXISTS method_deviation (
    slide_id VARCHAR, overlap_pct DOUBLE, nms_method VARCHAR, block_size INT,
    block_x INT, block_y INT,
    aitchison_distance DOUBLE,
    mahalanobis_distance DOUBLE,
    PRIMARY KEY (slide_id, overlap_pct, nms_method, block_size, block_x, block_y)
);
"""


def _default_db() -> "Path":
    """Runtime lookup of DB_PATH — respects monkeypatching."""
    import sys
    return sys.modules[__name__].DB_PATH


def init_db(db_path=None) -> None:
    """Create all tables (idempotent)."""
    if db_path is None:
        db_path = _default_db()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for stmt in _SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                con.execute(stmt)


@contextlib.contextmanager
def connect(read_only: bool = False, db_path=None) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield an open DuckDB connection, closed on exit."""
    if db_path is None:
        db_path = _default_db()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def replace_into(con, table: str, view: str) -> None:
    """Upsert a registered relation `view` into `table`, PK-agnostically.

    A standard DB has a PRIMARY KEY on every results table → use INSERT OR REPLACE
    (idempotent, resume-safe). The memory-rebuilt local DB recreates the big tables
    (tile_counts, block_counts, …) WITHOUT their PK to avoid the ART-index OOM on a
    16 GB box; INSERT OR REPLACE errors there ("no UNIQUE/PRIMARY KEY constraints"),
    so fall back to a plain INSERT. The no-PK path is NOT resume-safe — it assumes
    the table is rebuilt from empty (a re-run would duplicate rows).
    """
    has_pk = con.execute(
        "SELECT COUNT(*) FROM duckdb_constraints() "
        "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
        [table],
    ).fetchone()[0] > 0
    verb = "INSERT OR REPLACE INTO" if has_pk else "INSERT INTO"
    con.execute(f"{verb} {table} SELECT * FROM {view}")


def get_status(db_path=None) -> str:
    """Return a human-readable progress summary."""
    lines: list[str] = []
    try:
        with connect(read_only=True, db_path=db_path) as con:
            for table in [
                "slides", "raw_detections", "cell_counts", "tile_counts",
                "timing", "timing_clean",
                "block_counts", "block_consensus", "method_deviation",
                "checkpoint",
            ]:
                try:
                    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    lines.append(f"  {table:<30} {n:>10,} rows")
                except Exception:
                    lines.append(f"  {table:<30} {'(missing)':>10}")
    except Exception as exc:
        return f"Cannot open database: {exc}"
    return "Database status:\n" + "\n".join(lines)
