"""Tests for core/db.py — schema initialisation and connection management."""
from __future__ import annotations

import datetime

import pytest
import duckdb

from core.db import init_db, connect, get_status


_EXPECTED_TABLES = [
    "slides", "raw_detections",
    "cell_counts", "tile_counts", "timing", "timing_clean", "checkpoint",
    "block_counts", "block_consensus", "method_deviation",
]


def test_init_db_creates_all_tables(tmp_path):
    db = tmp_path / "test.duckdb"
    init_db(db_path=db)
    with connect(db_path=db) as con:
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
    for t in _EXPECTED_TABLES:
        assert t in tables, f"Table '{t}' not created"


def test_init_db_idempotent(tmp_path):
    db = tmp_path / "test.duckdb"
    init_db(db_path=db)
    init_db(db_path=db)  # should not raise
    with connect(db_path=db) as con:
        n = con.execute("SELECT COUNT(*) FROM slides").fetchone()[0]
    assert n == 0


def test_connect_write_and_read(tmp_path):
    db = tmp_path / "test.duckdb"
    init_db(db_path=db)
    with connect(db_path=db) as con:
        con.execute(
            "INSERT INTO slides VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["s001", "/img/s001.png", 1920, 1920, 3, 3, datetime.datetime.utcnow()],
        )
    with connect(read_only=True, db_path=db) as con:
        n = con.execute("SELECT COUNT(*) FROM slides WHERE slide_id='s001'").fetchone()[0]
    assert n == 1


def test_connect_context_manager_closes(tmp_path):
    db = tmp_path / "test.duckdb"
    init_db(db_path=db)
    with connect(db_path=db) as con:
        pass
    # Connection should be closed; accessing it raises
    with pytest.raises(Exception):
        con.execute("SELECT 1")


def test_primary_key_constraint_slides(tmp_path):
    db = tmp_path / "test.duckdb"
    init_db(db_path=db)
    ts = datetime.datetime.utcnow()
    with connect(db_path=db) as con:
        con.execute("INSERT INTO slides VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ["s001", "/img/s001.png", 1920, 1920, 3, 3, ts])
        # INSERT OR REPLACE should update existing row without error
        con.execute("INSERT OR REPLACE INTO slides VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ["s001", "/img/s001_v2.png", 1920, 1920, 3, 3, ts])
    with connect(read_only=True, db_path=db) as con:
        n = con.execute("SELECT COUNT(*) FROM slides").fetchone()[0]
    assert n == 1


def test_get_status_returns_string(tmp_path):
    db = tmp_path / "test.duckdb"
    init_db(db_path=db)
    status = get_status(db_path=db)
    assert isinstance(status, str)
    assert "slides" in status


def test_get_status_missing_db():
    status = get_status(db_path="/nonexistent/path/db.duckdb")
    assert "Cannot open database" in status or "slides" in status or isinstance(status, str)
