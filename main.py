"""Top-level CLI entry point (`balc`)."""
import sys

# Force UTF-8 on the console streams. On Windows a redirected/piped stdout defaults
# to cp1252, which can't encode the rich output / status glyphs (✓, …, box-drawing)
# and raises UnicodeEncodeError. reconfigure() mutates the existing stream in place,
# so the shared rich Console (created on import) picks it up. Safe no-op on Mac.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import typer

from core.db import init_db, get_status
from core.loader import load_all

import nms.cli as nms_cli

app = typer.Typer(help="BAL cytospin NMS robustness analysis CLI")

app.add_typer(nms_cli.app, name="nms")


@app.command("init-db")
def cmd_init_db() -> None:
    """Initialise DuckDB schema (idempotent)."""
    init_db()
    typer.echo("Database schema initialised.")


@app.command("load")
def cmd_load(
    slide: str = typer.Option(None, help="Load only this slide_id"),
) -> None:
    """Load CSV detections into DuckDB (idempotent)."""
    load_all(slide_filter=slide)


@app.command("status")
def cmd_status() -> None:
    """Print row counts of the result tables."""
    typer.echo(get_status())


if __name__ == "__main__":
    app()
