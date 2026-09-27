"""Shared rich console + a single progress style for every long-running command.

Inspired by the 2dNMS pipeline's look: a coloured Panel header, then a Progress
with spinner · description · bar · M/N · % · elapsed · ETA. Keeping it in one
place means `balc load` and `balc p1 run` (and anything else) look identical.

Off a real terminal (tests, piped output) rich degrades gracefully to occasional
plain updates, so this is safe to use unconditionally.
"""
from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

# One shared console for the whole app.
console = Console()


class RateColumn(ProgressColumn):
    """Throughput as completed items per second (method-runs/s, files/s, …)."""

    def render(self, task) -> Text:
        speed = task.finished_speed or task.speed
        if not speed:
            return Text("–/s", style="progress.data.speed")
        return Text(f"{speed:,.1f}/s", style="progress.data.speed")


def pipeline_progress(transient: bool = False, disable: bool = False) -> Progress:
    """A Progress laid out as:
    ⠋ <description> ━━━━ 123/456 · 27% · 12.3/s · 0:00:12 · 0:00:34.

    `disable=True` silences the live display (for programmatic / non-interactive use).
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),               # expand to fill the terminal width
        MofNCompleteColumn(),
        TaskProgressColumn(),                    # percentage
        RateColumn(),                            # throughput (items/s)
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
        disable=disable,
        refresh_per_second=10,
    )
