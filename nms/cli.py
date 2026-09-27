"""CLI commands (registered as `balc nms ...`)."""
from __future__ import annotations

import logging
import time

import typer

from config import RESULTS_DIR, OVERLAP_PCTS, EQUIV_MARGIN_ALR
from core.blocks import ALL_BLOCK_SIZES
from core.progress import console, pipeline_progress

logger = logging.getLogger(__name__)
app = typer.Typer(help="NMS benchmark via inter-method consensus")


@app.command("run")
def cmd_run(
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel workers"),
    slide: str = typer.Option(None, "--slide", help="Run only this slide_id"),
) -> None:
    """Resumable 76x9x31 NMS run -> cell_counts, tile_counts, timing."""
    from nms.process import run_all
    run_all(n_workers=workers, slide_filter=slide)


@app.command("blocks")
def cmd_blocks(
    overlap: float = typer.Option(None, "--overlap", help="Run for one overlap value only"),
) -> None:
    """Aggregate tile_counts -> block_counts for all block sizes."""
    from nms.consensus_analysis import build_block_counts
    total = 1 if overlap is not None else len(OVERLAP_PCTS)
    with pipeline_progress() as progress:
        task = progress.add_task("[cyan]Block counts (per overlap)[/cyan]", total=total)
        build_block_counts(overlap_pct=overlap, tick=lambda: progress.advance(task))
    console.print("[green]✓[/green] Block counts built.")


@app.command("consensus")
def cmd_consensus(
    overlap: float = typer.Option(None, "--overlap", help="Compute for one overlap only"),
) -> None:
    """Compute geometric-median consensus, method deviations, and Krippendorff alpha."""
    from nms.consensus_analysis import (
        compute_consensus_per_block,
        compute_method_deviations,
        scale_dependence_curve,
    )

    overlaps = [overlap] if overlap is not None else OVERLAP_PCTS
    with pipeline_progress() as progress:
        task = progress.add_task(
            "[cyan]Consensus + deviations[/cyan]",
            total=len(overlaps) * len(ALL_BLOCK_SIZES),
        )
        for op in overlaps:
            for bs in ALL_BLOCK_SIZES:
                progress.update(task, description=f"[cyan]consensus[/cyan] [yellow]bs={bs} ov={op:.1%}[/yellow]")
                compute_consensus_per_block(bs, op)
                compute_method_deviations(bs, op)
                progress.advance(task)

    n_curve = len(ALL_BLOCK_SIZES) * len(OVERLAP_PCTS)
    with pipeline_progress() as progress:
        task = progress.add_task("[cyan]Scale-dependence α[/cyan]", total=n_curve)
        curve = scale_dependence_curve(tick=lambda: progress.advance(task))
    out = RESULTS_DIR / "scale_dependence_curve.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out, index=False)
    console.print(f"[green]✓[/green] Saved {out}")

    from nms.zero_sensitivity import zero_sensitivity_curve
    with pipeline_progress() as progress:
        task = progress.add_task("[cyan]Zero-replacement sensitivity α[/cyan]", total=n_curve)
        zs = zero_sensitivity_curve(tick=lambda: progress.advance(task))
    zs_out = RESULTS_DIR / "zero_sensitivity_curve.csv"
    zs.to_csv(zs_out, index=False)
    if not zs.empty and zs["gap"].notna().any():
        worst = zs.loc[zs["gap"].abs().idxmax()]
        console.print(
            f"[green]✓[/green] Saved {zs_out} — largest α artefact: gap={worst['gap']:.3f} "
            f"(as_is={worst['alpha_as_is']:.3f} → common_scale="
            f"{worst['alpha_common_scale']:.3f}) at block_size={int(worst['block_size'])}, "
            f"overlap={worst['overlap_pct']:.3f}"
        )
    else:
        console.print(f"[green]✓[/green] Saved {zs_out}")


@app.command("outliers")
def cmd_outliers() -> None:
    """Identify statistical outlier methods via Mahalanobis distance."""
    from nms.outlier_detection import outlier_summary
    with pipeline_progress() as progress:
        task = progress.add_task(
            "[cyan]Outlier scan (per setting)[/cyan]",
            total=len(ALL_BLOCK_SIZES) * len(OVERLAP_PCTS),
        )
        df = outlier_summary(tick=lambda: progress.advance(task))
    out = RESULTS_DIR / "outlier_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"[green]✓[/green] Saved {out}")
    flagged = (
        df[df["frac_settings_flagged"] > 0.05]["nms_method"].tolist()
        if not df.empty else []
    )
    if flagged:
        console.print(f"[yellow]Flagged outlier methods:[/yellow] {flagged}")
    else:
        console.print("No methods flagged as outliers.")


@app.command("weighted-summary")
def cmd_weighted(
    overlap: float = typer.Option(0.0, "--overlap", help="Overlap fraction to summarise"),
) -> None:
    """Count-weighted slide-level ILR mean per method."""
    from nms.weighted_summary import weighted_method_means
    with console.status("[cyan]Computing count-weighted means…[/cyan]"):
        df = weighted_method_means(overlap)
    out = RESULTS_DIR / "weighted_method_means.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"[green]✓[/green] Saved {out}")


@app.command("baseline")
def cmd_baseline(
    overlap: float = typer.Option(0.0, "--overlap", help="Overlap fraction to test"),
    reference: str = typer.Option(
        "iou_grid_n1", "--reference",
        help="A-priori standard-NMS reference for the focused no-NMS-equivalence verdict"),
) -> None:
    """Explicit no-NMS-vs-NMS thesis: Friedman across methods + equivalence vs 'none'."""
    from nms.baseline_comparison import baseline_summary
    with console.status("[cyan]Baseline: Friedman + bootstrap equivalence vs 'none'…[/cyan]"):
        res = baseline_summary(overlap, reference_method=reference)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res["friedman"].to_csv(RESULTS_DIR / "baseline_friedman.csv", index=False)
    res["equivalence"].to_csv(RESULTS_DIR / "baseline_equivalence.csv", index=False)

    # Headline: focused, a-priori no-NMS vs standard reference (consistency, not accuracy).
    if res["reference_equivalent"]:
        console.print(
            f"[green]✓ HEADLINE @ overlap {overlap}:[/green] no-NMS is EQUIVALENT to "
            f"'{reference}' for all labels (±{EQUIV_MARGIN_ALR} ALR) — the differential "
            f"is robust to running standard NMS vs none.")
    else:
        ref_tbl = res["reference_equivalence"]
        bad = (ref_tbl[~ref_tbl["equivalent"]]["label"].tolist()
               if not ref_tbl.empty else [f"<'{reference}' not found>"])
        console.print(
            f"[yellow]✗ HEADLINE @ overlap {overlap}:[/yellow] no-NMS NOT equivalent to "
            f"'{reference}' for: {bad}")

    # Context: strict all-27-methods verdict.
    verdict = ("EVERY NMS method is equivalent to 'none'"
               if res["all_equivalent"]
               else "some methods differ from 'none' — see baseline_equivalence.csv")
    console.print(f"[dim]Context (all methods vs none): {verdict}[/dim]")


@app.command("timing")
def cmd_timing(
    n_slides: int = typer.Option(5, "--n-slides", help="slides to sample (× all overlaps)"),
    repeats: int = typer.Option(3, "--repeats", help="timed repeats per method"),
) -> None:
    """Controlled single-process timing of all 31 variants → timing_clean.

    Use this — not the parallel-run `timing` table — for reported time savings.
    For a pristine measurement set OMP_NUM_THREADS=1 etc. before launching.
    """
    from nms.timing import run_timing, timing_summary
    n = run_timing(n_slides=n_slides, repeats=repeats)
    if not n:
        typer.echo("No timing rows written (load data first).")
        return
    summary = timing_summary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "timing_clean_summary.csv", index=False)
    typer.echo(summary.to_string())


@app.command("method-similarity")
def cmd_method_similarity(
    overlap: float = typer.Option(0.0, "--overlap"),
) -> None:
    """Method-distance matrix + dendrogram (context for Krippendorff's α)."""
    from nms.method_similarity import method_distance_matrix, plot_dendrogram
    with console.status("[cyan]Computing method-distance matrix…[/cyan]"):
        dm = method_distance_matrix(overlap)
    if dm.empty:
        console.print("[red]No cell_counts — run `nms run` first.[/red]")
        raise typer.Exit(1)
    (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    dm.to_csv(RESULTS_DIR / "method_distance_matrix.csv")
    plot_dendrogram(dm, RESULTS_DIR / "plots" / "method_dendrogram.png")
    console.print(f"[green]✓[/green] Saved method-distance matrix + dendrogram → {RESULTS_DIR}")


@app.command("plots")
def cmd_plots(
    overlap: float = typer.Option(0.0, "--overlap"),
) -> None:
    """Generate all figures."""
    from nms.consensus_analysis import scale_dependence_curve
    from nms.outlier_detection import outlier_summary
    from nms.weighted_summary import weighted_method_means
    from nms.plots import make_all_plots

    with console.status("[cyan]Generating figures…[/cyan]"):
        curve = scale_dependence_curve()
        wmeans = weighted_method_means(overlap)
        out_df = outlier_summary()
        make_all_plots(curve, wmeans, out_df)
    console.print("[green]✓[/green] All plots saved.")


@app.command("benchmark")
def cmd_benchmark(
    slide: str = typer.Argument(..., help="slide_id to benchmark"),
    overlap: float = typer.Option(0.0, "--overlap"),
) -> None:
    """Time all 31 NMS variants on one slide."""
    import time
    from core.db import connect
    from core.nms import NMS_REGISTRY
    from config import CONF_THRESH

    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT * FROM raw_detections WHERE slide_id=? AND overlap_pct=? AND confidence>=?",
            [slide, float(overlap), CONF_THRESH],
        ).fetchdf()

    if df.empty:
        typer.echo(f"No detections found for slide={slide} overlap={overlap}")
        raise typer.Exit(1)

    typer.echo(f"Benchmarking {len(NMS_REGISTRY)} methods on {len(df)} detections…")
    results = []
    for name, fn in sorted(NMS_REGISTRY.items()):
        t0 = time.process_time()
        keep = fn(df)
        elapsed = time.process_time() - t0
        results.append((name, elapsed, int(keep.sum())))
        typer.echo(f"  {name:<35} {elapsed*1000:7.1f} ms -> {int(keep.sum()):6d} survivors")

    results.sort(key=lambda x: x[1])
    typer.echo(f"\nFastest: {results[0][0]} ({results[0][1]*1000:.1f} ms)")
    typer.echo(f"Slowest: {results[-1][0]} ({results[-1][1]*1000:.1f} ms)")


@app.command("all")
def cmd_all(
    workers: int = typer.Option(1, "--workers"),
) -> None:
    """Run full pipeline: run -> blocks -> consensus -> outliers -> weighted-summary -> plots."""
    from rich.table import Table

    # Typer's @app.command returns the undecorated function, so each command is a
    # plain callable. Invoke them directly with explicit args (no Click context
    # needed — typer.get_current_context() does not exist). A console.rule banner +
    # per-stage timing gives a clear timeline without a persistent Live display that
    # would clash with each step's own progress bar.
    stages = [
        ("run", lambda: cmd_run(workers=workers, slide=None)),
        ("blocks", lambda: cmd_blocks(overlap=None)),
        ("consensus", lambda: cmd_consensus(overlap=None)),
        ("outliers", cmd_outliers),
        ("weighted-summary", lambda: cmd_weighted(overlap=0.0)),
        ("baseline", lambda: cmd_baseline(overlap=0.0)),
        ("method-similarity", lambda: cmd_method_similarity(overlap=0.0)),
        ("plots", lambda: cmd_plots(overlap=0.0)),
    ]
    results: list[tuple[str, str, float]] = []
    t_pipeline = time.perf_counter()
    for i, (name, fn) in enumerate(stages, 1):
        console.rule(f"[bold cyan]Step {i}/{len(stages)} · {name}[/bold cyan]")
        t0 = time.perf_counter()
        try:
            fn()
            status = "[green]ok[/green]"
        except (typer.Exit, Exception) as exc:  # noqa: BLE001 — record and keep going
            status = "[red]failed[/red]"
            console.print(f"[red]✗ {name} failed:[/red] {exc!r}")
        results.append((name, status, time.perf_counter() - t0))

    total = time.perf_counter() - t_pipeline
    table = Table(title="Pipeline — summary", header_style="bold")
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Elapsed", justify="right")
    for name, status, elapsed in results:
        table.add_row(name, status, f"{elapsed:.1f}s")
    table.add_row("[bold]total[/bold]", "", f"[bold]{total:.1f}s[/bold]")
    console.print(table)
