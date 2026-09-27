"""Export the slide-level cell counts (results/slide_counts.csv) from the DB.

One row per (slide, tile overlap, post-processing variant, cell type) with the
number of retained detections. This is the only input needed to reproduce every
slide-level (whole-slide) result of the study; see
scripts/reproduce_from_slide_counts.py.

Slide identifiers are replaced by pseudonyms S01..S76. The pseudonyms are
assigned in the sort order of the original identifiers so that the row order
seen by the fixed-seed bootstrap is unchanged and published intervals reproduce
exactly. The mapping itself is not exported.

Run:  python scripts/export_slide_counts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import connect

OUT = ROOT / "results" / "slide_counts.csv"


def main() -> None:
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT slide_id, overlap_pct, nms_method, label, count "
            "FROM cell_counts ORDER BY slide_id, overlap_pct, nms_method, label"
        ).fetchdf()
    slides = sorted(df["slide_id"].unique())
    width = len(str(len(slides)))
    pseudo = {s: f"S{i:0{width}d}" for i, s in enumerate(slides, start=1)}
    df["slide_id"] = df["slide_id"].map(pseudo)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df):,} rows, {len(slides)} slides)")


if __name__ == "__main__":
    main()
