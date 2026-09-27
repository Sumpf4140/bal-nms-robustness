"""Block-level analysis that produced the published agreement, outlier and
equivalence results.

Starting from a database in which `balc nms run` has filled tile_counts and
cell_counts for all 31 variants, this script

  1. aggregates tile counts to all analysis scales (block_counts, 31 variants),
  2. restricts the method set to No-NMS + the 24 deduplication variants
     (edge-crop variants are excluded from the consensus),
  3. computes the per-block consensus and leave-one-out deviations,
  4. writes results/scale_dependence_curve.csv (Table 2, Table S5) and
     results/outlier_summary.csv (Table 3),
  5. writes results/equivalence_all_overlaps.csv (Table 1, Table S4; all 30
     variants versus No-NMS at every overlap).

The consensus/deviation tables are recreated without primary keys: with ~2x10^8
block rows the primary-key index does not fit in 16 GB of RAM. Every analysis
function called is the standard one from nms.

Run:  python scripts/reproduce_analysis.py
"""
import contextlib
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.db as dbm
import core.nms as core_nms
from core.db import _default_db

DBP = Path(_default_db())
_orig = dbm.connect


@contextlib.contextmanager
def _tuned(read_only=False, db_path=None):
    with _orig(read_only=read_only, db_path=db_path) as con:
        try:
            con.execute("SET preserve_insertion_order=false")
            con.execute("SET memory_limit='9GB'")
            con.execute(f"SET temp_directory='{DBP.parent.as_posix()}'")
        except Exception:
            pass
        yield con


dbm.connect = _tuned
import nms.consensus_analysis as ca  # noqa: E402

ca.connect = _tuned

from config import OVERLAP_PCTS, RESULTS_DIR  # noqa: E402
from core.blocks import ALL_BLOCK_SIZES  # noqa: E402

t0 = time.time()


def el():
    return f"[{(time.time() - t0) / 60:5.1f}m]"


# 1. block_counts from tile_counts (31 variants)
with _orig() as con:
    con.execute("DELETE FROM block_counts")
ca.build_block_counts()
with _orig(read_only=True) as con:
    nbc = con.execute("SELECT COUNT(DISTINCT nms_method) FROM block_counts").fetchone()[0]
print(f"{el()} block_counts rebuilt: {nbc} distinct methods (expect 31)", flush=True)

# 2. consensus set = No-NMS + 24 deduplication variants
for k in [m for m in list(core_nms.NMS_REGISTRY) if m.startswith("iou_edgecrop")]:
    del core_nms.NMS_REGISTRY[k]
assert len(core_nms.NMS_REGISTRY) == 25, len(core_nms.NMS_REGISTRY)
print(f"{el()} registry restricted to {len(core_nms.NMS_REGISTRY)} methods", flush=True)

# 3. consensus + deviations (tables recreated without primary keys)
with _orig() as con:
    for t, cols in {
        "block_consensus": ("slide_id VARCHAR, overlap_pct DOUBLE, block_size INT, block_x INT, "
                            "block_y INT, consensus_alr_Mac DOUBLE, consensus_alr_Neu DOUBLE, "
                            "consensus_alr_Eos DOUBLE, n_methods_in_consensus INT, "
                            "n_block_total INT"),
        "method_deviation": ("slide_id VARCHAR, overlap_pct DOUBLE, nms_method VARCHAR, "
                             "block_size INT, block_x INT, block_y INT, "
                             "aitchison_distance DOUBLE, mahalanobis_distance DOUBLE"),
    }.items():
        con.execute(f"DROP TABLE IF EXISTS {t}")
        con.execute(f"CREATE TABLE {t} ({cols})")

for ov in OVERLAP_PCTS:
    for bs in ALL_BLOCK_SIZES:
        ca.compute_consensus_per_block(bs, ov)
        ca.compute_method_deviations(bs, ov)
    print(f"{el()} consensus+deviations ov={ov:.3f}", flush=True)

# 4. agreement (Table 2 / S5) and outliers (Table 3)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ca.scale_dependence_curve().to_csv(RESULTS_DIR / "scale_dependence_curve.csv", index=False)
print(f"{el()} wrote scale_dependence_curve.csv", flush=True)
from nms.outlier_detection import outlier_summary  # noqa: E402

outlier_summary().to_csv(RESULTS_DIR / "outlier_summary.csv", index=False)
print(f"{el()} wrote outlier_summary.csv", flush=True)

# 5. equivalence versus No-NMS, all overlaps (Table 1 / S4)
from nms.baseline_comparison import equivalence_vs_none  # noqa: E402

allr = pd.concat([equivalence_vs_none(ov) for ov in OVERLAP_PCTS], ignore_index=True)
allr.to_csv(RESULTS_DIR / "equivalence_all_overlaps.csv", index=False)
print(f"{el()} wrote equivalence_all_overlaps.csv ({len(allr)} rows)", flush=True)
print(f"{el()} DONE", flush=True)
