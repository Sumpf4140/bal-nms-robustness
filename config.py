"""Project-wide constants. Import from here, never hardcode."""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
IMAGES_DIR   = PROJECT_ROOT / "images"
DATA_DIR     = PROJECT_ROOT / "data"

# The results database (DuckDB) is kept outside the repository; override with
# BALC_DB_PATH. The full run over 76 slides x 9 overlaps x 31 variants produces
# a multi-GB database, so place it on a local disk rather than a synced folder.
DB_PATH = (
    Path(os.environ["BALC_DB_PATH"]).expanduser()
    if os.environ.get("BALC_DB_PATH")
    else Path.home() / ".balc" / "results.duckdb"
)
P1_REPORTS   = PROJECT_ROOT / "results"

SLIDE_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# ── Cell labels (canonical order; never reorder) ───────────────────────
LABELS        = ["Makrophage", "Lymphozyt", "NeutrophilerGranulozyt", "EosinophilerGranulozyt"]
ALR_REF_LABEL = "Lymphozyt"
# Canonical short names for column suffixes — keep consistent across every output.
LABEL_ABBR = {
    "Makrophage": "Mac",
    "Lymphozyt": "Lym",
    "NeutrophilerGranulozyt": "Neu",
    "EosinophilerGranulozyt": "Eos",
}

# ── CNN / NMS thresholds ──────────────────────────────────────────────
CONF_THRESH      = 0.440
IOU_THRESH       = 0.45
NN_DIST_PX       = 20
DBSCAN_EPS_PX    = 20
DBSCAN_MIN_SAMP  = 1
# iou_edgecrop_*: a detection whose box lies within this many px of any
# tile edge is dropped before the IoU NMS step (the clipped "half" cells —
# recovered whole from the neighbouring tile when tiles overlap).
EDGE_MARGIN_PX   = 5

# ── Tile geometry ─────────────────────────────────────────────────────
TILE_WIDTH_PX    = 640
TILE_HEIGHT_PX   = 640
TILE_GRID_TOLERANCE = 0.99
OVERLAP_PCTS     = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.50]

# ── Block-grid family (analysis scales) ───────────────────────────────
BLOCK_SIZES         = [1, 2, 3, 4, 5]    # n×n tiles per block
INCLUDE_FULL_SLIDE  = True               # whole slide as one block (block_size = 0)
BLOCK_MIN_CELLS     = 500                # below this, block is excluded from consensus
# The ≥BLOCK_MIN_CELLS filter must select block *positions* method-independently,
# otherwise it admits only the high-count methods at a position and silently drops
# every other method → unbalanced panels. The reference count is taken under this
# method (the no-suppression upper bound), and the same positions are then used
# for every variant.
BLOCK_FILTER_REFERENCE = "none"

# ── Statistics ────────────────────────────────────────────────────────
N_BOOTSTRAP        = 1000
ALPHA              = 0.05
RNG_SEED           = 20260501
# Equivalence margin (alr units): a paired alr difference whose 90% bootstrap CI
# lies within ±EQUIV_MARGIN_ALR is declared equivalent. ±0.20 natural-log units
# correspond to a −18.1% to +22.1% change in the cell-type/lymphocyte ratio.
EQUIV_MARGIN_ALR   = 0.20
