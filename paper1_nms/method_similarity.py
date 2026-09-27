"""Paper 1: structure of the method space.

Krippendorff's α treats the methods as independent raters, but many are
near-identical transforms of the same detections (e.g. iou_grid_n1…n5). This
module quantifies that: the mean pairwise Aitchison distance between methods'
slide-total compositions, plus a dendrogram of the method "families". Report it
alongside α so agreement isn't read as if the raters were independent.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import LABELS
from core.compositional import cmult_repl, ilr
from core.db import connect

logger = logging.getLogger(__name__)


def method_distance_matrix(overlap_pct: float = 0.0) -> pd.DataFrame:
    """Symmetric (M×M) mean pairwise Aitchison distance between methods,
    averaged over slides. Indexed and columned by method name."""
    with connect(read_only=True) as con:
        df = con.execute(
            "SELECT slide_id, nms_method, label, count FROM cell_counts WHERE overlap_pct=?",
            [float(overlap_pct)],
        ).fetchdf()
    if df.empty:
        return pd.DataFrame()

    piv = (df.pivot_table(index=["slide_id", "nms_method"], columns="label",
                          values="count", aggfunc="sum", fill_value=0)
             .reindex(columns=LABELS, fill_value=0))
    coords = pd.DataFrame(ilr(cmult_repl(piv.to_numpy(dtype=float))), index=piv.index)
    methods = sorted({m for _, m in piv.index})
    m_idx = {m: i for i, m in enumerate(methods)}
    M = len(methods)

    sums = np.zeros((M, M))
    counts = np.zeros((M, M))
    for slide, sub in coords.groupby(level=0):
        vecs = {m: sub.loc[(slide, m)].to_numpy() for (_, m) in sub.index}
        present = list(vecs)
        for a in present:
            ia = m_idx[a]
            for b in present:
                d = float(np.linalg.norm(vecs[a] - vecs[b]))
                sums[ia, m_idx[b]] += d
                counts[ia, m_idx[b]] += 1
    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, sums / counts, np.nan)
    return pd.DataFrame(mean, index=methods, columns=methods)


def plot_dendrogram(dist_df: pd.DataFrame, path) -> None:
    """Average-linkage dendrogram of the method-distance matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    D = dist_df.to_numpy(dtype=float)
    D = (D + D.T) / 2.0
    D = np.nan_to_num(D, nan=np.nanmax(D))
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")

    fig, ax = plt.subplots(figsize=(10, 9))
    dendrogram(Z, labels=list(dist_df.index), orientation="right", ax=ax)
    ax.set_xlabel("mean Aitchison distance")
    ax.set_title("NMS method similarity (slide-total composition)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
