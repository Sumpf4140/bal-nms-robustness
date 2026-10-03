# Does non-maximum suppression change the cell differential?

Code, analysis scripts and intermediate results for the study *"Does non-maximum
suppression change the cell differential? A robustness and equivalence analysis of
detector post-processing in quantitative cytology"* (manuscript under review; see
[Citation](#citation)).

The study applies 31 post-processing variants (No-NMS, 24 deduplication variants,
6 edge-crop variants) to the output of one fixed cell detector on 76 bronchoalveolar
lavage (BAL) cytospin slides, at nine tile-overlap fractions, and asks whether the
choice changes the slide-level differential cell count. The analysis is compositional
(ilr/alr log-ratios, Aitchison geometry). Because there is no manual reference
count, the results describe agreement and robustness between post-processing
choices, not accuracy.

## Contents

```
core/                  post-processing and statistics library
  nms.py               the 31 post-processing variants (NMS_REGISTRY)
  loader.py            detector CSV -> database; tile-relative -> slide coordinates
  blocks.py            aggregation of retained detections to analysis scales
  compositional.py     zero replacement (CZM), ilr/alr, Aitchison distance
  consensus.py         geometric-median consensus, Krippendorff's alpha, Mahalanobis
  stats.py             bootstrap equivalence (TOST), Friedman, Holm, weights
nms/                   analysis pipeline (run, consensus, outliers, equivalence, timing)
scripts/
  reproduce_from_slide_counts.py   re-derive the slide-level results from results/ alone
  reproduce_analysis.py            block-level pipeline behind Tables 1-3, S4, S5
  revision_analyses.py             additional analyses requested in peer review
  export_slide_counts.py           writes results/slide_counts.csv from the database
results/               intermediate outputs underlying the reported statistics
tests/                 unit tests on synthetic data (no study data needed)
```

## Post-processing variants

Every family is applied at six suppression scopes: single tile (`grid_n1`), sliding
n×n tile blocks with one-tile overlap and sequential carry-over (`grid_n2` …
`grid_n5`), and the whole slide (`global`).

| Manuscript name | Code key | Rule |
|---|---|---|
| No-NMS | `none` | no suppression (confidence-filtered detector output) |
| IoU | `iou_<scope>` | greedy IoU ≥ 0.45, class-agnostic, highest confidence kept |
| nn-distance | `nn_dist_<scope>` | same-class centroids within 20 px, highest confidence kept |
| nn-centre | `nn_center_<scope>` | as nn-distance, radius = 2 × SD of box size ((w+h)/2), per class and slide |
| nn-cluster | `nn_cluster_<scope>` | connected components of the 20 px same-class centroid graph (DBSCAN, minPts = 1); highest-confidence member kept |
| edge-crop | `iou_edgecrop_<scope>` | drop detections within 5 px of their tile edge, then IoU |

Pseudocode for the scoped suppression with carry-over is given as Algorithm S1 in the
Supporting Information; the implementation is `core/nms.py::_grid_overlap_suppress`.

## Quick check without the raw data

`results/slide_counts.csv` holds the number of retained detections per slide, tile
overlap, variant and cell type (76 × 9 × 31 × 4 rows). It is sufficient to reproduce
every slide-level result:

```bash
python -m pip install -r requirements.txt
python scripts/reproduce_from_slide_counts.py
```

This recomputes the equivalence of all 30 variants against No-NMS at all nine
overlaps (Table 1, Table S4), the Friedman tests (Results) and the cohort proportions
of Table S1, and compares them with the published values. Expected output:

```
equivalence: 810 comparisons, max |deviation| = 9.71e-17, equivalence decisions identical: True
  primary, Makrophage               Δ = -0.065 (90% CI -0.097 to -0.056)  equivalent: True
  primary, NeutrophilerGranulozyt   Δ = +0.063 (90% CI +0.042 to +0.083)  equivalent: True
  primary, EosinophilerGranulozyt   Δ = +0.116 (90% CI +0.095 to +0.139)  equivalent: True
friedman: chi2(24) = 1170.1-1336.7, W = 0.64-0.73, n = 76; max |deviation| = 2.27e-13
Table S1 proportions (%): Makrophage 64.8, Lymphozyt 31.8, NeutrophilerGranulozyt 2.6, EosinophilerGranulozyt 0.8
```

Differences are always computed as Δ = variant − No-NMS. Cell-type labels are the
detector's class names (Makrophage, Lymphozyt, NeutrophilerGranulozyt,
EosinophilerGranulozyt); lymphocytes are the alr reference.

## Full pipeline from detector output

Input is one CSV per slide and overlap, exported from the detector before its
built-in NMS and filtered only at confidence ≥ 0.44:

```
data/<overlap>_overlap_raw/<slide_id>.csv     overlap in percent: 0, 2.5, 5, 7.5, 10, 15, 20, 25, 50
images/<slide_id>.tif                         whole-slide image (dimensions only)
```

Required columns: `label, confidence, x1, y1, x2, y2, x_correct, y_correct`, with
box coordinates relative to the 640 × 640 px tile and `x_correct, y_correct` the
tile's origin on the slide.

```bash
python main.py init-db
python main.py load
python main.py nms run --workers 4          # all variants x overlaps -> cell/tile counts
python scripts/reproduce_analysis.py    # analysis scales, consensus, alpha, outliers, equivalence
python scripts/revision_analyses.py        # additional analyses (results/revision/)
```

The database location defaults to `~/.balc/results.duckdb` (override with
`BALC_DB_PATH`). The full study produces a database of roughly 15 GB.
`python main.py nms --help` lists the individual pipeline stages, including the
controlled single-process timing (`nms timing`).

## Results

| File | Content | Manuscript |
|---|---|---|
| `slide_counts.csv` | retained detections per slide × overlap × variant × cell type | input to all slide-level results |
| `equivalence_all_overlaps.csv` | median Δ and 90% bootstrap CI, 30 variants × 3 log-ratios × 9 overlaps | Table 1, Table S4, Figure 4 |
| `scale_dependence_curve.csv` | Krippendorff's α with 95% CI and eligible blocks per analysis scale × overlap | Table 2, Figure 3, Tables S2, S5 |
| `outlier_summary.csv` | Westfall–Young outlier flags and mean upper-tail frequency | Table 3 |
| `weighted_method_means.csv` | count-weighted mean composition per variant (0% overlap) | Table S1 |
| `revision/A_margin_sensitivity_*.csv` | equivalence at margins ±0.10, ±0.15, ±0.20 | Discussion |
| `revision/C_zero_frequencies.csv` | zero counts by analysis scale × variant | Table S3 |
| `revision/D_zero_replacement_sensitivity.csv` | equivalence under alternative zero replacement | Table S6 |
| `revision/F_edgecrop_vs_iou.csv` | edge-crop versus matching IoU-only variant | Results (edge crop) |
| `revision/G_slide_level_distribution.csv` | per-slide distribution of Δ | Results |
| `revision/G_slide_level_all.csv` | per-slide distribution and slides outside ±0.20, all variants, log-ratios and overlaps | Results, Table S4 |
| `revision/H_table3_*.csv` | upper-tail frequency and largest-deviation share | Table 3 |
| `revision/H_no_nms_slide_level.csv` | slide-level sign test of the No-NMS outlier finding | Methods (outlier analysis) |
| `revision/I_class_retention.csv` | class-specific retention relative to No-NMS | Discussion |
| `revision/I_class_retention_per_slide.csv` | per-slide retention by cell type, all variants and overlaps | Discussion |
| `revision/J_friedman_25methods.csv` | Friedman statistics across the 25 variants | Results |

Weights in `weighted_method_means.csv` are the per-slide detection counts (w = N).

## Tests

```bash
python -m pip install pytest scikit-learn
python -m pytest
```

The tests run on synthetic data in a temporary database and do not need any study
data. `tests/test_consensus.py` includes an optional cross-check of Krippendorff's α
against the `krippendorff` package (skipped if it is not installed).

## Data availability

Whole-slide images and raw detections are not included. Slide identifiers in
`results/` are pseudonyms (S01–S76) assigned in the sort order of the original
identifiers, which keeps the fixed-seed bootstrap resampling identical to the
published analysis.

## License

This code is released under the GNU General Public License, version 3 or later (see [LICENSE](LICENSE)).

## Citation

If you use this code or the results, please cite the paper:

> [AUTHORS]. Does non-maximum suppression change the cell differential? A robustness
> and equivalence analysis of detector post-processing in quantitative cytology.
> *[JOURNAL]*. [YEAR];[VOLUME]([ISSUE]):[PAGES]. doi:[DOI](https://doi.org/[DOI])

```bibtex
@article{[CITATION_KEY],
  author  = {[AUTHORS]},
  title   = {Does non-maximum suppression change the cell differential? A robustness and
             equivalence analysis of detector post-processing in quantitative cytology},
  journal = {[JOURNAL]},
  year    = {[YEAR]},
  volume  = {[VOLUME]},
  number  = {[ISSUE]},
  pages   = {[PAGES]},
  doi     = {[DOI]}
}
```
