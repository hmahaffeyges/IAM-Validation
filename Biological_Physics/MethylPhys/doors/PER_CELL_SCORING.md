# The per-cell reading already exists — what is missing is a healthy band

**Recorded 2026-09-26**, after the author stated the architecture plainly: *"it should never be pooled into a
class before deconvolution. We should find the cells there and then score them based on their architecture
class's H_min"*, and *"I only care about the cell score."*

**That is what the chain already does, and it has all along.** Verified in the source rather than recalled:

`Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py`, `score_per_celltype` — docstring verbatim:

> *"Score all 115 cell-type A-scores for one patient. Each cell type's H_min is looked up via its class
> membership."*

```
cls = celltype_to_class.get(ct)
result = _score_one(beta_series, markers, h_min_by_class[cls])
```

So every one of the 115 cell types is scored **on its own discriminative markers**, against **the H_min of
the architecture class it belongs to**. `stage_a_cells` returns
`{cell: {A, coverage, confidence, status, class, fraction, present}}` for all of them, and the report bundle
carries them as `cells_all`. Nothing is pooled before deconvolution; the cell-type loci are per cell type,
not per class. The per-cell formula is `mean_i H(β_i)/H_min`, and the module asserts against the pooled form
on import (LESSON-ASCORE-02).

## So what is the pooled class score for, and what good does it do?

The pooled class gauge is a **separate, later stage** — `stage_b_identity` — which builds only two groups,
immune and haematopoietic progenitor, and is the thing that receives a laboratory zero, a band, a placement
and a tier. It is defensible exactly where it is used: immune in whole blood, where the class is homogeneous
(51 cell types of one lineage) and dominant (0.90–0.97 of the specimen), so the pooled profile is close to
the specimen's real composition.

It is **not** defensible anywhere else, and the chain does not use it anywhere else: `secretory` pools five
unrelated organs, `cycling` pools colon with bladder and skin basal, and `stem_adult` and `stem_pluri` are
single cell types wearing a class label.

## The actual gap

**The per-cell A-scores have no healthy band.** Only immune has a measured band (width 0.0524, pooled
p10–p90); every other entry in [`iamatlas_gauge_identity_loci_v1_0.json`](../chain/Runtime%20Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json) carries
`band_status: "UNMEASURED PLACEHOLDER"`. A per-cell A can therefore be computed but not *placed* — there is
no healthy distribution to say whether a given cell's reading is ordinary or not.

**That is a calibration task on data already in hand, not a rebuild.** A healthy band for a cell type is the
distribution of that cell's A across healthy arrays where the cell is present above its floor. The 732-array
Uppsala panel now calibrating is exactly the substrate for it, and the 318 already-published arrays are a
four-laboratory cross-check.

This is the next procedure to write, and it is the one that turns 115 computed numbers into 115 readable
ones.
