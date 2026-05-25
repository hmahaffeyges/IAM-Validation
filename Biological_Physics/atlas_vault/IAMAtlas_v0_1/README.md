# IAMAtlas v0.1

The unified IAM cell-type methylation reference atlas. This is the single artifact the EDEAR cellular-fidelity tool and the deconvolver consume at runtime. It replaces the older borrowed-reference approach (Xu panel / EpiDISH / NNLS-against-Loyfer) with one matrix built by a hierarchical Bayesian model anchored to the IAM information-floor framework.

## What this is, in one paragraph

A matrix of posterior methylation estimates: for every CpG site on the array, the model estimates the methylation level (and uncertainty) for each cell type and each of the 8 IAM architecture classes. A customer's methylation sample is scored against this matrix to estimate which cell types and classes their cell-free DNA came from, and how far each class sits from its information-maintenance floor (the A-score / cellular thermometer reading).

## Files in this folder

| File | What it is |
|---|---|
| `IAMAtlas.csv.xz` | The atlas matrix, xz-compressed (LFS-tracked). Decompress before use. |
| `IAMAtlas_celltype_to_class.json` | Maps each of the 133 cell types to its architecture class. The deconvolver needs this for class-level aggregation. |
| `merge_iamatlas_v0_1.py` | The script that built `IAMAtlas.csv` from the 8 per-class MCMC outputs. Kept here so a future rebuild (after adding cell types / atlases) is reproducible. |
| `README.md` | This file. |

To decompress:
```python
import lzma, shutil
with lzma.open("IAMAtlas.csv.xz","rb") as f, open("IAMAtlas.csv","wb") as o:
    shutil.copyfileobj(f, o)
```

## Matrix shape and contents

- **Rows:** 483,092 CpGs (one row per CpG; the full atlas universe).
- **Columns:** 300 total —
  - `cpg_id`
  - **32 class-level brightness columns** = 8 architecture classes x 4 stats (`<class>_mean`, `_sd`, `_ci_lo`, `_ci_hi`).
  - **266 per-cell-type columns** = 133 cell types x 2 stats (`<celltype>_mean`, `<celltype>_sd`).
  - `n_classes_with_data` (how many of the 8 classes have a value at that CpG).
- **Empty cells:** where a cell type was never measured at a given CpG, the cell is left empty (not zero, not a guessed value). This is deliberate — no fabricated estimates. Consumers must treat empty as "no data," not as a low value.
- **Uncompressed size:** ~1,181,404 KB (~1.15 GB). Compressed: 207,534,592 bytes (~5.8x).

## The 8 architecture classes and their cell counts in v0.1

| Class | Cell types (v0.1) | Notes |
|---|---|---|
| immune | 63 | Richest class; lymphoid + myeloid lineages, multiple atlases. |
| cycling | 20 | Gut/skin/lung/bladder epithelia and similar high-turnover tissue. |
| secretory | 20 | Glandular: breast, liver, pancreas, prostate, gastric, thyroid. |
| terminal | 12 | Post-mitotic: neurons, cardiomyocytes, glia (incl. Gasparoni Glia). |
| progenitor | 11 | GMP/CMP/MEP/MPP/OPC and related committed progenitors. |
| stromal | 5 | adipocyte, endothelial, fibroblast, smooth_muscle, stromal_other. See stromal note below. |
| stem_pluri | 1 | Single collapsed pluripotent cell type in v0.1 (1 atlas). |
| stem_adult | 1 | Single collapsed adult-stem cell type (HSC) in v0.1 (1 atlas). |

Total: 133 cell types. Per-class cell membership is given exactly by `IAMAtlas_celltype_to_class.json`.

## How it was built

Each architecture class was estimated independently by a hierarchical Bayesian model (`iamatlas_v0_1_mcmc_batched*.py`, in the production-data area of the repo) run over batches of CpGs. For each class the model ingests every observation in `iamatlas_mcmc_inputs.csv` for that class (one row per CpG x cell type x source atlas), and produces posterior mean / sd / 95% CI per CpG per cell type. The per-class outputs (`iamatlas_v0_1_<class>_brightness.csv`, `_per_celltype.csv`, `_result.json`) are then combined by `merge_iamatlas_v0_1.py` into this single matrix.

The input was ~11.9 million observation rows (long format, the same CpGs measured across many cell types and atlases); the model collapses these into the 483,092-row wide matrix.

## CpG universe: 450K-level, not full EPIC

The 483,092-CpG universe is essentially the Illumina HumanMethylation450 (450K) set — the reliable intersection of CpGs that the source reference atlases share. The newer EPIC/850K array reads ~865,000 CpGs (most of the 450K set plus ~400,000 additional sites, many in enhancer regions). Those EPIC-only sites are NOT in v0.1 because the current source-atlas pool does not measure them in enough cell types to anchor estimates.

Practical consequences:
- Customer samples on either 450K or EPIC platforms can be scored — the deconvolver intersects the customer's CpGs with the atlas's informative CpGs (the ~483K backbone).
- EPIC-only sites are simply not used in v0.1. Adding them is a v0.2+ expansion (requires EPIC-native source atlases; see roadmap STEP 9), and would mainly sharpen per-cell-type / disease discrimination rather than transform the class-level thermometer.

## Stromal note (5-cell v0.1 history)

Stromal was sealed as a 5-cell class (adipocyte, endothelial, fibroblast, smooth_muscle, stromal_other) using a model fix (parameters restricted to observed cpg-celltype pairs; per-pair logit-mean prior removing a sigma/kappa identifiability ridge). Convergence: 0 divergences across all 16 batches, Pearson 0.934, MAE 0.078. ESS is soft on smooth_muscle and stromal_other (each supported by only 2 atlases) — those two carry wide credible intervals (honest uncertainty, not error). Four sparse single-atlas cells (placenta, astrocyte, stellate, pericyte) were dropped from v0.1 and are flagged for reintroduction in v0.2 when better-supported atlases arrive.

## How the deconvolver uses this atlas

See `../deconvolver/`. In brief: the deconvolver loads this matrix and the celltype-to-class JSON, filters informative CpGs, intersects them with a customer's CpGs, solves a constrained non-negative least-squares for cell-type fractions, and aggregates to per-class fractions / A-scores.

## Provenance

- Build date: 2026-05-23 (stromal sealed) / merged 2026-05-25.
- SHA-256 of `IAMAtlas.csv.xz`: see `../INVENTORY.json`.
- Source atlases: see the per-atlas folders under `../stage2_cell_of_origin/` and `../stage3_immune_fraction/` and their entries in `../INVENTORY.json`.
- Repository: hmahaffeyges/IAM-Validation.

## v0.2+ expansion path (no rebuild of the product, just a deeper atlas)

Adding an atlas: acquire data, bridge to the array if needed, map its cell types to architecture classes, add rows to `iamatlas_mcmc_inputs.csv`, re-run the hierarchical MCMC for the affected class(es), re-run `merge_iamatlas_v0_1.py`, version the result in this vault with a new SHA. Standing expansion targets are listed in the master roadmap (STEP 9): full Loyfer 2023 at native resolution, Roadmap Epigenomics, MARLIN/Capper leukemia, EPIC-native enhancer coverage, and eventually IAM-specific reference data measured in-house.
