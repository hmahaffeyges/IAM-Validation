# IAMAtlas v0.1 Production Data

Durable storage for the IAMAtlas v0.1 production MCMC pipeline — inputs,
build script, batch outputs, and the stromal re-run package.

**Why this directory exists.** Files here are too large for GitHub web upload
(>25 MB) and too large for the project file system (>30 MB). They live in git
LFS so the next session of Walther — or the next instance of Heath after
6 months of context loss — can read them without needing Heath to re-upload.

## Contents

```
iamatlas_production_data/
├── README.md                              (this file)
├── iamatlas_v0_1_mcmc_batched.py          Production MCMC build script
├── merge_iamatlas_v0_1.py                 Per-class merge script
├── inputs/
│   ├── iamatlas_mcmc_inputs.csv           698 MB raw inputs (10.9M rows)
│   ├── iamatlas_mcmc_inputs.csv.xz         19 MB xz-compressed (decompresses to .csv)
│   ├── iamatlas_cpg_coverage_per_atlas.csv 47 MB
│   └── iamatlas_cpg_universe.csv           5.6 MB
├── outputs/v0_1/
│   ├── iamatlas_v0_1_secretory_per_celltype.csv      137 MB
│   ├── iamatlas_v0_1_stem_adult_per_celltype.csv      14 MB
│   ├── iamatlas_v0_1_stem_pluri_per_celltype.csv      14 MB
│   ├── iamatlas_v0_1_stromal_per_celltype.csv         7.3 MB (FAILED RUN)
│   ├── iamatlas_v0_1_terminal_per_celltype.csv        3.4 MB
│   └── *_result.json                                   convergence summaries
├── stromal_rerun/
│   ├── STROMAL_RERUN_README.md            Operational instructions for re-run
│   └── harmonize_stromal_labels.py        Label harmonization script
└── terminal_addition/
    ├── GASPARONI_STAGING_README.md        Operational instructions for terminal addition
    ├── stage_gasparoni_for_terminal.py    Source-extraction script (already run)
    └── gasparoni_terminal_addition.csv    61 MB, 957,638 rows ready to append
```

## MCMC inputs schema

`iamatlas_mcmc_inputs.csv`:
```
cpg_id, atlas_source, cell_type, arch_class, beta_observed, n_donors, weight
```

10,938,663 rows across 8 architecture classes (terminal, secretory,
stem_pluri, stem_adult, stromal, cycling, progenitor, immune).

## Production run status (as of 2026-05-07)

Completed runs (from `outputs/v0_1/`):

| class       | R-hat | ESS  | div | Pearson | n_obs   | status                  |
|-------------|-------|------|-----|---------|---------|-------------------------|
| terminal    | 1.01  | 1493 |   0 | 0.799   |  30,895 | clean                   |
| stem_pluri  | 1.01  |  501 |   2 | 0.919   | 482,421 | clean                   |
| stem_adult  | 1.01  |  723 |   0 | 0.904   | 482,421 | clean                   |
| secretory   | 1.02  | 1023 |   0 | 0.791   |1,211,597| clean                   |
| stromal     | 3.67  |    4 |   6 | 0.897   |  96,733 | FAILED — see stromal_rerun/ |
| cycling     |   —   |   —  |   — |    —    |    —    | running                 |
| progenitor  |   —   |   —  |   — |    —    |    —    | queued                  |
| immune      |   —   |   —  |   — |    —    |    —    | queued                  |

## Stromal failure root cause and fix

The original stromal run failed because 17 cell-type labels were spread
across 17 atlases with most cell types supported by only 1-2 atlases. Many
of those labels are atlas-specific naming variants of the same biological
cell type (EC, Endo, Vascular_endothelial_cells, endothelial all = endothelial
cells). The model treated each label as a distinct cell type with its own
posterior, creating identifiability degeneracies — the chain could put the
endothelial signal into any of four interchangeable buckets, and never
converged.

`stromal_rerun/harmonize_stromal_labels.py` consolidates the 17 labels into
9 canonical cell types (endothelial: 13 atlases; fibroblast: 11; adipocyte:
4; smooth_muscle: 2; stromal_other: 2; pericyte/stellate/astrocyte/placenta:
1 each). After harmonization, the existing MCMC build script can be re-run
on `iamatlas_mcmc_inputs_stromal_harmonized.csv` with tightened sampler
config (target_accept=0.99, tune=2000, draws=2000) and converge cleanly.

See `stromal_rerun/STROMAL_RERUN_README.md` for full procedure.

## Terminal class addition (Gasparoni 2018)

`terminal_addition/gasparoni_terminal_addition.csv` contains 957,638 rows
extracted from Gasparoni et al. 2018 (GSE66351, occipital cortex FANS-sorted
brain methylation atlas, n=16 donors, HM450 platform). Closes the data-imbalance
between terminal class (~31K rows) and the rest of the architecture-class pool.
After append: terminal grows to ~988K rows, comparable to secretory/cycling/
progenitor scale.

Cell-type label mapping (Heath's decision, 2026-05-06):
  - `cortical_neuron` → `Cortical_neurons` (matches Loyfer's existing label)
  - `cortical_glia`   → `Glia` (new terminal-class cell type)

Atlas source: `gasparoni_2018` (internal provenance, not customer-facing).

DO NOT APPEND until the main MCMC run (cycling/progenitor/immune) completes.
See `terminal_addition/GASPARONI_STAGING_README.md` for full procedure.

## Working with this directory

**Cloning:** standard `git clone` works but pulls all LFS data (~800 MB).
For partial clones use `git lfs install --skip-smudge` before clone, then
`git lfs pull --include="<path>"` for the specific files needed.

**Decompressing the inputs file:** if you only have the .xz version,
```
xz -d iamatlas_mcmc_inputs.csv.xz
```
Yields the 698 MB raw CSV.

**Adding new outputs:** drop new files in `outputs/v0_1/` (or future
`outputs/v0_2/` for next iteration). LFS picks them up automatically via
the `.gitattributes` rule at the repo root.
