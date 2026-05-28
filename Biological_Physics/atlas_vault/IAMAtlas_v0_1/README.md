# IAMAtlas v0.1 — REBUILD edition (May 2026)

The unified IAM cell-type methylation reference atlas. This is the single artifact the EDEAR cellular-fidelity tool (the **Cellular Performance Gauge / CPG**) and the deconvolver consume at runtime.

**This folder contains the REBUILD edition** of IAMAtlas v0.1. The earlier collapsed build was retired in May 2026 after the flatness problem was identified and fixed. See `IAMAtlas_FLATNESS_LESSON.md` in this folder for the full account; the one-sentence summary is below.

---

## What is the REBUILD edition

In the original v0.1 build, all eight class-MCMCs converged cleanly (R-hat ~ 1.01, ESS in the hundreds, zero divergences) — but the model had a latent identifiability ridge that collapsed per-cell-type estimates onto the class mean. The result was an atlas in which the cell types within each class came out nearly identical to each other in the output, even though their inputs were genuinely different. The class-level brightness was usable; the per-cell-type layer was not. The collapse was invisible to every standard convergence check, because the chains genuinely converged — to a flattened answer.

The REBUILD edition was produced by running the FIXED MCMC script (`iamatlas_v0_1_mcmc_batched_FIXED.py`) on all eight classes, replacing the collapsed output entirely. Every class was independently distinctness-tested on its output (pairwise mean-absolute-difference between cells within a class) before being accepted into this folder. The full record of what went wrong, why diagnostics didn't catch it, and what to watch for in any future rebuild lives in `IAMAtlas_FLATNESS_LESSON.md`. **Read that file before re-running any class.**

---

## Files in this folder

| File | What it is |
|---|---|
| `IAMAtlasREBUILD.csv.xz` | The atlas matrix, xz-compressed (Git LFS). Decompress before use. |
| `IAMAtlasREBUILD_celltype_to_class.json` | Maps each cell type to its architecture class. The deconvolver needs this for class-level aggregation. |
| `IAMAtlasREBUILD_provenance.json` | Full build metadata: dates, pipeline steps, H_min values, per-class distinctness test outcomes. |
| `IAMAtlas_FLATNESS_LESSON.md` | **READ BEFORE ANY REBUILD.** The flatness problem, the fix, the diagnostic trap to avoid, the rebuild settings. |
| `class_archives/` | Eight per-class `.tar.xz` archives (LFS). Each contains that class's `per_celltype.csv` (post-reconciliation), `brightness.csv`, and `result.json`, plus the raw pre-reconciliation `per_celltype.csv` under `/raw/` for provenance. **Use these when re-running a single class** without rebuilding the others. |
| `iamatlas_v0_1_mcmc_batched_FIXED.py` | The canonical MCMC build script. Use this for any rebuild. The OLD `iamatlas_v0_1_mcmc_batched.py` is retired and not in this folder. |
| `reconcile_duplicates.py` | Resolves the duplicate-named cell types that the rebuild surfaced. Run between the per-class MCMC and the merge. See "How REBUILD was produced" below. |
| `merge_iamatlas_v0_1_REBUILD.py` | Combines the 8 per-class outputs into the unified `IAMAtlasREBUILD.csv`. Reads the reconciled inputs from the previous step. |
| `compact_atlas.py` | xz-compacts the merged atlas and tarballs each class's three files into `class_archives/`. |
| `README.md` | This file. |

To decompress the atlas:
```python
import lzma, shutil
with lzma.open("IAMAtlasREBUILD.csv.xz", "rb") as f, open("IAMAtlasREBUILD.csv", "wb") as o:
    shutil.copyfileobj(f, o)
```

---

## Matrix shape and contents

- **Rows:** 483,092 CpGs (one row per CpG; the full atlas universe).
- **Columns:** 264 total —
  - `cpg_id`
  - **32 class-level brightness columns** = 8 architecture classes × 4 statistics (`<class>_mean`, `_sd`, `_ci_lo`, `_ci_hi`).
  - **230 per-cell-type columns** = 115 cell types × 2 statistics (`<celltype>_mean`, `<celltype>_sd`).
  - `n_classes_with_data` (how many of the 8 classes have a value at that CpG).
- **Empty cells:** where a cell type was never measured at a given CpG, the cell is left empty (not zero, not a guessed value). Consumers must treat empty as "no data," not as a low value.
- **Uncompressed size:** ~577 MB. Compressed (.csv.xz): 96 MB.

The cell-type count dropped from 133 in the retired collapsed build to 115 in the REBUILD edition. The reduction is entirely from the duplicate-reconciliation step (described in "How REBUILD was produced" below), not from cell-type loss — biologically distinct cells were preserved.

---

## The 8 architecture classes and cell counts in the REBUILD edition

| Class | Cells | H_min | Notes |
|---|---|---|---|
| immune | 51 | 0.838889 | Richest class; lymphoid + myeloid lineages, multiple atlases. 12 `a`-prefix Salas-naming duplicates merged with their plain-name twins via coverage-weighted inverse-variance pooling. Reduced from 63 → 51. |
| cycling | 19 | 0.8561 | Gut/skin/lung/bladder/gastric epithelia and similar high-turnover tissue. `lung` (n=252, duplicate of `Lung_cells` n=6105) dropped; `small_intestine` (no twin) kept at low confidence. 20 → 19. |
| secretory | 18 | 0.8433 | Glandular: breast, liver, pancreas, prostate, gastric, thyroid. `hepatocyte` and `mammary` lowercase thin duplicates dropped (have well-covered capitalized twins). 20 → 18. |
| terminal | 9 | 0.7728 | Post-mitotic: neurons, cardiomyocytes, glia. `brain`/`heart`/`skeletal` lowercase thin duplicates dropped; `neuron` kept (biologically distinct from NeuMa). 12 → 9. |
| progenitor | 11 | 0.8522 | GMP/CMP/MEP/MPP/OPC and related. No reconciliation needed. Sparse cells (erythroblast n=20, megakaryocyte n=20, nRBC n=174) kept at low confidence per two-tier design. |
| stromal | 5 | 0.8630 | adipocyte, endothelial, fibroblast, smooth_muscle, stromal_other. The original FIXED-script build from 2026-05-23; not rebuilt because already correct. |
| stem_pluri | 1 | 0.9822 | Single collapsed pluripotent cell type. |
| stem_adult | 1 | 0.8737 | HSC. |

Total: 115 cell types. Per-class membership is given exactly by `IAMAtlasREBUILD_celltype_to_class.json`.

H_min values frozen 2026-04-06; same values as the retired build (the MCMC re-run did not affect H_min, which is derived from first principles, not fit).

---

## Per-class distinctness verification

Each rebuilt class was verified by pairwise mean-absolute-difference between cell columns on shared CpGs (≥200 shared per pair). The test is the judge, not R-hat — see `IAMAtlas_FLATNESS_LESSON.md`.

| Class | Distinctness result |
|---|---|
| stem_pluri | Single-cell; per-CpG signal/noise ratio ~41 (vs ~0.5 when collapsed). |
| stem_adult | Single-cell; per-CpG signal/noise ratio ~41. |
| progenitor | 11 cells. Median pairwise mean\|diff\| 0.043; max 0.38. Adjacent myeloid progenitors (CMP/GMP/MPP/L-MPP) correctly close (~0.015–0.024, corr ~0.99). Neural precursors (OPC, NeuIm) correctly distant from blood progenitors (~0.34–0.38, corr ~0.09). |
| cycling | 19 cells (post-drop). Median 0.10; max 0.40. Gastric undiff trio (Antrum/Corpus/Fundus_undiff) correctly clustered (~0.020–0.024). |
| secretory | 18 cells (post-drop). Median 0.115; max 0.39. Gastric diff trio clustered (~0.019); pancreatic beta vs duct correctly resolvable; hepatocyte distant from gastric. |
| terminal | 9 cells (post-drop). Median 0.31; max 0.44. Cortical_neurons vs Glia correctly close (0.07, both CNS terminal cells). The highest-distinctness class — these are end-state identity-locked tissues, as expected at H_min = 0.7728. |
| stromal | 5 cells. Original FIXED-script build; distinctness verified at seal time (2026-05-23). |
| immune | 51 cells (post-merge). 56% of pairs testable; of those, 92% above 0.05; median 0.18; max 0.48. The 12 a-prefix merges removed all of the "near-identical" pairs that surfaced before reconciliation. |

The convergence-diagnostic profile of the rebuild is documented at length in `IAMAtlas_FLATNESS_LESSON.md` Section 3: the FIXED script runs at R-hat 1.4–2.6 / ESS 5–9 / zero divergences on multi-cell classes, which is the expected "high R-hat but the means are correct" signature. **Do not chase R-hat down by changing sampler settings — doing so risks re-collapsing the per-cell-type layer.**

---

## How REBUILD was produced (the pipeline)

The end-to-end rebuild used a three-stage pipeline. Each script in this folder corresponds to one stage and is fully reproducible.

**Stage 1 — Per-class MCMC** (`iamatlas_v0_1_mcmc_batched_FIXED.py`): each of the eight classes was independently fit by hierarchical Bayesian MCMC against the long-format observations in `iamatlas_mcmc_inputs.csv`. Settings: tune 1000, draws 1000, chains 4, target_accept 0.95. Batch size 5000 for all classes except immune (1500 — at 5000 it OOMs on 32 GB). Output per class: `<class>_brightness.csv` (class-level β per CpG), `<class>_per_celltype.csv` (per-cell-type β per CpG), `<class>_result.json` (convergence metadata).

**Stage 2 — Duplicate reconciliation** (`reconcile_duplicates.py`): a step the original v0.1 pipeline didn't include. The rebuild's distinctness tests surfaced duplicate-named cell types — particularly the 12 a-prefix variants in immune (aCD8Tmem vs CD8Tmem etc.) which were Salas-atlas naming variants of the same biological cells (corr = 1.000, mean\|diff\| < 0.005 on tested CpGs). The reconciliation step merges those pairs via coverage-weighted inverse-variance pooling, and drops thin lowercase variants whose better-covered capitalized twins exist. The policy is encoded in the script itself (constants at the top). Output: `<class>_per_celltype.RECONCILED.csv` for each class, plus `RECONCILIATION_REPORT.json` summarizing the decisions.

**Stage 3 — Merge and compact** (`merge_iamatlas_v0_1_REBUILD.py` then `compact_atlas.py`): the merge step reads the reconciled per-cell-type files plus the original brightness files and builds the unified `IAMAtlasREBUILD.csv` (32 class columns + 230 per-cell columns + meta). The compact step then xz-compresses the merged atlas and tarballs each class's three files into `class_archives/` for re-runnability.

To run the entire pipeline from the eight raw per-class outputs:
```
python reconcile_duplicates.py --in_dir <output_folder>
python merge_iamatlas_v0_1_REBUILD.py --in_dir <output_folder>
python compact_atlas.py
```

---

## Re-running a single class (the common case)

If a class needs a re-run — to add an atlas, expand the cell list, or revisit a sparse cell with new coverage — start from that class's tarball in `class_archives/`. Each tarball contains the canonical reconciled per_celltype.csv plus the raw pre-reconciliation version under `/raw/`, so you can decide whether to re-reconcile or start from scratch.

The rebuild procedure for a single class is:

1. Extract `class_archives/<class>_v0_1_REBUILD.tar.xz`.
2. Add new observation rows to `iamatlas_mcmc_inputs.csv` (in the production-data area of the repo) for the class being re-run.
3. Run `iamatlas_v0_1_mcmc_batched_FIXED.py --classes <class> --inputs iamatlas_mcmc_inputs.csv --out_dir <new_output_folder> --batch_size 5000` (or `1500` for immune). The settings are documented in `IAMAtlas_FLATNESS_LESSON.md` "REBUILD SETTINGS USED."
4. Distinctness-test the new `per_celltype.csv` against the prior class results before merging. The test procedure is in `IAMAtlas_FLATNESS_LESSON.md`.
5. Re-run `reconcile_duplicates.py` (if any new duplicates surface), `merge_iamatlas_v0_1_REBUILD.py`, and `compact_atlas.py`.
6. Version the new atlas in this vault with a new SHA. Bump to `IAMAtlas_v0_2/` etc. when the change is significant.

---

## CpG universe: 450K-level, not full EPIC

The 483,092-CpG universe is essentially the Illumina HumanMethylation450 (450K) set — the reliable intersection of CpGs that the source reference atlases share. The newer EPIC/850K array reads ~865,000 CpGs (most of the 450K set plus ~400,000 additional sites, many in enhancer regions). Those EPIC-only sites are not in v0.1 because the current source-atlas pool does not measure them in enough cell types to anchor estimates.

Practical consequences:
- Customer samples on either 450K or EPIC platforms can be scored — the deconvolver intersects the customer's CpGs with the atlas's informative CpGs (the ~483K backbone).
- EPIC-only sites are simply not used in v0.1. Adding them is a v0.2+ expansion (requires EPIC-native source atlases) and would mainly sharpen per-cell-type / disease discrimination rather than transform the class-level thermometer.

---

## How the deconvolver uses this atlas

See `../deconvolver/`. In brief: the deconvolver loads `IAMAtlasREBUILD.csv` and `IAMAtlasREBUILD_celltype_to_class.json`, filters informative (high-discrimination) CpGs, intersects them with a customer's CpGs, solves a constrained non-negative least-squares for cell-type fractions, and aggregates to per-class fractions. The class-level reading is the load-bearing wellness output (the CPG / thermometer). The per-cell-type breakdown is the indicative tier — reliable where coverage is good, lower-confidence where the source atlas was thin.

---

## Provenance

- Stage 1 (per-class MCMC) run dates: 2026-05-25 to 2026-05-28 on Heath's son's 16-core / 32 GB workstation.
- Stage 2 (reconciliation) and Stage 3 (merge + compact): 2026-05-28.
- SHA-256 of `IAMAtlasREBUILD.csv.xz`: `41b7c16f043bce96e085a2b8b4e709efd2b862af9de8dbe9a8646e9fb94c32ee` (see `../INVENTORY.json` for full SHA inventory).
- Source atlases: see the per-atlas folders under `../stage2_cell_of_origin/` and `../stage3_immune_fraction/` and their entries in `../INVENTORY.json`.
- Repository: `hmahaffeyges/IAM-Validation`.

The retired collapsed v0.1 atlas (`IAMAtlas.csv.xz`, sha256 `a11f62b0…`) has been removed from this folder. The corresponding `MCMC_BUILD_LESSONS.md` (which incorrectly stated that the seven non-stromal classes were "valid as-is and not re-run") has been replaced by `IAMAtlas_FLATNESS_LESSON.md`. Both removed artifacts remain in the repository's git history at any commit before this REBUILD push, should they ever need to be inspected.
