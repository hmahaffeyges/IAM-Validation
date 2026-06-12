# Stromal Re-Run Package

**Date:** 2026-05-07
**Author:** Walther / Heath
**Purpose:** Fix the stromal MCMC failure (R-hat 3.67, ESS 4) before launching
the re-run.

---

## What was wrong

The original stromal run failed catastrophically. R-hat 3.67 means the chains
never mixed. ESS 4 means effectively no independent posterior samples were
drawn. Six divergences confirmed the sampler was hitting pathological geometry.

**Root cause:** the stromal class was fed 17 cell-type labels across 17 atlases
where most cell types were supported by only 1-2 atlases. Many of those 17
labels are atlas-specific naming variants of the same biological cell type:

  - EC (11 atlases) + Endo (2 atlases) + Vascular_endothelial_cells (1 atlas) +
    endothelial (1 atlas) = 15 atlases all measuring **endothelial cells**
  - FB (1 atlas) + Fib (9 atlases) + fibroblast (1 atlas) = 11 atlases all
    measuring **fibroblasts**
  - Adipocytes + Fat (2 atlases) + adipose = 4 atlases all measuring **adipocytes**
  - SM + SMC = 2 atlases for **smooth muscle**

The model treated each label as a distinct cell type with its own posterior,
creating identifiability degeneracies — the chain could put the endothelial
signal into any of the four endothelial-named buckets without changing the
likelihood. The chains wandered across these degeneracies forever and never
converged.

**Diagnosis confirmed by inspecting iamatlas_v0_1_stromal_per_celltype.csv:**
the posterior means for EC (0.3646), Endo (0.3616), Vascular_endothelial_cells
(0.3746), and endothelial (0.3628) are all the same value within posterior
uncertainty — the same biological signal split across four labels.

---

## What this package does

**Step 1: harmonize_stromal_labels.py** consolidates duplicate-named labels in
the stromal class into canonical cell types. After harmonization, stromal goes
from 17 sparsely-supported labels to **9 well-supported canonical cell types**:

| canonical       | n_atlases | source labels merged                        |
|-----------------|-----------|---------------------------------------------|
| endothelial     | 13        | EC + Endo + Vascular_endothelial_cells + endothelial |
| fibroblast      | 11        | FB + Fib + fibroblast                       |
| adipocyte       | 4         | Adipocytes + Fat + adipose                  |
| smooth_muscle   | 2         | SM + SMC                                    |
| stromal_other   | 2         | Stromal (colonref + lungref catch-all)      |
| pericyte        | 1         | Peri                                        |
| stellate        | 1         | Stellate                                    |
| astrocyte       | 1         | Astro (kept in stromal per Heath's call)    |
| placenta        | 1         | placenta                                    |

This breaks the identifiability problem. The model now sees a single endothelial
cell type informed by 13 atlases (instead of four near-duplicate cell types
each informed by 1-15 atlases), and similarly for fibroblast and adipocyte.

**No new atlases are added.** This pipeline uses only the data already in
`iamatlas_mcmc_inputs.csv`. Atlas vault expansion (e.g., adding GSE142439 or
GSE51954 for additional fibroblast coverage) is a separate v0.2 effort
documented in the surveillance notes.

---

## How to run

### Prerequisites
- Anaconda Prompt or any Python 3 environment
- `iamatlas_mcmc_inputs.csv` in the current directory
- The same MCMC build script you used for the original runs

### Step 1 — Harmonize the labels (5 seconds)

```
python harmonize_stromal_labels.py
```

This reads `iamatlas_mcmc_inputs.csv` and writes
`iamatlas_mcmc_inputs_stromal_harmonized.csv` with stromal-class cell_type
entries remapped. **All other arch_class rows pass through unchanged** —
cycling, immune, secretory, etc. are not affected.

The script prints a before/after summary so you can verify the mapping
before proceeding. If any unmapped stromal labels appear in the warning
section, stop and add them to `STROMAL_LABEL_MAP` in the script before
proceeding.

### Step 2 — Smoke test on a single batch (~30 min)

Re-point your existing MCMC build script at the harmonized inputs file:

  Original: input_csv = 'iamatlas_mcmc_inputs.csv'
  New:      input_csv = 'iamatlas_mcmc_inputs_stromal_harmonized.csv'

Run only the **first batch** of stromal (modify the batch loop to break after
batch 1) with tightened sampler config:

  target_accept = 0.99   (was 0.85 default)
  tune          = 2000   (was 1000 default)
  draws         = 2000   (was 1000 default)

**Expected outcome:** R-hat <= 1.02, ESS >= 400, divergences <= 5.

If smoke test PASSES, proceed to Step 3.
If smoke test FAILS, do not run Step 3. Capture the failure mode (R-hat,
ESS, divergence count, posterior mean drift across chains) and we re-evaluate.

### Step 3 — Full stromal re-run (~6-8 hours)

Same MCMC build script, same harmonized inputs, same tightened sampler config,
all 5 batches.

Expected output files (overwriting the failed originals):
  iamatlas_v0_1_stromal_brightness.csv
  iamatlas_v0_1_stromal_per_celltype.csv
  iamatlas_v0_1_stromal_result.json

### Step 4 — Verify convergence

Open the new `iamatlas_v0_1_stromal_result.json` and confirm:
  - convergence.rhat_max <= 1.02
  - convergence.ess_min >= 400
  - convergence.n_diverging <= 20  (across all 4000 samples)

If any of these fail, the harmonization wasn't sufficient and we need to
revisit the sampler config or the cell-type structure further.

---

## What this does NOT do

- **Does not modify any other class.** Cycling, immune, progenitor, secretory,
  terminal, stem_pluri, stem_adult inputs are unchanged. The chain runs
  currently in progress / queued are unaffected.
- **Does not modify the MCMC build script.** You re-point it at the harmonized
  inputs file and tighten three sampler hyperparameters. The model code itself
  is unchanged.
- **Does not add new atlases.** This is purely a label-cleanup operation on
  existing inputs. Future atlas additions (GSE142439 fibroblast EPIC, GSE51954
  dermis 450K, etc.) are tracked separately for v0.2 atlas vault expansion.
- **Does not modify the H_min anchor.** The class anchor for stromal stays at
  0.8630 per the architectural class definitions. The re-run produces a tighter
  posterior estimate of the per-cell-type means within that class, not a
  different class anchor.

---

## Why this works

The chains don't care about names. They care about patterns. With 13 atlases
all carrying the same endothelial methylation pattern, when those atlases all
write to a single canonical "endothelial" entry, the model has overwhelming
evidence for what the endothelial posterior should look like. Same for
fibroblast (11 atlases) and adipocyte (4 atlases).

Single-atlas cell types (pericyte, stellate, astrocyte, placenta) remain
weakly supported, which means their per-cell-type posterior estimates will
have wider credible intervals — but that's honest uncertainty rather than the
chain-breaking degeneracy that killed the original run. The model can fit
single-atlas cell types when there's no competing label trying to claim the
same signal.

---

## Files in this package

  harmonize_stromal_labels.py    The harmonization script
  STROMAL_RERUN_README.md        This file
