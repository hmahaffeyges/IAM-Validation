# PreIAMAtlas_Build

> **STUB — full README in progress.** This file stands up the folder and states the two governing principles; the complete write-up is being prepared.

This folder holds the **pre-IAMAtlas-build development phase** of the biophysics track — the foundational work that asked two questions, in order:

1. Does IAM's law apply to the methylome at all?
2. What is the language of cellular methylation in disease?

It is kept as a distinct phase, separate from the current production chain, so the **methods do not get mixed**. The current, production IAMAtlas-post-build chain lives only in `../AstroGenetics/CPG_CMB_v1/` and is not part of this folder.

## Contents

- **`Preliminary_Test_Results/`** — the evidence reports and VAL runs from this phase (`validation_runs/`, `evidence/`, `validation/`, `post_build_evidence/`). Named *preliminary* deliberately: these are early results, not final, prospectively validated claims.
- **`RETIRED_Phase1_PreBuild_Cards/`** — the **pre-build disease cards**. These are retained, not discarded: they are run through the new chain of custody (`../AstroGenetics/CPG_CMB_v1/`) as a **reference** for comparison.
- **`iamatlas_production_data/`** — the v0.1 atlas-build machinery (per-class MCMC scripts, merge, deconvolution, cards, EDEAR roadmap) used to produce the atlas.
- **`atlas_vault/`** — the pre-build vault/runtime structure (read-only history).
- **`chain_of_custody/`** — the L4 component-separation and L9 null-suite chain-integrity scaffolding from this phase.
- **`scripts/`** — pre-build utility scripts.

The current production IAMAtlas-post-build chain — including the live atlas, its provenance, and the brightness/class archives — lives only in `../AstroGenetics/CPG_CMB_v1/` and is intentionally **not** part of this folder.

## Two governing principles (these hold across the whole biophysics track)

1. **Derived, not comparison.** The method scores `A = H(v) / H_min` against *derived* architectural floors. It is **not** a comparison/deconvolution against a reference panel, and it does not pool cohorts. Do **not** import Loyfer/Moss reference-atlas framing, pooled-cohort centroids, or Mahalanobis-distance-to-a-population vocabulary — that is a different paradigm and conflating the two is an error.
2. **No foregrounds subtracted.** The production chain subtracts no age/sex/smoking/batch foreground. Smoking-, age-, and sex-driven methylation change is part of the cellular departure the score measures — removing it removes signal. Intake facts are report annotations, never operands in the score.
