# IAMAtlas MCMC — The Flatness Problem and the Fix
### READ THIS BEFORE REBUILDING ANY CLASS

**Audience:** the next person, AI, or future-Heath who re-runs the IAMAtlas MCMC to add
cell types / atlases or rebuild a class.
**Written:** 2026-05-25, during the v0.1 rebuild, while the lesson was fresh.
**Status:** Sections 1, 2, and 3 are all CONFIRMED (Section 3 confirmed by progenitor,
the first finished multi-cell class — see below).

---

## THE ONE RULE (if you read nothing else)

**Judge a rebuilt class by whether its cell types come out DISTINCT in the output
`per_celltype.csv` — NEVER by R-hat alone.**

R-hat misled us twice in opposite directions:
- The OLD script showed R-hat ~1.01 (looks perfect) while the atlas was BROKEN.
- The FIXED script shows R-hat ~1.4-2.6 (looks alarming) while the atlas is FINE.

The convergence diagnostics (R-hat, ESS, divergences) cannot see the failure that
actually matters here. Only the output can. Always run the distinctness test (bottom
of this doc).

---

## 1. THE FLATNESS PROBLEM — what went wrong (CONFIRMED)

The original v0.1 atlas was built with `iamatlas_v0_1_mcmc_batched.py` (the OLD script).
Its per-pair model was:

    z        = Normal(0, 1, shape=(n_cpg, n_ct))
    mu_logit = mu_class_logit + sigma_class_logit * z

Two compounding problems:

- **The ridge (the killer).** `sigma_class_logit` (spread of per-CpG values) and
  `log_kappa` (per-atlas observation precision) trade off against each other — the data
  identifies only their combination, not each alone. So `sigma_class_logit` collapsed
  toward 0. When it hits 0, `mu_logit -> mu_class_logit` for EVERY cell type.
- **The swamp.** `z` was shaped over the full (n_cpg x n_ct) grid, creating parameters
  for (cpg, celltype) pairs that were never observed — a high-dimensional region the
  sampler cannot constrain.

**What the collapse produced:**
- Every cell type within a class came out NEARLY IDENTICAL. Example (immune, old build):
  CD4_T-cells, CD8_T-cells, Mono all equal to ~4 decimals (mean|diff| ~0.0001, corr ~1.0).
  Terminal: all 12 cell types identical in the output, even though their INPUTS differed
  by up to 0.50 mean|diff|.
- Class-level brightness went nearly flat: ~0.47 genome-wide, std ~0.005 — which is
  BELOW the per-CpG posterior SD of ~0.010. Signal smaller than noise.

**Why it was invisible (the dangerous part):**
A collapsed/flat posterior is TRIVIALLY easy to converge to. The old build reported
R-hat ~1.01, ESS in the hundreds, 0 divergences, status "complete." Every standard QC
check passed. The break only became visible when comparing TWO CELL TYPES WITHIN A CLASS
— which no routine check did, until the deconvolver needed to tell cells apart.

**Do not trust toy-data equivalence checks.** A check that the FIXED model "reproduces
the old model at corr 0.99996" was run on BALANCED TOY data, where `sigma_class_logit`
IS identifiable. It never exercised the production failure mode (real, imbalanced,
sparse coverage). It gave false confidence that the 7 old-script classes were fine.
They were not — they were flat.

---

## 2. THE FIX (CONFIRMED)

Use **`iamatlas_v0_1_mcmc_batched_FIXED.py`** for every build/rebuild. Its model:

- Defines parameters only over OBSERVED (cpg, celltype) pairs — kills the swamp.
- Replaces the z / sigma_class_logit construction with a DIRECT per-pair prior:

      mu_logit = Normal(mu = mu_class_logit, sigma = 3.0, shape = n_pairs)

  — kills the ridge (there is no `sigma_class_logit` left to collapse).

**Confirmed on the rebuild (real production data, not toy):**
- stem_pluri / stem_adult outputs: full 0-1 range, std ~0.34, per-CpG signal/noise
  ratio ~41 (vs ~0.5 when collapsed). Only ~1% of CpGs near the old flat 0.47.
- terminal slice (FIXED): Cortical_neurons vs Left_atrium 0.10 apart (was 0.0001).
- immune slice (FIXED): CD4_T vs CD8_T 0.15 apart, CD4_T vs Mono 0.42 apart
  (all were identical under the old script).

The OLD script is kept in the repo for provenance only. Never use it to build.

---

## 3. WHAT THE FIXED SCRIPT LOOKS LIKE WHILE RUNNING — do NOT panic (CONFIRMED)

The FIXED script produces **high R-hat (1.4-2.6) and low ESS (5-9)** on multi-cell
classes, with **zero divergences on every batch**. This is the opposite of the old
script's pretty 1.01. THIS IS EXPECTED. Do NOT change sampler settings to force R-hat
down — doing so risks re-introducing the collapse you just fixed.

**Interpretation:** the high R-hat reflects slow mixing on the variance/uncertainty
parameters under the deliberately wide `sigma=3.0` prior, while the per-pair MEANS
(which scoring actually uses) come out correct and distinct. Zero divergences across
every batch = healthy sampler geometry, not pathology. (Contrast: the original stromal
failure that forced the fix threw thousands of divergences. None here.)

**CONFIRMED by progenitor (2026-05-25), the first finished MULTI-CELL class:**
progenitor ran all 97 batches at R-hat 1.4-2.6, ESS 5-9, zero divergences — the
"alarming" profile. Its output per_celltype.csv (11 cell types) tested DISTINCT:
  - Zero of 25 cell-pairs near-identical (none below mean|diff| 0.005).
  - Median pairwise mean|diff| 0.043; max 0.38; 10 of 25 pairs above 0.05.
  - Every cell spans the full 0.01-0.99 range, std ~0.34 (real signal, not flat 0.47).
  - Biologically correct: the closest pairs are adjacent myeloid progenitors
    (CMP/GMP/MPP/L-MPP, mean|diff| 0.015-0.024, corr ~0.99 — genuinely similar lineage
    stages, correctly placed close but still separable). The most distinct pairs are
    neural precursors (OPC, NeuIm) vs blood progenitors (mean|diff| 0.34-0.38,
    corr ~0.09 — correctly separated lineages).
So: high R-hat on the FIXED script is BENIGN for the means. The single-cell stems showed
the same R-hat profile with excellent output (signal 41x noise); progenitor now confirms
it holds for a multi-cell class with biologically sensible cell separation. Trust the
output, not the R-hat.

**Caveat for sparse cells:** cell types with very low atlas coverage (in progenitor:
erythroblast, megakaryocyte, nRBC — tens to low-hundreds of CpGs in a sample) produce
real-looking values but cannot be distinctness-tested as confidently as well-covered
cells. This is a data-coverage limitation, not a build failure. The deconvolver's
two-tier design (reliable class-level + indicative cell-level) already handles this:
well-covered cells deconvolve reliably; sparse cells are lower-confidence.

---

## THE DISTINCTNESS TEST (the judge — run on every finished class)

Load the class's `per_celltype.csv`. For every PAIR of cell types, on the CpGs they
both have data for (>=200 shared), compute mean-absolute-difference and correlation:

- **DISTINCT** — no pair near-identical (none below ~0.005), median mean|diff| well
  above 0.01, several pairs > 0.05 -> the class is REAL and usable.
- **IDENTICAL / SMEARED** — pairs at ~0.0001, correlations ~1.0 -> COLLAPSED. Do not use.

Note: genuinely similar cell lineages (e.g. adjacent progenitors) SHOULD be close
(mean|diff| ~0.015, corr ~0.99) — that is correct biology, not collapse. Collapse is
~0.0001 / corr 1.000 across ALL pairs including ones that should differ.

For SINGLE-cell classes (stem_pluri, stem_adult): no cells to compare; instead confirm
the per-CpG signal spans the full range (std ~0.3, NOT ~0.005) and signal/noise >> 1
(we saw ~41).

---

## REBUILD SETTINGS USED (for reproducibility)

- Script: `iamatlas_v0_1_mcmc_batched_FIXED.py`
- batch_size 5000 for all classes EXCEPT immune, which uses **1500** (63 cell types /
  ~5M observations; OOMs at 5000 — a hard lesson the first time).
- tune 1000, draws 1000, chains 4, cores 4 (hard-coded), target_accept 0.95.
- Machine: 16 logical cores / 32 GB RAM. Ran the 6 non-immune classes in parallel,
  3 at a time (12 cores, ~8.5 GB free with 3 running). Did NOT add a 4th while 3 ran
  (8.5 GB free is too thin a margin to risk an OOM mid-run).
- **immune runs LAST and ALONE.** Note: a single run uses only 4 cores (cores=4 hard-
  coded), so on a 16-core machine 12 cores sit idle for ~8 days. This is accepted as the
  SAFE path on 32 GB — fixing it requires both a code change AND more RAM (see below).
- arviz note: the FIXED script calls `az.summary(..., stat_focus="mean")`. arviz >= 0.15
  supports this (build machine had 0.23.4). On older arviz, remove the `stat_focus`
  argument; it only affects the convergence summary, not the model or saved posteriors.

---

## FUTURE SPEEDUP — immune CpG-chunk parallelism (NEW-MACHINE ONLY)

immune CANNOT be split by CELL TYPE into separate runs — all 63 cells share one class
anchor (mu_class_logit, the H_min floor); separate sub-runs would compute different
baselines and break the common scale.

immune CAN be split by CpG-CHUNK into parallel runs — CpGs are independent, and each
chunk fits all 63 cells against the same shared anchor (the anchor is a property of the
cells, identical across chunks, so it comes out consistent). Merge chunk outputs after.
~4-8 parallel chunks ≈ 4-8x faster (8 days -> ~1-2 days).

This requires (a) a small `--chunk i of N` (or --cpg_start/--cpg_end) edit to the arg
parser + batch loop (model untouched), and (b) enough RAM to hold multiple 63-cell
chunks at once. On 32 GB this OOMs. **Only do multi-chunk immune on a high-RAM machine
(64+ GB, ideally 256+ GB).** On a 500 GB / 64-core box, run 8 chunks and immune finishes
in a fraction of a day. Stage the code edit then, not before.

---

## WHEN PUSHING TO THE REPO (checklist)

- Fold sections 1-3 into `MCMC_BUILD_LESSONS.md` and reference from the atlas README.
- CORRECT the prior claim in MCMC_BUILD_LESSONS that the 7 old-script classes are
  "valid as-is, converged cleanly, not re-run." They were FLAT and have been rebuilt
  with the FIXED script. State that plainly so it is never repeated.
- Confirm the FIXED script in the repo is byte-identical to the one used for the rebuild.
- Update INVENTORY.json (new atlas SHA, new per-class output archive).
