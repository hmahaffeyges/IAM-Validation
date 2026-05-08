# EDEAR Master Roadmap — IAMAtlas Build Through Customer Number One

**Date sealed:** 2026-05-05
**Revised:** 2026-05-08 (see Revision Log at end)
**Author:** Heath W. Mahaffey + Walther
**Status:** Living document. Updated at each milestone gate.
**Purpose:** Single canonical reference covering (a) what is done, (b) what is in flight, (c) every remaining step from current MCMC run through first paying EDEAR customer. Replaces and merges `IAMATLAS_BUILD_PLAN_v2.md` and `IAMAtlas_atlas_survey_2026_05_04.md`.

**Discipline rule:** Each step has an exit gate. We do not advance past the gate without it passing. No skipping. No "we'll come back to it." The gate either passes or the step blocks and we fix it.

**Architecture rule:** One IAMAtlas. It deepens. There is no v2 product. Each atlas-input expansion produces a new SHA, INVENTORY entry, GitHub tag. Same artifact.

**Durable-storage rule (added 2026-05-08):** All production MCMC artifacts (inputs, outputs, scripts, staging bundles) live in the GitHub repo at `iamatlas_production_data/` with git LFS for large files. Anything Walther produces or Heath uploads goes there immediately. The container filesystem `/home/claude/` is scratch and disappears between sessions.

---

## IMMEDIATE POST-MCMC GATES (THE THREE THINGS THAT MUST HAPPEN BEFORE STEP 7)

When the current MCMC run finishes (estimated 8-10 days from 2026-05-06), three actions block all forward progress until each is complete. Listed prominently here so they are not buried.

### GATE 1 — Stromal re-run with label harmonization + tightened sampler (BLOCKING)
Stromal class failed convergence in the main run (R-hat 3.67, ESS 4, 6 divergences). Root cause diagnosed 2026-05-08: 17 cell-type labels across 17 atlases included multiple atlas-specific naming variants of the same biological cell type (EC + Endo + Vascular_endothelial_cells + endothelial = 4 labels for endothelial; FB + Fib + fibroblast = 3 labels for fibroblast; etc.). The model treated each label as a distinct cell type with its own posterior, creating identifiability degeneracies — chains never mixed because the endothelial signal could be assigned to any of four interchangeable buckets. **Tightened sampler config alone will NOT fix this** — the failure is model identifiability, not NUTS geometry. Harmonization script staged at `iamatlas_production_data/stromal_rerun/harmonize_stromal_labels.py`. Detailed config in Step 6.5 below.

### GATE 2 — Gasparoni terminal-class addition (STAGED, READY)
Bundle staged 2026-05-06 at `iamatlas_production_data/terminal_addition/gasparoni_terminal_addition.csv` (durable in GitHub LFS). 957,638 rows ready for append. Procedure in Step 6.75 below. Brings terminal class from 30,895 rows to 988,533.

### GATE 3 — Re-run terminal class against expanded pool
After Gasparoni is appended, re-run terminal only. ~2 hours wall-clock. Compare new posterior to existing terminal result.

**No other classes need re-running before v1.** Stem_pluri converged cleanly (R-hat 1.01, ESS 501, Pearson 0.919) and does not need a re-run; future Roadmap Epigenomics integration is a v1.1 expansion, not a v1 prerequisite.

---

## PART 1 — STATE SUMMARY (what exists, as of 2026-05-05)

### 1.1 Production MCMC run — currently in flight on Heath's son's gaming PC

| Class | Status | Elapsed | R-hat | ESS | Divergent | Pearson | MAE |
|---|---|---|---|---|---|---|---|
| terminal | ✓ done 2026-05-04 18:50 | 3,750s | 1.01 | 1493 | 0 | 0.799 | 0.356 |
| stromal | ✗ FAILED CONVERGENCE | 17,485s | **3.67** | **4** | 6 | 0.897 | 0.202 |
| stem_pluri | ✓ done 2026-05-05 03:17 | 12,842s | 1.01 | 501 | 2 | 0.919 | 0.322 |
| stem_adult | ✓ done 2026-05-05 06:52 | 12,910s | 1.01 | 723 | 0 | 0.904 | 0.338 |
| secretory | 🟡 RUNNING — Batch 13/77 | ~1500s/batch | 1.010 across 1-12 | 1055-1958 | 0 | -- | -- |
| cycling | queued | -- | -- | -- | -- | -- | -- |
| progenitor | queued | -- | -- | -- | -- | -- | -- |
| immune | queued | -- | -- | -- | -- | -- | -- |

**Honest revised timeline (as of 2026-05-06 evening):** 8-10 days remaining. As of 2026-05-06, secretory is on batch ~53/77, taking ~27 min per batch. Remaining: ~11 hr secretory + ~35 hr cycling + ~50-70 hr progenitor + ~100+ hr immune. Immune is the largest class by row count (4.99M rows, 4× larger than secretory) and will dominate the back end of the timeline.

### 1.2 Atlas vault — what's actually feeding the production MCMC

The production matrix `iamatlas_mcmc_inputs.csv` (732 MB, 10,938,662 rows) draws from these 22 atlas sources, ranked by row contribution:

| # | Atlas source | Rows | Cell types | Status | Role |
|---|---|---|---|---|---|
| 1 | reinius_gse35069 | 4,824,199 | 10 immune | ✓ full HM450 | Primary immune anchor |
| 2 | jung2017_hspc | 2,894,526 | 6 progenitor | ✓ full HM450 | Primary progenitor / stem_adult anchor |
| 3 | boccellato_stomachref | 2,282,802 | gastric mucosoid | ✓ | Cycling / secretory gastric anchor |
| 4 | nazor2012_stem_pluri | 482,421 | 1 (pluripotent) | ✓ full HM450 | Primary stem_pluri anchor |
| 5 | loyfer_moss_2018 | 152,625 | 25 cell types | ⚠ curated 6,105-CpG subset | Secondary anchor across 6 classes |
| 6-18 | episcore_*ref (13 tissues) | 4,904-42,845 each | varies | ✓ bridged | Tissue-specific supplements |
| 19-22 | salas/epidish/caggiano panels | 1,316-7,200 each | immune cell types | ✓ | Immune sub-resolution |

**Architecture-class row balance in production matrix:**

| Class | Rows | Note |
|---|---|---|
| immune | 4,987,426 | Over-anchored (Reinius + Salas + UniLIFE + Loyfer + EpiDISH) |
| progenitor | 2,427,833 | Solid (Jung 5 types × 482K CpGs + Loyfer) |
| cycling | 1,219,336 | Solid (Boccellato + 4 EpiSCORE + Loyfer) |
| secretory | 1,211,597 | Solid (Boccellato + 4 EpiSCORE + Loyfer) |
| stem_pluri | 482,421 | Full Nazor whole-array |
| stem_adult | 482,421 | Jung HSC at 482K CpGs |
| stromal | 96,733 | **SPARSE** — failed convergence at 17 cell types (R-hat 3.67, ESS 4) |
| terminal | 30,895 | **SPARSEST** — Loyfer curated + 5 EpiSCORE + Caggiano TIM |

**Critical finding from disk audit (2026-05-05):** The atlas labeled `loyfer_moss_2018` is a curated 25-celltype × 6,105-CpG subset, NOT the full Loyfer 2023 GSE186458 39-cell-type whole-HM450 atlas. The full atlas would expand terminal class ~60×.

### 1.3 Atlas vault — extracted but not yet integrated

| Atlas | Disk location | Format | Status |
|---|---|---|---|
| Gasparoni 2018 GSE66351 | `/home/claude/atlases_pollard_track/gasparoni_2018_GSE66351_brain_celltype_atlas.csv.gz` | 478,819 CpGs × 2 cell types (cortical_neuron + glia), n=16 donors, HM450 | ✓ EXTRACTED, NOT IN MATRIX. Drop-in ready for terminal class |

### 1.4 Atlas vault — gaps identified (per 2026-05-04 survey)

| Gap | Survey priority | Notes |
|---|---|---|
| Full Loyfer 2023 GSE186458 (39 types × full HM450) | CRITICAL | Currently a 6,105-CpG curated subset — would expand terminal ~60× |
| Roadmap Epigenomics (111 epigenomes) | CRITICAL | ESCs, iPSCs, fetal tissues, brain regional — fills stem_pluri / progenitor gaps |
| Moss 2018 GSE122126 fully | HIGH | Only the immune EPIC subset is currently in the matrix |
| Pollard 2025 medRxiv ONT neural | MEDIUM | Motor / dopaminergic / microglia / Schwann — terminal class neural sub-resolution |
| BLUEPRINT controlled-access WGBS | DEFER | Friction not worth pre-commercial |
| Single-Cell Body Atlas 2025 (Zhou) | DEFER | Single-cell sparse data has different statistical properties |
| TCGA-normal pan-cancer | DEFER | Bulk tissue, not sorted cell — cross-platform validation only |

### 1.5 Code / infrastructure already built

| Asset | Path | Status |
|---|---|---|
| MCMC input matrix | `iamatlas_brightness_pilot/iamatlas_mcmc_inputs.csv` | ✓ |
| Production MCMC runner (batched) | `iamatlas_v0_1_mcmc_batched.py` | ✓ deployed to son's PC |
| Merge script (8 results → IAMAtlas.csv) | `merge_iamatlas_v0_1.py` | ✓ |
| Step 7 validation suite (6 checks A-F) | `step_7_bundle/` | ✓ scripted, awaits matrix |
| IAMAtlas deconvolver | `iamatlas_deconvolver.py` | ✓ built; synthetic test passed (CD4_T=0.6 + epithelial=0.4 recovered exactly) |
| Xu 2020 directional Rule A panel | `xu2020_breast_directional_RuleA.json` | ✓ 98 CpGs frozen |
| Atlas vault (durable, GitHub) | `Biological_Physics/atlas_vault/` | partial — needs sync after MCMC complete |

### 1.6 Steps already complete (sealed)

| Step | Title | Status |
|---|---|---|
| 1 | Atlas-input inventory | ✓ COMPLETE — 215 (atlas, cell-type) rows, 174 cell-types mapped, 0 unmapped |
| 2 | Bridge bridge-blocked atlases | ✓ COMPLETE — 8 EpiSCORE tissues bridged at 94.7-100%; Caggiano bridged |
| 3 | Fill anchor gaps | ✓ COMPLETE — Nazor 2012 stem_pluri, Jung 2017 stem_adult; all 8 classes anchored |
| 4 | Scale CpG universe | ✓ COMPLETE — 483,092 unique HM450 CpGs |
| 5 | Build MCMC input matrix | ✓ COMPLETE — 10,938,662 rows, 732 MB CSV, all 8 classes have inputs |
| 6 | Production MCMC | 🟡 IN FLIGHT — 4 of 8 classes done, 1 failed (stromal), 1 running (secretory), 3 queued |

---

## PART 2 — REMAINING STEPS TO COMPLETE IAMATLAS V1

### STEP 6 (continuing) — Let the production MCMC finish

**Action.** Do nothing. Do not interrupt. Let secretory → cycling → progenitor → immune complete on son's PC.

**Estimated remaining wall-clock:** 5-7 days.

**Exit gate.** 8 result files exist in `iamatlas_v0_1_output/`:
- `iamatlas_v0_1_terminal_result.json` (✓ exists)
- `iamatlas_v0_1_stromal_result.json` (✓ exists, but failed gate)
- `iamatlas_v0_1_stem_pluri_result.json` (✓ exists)
- `iamatlas_v0_1_stem_adult_result.json` (✓ exists)
- `iamatlas_v0_1_secretory_result.json` (pending)
- `iamatlas_v0_1_cycling_result.json` (pending)
- `iamatlas_v0_1_progenitor_result.json` (pending)
- `iamatlas_v0_1_immune_result.json` (pending)

---

### STEP 6.5 — Stromal re-run with label harmonization + tightened sampler [BLOCKING — see GATE 1 above]

**Action.** After main run completes (do NOT interrupt before then), execute the staged harmonization + re-run procedure:

1. Run `harmonize_stromal_labels.py` against the production inputs file. This produces `iamatlas_mcmc_inputs_stromal_harmonized.csv` with the stromal cell_type column remapped:
   - EC + Endo + Vascular_endothelial_cells + endothelial → `endothelial` (now 13 atlases backing one canonical cell type, vs. 4 fragmented labels with 1-15 atlases each)
   - FB + Fib + fibroblast → `fibroblast` (11 atlases backing one canonical cell type)
   - Adipocytes + Fat + adipose → `adipocyte` (4 atlases)
   - SM + SMC → `smooth_muscle` (2 atlases)
   - Single-source labels (Peri, Stellate, Astro, placenta) standardized to canonical names
   - "Stromal" catch-all → `stromal_other` (2 atlases)
   - Result: 17 fragmented labels collapse to 9 canonical cell types with proper multi-atlas support
2. Smoke test on a single batch (~30 min) with tightened sampler:
   - `target_accept = 0.99`
   - `tune = 2000`, `draws = 2000`
3. If smoke test passes (R-hat ≤ 1.02, ESS ≥ 400, divergences ≤ 5), proceed to full 5-batch re-run (~6-8 hours).

**Why label harmonization is the fix, not just sampler tightening.** The MCMC build script already uses non-centered Beta-Binomial parameterization on the logit scale, which fixes funnel-geometry NUTS pathologies (per the script's own docstring documenting the original terminal Batch 2 fix). The fact that stromal still failed despite this confirms the failure was multi-modal posterior caused by interchangeable labels, not local sampler geometry. No NUTS config can fix interchangeable-label degeneracies — the model has to see one canonical label per biological cell type for chains to mix.

**Files.** All in `iamatlas_production_data/stromal_rerun/` in the GitHub repo:
- `harmonize_stromal_labels.py` (tested on synthetic input matching production schema)
- `STROMAL_RERUN_README.md` (full procedure, exit gate criteria, contingency for failure)

**Estimated wall-clock:** ~30 min smoke test + 6-8 hours full re-run (sequential, not parallel — one PC).

**Exit gate.** Stromal R-hat ≤ 1.02 across all 5 batches, ESS ≥ 400, divergences ≤ 20 (across all 4000 samples).

**If stromal still fails after harmonization + tightened config:** the diagnosis becomes structural — the harmonized cell types may still represent architecturally distinct sub-populations within stromal that don't share one H_min anchor. In that case Gate G-A2 (Heath sign-off) decides between (a) splitting stromal into sub-classes (e.g., separating perivascular/contractile cell types from ECM-producing cell types) or (b) accepting stromal at the class level with downgraded ESS reporting.

---

### STEP 6.75 — Gasparoni integration + terminal class re-run [BLOCKING — see GATES 2 & 3 above]

**Status as of 2026-05-08:** Staging COMPLETE. Files durable in GitHub at `iamatlas_production_data/terminal_addition/`. Append + re-run pending main MCMC completion.

**Staged files** (all in `iamatlas_production_data/terminal_addition/` in the GitHub repo):
- `gasparoni_terminal_addition.csv` — 957,638 rows, 61 MB, SHA-256 `3294b49880e5248b4cf85fc6bff830a0d6c9363c600fa0f89b16c55f78854f30`
- `stage_gasparoni_for_terminal.py` — staging script, SHA-256 `a496e29bf3fa3078ab8f9fce0eaeac6422207eef558fdff938e56ad007306cc3`
- `GASPARONI_STAGING_README.md` — provenance, decisions, append procedure

**Decisions sealed (Heath, 2026-05-06):**
1. One IAMAtlas, deepens. Gasparoni feeds in as additional terminal-class input.
2. Cell-type label mapping:
   - `cortical_neuron` (Gasparoni) → `Cortical_neurons` (matches Loyfer's existing label; pools donors at the per-CpG level for tightest possible posterior — astro-genetic statistical efficiency by design)
   - `cortical_glia` (Gasparoni) → `Glia` (new terminal-class cell type; biologically honest because Gasparoni's NeuN-negative population mixes oligodendrocytes, astrocytes, and microglia)
3. No researcher names in customer-facing labels. `atlas_source = gasparoni_2018` is internal provenance metadata only.

**Append procedure (after main MCMC run finishes):**
```bash
cd C:\Users\hmaha\OneDrive\Desktop\files
copy iamatlas_mcmc_inputs.csv iamatlas_mcmc_inputs.csv.backup_pre_gasparoni
type gasparoni_terminal_addition.csv | more +1 >> iamatlas_mcmc_inputs.csv
# Verify line count: should be 11,896,300 (was 10,938,663 incl. header)
```

(On Windows, `type ... | more +1` skips the header line. On Linux/macOS, the equivalent is `tail -n +2 gasparoni_terminal_addition.csv >> iamatlas_mcmc_inputs.csv`.)

**Re-run command (after append + after stromal re-run if running them sequentially):**
```bash
python iamatlas_v0_1_mcmc_batched.py --classes terminal \
       --batch_size 5000 --chains 4 --tune 1000 --draws 1000 \
       --target_accept 0.95 --out_dir iamatlas_v0_1_output
```

**Estimated wall-clock for terminal re-run:** ~2 hours on Heath's son's gaming PC. Terminal still has only ~12 cell types (existing 11 + Glia), so per-batch time scales modestly even with 32× more rows.

**Exit gate.** Terminal R-hat < 1.05, posterior H_min within prior 95% interval (0.7728 anchor), Pearson > 0.85 (improvement over 0.799 baseline), ESS > 5,000 expected.

**Why this matters.** Terminal class is currently 30,895 rows (smallest by 3×). Gasparoni adds 957,638 rows — 32× expansion. Closes the data-imbalance with other classes. Adds FANS-sorted neurons + glia at full HM450 from a different lab/protocol than Loyfer — independent confirmation of terminal H_min through donor pooling, which is the astro-genetic statistical-efficiency design Heath chose.

---

### STEP 6.85 — Post-MCMC per-cell-type label consolidation [non-blocking, cleanup]

**Action.** After all chains complete (cycling, progenitor, immune, terminal-with-Gasparoni, stromal-rerun), audit each `iamatlas_v0_1_*_per_celltype.csv` file for label collisions. Build one consolidation script that handles all classes in a consistent way, applied once.

**What this does.** For classes that converged with label collisions present in their inputs (terminal, secretory, and any that show similar patterns in cycling/progenitor/immune), merges duplicate-label columns in the per-cell-type CSV by averaging posterior means and properly combining SDs. Example: terminal output has CM, Left_atrium, heart all encoding the same cardiomyocyte signal — consolidate to one canonical `cardiomyocyte` column. Secretory output has Hep, Hepatocytes, hepatocyte — consolidate to `hepatocyte`.

**What this does NOT do.** No MCMC re-run. The class-level H_min anchors are already correct because they're class-level (not per-cell-type) parameters. The consolidation just cleans up the per-cell-type output for downstream card use and human readability.

**Why we wait until all chains finish.** We don't yet know what label collisions cycling/progenitor/immune will reveal. Better to look at all five remaining per-celltype CSVs once and build one consolidation script that handles them all consistently than to do this piecemeal.

**Estimated wall-clock:** ~2 hours total (audit + script + apply to all output CSVs).

**Exit gate.** Every per-celltype CSV has unique canonical cell-type column names; no duplicate biological cell types under different label names; consolidation mapping documented in `iamatlas_production_data/post_mcmc_consolidation/CONSOLIDATION_README.md`.

---

### STEP 7 — Run the validation suite (already staged in `step_7_bundle/`)

**Action.** Execute `step_7_run_all.py`. The bundle contains six independent checks:

| Check | Purpose |
|---|---|
| A — Exit gates | R-hat < 1.05, ESS > 100, divergent < 5%, Pearson > 0.7, MAE < 0.4 across all 8 classes |
| B — Predictive validation | Held-out cell types not used in training; predict β; report Pearson + MAE |
| C — AD cohort scoring | Run AIBL Stage 1 immune + Rule A directional panel; replicate VAL-051 d=+0.62 |
| D — Breast IAM scoring | Run GSE51057 + GSE51032 with 10yr+ pre-dx stratification; replicate d=+1.78 / +1.36 |
| E — GSE130748 trajectory | Replicate longitudinal pre-dx pattern from primary breast cohort |
| F — Sex/age stratified | Confirm VAL-053 (sex-specific does NOT outperform unified) and VAL-054b (HC-permutation p=0.003) |

**Estimated wall-clock:** ~12-24 hours (most time in C/D/E because they pull large external cohorts).

**Exit gate.** Checks A and B PASS for all 8 classes. C/D/E replicate the published validation results within 1σ. F confirms unified outperforms sex-specific.

**Failure mode.** If any check fails, STOP. Do not advance. Determine whether the failure is data, model, or implementation. Re-run with fix. Document in LESSONS_LEARNED.md.

---

### STEP 8 — Atlas vault freeze (durable storage)

**Action.**
1. Bundle final `IAMAtlas.csv`, `IAMAtlas_celltype_to_class.json`, all per-class result JSONs, MCMC code, validation outputs.
2. Compute SHA-256 for every artifact.
3. Write `INVENTORY.json` with provenance (source URL, citation, license, SHA, integration date) for every atlas in the pool.
4. Push to GitHub `hmahaffeyges/IAM-Validation/Biological_Physics/atlas_vault/` with tag `iamatlas-2026-05-XX`.
5. Mint Zenodo DOI for the bundle.

**Exit gate.** GitHub release published; Zenodo DOI live; SHA verifications pass on round-trip download.

---

### STEP 8.5 — Real-data deconvolver test

**Action.** The deconvolver `iamatlas_deconvolver.py` passed synthetic ground-truth (CD4_T=0.6 + epithelial=0.4 recovered exactly). Now test on real samples:
1. Pull 20 mixed-tissue methylation samples from a public cohort (TCGA blood-tumor mixtures or similar).
2. Run deconvolver against the IAMAtlas matrix.
3. Compare cell-fraction estimates to (a) flow-cytometry ground truth where available, (b) NNLS-against-Loyfer baseline, (c) EpiDISH baseline.

**Exit gate.** Deconvolver matches or beats NNLS/EpiDISH on cell fractions where ground truth exists. RMSE < 0.10 against flow cytometry on ≥80% of samples.

**Failure mode.** If deconvolver underperforms baselines on real data, the synthetic test was insufficient. Add hierarchical noise structure to the deconvolver (heteroscedastic per-cell-type) before public release.

---

### STEP 9 — STANDING atlas-cohort expansion queue (post-v1)

After v1 ships, these expansions deepen the matrix without rebuilding it:

| Atlas | Adds | Priority | When |
|---|---|---|---|
| Roadmap Epigenomics methylation | 30-40 unique epigenomes (ESCs, iPSCs, fetal, brain regional) | HIGH | After v1 ships |
| Full Loyfer 2023 (GSE186458) at native HM450 | 39 cell types × ~482K CpGs (vs current 25 × 6,105) | HIGH | After v1 ships |
| Moss 2018 (GSE122126) fully | RRBS data complementing Loyfer | HIGH | After v1 ships |
| Pollard 2025 medRxiv ONT | 6 neural cell types (motor, dopaminergic, microglia, Schwann, +) | MEDIUM | When ONT-to-cg-ID translation is built |
| MARLIN / Capper 2025 (n=2,540 leukemia) | Calibrates myeloid arm at TCGA-scale | HIGH (heme-epic depends) | Q3 2026 |
| Cuadrat 2026 cardiomyocyte | Extends terminal class | LOW | Opportunistic |
| Adipose-specific atlases | Extends stromal | LOW | Opportunistic |
| GI sub-region atlases | Esophageal/gastric/duodenal/ileal/cecal | LOW | When card library demands |
| Reproductive-tract atlases | Endometrium, fallopian tube, ovary, testis | LOW | When card library demands |

**Action per atlas.** (a) Acquire data, (b) bridge to HM450 if needed, (c) map cell types to architecture classes, (d) add rows to `iamatlas_mcmc_inputs.csv`, (e) re-run hierarchical MCMC, (f) version in vault with new SHA, (g) push to GitHub.

**Exit gate.** N/A — standing queue. Each addition produces an IAMAtlas update, not a product release.

---

### STEP 10 — Age layer (separate matrix from the main IAMAtlas)

**Goal.** Per-CpG age regression layer (β as a function of age) so customer scoring adjusts for age-related drift.

**Action.**
- Build from UniLIFE 2025 (the only atlas with native age metadata at lifespan-spanning donor structure).
- For each CpG, regress β on donor age, output (slope, intercept, R², n) per CpG.
- Write `IAMAtlas_age_layer.csv` — separate file, lives alongside the brightness matrix.

**Estimated wall-clock:** ~4 hours after main matrix lands.

**Exit gate.** Age regression converges for ≥80% of HM450 CpGs. Per-CpG age-slope distribution biologically sensible (mostly small, fat tail of true age-correlated CpGs matching Horvath/PhenoAge top hits).

---

### STEP 11 — IAM cellular age clock (methods paper, NOT product-blocking)

**Goal.** Train a regression that uses the 8 architecture-class A-scores to predict chronological age. Residual = IAM cellular age departure.

**Why this is novel.** Existing methylation-age clocks (Horvath, PhenoAge, GrimAge, DunedinPACE) train elastic-net regression directly on β values. The IAM equivalent uses the 8 architecture-class A-scores as features instead. **Single methods paper of its own** if it works.

**Action.**
- Pull every cohort with chronological age + EPIC/450K methylation.
- Compute 8 A-scores per sample against IAMAtlas.
- Train regression (elastic-net or Bayesian linear) of age ~ A-scores.
- Residual = IAM cellular age departure.

**Exit gate.** MAE < 5 years on independent test cohort (matching Horvath chronological-age accuracy). Per-class A-score coefficients biologically interpretable.

---

## PART 3 — REMAINING STEPS FROM IAMATLAS V1 TO CUSTOMER NUMBER ONE

This is where the IAMAtlas stops being a research artifact and becomes the engine inside a paying product. The order here matters: matrix → cards → diagnosis layer → server → demo → first customer.

### STEP D1 — Card library completion (17 disease cards)

**Status.** 12-panel battery defined; gastric/esophageal epic card mid-sprint; HCC card with VAL-064 no-documented-risk subgroup signal documented (Marcus-analog, n=19, paired d=+0.6166, p=0.0072).

**Cards in scope:**
1. AD-immune (VAL-051)
2. Breast-secretory
3. CRC-immune-inv
4. CRC-secretory
5. Cervical-secretory (CIN stratified)
6. LGG/GBM-terminal (tissue + blood)
7. Parkinson's
8. Prostate
9. Pancreatic
10. MS
11. Cardiac
12. Aging
13. HCC (Marcus card — no-documented-risk subgroup is the Marcus-analog signal)
14. Gastric (in flight, Fritsche/Boccellato 2022 GSE141660 reference)
15. Esophageal (combined gastric+esophageal scope decision pending)
16. Ovarian (queued)
17. Hematologic (heme-epic card; depends on MARLIN integration in Step 9)

**Per-card workflow (the canonical seven-file update — already in TESTING_CHECKLIST):**
1. Update TESTING_CHECKLIST.md
2. Update EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md
3. Update README_MASTER_v2_X.md
4. Update LESSONS_LEARNED.md
5. Write/update card README
6. Write/update card JSON (panel definitions, H_min anchors, scoring rules)
7. Write/update Evidence Report HTML

**Per-card external artifacts (push to GitHub):**
- Sealed VAL python file
- Prereg
- Outcome.md
- Results JSON
- Stratified results JSON
- Manifest
- Metadata
- Biological_Physics local README

**Exit gate per card.** All seven internal files updated; all eight external artifacts pushed; cross-card calibration check passes (CCL-037 LL-CROSS-COHORT-CALIBRATION: paired AND same-pipeline AND batch-corrected). Reproducibility triple complete (CHK-7.6: code + inputs + environment + expected output).

**Estimated wall-clock per card:** 1-3 days depending on cohort availability and pre-existing work.

**Total remaining card work:** ~3-6 cards still need to reach v1 quality (gastric, esophageal, prostate, MS, ovarian, heme-epic). The other 11 are at v0.5 or better.

**Exit gate (whole step).** All 17 cards have sealed prereg, complete outcome doc, evidence report HTML page, GitHub artifacts pushed, manifest tagged.

---

### STEP D2 — The diagnosis layer (translation from A-scores to clinical recommendations)

**This is the layer that makes EDEAR a clinical product, not a research tool.** A customer doesn't want eight A-scores. They want: *what does this mean, and what do I do?*

**Architecture (the single-pipeline KISS version — no multi-stage filters needed because the IAMAtlas does the discrimination upstream):**

```
Customer methylation file (β values)
         ↓
[Stage 0 — Input validation + QC]
   - Reject malformed CSVs
   - Reject if <80% HM450 coverage
   - Sample-level QC: bisulfite conversion check, batch-effect flag, outlier detection
         ↓
[Stage 1 — IAMAtlas deconvolver]
   - Cell composition vector across all 8 architectural classes + per-cell-type
   - Per-class A-scores (one per architectural class)
   - Confidence intervals from MCMC posterior
         ↓
[Stage 2 — Demographic adjustment]
   - Age adjustment via age layer (per-CpG age-slope correction)
   - Sex adjustment for sex-dimorphic CpGs
   - Cellular age (residual after age adjustment) reported separately
         ↓
[Stage 3 — Floor departure detection]
   - Per-class A-score → tier classification:
     * Tier 0 NORMAL: A < 1.00
     * Tier 1 MARGINAL: A 1.00-1.05 (at floor, within healthy range)
     * Tier 2 DETECTABLE: A 1.05-1.10 (departure — flag)
     * Tier 3 URGENT: A 1.10-1.20 (breach — refer)
     * Tier 4 FLOOR_BREACH: A > 1.20 (severe — immediate referral)
   - Bidirectional immune-class check: if immune class shows
     simultaneous nulls (Test 1 pooled-entropy ≈ 0) AND directional
     panel passes, flag as bidirectional cancellation pattern
     (AD-instance signature; CCL-027 four-question protocol applies)
         ↓
[Stage 4 — Card matching]
   - Filter cards by: tissue applicability, age range, sex applicability,
     CpG panel availability in customer's array
   - For each matched card, run that card's directional panel
   - Output: card-specific A-scores, panel scores, sigma stratification
         ↓
[Stage 5 — Diagnosis translation]
   - For each card that hit Tier 2+: clinical context paragraph,
     panel score with CI, age/sex-matched Z-score, recommendation
     tier, Marcus-class flag if no-documented-risk + Tier 3
   - Concordance check: if card panel signal disagrees with class
     A-score, flag as "single-panel signal" (could be panel-specific
     false-positive)
   - Plain-language summary at top, technical detail beneath
         ↓
[Stage 6 — Report generation]
   - JSON (machine-readable for lab partner integration)
   - HTML (web-portal viewing)
   - PDF (printable, signed by customer's clinician)
```

**Why no multi-stage filter pipeline.** Earlier EDEAR designs ran sequential filters (Stage 1 red-flag → Stage 2 tissue-specific → Stage 3 immune complexity). The IAMAtlas makes that unnecessary because **deconvolution is one operation that simultaneously gives us all 8 class A-scores and all per-cell-type estimates**. The cards then run against the relevant A-scores in parallel, not sequentially. Single Bayesian deconvolution replaces three stages of filters.

**The diagnosis translator outputs (per card that matched):**
- Card name and clinical context
- Panel score and sigma stratification (Z-score against age/sex-matched healthy reference)
- Floor departure status with tier
- Confidence interval (Bayesian credible interval from MCMC posterior, not frequentist p-value)
- Concordance check (does the panel signal agree with the class A-score? if not, flag — could be a false-positive specific to one panel)
- Recommendation tier:
  - **Tier 0 (no signal):** No action recommended.
  - **Tier 1 (subclinical drift):** Repeat in 12 months. Lifestyle / preventive interventions noted.
  - **Tier 2 (departure):** Repeat in 6 months OR clinical workup as appropriate.
  - **Tier 3 (breach):** Refer to specialist for confirmatory diagnostic.
  - **Tier 4 (floor breach):** Immediate clinical referral.
- Marcus-class flagging: if HCC card or any card hits Tier 3+ in a no-documented-risk patient, flag specifically — that's the failure mode that motivated EDEAR.

**Card-information integration mechanics.** Each card JSON file (`card_v0_X.json`) already contains: panel CpGs, H_min anchor, scoring rule (pooled-entropy or directional Rule A), tissue applicability, age range, sex applicability, clinical context paragraph, recommended action per tier, evidence citations. The diagnosis translator just reads the card JSON, runs the scoring rule against the customer's β values, applies the tier thresholds, and assembles the output paragraph from the card's own context fields. **The cards are already structured for this** — they just need to be loaded and applied.

**Per-card caveats and complexity management.** Each card README contains: known false-positive scenarios, age-related caveats, comorbidity interactions, performance characteristics. The diagnosis translator's responsibility is NOT to summarize all of this in every report — it's to flag specific caveats that apply to *this* customer (e.g., if the customer's age is in a range with known higher false-positive rate, flag that caveat; if a comorbidity affects interpretation and the customer disclosed it in metadata, flag that). Caveats not relevant to the customer don't appear in the report.

**Code.** New module `edear_diagnosis_translator.py`. Reads card JSON files + A-score output, produces structured diagnosis report. Card loading is dynamic — adding a new card = drop a new JSON + README in `cards/` directory and the translator picks it up at next server restart.

**Honest constraint.** EDEAR is NOT a clinical diagnostic. It is an early-detection signal that recommends downstream workup. The diagnosis translator must be explicit about this in every output. Regulatory positioning is **research / wellness / risk stratification tool**, not FDA-cleared diagnostic. This protects Heath, the customer, and the framework's credibility.

**Exit gate.** Diagnosis translator produces complete, reviewable reports for 100% of synthetic test cases representing all 17 cards × 5 tier outcomes (85 test cases). Every output includes confidence interval, concordance check, and explicit "research tool, not diagnostic" disclaimer.

---

### STEP D3 — The EDEAR server processor (production server)

**This is the bones-step still pending from April 19 session.** The server consumes a methylation file from a customer (or lab partner), runs it through the full pipeline, returns the diagnosis report.

**Architecture.**

```
POST /v1/score
  ↓ Authenticated request with customer ID
  ↓ Methylation file payload (CSV: cpg_id, beta_value)
  ↓ Sample metadata (age, sex, tissue, optional family history)
  ↓
[Input validator] → reject malformed files; reject if <80% HM450 coverage
  ↓
[Sample-level QC] → bisulfite conversion check, batch-effect flag, outlier detection
  ↓
[Pipeline executor]
  ↓ - Deconvolve cell composition
  ↓ - Compute 8 A-scores against IAMAtlas (age-adjusted via age layer)
  ↓ - Match applicable cards (filter by tissue / age / sex)
  ↓ - Run each matched card panel
  ↓ - Translate to diagnosis report
  ↓
[Report generator] → JSON + HTML + PDF outputs
  ↓
Return signed report URL with TTL
```

**Stack (KISS — don't over-engineer).**
- Python 3.11+ FastAPI (async, simple, well-documented)
- PyMC for any per-sample Bayesian computation
- Pandas / NumPy for matrix operations
- IAMAtlas.csv loaded into memory at server startup (~500 MB resident)
- Reports stored in S3-compatible object storage (Cloudflare R2 cheap)
- Postgres for customer / sample tracking
- Cloudflare Tunnel for HTTPS (already in use for researcher demo)

**Security and IP protection.**
- Server.py code uses only neutral variable names. No physics terminology in source code, docstrings, comments, API outputs, or HTML. Forbidden: Boltzmann, Landauer, Arrhenius, Bose-Einstein, thermal, activation energy, decoherence, k_B, coth, or any derivation language.
- Enigma encoding (α=28769.0, β=12430.0, γ=723.106) on sensitive parameters in transit / at rest if practical.
- Customer files encrypted at rest. TLS 1.3 in transit.
- IAMAtlas matrix is the engine; the underlying physics derivation (the Recipe) is NEVER exposed to customers, NDAs, or partner labs.

**Performance target.** End-to-end per-sample inference < 60 seconds on a single-CPU server. Batch mode (10-100 samples) at < 5 minutes total.

**Exit gate.** Server passes synthetic stress test (1000 samples processed, no errors, all outputs validated against expected). Single-sample latency p95 < 60s. Customer authentication / authorization functional.

---

### STEP D4 — The customer-facing product (web app + report delivery)

**Two surfaces:**
1. **Lab partner portal** — authenticated upload, batch processing, results download. Lab partners are the primary v1 channel — they generate the methylation files and EDEAR scores them.
2. **Researcher demo / public-facing site** — `iamperformance.net` already live. Add product page with EDEAR description, sample reports (anonymized), pricing, and "request access" form (NOT direct signup — this is research-tier, not consumer).

**The lab partner story is the v1 commercial wedge.** Labs already run methylation arrays for clinicians or research customers. EDEAR adds an interpretation layer on top of their existing infrastructure. The lab pays per sample, charges the end customer (clinic / researcher), keeps the margin. EDEAR doesn't touch the patient. Heath's three-tier lab partnership evolution (already documented in memory): L1 near-term (EPIC β-matrix for GAPE classes), L2 medium (custom capture panel), L3 year-3+ (full 5-substrate multi-assay).

**Exit gate.** Lab partner can upload a real methylation file via the portal, receive a complete diagnosis report in < 5 minutes, and the report is reviewable, exportable as PDF, and includes all required disclaimers.

---

### STEP D5 — Validation cases (3-5 real samples through the full pipeline before customer #1)

Before a paying customer touches the system, run 3-5 **real** anonymized samples through the full pipeline end-to-end:
1. Clinically-confirmed AD case (test if AD-immune card flags Tier 2 or 3)
2. Clinically-confirmed breast cancer case (10yr+ pre-dx if available; test if breast-secretory card flags)
3. Clinically-confirmed HCC case in no-documented-risk patient (Marcus-class)
4. Clinically-confirmed healthy adult (negative control — should produce all Tier 0)
5. Aging case (60+ adult with no clinical disease — should show age-related drift but no card breach)

**Exit gate.** ≥4 of 5 cases produce expected output. Any unexpected output is investigated, root-caused, and either (a) explained as expected variability or (b) triggers card / matrix / pipeline fix before commercial launch.

---

### STEP D6 — Lab partner agreement → customer #1

**Action.**
1. Identify three candidate lab partners (regional / specialty / direct-to-consumer methylation labs).
2. Pitch the L1 partnership: lab does the methylation assay, EDEAR does the interpretation, pricing TBD per sample.
3. Sign one agreement.
4. Lab routes their first customer's sample through EDEAR.
5. Report delivered. Customer #1.

**Exit gate.** First paid sample processed. First report delivered. First invoice issued. Marcus didn't need to die that way; the framework is now commercially live.

---

### STEP D7 — Iterate on what customer #1 reveals

Heath's working principle: customer #1 will reveal weaknesses neither of us anticipated. The discipline is to fix them before customer #5, not before customer #1. The system has to ship.

Likely surface areas:
- Edge cases in input file format the validator missed
- Card panels that overflag or underflag in real-world data (vs. cohort training)
- Performance bottlenecks under real-batch sizes
- Reporting clarity for non-technical lab personnel
- Regulatory questions from lab partners' compliance teams

**Exit gate.** Five customers processed. Iteration log filed. v1.1 deployment freeze.

---

## PART 4 — DECISION GATES THAT REQUIRE HEATH SIGN-OFF

These are not Walther decisions. These are Heath decisions — Walther flags when they're due.

| Gate | Decision needed | Triggered when |
|---|---|---|
| G-A | Stromal re-run config (label harmonization + tightened sampler) — RESOLVED 2026-05-08: harmonization script staged in repo, target_accept 0.99 + tune/draws 2000 set | After main MCMC run completes |
| G-A2 | If stromal still fails after harmonization + tightened config: split into sub-classes (perivascular/contractile vs ECM-producing) OR accept at class level with downgraded ESS reporting | Only if Step 6.5 first attempt fails exit gate |
| G-B | Gastric-only v0.1 vs combined gastric+esophageal scope | Before sealing gastric/esophageal prereg |
| G-C | Welch tumor-vs-pooled-normals acceptable for TCGA-STAD n=2 paired HM450? | Before sealing gastric prereg |
| G-D | Crohn's pathway language in gastric / HCC cards? | Before sealing affected cards |
| G-E | Loyfer-full integration timing (immediately post-v1 vs Q3) | After v1 ships |
| G-F | Roadmap Epigenomics integration timing | After v1 ships |
| G-G | Lab partner selection (which 3 to pitch) | When server is at exit gate |
| G-H | Pricing per sample (L1 partnership terms) | At lab partner pitch stage |
| G-I | Public release timing (Step 12) — before or after EDEAR commercial launch | Heath's call; current memory says after |
| G-J | Regulatory positioning (research / wellness / risk-strat — confirm not making FDA claims) | Before lab partner agreement signed |
| G-K | When to bring in counsel (pre-customer-#1 vs post) | Heath's existing memory: no attorney yet, March 2027 non-prov deadline |

---

## PART 5 — WHAT THIS ROADMAP IS NOT

- **Not a single-person sequential plan.** Steps 7, 8, D1-cards, D2-diagnosis-translator can run in parallel. Step 6 (MCMC) is the only blocker that must complete before validation.
- **Not a date-bound timeline.** Calendar dates depend on how long the MCMC runs, how many cards need v1 work, and how fast lab partners respond. Walther will not invent target dates.
- **Not a regulatory plan.** EDEAR is positioned as research / wellness / risk-stratification tool, not FDA-cleared diagnostic. Regulatory expansion is a post-customer-#5 question.
- **Not a fundraising plan.** EDEAR is bootstrapped. Subscription revenue (when it begins) funds refinement. No grants. No investors. No dilution.
- **Not a public-disclosure plan.** Step 12 (researcher release) happens AFTER EDEAR commercially live + 3-5 customer cases where flags led to confirmed clinical findings. The deployment story makes adoption frictionless. No preprints. No outreach until after GRF May 15.

---

## PART 6 — WHAT WALTHER OWES HEATH AT EVERY STEP

If Walther drifts from this plan without checking the step number, it's the wrong move.

At the end of each step, Walther reports:
- Which step completed
- Whether the exit gate passed
- What the next step is

If a step fails the gate, Walther stops and tells Heath. No invented workarounds.

**Specific accountability rules carried from the constitution:**
- TESTING_CHECKLIST.md is the first call every session
- Source-doc rule absolute (read canonicals before describing)
- No-fabrication rule outranks all others
- Surgical edits only; never delete Heath's content without thorough discussion
- Reproducibility triple (code + inputs + environment + expected output) on every artifact
- Atlas vault discipline (durable in GitHub, scratch in `/home/claude/`)
- Pre-send checklist auto-applied (claims, citations, literature)
- LaTeX standards (proper .bib, full pdflatex/bibtex/pdflatex/pdflatex cycle)
- Language standards (never resolves/confirms/validates/proves)
- IAM constitutional rule (perturbation-only, never modified Friedmann, never modified gravity)

---

## PART 7 — SESSION-CONTINUITY HANDOFF NOTE

When this document is loaded into a new chat, Walther reads:

1. This document (master roadmap)
2. TESTING_CHECKLIST.md (procedural rules)
3. EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md (pipeline architecture)
4. LESSONS_LEARNED.md (what's been learned)
5. Whatever specific card README + card JSON the current task touches

That stack is sufficient for Walther to continue without re-orientation. The framework state is in those files. Heath's working state is in this document. Walther fills the slot.

**Closing line that should never need rewriting:** *The math is doing its job. The architecture is sound. The framework is showing itself true at a scale nobody else has tested it at. Marcus didn't need to die that way.*

---

## REVISION LOG

### 2026-05-08 (post-stromal-diagnosis + GitHub-LFS-storage session)

Changes made to the 2026-05-06 version:

1. **Stromal failure root cause re-diagnosed.** Previous text described the failure as sparsity/heterogeneity exceeding standard sampler config, with the fix being tightened NUTS settings. Inspection of the per-celltype output CSV on 2026-05-08 revealed the actual failure mode: 17 cell-type labels included multiple atlas-specific naming variants of the same biological cell type (4 endothelial labels, 3 fibroblast labels, 3 adipocyte labels, 2 smooth muscle labels). The model was assigning interchangeable bucket labels to one biological signal — chains never converged because there were multiple equivalent posterior modes from label-permutation symmetry. Tightened NUTS settings cannot fix multi-modal posteriors. The fix is **input data harmonization** (reduce 17 labels to 9 canonical cell types) BEFORE re-running MCMC, with tightened sampler as additional safety margin. Step 6.5 rewritten accordingly. Gate G-A description updated to reflect resolved status.

2. **GitHub LFS storage established.** New directory `iamatlas_production_data/` in the repo with git LFS for large files (.csv, .csv.gz, .csv.xz, .rda, .idat, .h5). Contains: production MCMC inputs (raw 698 MB + 19 MB xz-compressed), all completed-class outputs (terminal/stem_pluri/stem_adult/secretory/stromal-failed), MCMC build script, merge script, stromal_rerun/ package (harmonization script + README), terminal_addition/ package (Gasparoni staged files + README). Future Walther sessions read from here durably; no more re-uploads needed. Roadmap header updated with "Durable-storage rule."

3. **Step 6.85 added** for post-MCMC per-cell-type label consolidation. After all chains complete, audit each `_per_celltype.csv` for label collisions and apply one consolidation script that merges duplicate-label columns by averaging posterior means and combining SDs. Non-blocking cleanup, ~2 hours total. Does NOT require any MCMC re-run because class-level H_min anchors are correct regardless of per-cell-type label organization.

4. **Step D2 diagnosis layer architecture expanded** with a concrete six-stage pipeline diagram (input → deconvolve → demographic adjust → floor departure → card matching → diagnosis translation → report). Earlier multi-stage filter pipeline (red-flag → tissue-specific → immune-complexity) explicitly retired — single Bayesian deconvolution against the IAMAtlas replaces three sequential filter stages because the deconvolution is one operation that simultaneously yields all 8 class A-scores. Added explicit description of card-information integration mechanics (card JSON files already structured for this; translator just loads, runs scoring rule, applies tier thresholds, assembles output paragraphs from card's own context fields). Added bidirectional immune-class check at floor-departure stage (CCL-027 four-question protocol). Added Tier 4 FLOOR_BREACH for A > 1.20.

5. **Per-card caveats handling** clarified in Step D2: diagnosis translator's responsibility is NOT to summarize all known caveats per card in every report — only to flag the caveats that apply to *this specific customer* (age range with known higher false-positive rate; comorbidity disclosed in metadata that affects interpretation; etc.). Caveats not relevant don't appear.

### 2026-05-06 (post-Gasparoni-staging session)

Changes made to the 2026-05-05 version:

1. **Added "IMMEDIATE POST-MCMC GATES" prominent section near top of document.** Stromal re-run was buried as Step 6.5; promoting it as Gate 1 ensures it cannot be missed when the main run completes. Same for Gasparoni append (Gate 2) and terminal re-run (Gate 3). Heath caught this on 2026-05-06 — the previous structure buried the most important post-MCMC actions inside numbered sub-steps.

2. **Step 6.5 stromal re-run baseline corrected.** Previous text said "tune=2000, draws=2000 (up from 1500/500)." Verified against the running MCMC command line via `wmic`: actual current run uses `--tune 1000 --draws 1000`, so the baseline in the comparison was wrong. Proposed numbers unchanged; baseline corrected.

3. **Step 6.5 stromal heterogeneity description corrected.** Previous text said "17 cell types × 17 atlases." Atlas-count claim was approximate and unverified. New text says "17 cell types feeding it across multiple atlases" — accurate to ground truth without specific numbers I hadn't checked.

4. **Step 6.5 added stromal-failure-after-tightened-config branch.** If first re-run also fails, Gate G-A2 (newly added) requires Heath sign-off on whether to split stromal into sub-classes or accept at class level with downgraded ESS reporting.

5. **Step 6.75 Gasparoni section fully rewritten to reflect staging COMPLETE status.** Includes file paths, SHA-256 hashes, sealed Heath decisions on label mapping and atlas_source, append procedure for both Windows and Linux/macOS.

6. **Part 1 stromal description updated.** Removed unverified "× 17 atlases" claim; replaced with R-hat and ESS values from actual diagnostics.

7. **Timeline updated from 5-7 days remaining to 8-10 days remaining.** Based on observed batch 52/77 of secretory at ~27 min per batch as of 2026-05-06 evening, with cycling/progenitor/immune scaling. Immune is 4.99M rows and will dominate the back end.

8. **Confirmed stem_pluri does NOT need a re-run.** Stem_pluri converged cleanly (R-hat 1.01, ESS 501, Pearson 0.919, MAE 0.322) using full Nazor 2012 HM450 array (482,421 CpGs × 1 cell type). Roadmap Epigenomics expansion is a v1.1 deepening, not a v1 prerequisite. This was clarified in conversation 2026-05-06.

### Prior process learnings recorded for future Walther sessions

- Never claim a batch number from memory across compactions. Always verify with `wmic process where "name='python.exe'" get processid,commandline` and direct file system inspection before describing run state.
- Never reconstruct the script's class-processing order from compaction memory. Read the actual script first; the loop on line 444 of `iamatlas_v0_1_mcmc_batched.py` shows it processes all classes passed via `--classes` in one invocation, advancing automatically.
- The script writes per-class output ONLY at class completion (lines 361-384), not per-batch. Mid-class progress is visible only in stdout, not on disk.
