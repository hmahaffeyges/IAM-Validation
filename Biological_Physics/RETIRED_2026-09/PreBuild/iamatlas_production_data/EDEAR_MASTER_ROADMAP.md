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
| stromal | ✗ FAILED CONVERGENCE — Step 6.5 re-run staged | 17,485s | **3.67** | **4** | 6 | 0.897 | 0.202 |
| stem_pluri | ✓ done 2026-05-05 03:17 | 12,842s | 1.01 | 501 | 2 | 0.919 | 0.322 |
| stem_adult | ✓ done 2026-05-05 06:52 | 12,910s | 1.01 | 723 | 0 | 0.904 | 0.338 |
| secretory | ✓ done 2026-05-07 05:22 | 167,328s (46 hr) | 1.02 | 1023 | 0 | 0.791 | 0.338 |
| cycling | 🟡 RUNNING (~28-30 hr remaining as of 2026-05-08) | 30 min/batch | -- | -- | -- | -- | -- |
| progenitor | queued | -- | -- | -- | -- | -- | -- |
| immune | queued | -- | -- | -- | -- | -- | -- |

**Honest revised timeline (as of 2026-05-08):** ~5-7 days remaining. Cycling at batch 20/77 → ~28 hr remaining; progenitor ~50 hr after cycling; immune ~100+ hr after progenitor. Immune is the largest class by row count (4.99M rows, 4× larger than secretory) and will dominate the back end of the timeline.

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
| 6 | Production MCMC | 🟡 IN FLIGHT — 5 of 8 classes done (terminal, stem_pluri, stem_adult, secretory, + stromal-failed-pending-rerun), 1 running (cycling), 2 queued (progenitor, immune) |

---

## PART 2 — REMAINING STEPS TO COMPLETE IAMATLAS V1

### STEP 6 (continuing) — Let the production MCMC finish

**Action.** Do nothing. Do not interrupt. Let cycling → progenitor → immune complete on son's PC. Secretory finished 2026-05-07 cleanly (R-hat 1.02, ESS 1023).

**Estimated remaining wall-clock:** 5-7 days.

**Exit gate.** 8 result files exist in `iamatlas_v0_1_output/`:
- `iamatlas_v0_1_terminal_result.json` (✓ exists)
- `iamatlas_v0_1_stromal_result.json` (✓ exists, but failed gate — see Step 6.5)
- `iamatlas_v0_1_stem_pluri_result.json` (✓ exists)
- `iamatlas_v0_1_stem_adult_result.json` (✓ exists)
- `iamatlas_v0_1_secretory_result.json` (✓ exists, R-hat 1.02 ESS 1023 zero divergences)
- `iamatlas_v0_1_cycling_result.json` (pending, ~28 hr remaining)
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

This is where the IAMAtlas stops being a research artifact and becomes the engine inside a paying product. EDEAR is positioned as a cellular health and wellness tracking product — not a clinical diagnostic. The order here matters: matrix → cards → report layer → server → website + customer portal → first customer.

### Wellness positioning — the master constraint for everything that follows (locked 2026-05-08)

The framing for the entire customer-facing product is cellular health and wellness tracking. This is the single most important positioning decision in the project and it shapes every downstream choice. Locking it here so every subsequent step inherits the same orientation.

**What EDEAR is.** A cellular health and wellness tracking product. Customers send a blood draw, EDEAR scores their cellular health across 8 architectural cell classes plus per-cell-type breakdowns, returns a cellular age relative to chronological age, and (with subscription) tracks the trajectory across multiple tests. The customer watches their cellular health respond in real time to lifestyle changes — weight loss, smoking cessation, exercise, stress management, dietary changes — at a resolution no smart watch or other consumer health product can match. The subscription is the product; trajectory is the value proposition.

**What EDEAR is not.** A clinical diagnostic. EDEAR does not diagnose disease. EDEAR does not name specific diseases as conclusions in customer reports. EDEAR does not recommend treatment. EDEAR does not use medical-action language ("refer to specialist", "immediate clinical referral", or equivalent) anywhere in customer-facing copy.

**How disease detection capability fits.** As a downstream consequence of measuring cellular architecture across all major cell classes, EDEAR is also capable of detecting methylation patterns that have been associated with various conditions in published research. When those patterns appear in a customer's results, the report flags them neutrally and points the customer to the website for educational content. The website explains what the pattern can indicate in research terms. The customer takes that information to their physician if they choose. This is mentioned in customer-facing copy, but always after the wellness lead and never as the primary positioning.

**Strong-recommendation language is conservative.** "We recommend you discuss this finding with your physician" fires only on cards where (a) the customer's A-score is in the SIGNIFICANTLY ELEVATED range and (b) the card has sealed VAL evidence supporting that magnitude as worth attention. ELEVATED findings get neutral website pointers, not physician-referral language. Most customers will see clean reports with everything in normal ranges — that is the dominant case and the report should land it cleanly.

**The runtime card is intentionally minimal.** Each card encodes only what the engine needs at runtime: identifier, architectural class, validated substrate (ccfDNA plasma only for launch), demographic gates, threshold ranges, website URL. No prose interpretation in the card. No conditional caveat tables. No clinical-action matrices. No message generation logic. Maybe 50-100 lines of JSON per card. Everything else lives on the website where it can be updated without touching the engine. The simpler the runtime, the fewer ways it can lie to a customer.

**The website carries the substance.** Each architectural class has a page on iamperformance.net explaining what those cells do, normal ranges, what departures can indicate (with research citations, not claims), lifestyle factors that affect that class, why trajectory matters. Each disease card has a page with the same structure. Educational content can evolve as research evolves; the runtime stays stable.

**The immune class still leads the interpretive narrative even in run-everything architecture.** The immune system sees everything happening anywhere in the body that's sustained enough to matter. A normal immune class with elevated organ tiles is one pattern; an elevated immune class without organ tiles is another; an elevated immune class with concordant organ tile elevation is a third. These mean different things, and the immune card's website page is what helps the customer understand which regime they're in. The runtime engine does not produce this interpretation — the website does.

**On customer expectations.** A customer with an elevated immune class and no organ tile elevation receives a report saying which immune cell types are showing departure from age-matched baseline. The report does not claim this points to any specific disease. The website explains that immune system changes can precede the development of certain conditions by months to years (with research citations) but that many things drive immune signals — recent illness, vaccination, stress, allergies, autoimmune conditions, sustained inflammation. The website recommends prudent action: continue routine checkups, consider lifestyle factors, continue EDEAR testing because trajectory is the real product. No disease-specific claims to a customer based on immune-only signal — we lack the cohort comparators to make those claims honestly.

**Regulatory posture follows naturally from the product framing.** Wellness positioning is the cleanest legal and regulatory posture available. EDEAR makes no medical claims about an individual customer; EDEAR provides measurements and points the customer to educational content. This places EDEAR in the same category as continuous glucose monitors for non-diabetics, methylation age clocks (Elysium, TruDiagnostic), wellness panels, fitness wearables — all of which avoid medical-device classification by leading with personal health optimization. Direct-to-consumer products that have run into trouble (early 23andMe, GRAIL Galleri's EU rollout, several startups) all led with the disease claim and added wellness framing as ancillary. Products that have stayed clean led with personal health optimization throughout.

**Commercial logic follows naturally.** The TAM for "health and wellness tracking" is the entire health-conscious population — hundreds of millions of people. The TAM for "cancer screening" is much smaller and dominated by traditional screening with insurance coverage. The subscription model only works in the wellness framing — you don't subscribe to a cancer test. Customer testimonials will be about cellular age dropping after lifestyle changes, immune class tightening as stress is managed, cycling class improving as fitness improves. Cancer-caught-early stories will happen and will be real benefits, but leading with them puts EDEAR in the wrong category and invites the wrong regulatory attention.

**KISS in the engine.** The engine does as little interpretation as possible. The engine measures, classifies range, matches cards, and produces structured output for the web portal to render. Every interpretive element lives on the website or in the customer's hands. The fewer interpretive decisions the engine makes per customer, the fewer ways it can produce a wrong flag or a false alarm. Complexity is moved out of the runtime and into the educational layer where it can evolve safely.

This positioning is the constraint every step in Part 3 must respect. If a downstream design choice conflicts with wellness positioning, the design choice changes — not the positioning.

### Operational details — runtime architecture, schema, and report (locked 2026-05-08)

These operational details flow directly from the wellness-first positioning above and from the working session that locked the immune-card-as-narrative-lead, the per-finding minimum-information rule, and the upstream-vs-downstream split for covariates and bidirectional patterns. They are captured here so subsequent sessions can build on them without rediscovering.

**The four-fields-per-finding rule.** Each card-firing in the customer report shows exactly four pieces of information: (a) the cell type or class name in plain language ("Hepatocyte" not "secretory_class.hepatocyte_subset"); (b) the score with credible interval, plotted against the age-matched normal range with a visual indicator and a plain-text range label (NORMAL / ELEVATED / SIGNIFICANTLY ELEVATED / BELOW NORMAL); (c) a one-sentence factual flag-summary string ("Your hepatocyte score is significantly elevated compared to your age-matched baseline") with no disease names and no interpretation; (d) a link to the relevant website page. When the finding is at SIGNIFICANTLY ELEVATED on a card with sealed VAL evidence supporting that magnitude, one additional sentence fires: "We recommend you discuss this finding with your primary care provider." That sentence never appears at lower magnitudes and never on cards lacking VAL anchor data at SIGNIFICANTLY ELEVATED.

**The runtime card schema (locked v1.0).** Each card JSON encodes:

```
schema_version            (string, "1.0")
card_id                   (string)
card_version              (string)
card_date                 (string, ISO date)
display_name              (string, customer-facing section header)
architectural_class       (string, one of 8 class names)
cell_types_of_interest    (list of strings, optional)
validated_substrates      (list of strings, currently only "ccfDNA_plasma")
demographic_gates         (object with age_min, age_max, sex)
covariate_dependencies    (list of strings naming covariates this card uses)
thresholds                (object, see threshold lookup section below)
report_strings            (object with the four flag_summary strings per range)
website_url               (string, link to this card's educational page)
validation_tier_label     (string, metadata)
```

That is the entire runtime card. ~50 lines of JSON for cards without covariate dependencies; ~100 lines for cards with covariate-keyed threshold tables. No prose interpretation, no conditional caveat tables, no clinical-action matrices, no message-generation logic. The engine has one loader, one matcher, one classifier, one assembler.

**Threshold lookup mechanics.** The `thresholds` block is keyed first by substrate (ccfDNA_plasma for launch, with tissue/urine reserved for future). For cards without covariate dependencies, the substrate block contains four numeric thresholds: NORMAL_max, ELEVATED_max, SIGNIFICANTLY_ELEVATED_min, BELOW_NORMAL_max — plus a boolean `VAL_anchor_supports_significant` that gates physician-referral language. For cards with covariate dependencies (e.g. lung-epic varying by smoking status, HCC varying by viral hepatitis status), the substrate block contains nested threshold tables keyed by covariate state ("current_smoker" / "former_smoker" / "never_smoker"). The engine reads the customer's covariate state from their profile and selects the matching threshold table. Cards that should fire on bidirectional patterns include an additional `bidirectional_pattern_fires_at` clause at the substrate-block level, which the engine consults against the bidirectional consistency flag from Stage 2.

**Customer profile / covariate intake.** Covariates that affect interpretation (recent illness, smoking status, viral hepatitis, HRT, pregnancy, current medications, recent vaccinations) are collected at the engine level, not at the card level. Each card declares its `covariate_dependencies` list. The engine collects the union of all card-declared covariates and presents them as the customer intake form (either through the lab partner channel or directly at iamperformance.net depending on channel). Customer answers are attached to the sample as profile metadata. At runtime, the engine reads the profile and feeds the relevant covariates to each card's threshold lookup. If a customer skipped a question, the engine uses the most permissive default and the report notes that the result is uncalibrated for that covariate, with a website link explaining why answering improves accuracy. The customer profile is the engine's job; no individual card stores customer answers or owns the intake form.

**Bidirectional patterns are computed upstream of cards.** Bidirectional methylation patterns (some CpGs gaining methylation while others lose it within the same panel or class — the AD-instance signature where mean β shifts approximately to zero but real signal exists in the directions) are handled at Stage 2, not at the card level. Stage 2's class A-score computation produces two outputs per class: the aggregate A-score (magnitude of departure from baseline) and a bidirectional consistency flag (boolean indicator of whether the directions of departure within the class are consistent or bidirectional). Cards that should fire on bidirectional patterns declare so via `bidirectional_pattern_fires_at`; most cards (cancers showing aggregate magnitude shifts) don't need this clause. The engine knows how to compute the consistency flag from Stage 2 output; cards just consume it.

**Cellular age reported at three levels.** Other consumer methylation products (Horvath-clock derivatives, GrimAge, PhenoAge, DunedinPACE, TruDiagnostic, Elysium) report a single biological age number averaged across all tissues — convenient but loses the heterogeneity that matters most. EDEAR reports cellular age at three nested levels: (a) overall cellular age as the headline number (weighted combination across the 8 architectural classes for customer-friendly comparison to other products), (b) per-class cellular age across all 8 classes ("your immune cells are operating at age 52, your secretory cells at age 45, your terminal cells at age 47"), (c) per-cell-type cellular age where IAMAtlas posteriors have the resolution. The customer who wants a simple headline gets it; the customer who wants the detailed picture gets it too. This three-level resolution is one of the strongest competitive differentiators of the product and follows naturally from the IAMAtlas architecture — single-clock products can't resolve more because their methods don't have the per-class structure. The website page on cellular age explains all three levels and why they matter.

**Card walkthrough plan (the order for Part 3 / Step D1).** The runtime cards (runtime-schema versions of the cookbook cards) get built one at a time in this order:

1. **Immune card first.** It's the most interpretively-loaded card and forces the schema to be tested against the hardest case before we lock it. If the schema fits immune cleanly, the rest of the catalog is fill-in-the-template work.
2. **HCC second.** Already drafted as a worked example in this section. Validates the schema against the simplest cancer card (single-tissue, ccfDNA-only, single-substrate, demographic-gate-light).
3. **CRC third.** Validates against a card with cycling-class focus rather than secretory, with multi-substrate considerations now stripped down to ccfDNA-plasma-only for launch.
4. **Breast fourth.** Validates against a card with covariate dependencies (HRT, parity) and a more complex thresholds table.

After four cards, if the schema doesn't need to grow, we lock v1.0 and the remaining 13 cards become straightforward fill-in. If the schema needs revision, we revise it before continuing.

For each card walkthrough, the deliverables are two artifacts: the runtime card JSON in the v1.0 schema, and an outline (not the page itself) of the corresponding website educational page. The website outline captures what the page needs to say so the work doesn't have to be rediscovered when iamperformance.net is built.

The cookbook cards in `cards/hcc/` and `cards/crc/` stay where they are as audit-trail documentation. The new runtime cards are derived from them but live separately (proposed location: `cards/runtime/` alongside `cards/hcc/` and `cards/crc/`, or as `cards/<disease>/runtime.json` inside each existing card subdirectory — to be decided when the immune card walkthrough surfaces the right convention).

**Watch item: Quantum Motion silicon-CMOS qubits.** Quantum Motion (London) raised $160M Series C on 2026-05-07 for silicon-transistor-based qubits — manufactured on standard 300mm CMOS lines via partnership with GlobalFoundries. World's first full-stack silicon quantum computer was unveiled in September 2025 and is now deployed at the UK National Quantum Computing Centre. Architecture suspends single electrons in transistor gaps and uses electron-spin states as qubits. If silicon-CMOS quantum computing scales the way Quantum Motion is betting (millions of qubits at $10-20M per system, vs current systems requiring multi-megawatt cryogenic infrastructure), it represents a different qubit modality than the superconducting systems QAPE was originally calibrated against. Silicon spin qubits have different decoherence characteristics, different gate operations, different temperature requirements. Future QAPE work may need to extend characterization coverage to this modality. Not a current blocker; flagged for awareness as the QAPE roadmap evolves.

### STEP D1 — Card library completion (17-card catalog)

**Status as of 2026-05-08.** Card list per Heath's 2026-05-08 evening review:

**Active cards with JSON in production (14):**
1. AD (`ad-immune/`)
2. Bladder (`bladder-epic/` v0.1)
3. Breast (`breast-epic/`)
4. Cardio (`cardio-epic/` — v0.1 sealed but DEFERRED pending Konigsberg/Cuadrat 2023 atlas integration)
5. Cervical (`cervical-epic/`)
6. Colon/Rectal (`crc-epic/`)
7. Gastric/Esophagus (`gastric-epic/`, `esophageal-epic/` — both v0.1 sealed 2026-05-02)
8. Glioma (`glioma-epic/`)
9. HCC Liver (`hcc-epic/` — Marcus card; no-documented-risk subgroup paired d=+0.6166, p=0.0072, n=19)
10. HEME Leukemia (`heme-epic/` — VAL-082 AML d=+3.71, strongest single-cohort effect)
11. Lung (`lung-epic/`)
12. Pancreatic (`pancreatic-epic/`)
13. Prostate (`prostate-epic/` v0.3)
14. Immune (`immune-atlas/` — not a disease per se; provides systemic-departure routing for AD-pattern, inflammaging vs autoimmune differential, and the heme-epic uniform-elevation re-route)

**Need JSON and more work (2):**
15. PSP — progressive supranuclear palsy. Currently surfaces as a tauopathy-specificity arm in ad-immune (VAL-057 GSE53740 PSP/CBD preserved 5/7 frozen directions). Promotion to standalone card requires a dedicated panel definition, H_min anchor, scoring rules, mandatory covariates, and atlas integration plan.
16. Kidney — kidney_epithelial is in the Stage 2 output space and KIRC+KIRP are validated cycling-class TCGA types in the Issue 002 build. v2.1 expansion table includes kidney-epic. JSON + card README + sealed VAL pending.

**Candidate, not yet built (1):**
17. Schizophrenia — discussed as a candidate disease. No card exploration yet. Requires landscape survey (cohort availability, expected architectural class signal, healthy comparator availability) before a build plan can be drafted.

**Per-card workflow (the canonical seven-file update — TESTING_CHECKLIST.md and README_MASTER_v2_7 are the authoritative cookbook references):**
1. Update TESTING_CHECKLIST.md
2. Update EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md
3. Update README_MASTER_v2_X.md
4. Update LESSONS_LEARNED.md
5. Write/update card README
6. Write/update card JSON (panel definitions, H_min anchors, scoring rules, full universal_reference block per CHK-5.7's 14-sub-key verification)
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

**Exit gate per card.** All seven internal files updated; all eight external artifacts pushed; cross-card calibration check passes (CCL-037 LL-CROSS-COHORT-CALIBRATION: paired AND same-pipeline AND batch-corrected). Reproducibility triple complete (CHK-7.6: code + inputs + environment + expected output). CHK-5.7 14-sub-key universal_reference block verification. CHK-5.8 atlases_used_and_deferred block populated.

**Estimated wall-clock per card:** 1-3 days depending on cohort availability and pre-existing work. PSP and Kidney are the active build targets. Schizophrenia is gated on landscape survey.

**Runtime card schema (added 2026-05-08, under wellness-first positioning).** The card files currently in `cards/` (hcc-epic v0.3, crc-epic v2.4) are cookbook documentation cards — they contain validation evidence summaries, sealed VAL prereg SHAs, lessons learned, CCL terminology disambiguation, and other audit-trail material. These are valuable as documentation but are not the right shape for the runtime engine to consume.

The runtime card schema (to be drafted as part of the Step D2 report builder work) is minimal. Each runtime card encodes only:
- Card identifier and architectural class
- Cell type(s) of interest beyond class level (when applicable)
- Validated substrate(s) — for launch, ccfDNA plasma only
- Demographic gates (age range, sex restriction) as Boolean rules
- A-score threshold ranges that map values to NORMAL / ELEVATED / SIGNIFICANTLY ELEVATED
- The URL of the website page that explains this card to customers

Everything else — disease description, mechanism summary, validation evidence, conditional caveats, lessons learned, atlas integration history — moves to either (a) the educational website page for that card, or (b) the cookbook canonicals (README_MASTER, LESSONS_LEARNED, TESTING_CHECKLIST, GAPE Evidence Report, validation_runs/).

The cookbook documentation cards stay in `cards/` as historical record and as the source material from which runtime cards will be derived. The runtime cards are produced as a separate artifact when the engine is built. One per disease, ~100-200 lines of JSON each, deployment-focused.

**Exit gate (whole step).** All 16 buildable cards reach sealed v0.1+ tier. Cardio-epic stays DEFERRED until Konigsberg/Cuadrat 2023 atlas-of-record is acquired and bridged. Schizophrenia lands as a v0.1 card OR is documented as a deferred candidate with explicit unblock dependency.

---

### STEP D2 — The report layer (translation from A-scores to customer-facing wellness report)

**Revised 2026-05-08 under wellness-first positioning.** This step previously framed EDEAR as a "clinical product, not a research tool" and described a "diagnosis translator" that produces clinical recommendations. That framing is wrong for the launch product. The corrected framing: this is the layer that translates raw A-scores into a customer-facing cellular health and wellness report. Disease findings, when they appear, are handed off to the customer with a pointer to educational content and a recommendation to discuss with their physician — not interpreted in the report itself.

**Architecture (the single-pipeline KISS version — no multi-stage filters needed because the IAMAtlas does the discrimination upstream):**

```
Customer methylation file (β values)
         ↓
[Stage 0 — Input validation + QC]
   - Reject malformed CSVs
   - Reject if <80% HM450 coverage
   - Sample-level QC: bisulfite conversion check, batch-effect flag, outlier detection
         ↓
[Stage 1 — IAMAtlas deconvolution]
   - Per-cell-type fractions across all cell types in IAMAtlas
   - Per-class A-scores (one per architectural class)
   - Confidence intervals from MCMC posterior
         ↓
[Stage 2 — Class A-scores]
   - Aggregate cell-type fractions into 8 architectural-class A-scores
   - Apply GAPE engine H_min anchors per class
   - Output A-score with credible interval per class
         ↓
[Stage 3 — Demographic adjustment]
   - Age adjustment via age layer (per-CpG age-slope correction)
   - Sex adjustment for sex-dimorphic CpGs
   - Cellular age (residual after age adjustment) reported separately as a headline product feature
         ↓
[Stage 4 — Floor departure detection]
   - Per-class A-score → tier classification:
     * NORMAL (within healthy range for age)
     * ELEVATED (outside healthy range, mild)
     * SIGNIFICANTLY ELEVATED (outside healthy range, strong)
     * BELOW NORMAL (homogenization or suppression)
   - Per-cell-type tier classification using same scheme
         ↓
[Stage 5 — Card matching]
   - Filter cards by: tissue applicability, age range, sex applicability,
     CpG panel availability in customer's array
   - For each matched card, evaluate its threshold ranges against the customer's
     A-scores and per-cell-type fractions
   - Output: card-level tier (NORMAL / ELEVATED / SIGNIFICANTLY ELEVATED) and
     the website URL for that card's educational page
         ↓
[Stage 6 — Report assembly]
   - Cellular health summary (8 class A-scores, plotted against age-matched normal
     ranges, with neutral/elevated/significantly-elevated visual indicators)
   - Cellular age vs chronological age
   - Per-cell-type breakdown for cell types with IAMAtlas posteriors
   - List of any cards that flagged ELEVATED or SIGNIFICANTLY ELEVATED, each with
     a link to its educational page on iamperformance.net
   - For SIGNIFICANTLY ELEVATED cards: a single sentence recommending the customer
     review the finding on the website and discuss it with their physician
   - Trajectory section (populated from re-test 2 onward) showing change since last test
         ↓
[Stage 7 — Report delivery]
   - JSON (machine-readable for lab partner integration)
   - HTML (web-portal viewing)
   - PDF (printable)
```

**Why no multi-stage filter pipeline.** Earlier EDEAR designs ran sequential filters (Stage 1 red-flag → Stage 2 tissue-specific → Stage 3 immune complexity). The IAMAtlas makes that unnecessary because deconvolution is one operation that simultaneously gives us all 8 class A-scores and all per-cell-type estimates. The cards then run against the relevant A-scores in parallel, not sequentially. Single Bayesian deconvolution replaces three stages of filters.

**The report's lead is cellular health and wellness, not disease findings.** The customer opens their report and sees their 8 architectural-class A-scores plotted against age-matched normal ranges. They see their cellular age relative to their chronological age. They see which cell types are tracking well and which are showing departure. This is the headline product. Disease-card flags appear further down as "items worth reviewing" with website links — not as the headline.

**The runtime card schema is minimal.** Each card encodes:
- Card identifier and the architectural class it monitors
- Cell type(s) of interest (when applicable beyond the class level)
- Validated substrate(s) — for launch, ccfDNA plasma only
- Demographic gates (age range, sex restriction) as Boolean rules
- Threshold ranges that map A-score values to NORMAL / ELEVATED / SIGNIFICANTLY ELEVATED
- The URL of the website page that explains this card to customers

That is the entire runtime card. No conditional caveat tables. No message-generation logic. No clinical-action matrices. No prose. The website carries everything else.

**The website carries the substance.** Each architectural class has a page on iamperformance.net explaining what those cells do, what normal A-score ranges look like, what elevation can indicate (with research citations, not claims), what lifestyle factors are known from research to affect that class, and how trajectory matters. Each disease card has a page explaining what the card looks for, what range of A-scores have been associated with what observations in research, and how to think about a flagged result. The customer reads about their result on their terms, with content that can be updated independently of the runtime engine.

**What "significantly elevated" means and who decides.** Threshold-setting is itself a clinical-grade judgment that needs to be defensible. For the launch:
- Cards with sealed VAL data (HCC, CRC, breast, lung, prostate, AD, glioma, heme, gastric, esophageal, bladder, pancreatic, cervical, immune) use card-specific thresholds derived from their own anchor cohort calibration
- Cards still in build (PSP, Kidney) use generic baseline-deviation thresholds (>2 SD from age-matched baseline = ELEVATED, >3 SD = SIGNIFICANTLY ELEVATED) until card-specific thresholds are calibrated, with the website page documenting that these are baseline-deviation thresholds rather than disease-specific calibrated thresholds

The strong-recommendation language ("we recommend you review this finding with your physician") fires only at SIGNIFICANTLY ELEVATED thresholds, not on every flag. ELEVATED findings get a neutral website pointer.

**Trajectory is the core subscription value proposition.** A single EDEAR report is a baseline. The product becomes meaningful when the customer re-tests. Every subsequent report shows change since baseline and change since last test. The website explains why trajectory matters and how to interpret it. This is the genuine product, and it is what makes the subscription model work.

**Code.** New module `edear_report_builder.py` (renamed from earlier "diagnosis_translator" framing). Reads card JSON files + Stage 4 output, produces structured wellness report. Card loading is dynamic — adding a new card = drop a new JSON in `cards/` and the website team adds a new educational page. Restart picks it up.

**Honest constraint.** EDEAR is positioned as a wellness and cellular health tracking product. The report does not diagnose. The report does not name diseases unless a customer clicks through to the website's educational content. The report does not recommend treatment. Customers who see SIGNIFICANTLY ELEVATED findings are pointed to the website and to their own physician for any clinical interpretation. This positioning protects Heath, the customer, and the framework's credibility — and it is also accurate to what the product actually does for most customers, who will use it for tracking rather than for disease detection.

**Exit gate.** Report builder produces complete, reviewable wellness reports for 100% of synthetic test cases representing the cellular-health summary baseline plus all production cards across NORMAL / ELEVATED / SIGNIFICANTLY ELEVATED tier outcomes. Every report leads with cellular health and cellular age. Card flags appear as secondary findings with website links. The strong-recommendation language fires only at SIGNIFICANTLY ELEVATED thresholds.

---

### STEP D3 — The EDEAR server processor (production server)

**This is the bones-step still pending from April 19 session.** The server consumes a methylation file from a customer (or lab partner), runs it through the full pipeline, returns the wellness report.

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
  ↓ - Deconvolve cell composition against IAMAtlas
  ↓ - Compute 8 A-scores against IAMAtlas (age-adjusted via age layer)
  ↓ - Match applicable cards (filter by tissue / age / sex)
  ↓ - Run each matched card's threshold check
  ↓ - Assemble cellular health and wellness report
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

### STEP D4 — The customer-facing product (web app + website + report delivery)

**Revised 2026-05-08 under wellness-first positioning.** This step previously described a "lab partner portal" as the v1 commercial wedge. That channel may still come into play later, but the customer-facing product for launch is structured around three surfaces:

1. **Customer-facing report (PDF + HTML web view).** What the paying customer receives. Leads with cellular health and cellular age. Eight architectural-class A-scores plotted against age-matched normal ranges with neutral/elevated/significantly-elevated visual indicators. Per-cell-type breakdowns where available. Trajectory section (populated from re-test 2 onward). Card flags (when present) appear as secondary findings with website links — never as the headline. Strong-recommendation language ("review this finding with your physician") fires only at SIGNIFICANTLY ELEVATED thresholds.

2. **Educational website (`iamperformance.net`).** Carries the substance that earlier card READMEs were trying to carry inside the runtime. Each architectural class has a page explaining what those cells do, what normal A-score ranges look like, what elevation can indicate (with research citations, not claims), what lifestyle factors are known to affect that class, and how trajectory matters. Each disease card has a page with the same structure: what the card looks for, what range of A-scores have been associated with what observations in research, and how to think about a flagged result. The website carries the messaging, the educational context, the research citations, and the disclaimers. This separation lets us update content without redeploying the runtime.

3. **Customer portal (subscription management + report archive + trajectory dashboard).** Customers log in to view their report archive, see their trajectory plotted across multiple tests, manage their subscription, and update their covariate intake (recent illness, medications, lifestyle changes between tests). The trajectory dashboard is where the subscription value lives — a customer who has tested four times sees their cellular health changing in real time as they pursue lifestyle goals.

**The wellness positioning shapes everything about how these surfaces present EDEAR.**

The website lead is cellular health and wellness tracking. The "What is EDEAR" page leads with: track your cellular health across 8 architectural cell classes with the most precise framework available. Watch your cellular age in real time. See how lifestyle changes affect your body at the cellular level before they show up anywhere else. Monitor trends across multiple tests with the subscription. The cancer detection capability is mentioned as a secondary capability further down the page: as a downstream consequence of measuring cellular architecture across all major cell classes, EDEAR is also capable of detecting methylation patterns that have been associated with various cancers and other conditions; when those patterns appear in your results we'll point them out so you can discuss them with your physician.

Marketing emphasizes health journeys. Customer testimonials are about people watching their cellular age drop after lifestyle changes, about people seeing their immune class tighten as they manage stress better, about people using EDEAR as a feedback mechanism for their fitness goals. The cancer-caught-early stories will happen and will be real benefits, but leading with them puts EDEAR in the wrong category and invites the wrong regulatory attention.

**Why the wellness positioning is also stronger commercially.**
- The total addressable market for "health and wellness tracking" is the entire health-conscious population — hundreds of millions of people. The TAM for "cancer screening" is much smaller and dominated by traditional screening with insurance coverage.
- The subscription model only makes sense in the wellness framing. Cancer screening subscribers don't make sense; wellness tracking subscribers do.
- The customer journey works in the wellness framing. A 38-year-old getting serious about their health buys a baseline test, gets normal numbers, subscribes to track changes as they hit fitness goals. Six months later their cellular age has dropped, their immune class has tightened, their cycling class shows reduced drift. They have something concrete to share. They retain. They refer.
- The wellness framing is also more honest. Most customers will use it for tracking, not for getting cancer diagnoses. Marketing it as a tracking product reflects what the product actually is at scale.

**Lab partners may still come into play.** The earlier roadmap framed labs as the v1 commercial wedge. That remains a viable channel — labs already run methylation arrays for clinicians or research customers, and EDEAR could add an interpretation layer on top of their existing infrastructure. But it is a B2B channel parallel to the direct-to-consumer wellness channel, not a replacement for it. Both can run.

**Exit gate.** Customer can receive a wellness report via the customer portal, view their cellular health summary and cellular age, click through to website pages for any flagged findings, and (after the second test) view their trajectory dashboard. Report includes wellness-positioning disclaimers, not clinical-diagnostic language. Strong-recommendation language fires only at SIGNIFICANTLY ELEVATED card thresholds.

---

### STEP D5 — Validation cases (3-5 real samples through the full pipeline before customer #1)

**Revised 2026-05-08 under wellness positioning.** The test cases below are framed around what the wellness report should produce, not around clinical confirmation. The goal is to verify the report layer behaves correctly across the spectrum of customer scenarios.

Before a paying customer touches the system, run 3-5 real anonymized samples through the full pipeline end-to-end:
1. Healthy adult baseline — should produce all-NORMAL cellular health summary, cellular age within ±2 years of chronological age, no card flags.
2. Aging case (60+ adult with no clinical disease) — should show some age-related drift in expected classes (terminal, cycling), cellular age modestly above chronological, no card flags at SIGNIFICANTLY ELEVATED.
3. Sample from a known AD case — verify AD-immune card fires at appropriate tier, report links to AD educational page on website, strong-recommendation language fires only if SIGNIFICANTLY ELEVATED.
4. Sample from a known breast pre-diagnostic cohort case — verify breast card behaves appropriately (immune class context plus organ tile elevation), report does not over-claim, website link is to breast educational page.
5. Sample from a known HCC case (Marcus-class no-documented-risk if available) — verify HCC card fires at SIGNIFICANTLY ELEVATED with strong-recommendation language, report points to HCC educational page on website.

**Exit gate.** ≥4 of 5 cases produce expected output. Reports lead with cellular health summary and cellular age. Card flags appear as secondary findings with website links. Strong-recommendation language fires only at SIGNIFICANTLY ELEVATED. Any unexpected output is investigated, root-caused, and either (a) explained as expected variability or (b) triggers card / matrix / pipeline fix before launch.

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
| G-J | Regulatory positioning (wellness / health-tracking — confirm not making FDA medical-device claims). Decision was made 2026-05-08: wellness-first positioning consistently across website, report, marketing. Lead is cellular health and cellular age tracking; disease detection is mentioned only as a secondary downstream capability. | Confirmed before customer-facing copy is finalized |
| G-K | When to bring in counsel (pre-customer-#1 vs post) | Heath's existing memory: no attorney yet, March 2027 non-prov deadline |

---

## PART 5 — WHAT THIS ROADMAP IS NOT

- **Not a single-person sequential plan.** Steps 7, 8, D1-cards, D2-report-builder can run in parallel. Step 6 (MCMC) is the only blocker that must complete before validation.
- **Not a date-bound timeline.** Calendar dates depend on how long the MCMC runs, how many cards need v1 work, and how fast lab partners respond. Walther will not invent target dates.
- **Not a regulatory plan.** EDEAR is positioned as a cellular health and wellness tracking product (not an FDA-regulated medical device). The product measures cellular health; disease detection is a secondary capability with findings handed off to the customer's physician via the educational website. Regulatory expansion (e.g., pursuing 510(k) clearance for specific disease-detection claims) is a post-customer-#5 question if and when it makes commercial sense.
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

### 2026-05-08 night (operational details captured — runtime schema, covariates, bidirectional, cellular age, walkthrough plan)

Captured operational details that came out of the working session on customer-facing report mechanics and the runtime card schema. These flow directly from the wellness-first master constraint locked earlier and are the answers to questions surfaced when working through how the immune-class-as-narrative-lead actually plays out in the schema.

What was added in a new "Operational details" subsection between the master-constraint section and Step D1:

1. **The four-fields-per-finding rule for the report.** Cell type name, score with credible interval and range label, one-sentence factual flag-summary string, website link. Plus a conditional physician-referral sentence at SIGNIFICANTLY ELEVATED on cards with VAL anchor support — never at lower magnitudes, never on cards lacking VAL evidence.

2. **The runtime card schema v1.0.** Locked at ~14 fields, ~50-100 lines per card. HCC drafted as a worked example. The schema accommodates demographic gates, covariate dependencies, and bidirectional pattern handling without bloating.

3. **Threshold lookup mechanics with covariate keying.** Cards with covariate dependencies (lung-epic varies by smoking status, HCC by viral hepatitis) use nested threshold tables keyed by covariate state. Engine reads customer profile and selects matching table.

4. **Customer profile / covariate intake handled at the engine level, not the card level.** Each card declares which covariates it needs; engine collects the union as the intake form; customer answers are sample metadata fed to all cards. Skipped questions default to most permissive thresholds with report-level notation that result is uncalibrated for that covariate.

5. **Bidirectional patterns computed upstream of cards in Stage 2.** Stage 2 produces both an aggregate A-score and a bidirectional consistency flag per class. Cards that should fire on bidirectional patterns declare so via a `bidirectional_pattern_fires_at` clause. Most cards don't need this; AD-immune is the canonical case.

6. **Cellular age reported at three levels: overall, per-class, per-cell-type.** This is one of the strongest competitive differentiators of the product. Other consumer methylation products report a single number; the IAMAtlas architecture lets us resolve all three levels.

7. **The card walkthrough plan: immune first, then HCC, CRC, breast.** Validates the schema against the most interpretively complex card before locking; then validates against simpler and progressively more covariate-dependent cards. After four cards, lock the schema and fill in the remaining 13 as routine work. Each walkthrough produces two artifacts: runtime card JSON in v1.0 schema, plus an outline (not the page itself) of the corresponding website educational page.

8. **Watch item: Quantum Motion silicon-CMOS qubits.** Logged with date and context — $160M Series C on 2026-05-07, world's first silicon full-stack quantum computer deployed at UK NQCC since September 2025, partnership with GlobalFoundries, electron-spin qubits in transistor gaps. If this modality scales it changes what QAPE needs to characterize. Not a current blocker; flagged for awareness.

What did NOT change:

- The wellness-first master constraint itself — unchanged. The operational details flow from it, they don't modify it.
- The 17-card catalog — unchanged.
- The IAMAtlas-only architectural rule, the GAPE engine as Heath-only IP, the deconvolution stage — unchanged.
- The card files in `cards/hcc/` and `cards/crc/` — unchanged. They remain as cookbook documentation. Runtime cards will be derived from them when the engine is built and live separately.
- The cookbook canonicals — unchanged.

This revision is documentation-only. It captures the schema design and walkthrough plan so subsequent sessions don't have to rediscover it. The next working session is the immune card walkthrough — first card to be derived into the v1.0 runtime schema, with its corresponding website page outline.

### 2026-05-08 night (wellness positioning master-constraint locked + further refinements)

Following the earlier wellness-positioning revision, an additional working session refined and locked the positioning as a master constraint that every downstream design decision inherits. The key refinements:

1. **Master-constraint section added at top of Part 3.** The wellness positioning is no longer just present in individual Part 3 steps — it's now the explicit master rule for the entire customer-product portion of the roadmap. The new section "Wellness positioning — the master constraint for everything that follows" makes the positioning, the rationale, and the operational consequences explicit before any Part 3 step is read. If a downstream design choice conflicts with wellness positioning, the design choice changes — not the positioning.

2. **No specific disease claims for individual customers based on cohort comparators we lack.** The breast 10-year-pre-diagnostic scenario surfaced a subtle overclaim risk. VAL-047 anchored on EPIC-Italy breast at >10 years pre-diagnostic, but we do not have equivalent 10-year pre-diagnostic methylation cohorts for most other cancers. We cannot honestly tell a customer "your immune class drift looks like the breast cancer 10-year pre-diagnostic pattern" because we don't have the comparator data to claim breast specificity at long pre-diagnostic windows. What we honestly have is: immune class drift in the absence of organ-specific signal can precede disease, sometimes by years, and the trajectory matters more than any single point. This is what the report says. Disease-specific claims to individual customers based on immune-only signal are explicitly out of bounds.

3. **Immune class as interpretive lead.** Even in run-everything architecture, the immune class is the right first read in the customer-facing narrative. The immune system sees everything happening anywhere in the body that's sustained enough to matter, anchors the "is anything happening at all" question, and differentiates between systemic and localized perturbations. A normal immune class with elevated organ tiles, an elevated immune class without organ tiles, and an elevated immune class with concordant organ tile elevation each mean different things — and the interpretive framing for each pattern lives on the immune card's website page, not in the runtime engine.

4. **Customer covariate handling for context, not engine logic.** Recent illness, vaccination, stress, allergies, autoimmune conditions, sustained inflammation from any cause all drive immune class signals. The customer reports these at intake. The website explains how to think about an elevated immune signal in the context of any reported drivers. The engine does not branch on covariates to produce different messages — it provides the measurement and points to the website. The customer interprets their result with the website as their guide.

5. **Cellular age as the headline product feature.** Many customers will engage with cellular age as the easiest-to-grasp single number. Cellular age is reported prominently. The trajectory of cellular age across re-tests is the primary commercial narrative. Lifestyle-change-to-cellular-age-shift is the customer success story EDEAR is built around.

6. **The runtime card schema is locked at minimal.** Six fields per card: identifier, architectural class plus optional cell-type focus, validated substrate (ccfDNA plasma only for launch), demographic gates, threshold ranges, website URL. Approximately 50-100 lines of JSON per card. No conditional caveat tables. No message-generation logic. No clinical-action matrices. Everything beyond these six fields lives on the website.

7. **Threshold-setting discipline.** Cards with sealed VAL data use card-specific thresholds derived from their own anchor cohort calibration. Cards still in build (PSP, Kidney) use generic baseline-deviation thresholds (>2 SD = ELEVATED, >3 SD = SIGNIFICANTLY ELEVATED) until card-specific thresholds are calibrated, with the website page documenting that these are baseline-deviation thresholds rather than disease-specific calibrated thresholds.

8. **"Why KISS in the engine" made explicit.** The simpler the runtime, the fewer ways it can produce a wrong flag or a false alarm. Complexity is moved out of the runtime and into the educational layer where it can evolve safely. This also means the website carries the burden of being very, very good — the customer's interpretation depends on the educational content. That content is real work but it's tractable and it's the right place for the work.

What did NOT change:

- The 17-card catalog from Heath's authoritative list — unchanged.
- The IAMAtlas-only architectural rule, the GAPE engine as Heath-only IP, the deconvolution stage — unchanged.
- The card files in `cards/hcc/` and `cards/crc/` — unchanged. They remain as cookbook documentation. Runtime cards will be derived from them when the engine is built.
- The cookbook canonicals (README_MASTER, LESSONS_LEARNED, TESTING_CHECKLIST, GAPE Evidence Report) — unchanged.

This revision is documentation-only. It does not change any code, any card content, any validation result, or any sealed prereg. It locks the positioning that was already implicit in the earlier wellness-shift revision and adds the customer-experience refinements that came out of working through the breast 10-year scenario, the immune card's role, and the runtime-versus-website split.

### 2026-05-08 night (wellness-first strategic positioning shift)

The product positioning was overhauled across the roadmap from "early-detection diagnostic with wellness as a side benefit" to "cellular health and wellness tracking product with disease detection as a secondary capability." This is the most significant strategic change since the document was sealed.

The trigger: Heath surfaced that the existing diagnosis-translator framing put EDEAR in an unhelpful regulatory and commercial category. The framework genuinely measures cellular health across 8 architectural classes and produces a cellular age estimate. Disease detection is downstream of that measurement, not separate from it. Most customers will use EDEAR for tracking their own health, not for getting cancer diagnoses. The marketing should reflect what the product actually does at scale.

What changed in this revision:

1. **Strategic positioning rule added near the top of the document.** EDEAR is positioned first and foremost as a cellular health and wellness tracking product. The subscription model is the core commercial value: tracking changes over time as customers pursue lifestyle goals. Disease detection is a real and valuable secondary capability but is not the lead.

2. **Step D1 (card library) gained a runtime card schema note.** The cards currently in `cards/` (hcc-epic, crc-epic) are cookbook documentation cards — useful as audit-trail and source material but not shaped for runtime consumption. The runtime card schema is minimal: card ID, architectural class, substrate, demographic gates, threshold ranges, website URL. Everything else (disease description, validation evidence, conditional caveats, lessons learned) lives on the educational website or in the cookbook canonicals.

3. **Step D2 (report layer) was rewritten end-to-end.** Renamed from "diagnosis layer" to "report layer." Pipeline restructured to 8 stages (0-7) with explicit deconvolution as Stage 1, class A-scores as Stage 2, separate report assembly as Stage 6, separate report delivery as Stage 7. The report leads with cellular health summary and cellular age. Card flags appear as secondary findings with website links. Strong-recommendation language fires only at SIGNIFICANTLY ELEVATED thresholds. The Tier 0/1/2/3/4 clinical-tier vocabulary was replaced with NORMAL / ELEVATED / SIGNIFICANTLY ELEVATED / BELOW NORMAL — wellness-appropriate language. Marcus-class flagging removed from the report layer (the Marcus motivation stays in the cookbook documentation; the runtime does not single out specific personal-history patterns for special handling). Module renamed from `edear_diagnosis_translator.py` to `edear_report_builder.py`.

4. **Step D4 (customer-facing product) was rewritten.** Three surfaces: customer-facing report (PDF + HTML), educational website (`iamperformance.net` carries the substance), customer portal (subscription management + report archive + trajectory dashboard). The website carries the messaging and educational content, the report stays minimal and measurement-focused. The earlier "lab partner portal as v1 commercial wedge" framing was demoted from primary channel to one of several possible channels — the launch is direct-to-consumer wellness, with B2B lab partnerships as a parallel option.

5. **Step D5 (validation cases) was reframed.** Test cases are now framed around what the wellness report should produce, not around clinical confirmation. Reports lead with cellular health summary; card flags appear as secondary findings with website links; strong-recommendation language fires only at SIGNIFICANTLY ELEVATED.

What did NOT change:

- The technical pipeline architecture (Stages 0 through 7) is correct and unchanged in substance. Names of some stages refined but the math and the data flow are the same.
- The card files in `cards/hcc/` and `cards/crc/` are unchanged. They remain as cookbook documentation. Runtime cards will be derived from them when the engine is built.
- The IAMAtlas-only architectural rule, the GAPE engine as Heath-only IP, the deconvolution stage — all unchanged.
- The 17-card catalog from Heath's authoritative list (2026-05-08 evening) — unchanged.
- The cookbook canonicals (README_MASTER, LESSONS_LEARNED, TESTING_CHECKLIST, GAPE Evidence Report) — unchanged. They continue to live where they live and carry what they carry.

Why this matters for what comes next: the runtime engine work in Step D2 is now substantially smaller and simpler than it would have been under the disease-diagnostic framing. Cards do not carry conditional caveat tables, message-generation logic, or per-firing-pattern clinical-action matrices. The website carries the educational content. The runtime stays small, stays testable, and stays correct. KISS at the engineering level matches KISS at the strategic level: simpler product, smaller code surface, fewer ways for things to go wrong, cleaner regulatory posture.

### 2026-05-08 late evening (Heath's authoritative card list)

The morning's revision corrected a fabricated 17-card list by replacing it with a 14-card catalog read from README_MASTER_v2_7. That correction was structurally right (read source docs, no fabrication) but produced a different error: it under-counted Heath's actual operational card catalog.

Heath's authoritative card list as of 2026-05-08 evening:

1. AD
2. Bladder
3. Breast
4. Cardio
5. Cervical
6. Colon/Rectal
7. Gastric/Esophagus
8. Glioma
9. HCC Liver
10. HEME Leukemia
11. Lung
12. Pancreatic
13. Prostate
14. PSP — needs JSON and more work
15. Kidney — needs JSON and more work
16. Immune — functions as a card for systemic-departure routing (AD-pattern, inflammaging vs autoimmune differential, heme-epic uniform-elevation re-route); not a disease per se
17. Schizophrenia — discussed as a candidate, no card exploration yet

The 14 documented in README_MASTER_v2_7 + amendments (the morning's reading) are accurate but incomplete. The Immune card was previously treated as a reference document rather than a card; per Heath, it functions as a card because it routes systemic-departure signals. PSP was previously a tauopathy-specificity arm in ad-immune (VAL-057); per Heath it warrants a standalone card. Schizophrenia is on the candidate list pending landscape survey.

Step D1 rewritten with all 17 cards, broken into three groups: 14 with JSON in production (cardio-epic noted as v0.1 sealed but DEFERRED), 2 needing JSON + more work (PSP, Kidney), 1 candidate (Schizophrenia).

Diagram `edear_pipeline_stages_4_6_detail.svg` regenerated with all 17. Color coding: pink for the 14 with production JSONs, amber for PSP and Kidney, gray for Schizophrenia.

Lesson logged for future Walther sessions: when reconciling a card list, the cookbook docs are necessary but not sufficient. Heath's working operational catalog includes cards that exist as routing constructs (Immune), cards being promoted from sub-arms (PSP), and candidates not yet documented (Schizophrenia). Always confirm the card list with Heath before committing it to a diagram or downstream document.

### 2026-05-08 evening (post-card-list-correction session)

Heath caught two errors in the morning's revision:

1. **Step D1 contained a fabricated 17-card list** (AD-immune, Breast-secretory, CRC-immune-inv, CRC-secretory, Cervical-secretory, LGG/GBM-terminal, Parkinson's, Prostate, Pancreatic, MS, Cardiac, Aging, HCC, Gastric, Esophageal, Ovarian, Hematologic). That list was assembled from rough memory rather than read from README_MASTER_v2_7. Several cards in it (Parkinson's, MS, Aging, Ovarian) do not exist in the cookbook. Some named ones used wrong identifiers (Breast-secretory → actual is breast-epic; LGG/GBM-terminal → actual is glioma-epic). This was a Rule 1 no-fabrication violation. Step D1 rewritten with the actual canonical 14-card catalog from README_MASTER_v2_7 §"The ten validated cards" + 2026-05-01 v2.6 amendment (bladder-epic v0.1 sealed) + 2026-05-02 gastric-esophageal-epic v0.1 sealing + cardio-epic v0.1 DEFERRED status + kidney-epic to-build per v2.1 expansion table.

2. **Multiple references claimed "secretory needs re-running"** (Step 6.85 implied, Step 6 implied via the 4-of-8-done count, exit gate language, etc.). Secretory completed cleanly 2026-05-07 at R-hat 1.02 / ESS 1023 / zero divergences. We do NOT re-run secretory. The only correct treatment for secretory is the post-MCMC label-consolidation pass in Step 6.85 (Hep + Hepatocytes + hepatocyte → hepatocyte), same as terminal needs. Production MCMC table updated to current state (5 of 8 classes done including secretory; cycling running; progenitor + immune queued). Step 6 action and exit gate updated. Timeline shortened to 5-7 days remaining (was 8-10).

3. **Diagram `edear_pipeline_stages_4_6_detail.svg` regenerated** with the actual 14-card catalog. Color coding: pink for active/sealed cards, gray for cardio-epic (DEFERRED) and kidney-epic (TO BUILD).

4. **Test case at Step D5 corrected** from "breast-secretory" to "breast-epic" (actual card identifier).

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
