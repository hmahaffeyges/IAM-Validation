# HANDOFF — the physics of methylation, for whoever picks this up

**What this field is called (2026-09-20): Physics of Methylation: Landauer Metrology** — measuring how far above the thermal noise quantum an information-writing process operates, against a fixed physical zero (H_min per cell class). Thermal noise is the unit (M = E_drive / k_B T), not the nuisance. Prior art: Sanchez & Mackenzie 2016 established that the methylome obeys Landauer's bound; Landauer metrology measures how far above it each cell class operates (Issue 003 §0b).


**Written 2026-09-19 at commit `46f9b77`.** This file is for a researcher arriving cold. It says where to start, what is sealed, what is open, and what not to do. It does not repeat the science; it points at where the science is.

## Start here, in this order
1. `Physics_of_Methylation/Issue003/IAMPerformance_GAPEIssue003_RC1.pdf` — **page 4 first** ("What this document claims, and what it does not"). Then §1.6 (what the cosmology tools found that cohorts could not) and §1.7 (the reporting rule). Everything else in the 280+ pages is reference.
2. `Physics_of_Methylation/Reproduction_Kit/README_FIRST.md` → `RUNBOOK.md`. Run `PROC_DECON_01.py` first. If it passes on your machine, the atlas and deconvolver are working; if it does not, stop and open an issue — nothing downstream is meaningful.
3. `Physics_of_Methylation/SOP/CPG_Chain_of_Custody_SOP_v2_0_0.md` — read the **SUPERSESSION LEDGER** before any stage section; stale sections are marked, not deleted.
4. `Testing_and_Code/VAL_INDEX.csv` — every validation ever run, with its path and stated verdict where one exists. Verdicts are *recorded*, not re-verified.

## What is sealed (reproduces from raw public data, on a machine that had never seen the project)
- **PROC-DECON-01** — deconvolver vs the project's own answer key (MAE 0.0004).
- **PROC-CAL-01** — raw IDAT → calibrated β through Stage 1, exact match to the cached betas, 11/11, both array types.
- **PROC-ANCHOR-01** — the sealed 648-sample breast foundation cohort (GSE51032 + GSE51057, 115 cell types) reproduced from raw GEO at r = 1.00000, on the repo-HEAD markers, under the per-locus statistic. Re-sealed anchors on the chrX-removed markers are in `foundation_cohort/anchors_v2_chrXremoved/`.
- **PROC-N7-01 / PROC-NILC-01 / PROC-SEP-01..03** — the synthetic-truth and cross-method tests. Read these before trusting any gauge reading (see "Do not").

## What is open (decided but not yet done)
- **PHASE 1 — the identity-loci healthy band.** Pre-registered and sealed: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/PREREG.md`. Until it runs, **the production class gauge has no valid band**: `cpg_conductor.stage_b_classes` computes H(β̄) over the class *marker union* and `age_reference_matrix.json` was compiled the same way, so they agree with each other, not with the biology (PROC-N7-01). Every conductor reading carries `gauge_surface = "marker_union"` so this cannot be missed.
- The myeloid arm (MDS / CML / CHIP) — unvalidated disease-matrix rows; Tool B (`CPG_Engine/Lineage_Splitter/`) is the instrument that would test them.
- Stage 6 cellular age is **not reportable** (pinned at its curve floor). Stage 0 intensity QCs are **DEFERRED** (hand-off from Stage 1 unwired).

## Do not
- **Do not compute group effect sizes (Cohen's d, AUC) on a cohort and call it a result.** The instrument reads each sample against a fixed physical reference; cohorts supply *direction only* (rule L-5). This error recurred ten-plus times across a year of AI-assisted work. If a cohort arrives with labels, the reflex is wrong.
- **Do not use the cell-type marker pool for anything but "which cell is this."** It is bimodal by construction. Used as a gauge surface it inflates A (PROC-N7-01); used as a deconvolution basis it is ill-conditioned (PROC-NILC-01); it caused the 2026-06-11 all-BREACH bug. Identity loci are the gauge surface; Walther's own class markers are the deconvolution basis.
- **Do not report a class gauge on a substrate where the class is not determined and present** (SOP §108, Issue 003 §1.7). On whole blood: immune is reported; progenitor + stem_adult as one haematopoietic-progenitor component; the other five composition-only. Three independent routes found stem_adult ≈ 0 in healthy blood.
- **Do not switch off a check because it disagrees** (RUNBOOK §11). N7 and the cross-method comparison run on every release; a disagreement gets a ledger row, never a disable flag. Two safeguards were once cut for tripping; both were right.
- **Do not read A absolutely against H_min from any pipeline's raw β without first mapping it onto the Roadmap scale** (`beta_scale_maps_v1.json`). H_min lives on the Roadmap scale; Stage-1 noob is +0.066 higher, GEO-processed EPIC +0.037. Within-pipeline comparisons never see this; the absolute gauge always does. Do not re-derive H_min per pipeline. (LESSON-SCALE-01)
- **Do not mix physical floors with cohort statistics.** H_min (per class × substrate, MCMC-derived, 40 values) is physics. A band is a percentile of a healthy cohort. A tier is a rule. Keep the three words separate (Issue 003, Chain Links glossary).
- **Do not trust a docstring over a measurement.** Two modules at HEAD each called the other the regression; the 2×2 test settled it (PROC-FORMULA-01). The repo is canonical; canonical does not mean correct.

## Where things live
See `Physics_of_Methylation/Reproduction_Kit/COMPONENT_MAP.md` — repo (edited), kit (frozen snapshot, never hand-edited), author's folder (large inputs and vault IP: the Recipe, `_gape_constants_private.py`, patents). Large inputs not in git: the decompressed atlas CSV (577 MB; the `.xz` is here), `betas_cache.pkl`, IDATs, GEO series matrices.

## The claim, in one paragraph
Not the largest atlas, not an atlas for sale, not detection of all diseases or any disease ten years early, not the myeloid cancers, not clinical readiness. Claimed: a per-class thermodynamic reference plus an MCMC atlas plus the CMB validation toolkit is a method biology did not have, and on limited public data with no funding it already resolves things a cohort comparison structurally cannot (§1.6). For the immune class in whole blood there are reproducible signals worth investigating with proper support. The record is written so you can break it.

*Nothing here is validated for patient care.*


---

**BETA SCALE (LESSON-SCALE-01, 2026-09-20).** H_min was calibrated by the G-002 MCMC on Roadmap/ENCODE reference β (GenomicStudio-normalised). The Atlas posteriors sit on that same scale. Other pipelines do NOT: on the 42,024 immune identity loci, healthy blood reads β̄ = 0.737 on the Roadmap/Atlas scale (A = 1.00), 0.774 on GEO author-processed EPIC (GSE51032 HC; A = 0.92), and 0.815 on Stage-1 noob from raw 450K IDATs (GSE87571; A = 0.82). The offset is additive (+0.066 β for Stage-1). Every within-pipeline comparison (Cohen d, ΔA, case-vs-control on one matrix) cancels this and never sees it — which is why 200 VALs never tripped on it and why the April 2026 VAL-003 output could say "ΔA valid within-pipeline; absolute thresholds require a pipeline-matched healthy reference." An ABSOLUTE reading of A against H_min requires the patient β to be mapped onto the Roadmap scale first: one affine map per pipeline, fit on healthy blood (`Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json`). The floors are not re-derived per pipeline — that would discard the MCMC confirmation. Three layers, keep them separate: FLOOR (Roadmap scale, MCMC, physics) → PIPELINE (affine map) → LAB (~0.01–0.02 A per cohort; plate/batch, N-plate). Record: `Testing_and_Code/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md`; Issue 003 RECON S1, §1.6.

**LAB ZERO (LAB-ZERO-02, 2026-09-20).** A patient's absolute A is read against FLOOR (H_min) + PIPELINE MAP (Stage 1s) + **LAB ZERO**. Four healthy whole-blood cohorts on one scale sit at 0 / +0.024 / −0.021 / −0.046 A (Uppsala / Karolinska / Munich / UCLA), each constant flat across age; the array's control probes predict the sign of every offset but the size of only the Swedish pair, so the lab zero is **not** modelled — it is measured: a per-lab healthy-control panel (20–30 arrays, once, through the same Stage 1; median mapped immune A subtracted; offset printed on every report), as CLSI EP28 prescribes. Record: `Testing_and_Code/VAL_PostAtlas/CPG_LABZERO_02_fourth_cohort_GSE111629/`.
