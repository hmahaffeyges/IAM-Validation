# HANDOFF — the physics of methylation, for whoever picks this up

**Written 2026-09-19 at commit `46f9b77`.** This file is for a researcher arriving cold. It says where to start, what is sealed, what is open, and what not to do. It does not repeat the science; it points at where the science is.

## Start here, in this order
1. `Physics_of_Methylation/Issue003/IAMPerformance_GAPEIssue003_DRAFT.pdf` — **page 4 first** ("What this document claims, and what it does not"). Then §1.6 (what the cosmology tools found that cohorts could not) and §1.7 (the reporting rule). Everything else in the 280+ pages is reference.
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
- **Do not mix physical floors with cohort statistics.** H_min (per class × substrate, MCMC-derived, 40 values) is physics. A band is a percentile of a healthy cohort. A tier is a rule. Keep the three words separate (Issue 003, Chain Links glossary).
- **Do not trust a docstring over a measurement.** Two modules at HEAD each called the other the regression; the 2×2 test settled it (PROC-FORMULA-01). The repo is canonical; canonical does not mean correct.

## Where things live
See `Physics_of_Methylation/Reproduction_Kit/COMPONENT_MAP.md` — repo (edited), kit (frozen snapshot, never hand-edited), author's folder (large inputs and vault IP: the Recipe, `_gape_constants_private.py`, patents). Large inputs not in git: the decompressed atlas CSV (577 MB; the `.xz` is here), `betas_cache.pkl`, IDATs, GEO series matrices.

## The claim, in one paragraph
Not the largest atlas, not an atlas for sale, not detection of all diseases or any disease ten years early, not the myeloid cancers, not clinical readiness. Claimed: a per-class thermodynamic reference plus an MCMC atlas plus the CMB validation toolkit is a method biology did not have, and on limited public data with no funding it already resolves things a cohort comparison structurally cannot (§1.6). For the immune class in whole blood there are reproducible signals worth investigating with proper support. The record is written so you can break it.

*Nothing here is validated for patient care.*
