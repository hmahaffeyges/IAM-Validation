# RETIRED Phase 1 Pre-Build Disease Cards

**Status: RETIRED — REFERENCE ONLY**
**Era: Pre-IAMAtlas, Pre-SOP v1.2 (Phase 1)**
**Date archived to repo: 2026-06-02**
**Pushed by:** Heath W. Mahaffey + Walther (Claude) — Phase 1 closure cleanup

---

## ⚠ Read this first

This folder contains 16 disease cards built during Phase 1 of the EDEAR / CPG project, **before the IAMAtlas REBUILD landed (2026-04-06)** and **before the SOP v1.2 chain-of-custody protocol was finalized (2026-05-31)**.

**These cards must NOT be replicated, scored against, or used in their current form.** They were authored against:

- The pre-REBUILD `IAMAtlas.csv.xz` (which had the documented flatness bug — see `IAMAtlas_FLATNESS_LESSON.md`)
- External panels (Xu-538, EpiSCORE, Loyfer, Salas, UniLIFE) used as substitutes for the per-cell-type fan-out that the post-build pipeline now does natively
- Pre-SOP scoring methods that did not include credible intervals, foreground subtraction, the L9 null suite, or the universal Mahalanobis hyper-volume departure summary
- An older disease signature matrix (v1.3 and earlier — current canonical is v1.5)

**The only use of this folder is biological context.** When building a new post-build card (analogous to how `breast-epic_card_v2.3.json` was bumped to `breast-epic_card_v3.0.json`), the corresponding RETIRED card here contains the cohort identifications, clinical claims, panel design rationale, and headline numbers that establish what the disease's signal looked like under the old pipeline. That information is the starting point for designing the new card's validation strategy — but every analytical artifact in this folder is to be re-derived against IAMAtlas REBUILD + Walther IAM Deconvolver + `walther_clinical_runtime/` modules before any production use.

## What this folder IS

A complete archive of Heath's Phase 1 work product — 16 disease cards, the retired evidence report, the prior cookbook docs (README_MASTER versions, LESSONS_LEARNED, TESTING_CHECKLIST, GAPE_Reproduction_Paper), the cross-card calibration TODO, and the EDEAR pipeline reference of that era. Everything that informed the design of the post-build instrument is here for reference and audit.

## What this folder is NOT

- **Not the current operational instrument.** That lives in `Biological_Physics/atlas_vault/walther_clinical_runtime/`.
- **Not the current scientific case.** That lives in `post_build_evidence/v3_CPG_IAMAtlas_Evidence_Report.html` and `v4_CPG_VAL_Inventory_Report.md`.
- **Not the current cards.** The only post-build card finalized to date is `walther_clinical_runtime/DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_0.json` (2026-06-02).

## Card index

| Card folder | Primary JSON | README | Status in post-build era |
|---|---|---|---|
| `AD/` | `ad-immune_card_v2.2.json` | `ad-immune_README.md` | **NEXT** active post-build VAL series. v2.2 + VAL057/091/092 are the reference material for building the new `ad-immune_card_v1_0.json` (post-build). |
| `Bladder/` | `bladder_epic_card_v0_2.json` | `bladder_epic_README_v0_2.md` | TO BUILD in future Family C sprint. |
| `Breast/` | `breast-epic_card_v2_3.json` | `breast-epic_README.md` | **SUPERSEDED 2026-06-02** by `breast-epic_card_v3_0.json` in the runtime folder. This Phase 1 copy retained for audit. |
| `Cardio/` | `cardio_epic_card_v0_3.json` | `cardio_epic_README_v0_3.md` | TO BUILD. |
| `Cervical/` | `cervical-epic_card_v0.1.json` | `cervical-epic_README.md` | TO BUILD. |
| `Colon:Rectal/` | `crc-epic_card_v2_4.json` | `crc-epic_README_v2_4_1.md` | TO BUILD. |
| `Gastric/` | `gastric_esophageal_epic_card_v0_1.json` | `README.md` | TO BUILD. Note: contains large `OLD/atlas_vault-2/` reference data (EpiSCORE, IDOL, Loyfer, UniLIFE companion panels) — all pre-IAMAtlas. |
| `Glioma/` | `glioma-epic_card_v0.2.json` | `glioma-epic_README.md` | TO BUILD. VAL092 also lives in `AD/` since glioma/AD share the terminal cortical neuron axis. |
| `HCC (Liver)/` | `hcc-epic_card_v0_3.json` | `hcc-epic_README.md` | TO BUILD. |
| `Heme (immune:Leukemia)/` | `heme-epic_card_v0.1.json` | `heme-epic_README.md` | TO BUILD. |
| `Immune Atlas Card/` | `immune-atlas_card_v0_3_2.json` | `immune-atlas_README_v0_3_2.md` | **Cross-cutting card.** Documents the 51-cell immune class split test (2026-05-04) that informed the post-build per-cell-type marker design. No direct replacement — the immune class is now scored cell-by-cell natively by the runtime. |
| `Kidney/` | — (landscape survey only) | — | Phase 0 cohort survey + Jeong 2026 bridge work. No card was built before IAMAtlas REBUILD. |
| `Lung Card/` | `lung-epic_card_v0_5.json` | `lung-epic_README_v0_5_1.md` | TO BUILD. |
| `PSP/` | — (README only) | `psp-epic_README_v0_1.md` | Pre-card concept doc. |
| `Pancreatic/` | `pancreatic-epic_card_v0.1.json` | `pancreatic-epic_README.md` | TO BUILD. |
| `Prostate/` | `prostate_epic_card_v0_3.json` | `prostate_epic_README_v0_3.md` | TO BUILD. |

## Top-level Phase 1 retired docs

- `RETIRED_Evidence_Report.html` — the Phase 1 era HTML evidence report. Superseded by `post_build_evidence/v3_CPG_IAMAtlas_Evidence_Report.html`.
- `edear_card_catalog_17.svg` — Phase 1 visual catalog of the 17 cards Heath was tracking. Reference only.

## `Z_OLD/` — Phase 1 cookbook artifacts

These are the Phase 1 cookbook documents — predecessors to the current SOP v1.2 + INVENTORY + build spec. Listed here as audit trail; **not to be used as the current protocol.**

| File | What it was | What replaces it now |
|---|---|---|
| `README_MASTER_v2_1.md` through `_v2_6.md` | Phase 1 master cookbook spec (5 versions across 2026-04 → 2026-05) | `walther_clinical_runtime/INVENTORY.md` (file-level reference) + `CPG_Chain_of_Custody_SOP_v1_2.md` (operational protocol) |
| `TESTING_CHECKLIST.md` | Phase 1 VAL testing checklist | Heath's local `v1_VAL_Test_Checklist.md` (until refreshed to v2 reflecting the SOP v1.2 chain) |
| `LESSONS_LEARNED.md` | Phase 1 lessons learned | Carried forward into MASTER_TRACKER §9 (Heath-only) and the SOP v1.2 narrative |
| `GAPE_Reproduction_Paper_v1.md` | Phase 1 reproduction-grade documentation | Superseded conceptually by the v3 evidence report's reproducibility quadruple format |
| `CROSS_CARD_CALIBRATION_TODO_v0_5.md`, `_v0_6.md` | Phase 1 cross-card calibration tracker | Folded into the v4 VAL inventory's Phase G/H/I/J ordering + the disease signature matrix v1.5 |
| `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md`, `_v2_a.md` | Phase 1 pipeline reference | `CPG_Chain_of_Custody_SOP_v1_2.md` (the SOP) + `walther_clinical_BUILD_SPEC_v1_1.md` (the build spec) |
| `universal_reference_block.py` | Phase 1 universal reference block template | Card JSONs in `walther_clinical_runtime/DISEASE_MAPS_CARDS/*/` now embed the universal_reference block inline (full-inline Option B per breast v2.3 design) |
| `update_all_cards_v2.1.py` | Phase 1 batch card updater | Each post-build card is bumped surgically and additively under SOP v1.2 — no global batch updater |

## Method-substitution table (what was used Phase 1 vs. what's used now)

| Phase 1 method | Post-build replacement |
|---|---|
| Pre-REBUILD `IAMAtlas.csv.xz` (flatness bug) | `IAMAtlasREBUILD.csv.xz` (483,092 CpGs × 8 classes × 115 cell types; SHA `41b7c16f...`; H_min frozen 2026-04-06) |
| `scipy.optimize.nnls()` ad-hoc | Walther IAM Deconvolver (marker-rank, streaming, 60%/80% gates, status codes) |
| Xu-538 panel for breast immune A-score | Still used Stage 1; supplemented by per-cell-type fan-out (115-cell A-scores) at Stage 2 |
| EpiSCORE BreastRef / EsoRef / OEref / etc. | IAMAtlas REBUILD per-cell-type markers v0_2 (one atlas, one deconvolver) |
| Loyfer 25-tile + Salas + UniLIFE separate runs | Single Walther + IAMAtlas pass produces all 115 cell-type A-scores natively |
| TODO 1.1/1.2/1.3/1.5 roadmap IDs in evidence_anchors | CPG-VAL-001/002/003/005/007 formal sealed-VAL IDs (with TODO IDs retained as aliases per disease matrix v1.5) |
| No L9 null suite | `cpg_null_runner.py` 7-test sealed null suite per VAL |
| No Mahalanobis universal summary | `iamatlas_mahalanobis_scoring.py` single-number Cohen's d departure-from-HC per patient |
| No age-axis foreground subtraction | `age_axis_foreground.py` + `IAMAtlas_age_layer.csv` (8,199 CpGs, 100% convergence) |
| No reproducibility quadruple per VAL block | Evidence Report VAL blocks now carry inline source code + inputs list with SHAs + environment + expected headline output |

## How to use this folder going forward

When building a new post-build card (per the MASTER_TRACKER §7 forward plan), the workflow is:

1. **Read the corresponding RETIRED card here** — the JSON, the README, the VAL outcomes — for biological context: what disease, what cohorts, what tissue, what stage, what clinical claim was being made, what windows in the natural history were being targeted.
2. **Identify what carries forward** — the cohort identifications (GSE accessions), the clinical claim, the literature anchors, the population stratification rationale. These are biology, not method, and most carry forward.
3. **Identify what does NOT carry forward** — every method, every panel choice, every scoring rule, every threshold, every H_min anchor. These were built against the old instrument and must be re-derived against IAMAtlas REBUILD + Walther + SOP v1.2.
4. **Run the new card through the SOP v1.2 chain** end-to-end (Stages 0–10 + L9 null suite) using `walther_clinical_runtime/` modules only.
5. **Package as the post-build card** at `DISEASE_MAPS_CARDS/{name}/` following the 6-file pattern: `{card}_card_json/{card}_card_v1_0.json` + `{card}_README.md` + `{card}_residual_maps/` with the three map CSVs + maps README.
6. **Add the row(s) to the disease signature matrix** (currently v1.5; will bump to v1.6+ as new cards land).
7. **Package the first VAL run** as the inaugural `Biological_Physics/validation_runs/CPG-VAL-NNN/` folder under v4 inventory protocol (reproducer script + sealed PREREG + results.json + per_sample.csv + outcome.md + cohort_manifest.json).

## Pointer

The forward sequencing — which card next, what order — lives in `MASTER_TRACKER.md` (Heath-only, not in this repo). The next active card per the 2026-06-02 decision is **AD-immune**, drawing from the `AD/` folder here.

## ⚠ Final reminder

**Nothing in this folder is part of the current operational instrument.** No tool in `walther_clinical_runtime/` reads from this folder. No production pipeline path passes through this folder. This is biological reference material and audit trail — nothing more.
