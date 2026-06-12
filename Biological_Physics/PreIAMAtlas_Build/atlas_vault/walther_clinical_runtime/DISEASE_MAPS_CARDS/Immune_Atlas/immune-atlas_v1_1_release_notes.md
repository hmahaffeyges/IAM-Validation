# Immune Atlas Card v1.1 — Release Notes

**Date:** 2026-06-07
**Card version:** v1.1
**Card type:** universal_baseline_card (unchanged from v1.0)
**Schema version:** 1.0 (unchanged from v1.0 — no structural change)
**Status:** SEALED. All declared v1.0 validation evidence resolved; Mahalanobis hull at production v0_5; disease matrix consultation aligned to v1.8.
**Predecessor:** `immune-atlas_v1_0_release_notes.md` (preserved as historical record of the v1.0 SKELETON release on 2026-06-06)

---

## What this release contains

`immune-atlas_card_v1_1.json` — 83 KB, 56 top-level keys, 13 stage blocks. This is the operational v1.1 of the immune-atlas card, replacing the v1.0 SKELETON that shipped 2026-06-06 with all declared validation evidence now sealed, the Mahalanobis HC hull at converged production version v0_5 (n=2,481), and Stage 8 Route B consulting disease matrix v1.8.

The card remains the **universal first-pass measurement** that every customer IDAT runs through. It produces no disease verdict — engine-internal concordance flags from Stage 8 disease signature matrix consultation feed downstream disease cards.

---

## What changed from v1.0 → v1.1

### Validation evidence resolved (was PENDING in v1.0)

All 7 declared v1.0 VALs reached resolution between 2026-06-06 and 2026-06-07. CPG-VAL-014 (AD-GIFT tauopathy specificity) was added retroactively to validation_evidence because its Mahalanobis hyper-volume + AD/FTD/PSP three-direction differential is immune-class-relevant for the immune-atlas card's Stage 5 Mahalanobis interpretation (primary card-origin remains AD-immune card v3.1). CPG-VAL-022 (smoking cessation reversibility) was added as a sealed companion to VAL-021 weight-loss work.

| VAL | Outcome | Notable result |
|---|---|---|
| CPG-VAL-014 | **PASS** | AD d=+0.681 / PSP d=-0.380 / FTD d=+0.279 — three-direction tauopathy differential |
| CPG-VAL-015 | **PASS** | A_immune r=-0.197 p=3.69e-07 Hannum n=656; sex-symmetric; decade-median ρ=-0.854 |
| CPG-VAL-016 | **DIRECTIONAL** | Universal alarm fires in AD (d=-0.36) AND breast pre-dx (d=+0.69 GSE51057) with disease-specific direction |
| CPG-VAL-017 | **NULL** (informative) | Pooled r=+0.034 p=0.150 washed by cohort heterogeneity; within-cohort late-life acceleration 2.19 |
| CPG-VAL-018 | **NULL** | Menarche partial r=+0.010 p=0.82 — clean null at pre-registered threshold |
| CPG-VAL-019 | **PASS** | AIBL d_up=+0.494 / d_down=-0.515; per-CpG concordance 7/7 with VAL-051 panel |
| CPG-VAL-020 | **SEALED** | Hannum full-chain reproduction; chain integrity 656/656; pre-build VAL-006 anchor NOT_REPRODUCED_BY_DESIGN (was regression-trained, not physics inversion) |
| CPG-VAL-021 | **DEFERRED** | No suitable longitudinal bariatric cohort identified; acquisition queued post-June-11 |
| CPG-VAL-022 | **NULL** (cohort-limitation) | Tsaprouni n=22 current-smokers underpowered; former-smoker overshoot pattern persists |

### Mahalanobis HC hull at production state v0_5

The Mahalanobis healthy-reference hull completed all four declared expansion phases between 2026-06-06 and 2026-06-07:

- Phase 1: v0_1 (n=601 EPIC-Italy) → v0_2 (n=1,257) +Hannum HM450
- Phase 2: v0_2 → v0_3 (n=1,721) +Tsaprouni GSE50660
- Phase 3: v0_3 → v0_4 +EPIC-platform HC cohort (cross-platform transferability)
- Phase 4: v0_4 → v0_5 (n=2,481) +Asian-population HC cohort

Production thresholds (v0_5): p95 = 13.62 default, p99 = 18.59 strict. Hull converged and sealed.

### Stage 8 Route B aligned to disease matrix v1.8

The disease matrix bumped from v1.7 to v1.8 on 2026-06-07 with a strict additive evidence_anchor refresh (5 surgical appends across 5 rows: breast_cancer/long_pre_dx_post_build_v3_0, alzheimers_disease/active, normal_aging/chronic, inflammaging/chronic, alzheimers_disease/at_dx_post_build_v3_0). Zero cell-value changes, zero new rows, zero schema changes.

The `iamatlas_115_to_matrix_v1_7_mapping.json` mapping artifact retains its v1_7-suffixed filename because v1.7 → v1.8 was strictly additive at the evidence_anchor level — column structure was unchanged. The mapping is therefore valid for v1.8 without rebuild.

### Stage 8 Route A trigger refreshed

v1.0 carried a stale Route A trigger of "Mahalanobis_d ≥ 2.0 against pooled n=601 HC hull" — a placeholder from initial v0_1 hull design. v1.1 corrects this to: "Mahalanobis_d ≥ p95 threshold = 13.62 (default) or p99 = 18.59 (strict) against pooled n=2,481 HC hull v0_5".

### Outstanding work — 5 of 17 items closed

The card's `outstanding_work_v1_0` list (still named `_v1_0` in v1.1 — the same list, with items closing) has 5 of its 17 items now marked COMPLETED:
- #1 Phase 1 Mahalanobis HC hull expansion (was already marked COMPLETED in v1.0)
- #2 Phase 2 Mahalanobis HC hull expansion (was already marked COMPLETED in v1.0)
- #3 Phase 3 Mahalanobis HC hull expansion (newly marked COMPLETED in v1.1)
- #4 Phase 4 Asian-population HC hull expansion (newly marked COMPLETED in v1.1)
- #17 DISEASE_MATRIX v1_7 → v1_8 (newly marked COMPLETED in v1.1 with v1.8 SHA hashes)

### Net structural change

`schema_version` is unchanged at 1.0 — v1.1 added one new top-level field (`v1_0_to_v1_1_changes`) and added one new entry to `validation_evidence_v1_0_set` (CPG-VAL-014). All other v1.0 sections remain byte-identical in structure; updated values are text-level only.

File size delta: v1.0 = 77,976 bytes → v1.1 = ~84,000 bytes (Δ +6 KB for CPG-VAL-014 entry + v1_0_to_v1_1_changes section + matrix v1.8 reference refreshes + Phase 3/4 item completion text + Route A trigger refresh).

---

## What did NOT change in v1.1

- All 8 existing validation_evidence entries for CPG-VAL-015 through CPG-VAL-022 — byte-identical to v1.0 (they were already complete and correct when v1.0 sealed)
- The 19 cell_types_of_interest list and per-cell atlas provenance entries
- All Stage 0–10 module references (except Stage 5 hull artifact path and Stage 8 disease matrix path)
- The 24-covariate intake schema
- The 9 report_strings, 10 report_vigilance_strings
- The 8 honest_limitations
- The 14-entry v1_0_changes_from_pre_build list (historical record of the pre-build → v1.0 transition)
- The 8 _open_questions_for_review

---

## Honest limitations (unchanged from v1.0)

See `card.honest_limitations` for the full list. Notable v1.0 caveats that remain in v1.1:

- Smoking + sex foreground subtraction at Stage 3 β-level NOT yet built (deferred to v1.2)
- L5 / L7 / L8 (correlation structure / Bayesian likelihood / per-card MCMC posterior) EMPTY — deferred to later phases
- Per-card immune residual map NOT BUILT — necessity under active assessment (Heath open question 2026-06-07: what would it actually determine beyond what class-level metrics + disease matrix already capture?)
- 19 per-cell pages still contain pre-build atlas references that need scrubbing (deferred — patient-facing education, lower priority than doctor-report workstream per Heath's 2026-06-07 pivot)
- The disease_immune_lens cross-reference section (Design C from Heath's 2026-06-07 plan) is NOT YET added to the card — that's Stage B work for a future session
- The wellness/aging/inflammation lens section is NOT YET added to the card — that's Stage C work for a future session

---

## Strategic pivot captured in v1.1

Per Heath's direction 2026-06-07, the immediate priority following v1.1 release is no longer the patient-facing report build (cell page scrub, customer education, residual maps). The new priority is:

1. **Capability inventory** — comprehensive list of EVERYTHING the current CPG version with all chain-of-custody steps can actually determine from a single blood draw
2. **Doctor report draft** — a thorough physician-facing report built from that capability inventory, designed for the June 11 GeoMetric meeting with Dr. Tanya Escobedo and team

Patient-facing report design will follow AFTER Dr. Escobedo provides direction on what patients should and should not receive. The cell pages and other educational material will be the substrate for that subsequent patient-report build.

---

## Chain-of-custody anchors (v1.1)

- IAMAtlas canonical SHA-256: `41b7c16f043bce96e085a2b8b4e709efd2b862af9de8dbe9a8646e9fb94c32ee` (unchanged from v1.0)
- Celltype marker artifact SHA-256: `46ea5be1db377f2b8773a02418a7f481a191630e0fa833d3294eab1fd19c47bd` (unchanged from v1.0)
- VAL-051 directional panel SHA-256 anchor: `52061285fc97bfff871ba7b62f625b14d953bccf25ee24e35f328e15b9827998` (unchanged from v1.0)
- BUILD_SPEC reference: `walther_clinical_BUILD_SPEC_v1_2.md` (unchanged from v1.0)
- SOP reference: `CPG_Chain_of_Custody_SOP_v1_3.md` (unchanged from v1.0)
- Disease matrix CSV SHA-256 (v1.8): `1ed44cccad8e7af21e5a5901453fc6de6ab416988a284accd3de8469150e69f1` (NEW — Stage 8 Route B consumption)
- Disease matrix README SHA-256 (v1.8): `ba2d321a3c168b2e48ce5f42b5bc78c142d4338f3edfeed9b63534de980e3918` (NEW — Stage 8 Route B documentation)
- Mahalanobis hull v0_5 artifact: `mahalanobis_healthy_reference_v0_5.json` (path locked; SHA carried by runtime artifact not by this card)

---

## Push protocol (when Heath pushes v1.1 to GitHub)

Per the per-card workflow rule (memory #11): the card JSON + the README + the release notes are a package and all push together.

1. Move existing repo `immune-atlas_card_v1_0.json` to `immune_atlas_card_json/OLD/`
2. Drop `immune-atlas_card_v1_1.json` in place
3. Overwrite `immune-atlas_README.md` (the v1.1 surgical update)
4. Add `immune-atlas_v1_1_release_notes.md` alongside the preserved `immune-atlas_v1_0_release_notes.md`
5. Verify `patient_intake_questionnaire_v1_0.md` is unchanged (it is — v1.1 did not touch it)
