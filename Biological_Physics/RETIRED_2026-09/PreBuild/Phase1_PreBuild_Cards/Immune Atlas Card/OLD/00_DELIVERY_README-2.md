# Immune-atlas card v0.2 → v0.3 promotion to Rosetta Reference Card
**Date:** 2026-04-30
**Trigger:** Heath request to promote card from v0.1/v0.2 differential-diagnosis engine to canonical Stage 1 interpretation engine + Stage 2 atlas cross-reference + Stage 3 OQ-2026-01 staging hub.

## Line-count audit

| File | v0.2 baseline | v0.3 final | Delta |
|---|---|---|---|
| immune-atlas_README.md | 268 lines | 1,194 lines | +926 |
| immune-atlas_card_v0_*.json | 561 lines (v0_2) | 1,496 lines (v0_3) | +935 |

JSON top-level keys: 21 → 35 (added 14 new keys; all 21 v0.2 keys preserved verbatim).

## What changed (high level)

### Header / framing
- Validation tier `reference_document` → `rosetta_reference_card`
- Card position #13 of 15 → #15 of 15 (last by dependency, first in operational importance)
- Card role added: `stage_1_interpretation_engine + stage_2_atlas_cross_reference + stage_3_oq_2026_01_staging_hub`

### Major NEW sections in README
- **§1.1 The three-stage diagnostic spine** (text-form architecture diagram)
- **§1.4 RUN-everything as the safety net for concurrent disease scenarios**
- **§2 Pre-test Integrity Protocol** — atlas calibration prerequisite, six data integrity checks (CHK-3.1, 3.1B, 3.1C, 3.2, 3.5, sample-group spot-check), biology consistency check (transferability vs cohort heterogeneity vs true biology null), demographics-as-mandatory-stratifiers (sex / age / smoking / ancestry / HPV / HIV / immunosuppression / pregnancy / treatment stage / specimen collection method per disease), bidirectional-as-default doctrine, atlas calibration prerequisite, ten failure-mode fingerprints, six biology-real patterns
- **§6 CCL-031 five-pattern taxonomy** — pooled-positive, pooled-negative compartment-flip, pooled-null + directional-pass, cross-disease direction difference, lineage-confirmed bidirectional (currently NONE)
- **§7 Three doctrine cases** — AD's gift (pooled-vs-directional), HCC's gift (substrate-as-discriminator), glioma's gift (orthogonal-vs-inverted)
- **§8 Cookbook doctrine that touches immune class** — CCL-006/019/023/027/028/030/031/032/039 verbatim rules + cards exhibiting each
- **§10 Stage 2 atlas registry** — every atlas EDEAR uses with sealed SHA + calibration anchor + use-tier; production atlases (Xu-538, Layered Moss+Loyfer, Salas IDOL, EpiDISH, Loyfer/Moss, ProstateRef, UniLIFE) and research-grade NOT calibrated (HeartRef, BreastRef, Caggiano TIM, EpiSCORE future bridges)
- **§11 CCL-027 four-question master cross-reference table** — every disease's documented answers consolidated centrally for the first time
- **§12 OQ-2026-01 immune-atlas staging hub** — canonical home moved here; what Test 2 needs, closest existing analogs, what unblocks when staging operationalizes, open atlas-coverage gaps blocking, cards first affected
- **§14 Cross-card syntheses** — expanded from 1 pair (prostate-vs-breast in v0.2) to 10 pairs covering all sealed and pending cards
- **§15 Open atlas-coverage gaps and biobank-gated next steps**
- **§17 Future v1.0 reorganization plan** — symptom-organized decision tree replacing v0.x disease-organized cross-reference with text-form decision tree

### Sections expanded
- **§5 Cross-reference table** — populated with v0.3 sealed numbers from cardio/pancreatic/heme/cervical cards (every CCL-027 answer fills in)
- **§13 Stage 3 sub-cell-type signatures** — expanded from prostate-only (v0.2) to 5 sealed cards (prostate VAL-118, breast VAL-095, glioma VAL-090, AD-immune exploratory, heme three-arm structure as closest OQ-2026-01 analog) plus 7 pending
- **§16 Mahalanobis differential ranker** — v0.1 3-dim (direction, magnitude, per-CpG pattern) → v0.3 5-dim (adds A_lymphoid, A_myeloid pending OQ-2026-01)
- **§22 Lessons learned** — v0.1 (CCL-019/020/021/022) preserved verbatim; v0.2 (DISC-PROSTATE-001/002/003, breast-LL-007, immune-atlas-LL-001) preserved verbatim; v0.3 added (immune-atlas-LL-002 through LL-008)

### Preserved verbatim (NOT touched)
- All v0.1 lessons (CCL-019/020/021/022) — preserved word-for-word
- All v0.2 additions (DISC-PROSTATE-001/002/003, breast-LL-007, immune-atlas-LL-001, prostate-vs-breast cross-card synthesis, prostate Stage 3 multi-atlas table) — preserved word-for-word
- §9 The four Stage-1-positive Stage-2-null pathways — Pathway 1 terminal, Pathway 2 hematologic, Pathway 3 cardiovascular, Pathway 4 unexplained drift — preserved verbatim with v0.3 cross-references added
- The Mahalanobis-distance differential ranker spec (v0.1 form) — preserved verbatim, expansion is additive
- Language discipline section — preserved verbatim with v0.3 CCL-031 additions
- File pointers — updated for v0.3 destinations only
- Card evolution plan — preserved verbatim with v0.3 + v1.0 additions

## File pointers for next sprint

- v0.3 README: `immune-atlas_README.md` (no version in filename, internal version 0.3)
- v0.3 JSON: `immune-atlas_card_v0_3.json` (matches v0.2 naming pattern)
- v0.2 originals preserved at `v0_2_originals_for_reference/` for diff comparison

## What this card now is

Per Heath's framing (sealed in §1.3, §1.4, §22 immune-atlas-LL-002):

> "This is the interpretation card for the red flag test that is the first test that alerts to a problem. The next tests run through ALL the atlases and matrices we have at our disposal and the individual disease cards translate what we then learn about the individual cell class and organ responses. Ultimately, the third stage is the full bi-directional immune test that takes a further deep dive and catches potentially blood cancers and multi-disease diagnoses."

The card sits at #15 of 15 because it cannot exist in full until everything before it has been tested. Once it does exist in full, every other card depends on it.

## What's next (out of scope for v0.3)

- Stage 3 multi-atlas runs for breast / CRC / lung / HCC / cervical / cardio / pancreatic (sealed VAL data exists, runs not yet executed)
- OQ-2026-01 Test 2 staging build (Salas IDOL-Ext panel-CpG cross-walk to Xu-538)
- Future EpiSCORE bridges (LungRef, KidneyRef, ColonRef, BrainRef, PancreasRef)
- Caggiano TIM Phase B calibration on substrate-matched healthy cohort
- Biobank-gated cohort applications (FitzGerald MCCS, Howard AA EPIC, UCSF AGS, GICC, UK Biobank)
- v1.0 reorganization to symptom-organized decision tree (triggers when all 15 cards have sealed v0.x+ entries)

