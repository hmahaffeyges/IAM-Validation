# EDEAR card library

This directory holds the publication-ready disease cards for the EDEAR
pipeline. Each card lives in its own subdirectory and is fully
self-contained per the v2.1 design rule:

> A new analyst loading ONLY `<disease>-card_v0.X.json` and the
> production GAPE engine can run the full Stage 1 + Stage 2 + Stage 3
> pipeline for that disease, without reading any other file.

## Card list (per Heath's authoritative 2026-05-08 catalog)

| # | Card | Subdirectory | Status |
|---|---|---|---|
| 1 | AD | `ad/` | JSON in production |
| 2 | Bladder | `bladder/` | v0.1 sealed 2026-05-01 |
| 3 | Breast | `breast/` | JSON in production |
| 4 | Cardio | `cardio/` | v0.1 sealed but DEFERRED (Konigsberg/Cuadrat 2023 atlas pending) |
| 5 | Cervical | `cervical/` | JSON in production |
| 6 | Colon/Rectal | `crc/` | JSON in production |
| 7 | Gastric/Esophagus | `gastric-esophageal/` | v0.1 sealed 2026-05-02 |
| 8 | Glioma | `glioma/` | v0.2 single_cohort_validated |
| 9 | HCC Liver | `hcc/` | multi_modal_validated (Marcus card) |
| 10 | HEME Leukemia | `heme/` | v0.1 single_cohort_validated (myeloid arm) |
| 11 | Lung | `lung/` | v0.2 multi_modal_validated |
| 12 | Pancreatic | `pancreatic/` | v0.1 cohort_screening_validated |
| 13 | Prostate | `prostate/` | v0.3 multi_modal_validated_plus_multi_atlas_calibrated |
| 14 | Immune | `immune/` | Routing card (systemic-departure differential) |
| 15 | PSP | `psp/` | TO BUILD — needs JSON + more work |
| 16 | Kidney | `kidney/` | TO BUILD — needs JSON + more work |
| 17 | Schizophrenia | `schizophrenia/` | CANDIDATE — landscape survey pending |

## What lives in each card subdirectory

Per the canonical per-card workflow (TESTING_CHECKLIST.md, README_MASTER):

- `README.md` — the card README (Block 1-20 per master spec)
- `<card>-card_v0.X.json` — the card JSON with full universal_reference block
- `prereg/` — sealed pre-registration files (one per VAL)
- `outcomes/` — outcome.md files (one per VAL)
- `results/` — results JSONs and stratified results JSONs
- `manifests/` — cohort manifests and clinical metadata
- `evidence_report.html` — the public-facing evidence report

VAL python scripts and per-sample CSVs live in `Biological_Physics/validation_runs/<VAL-XXX>/`
in the main repo (not duplicated here).

## What does NOT live in this directory

- The GAPE engine (proprietary; Heath-only IP)
- IAM physics derivations (proprietary; Heath-only IP)
- Anything that derives H_min from first principles
- Anything naming Landauer / Jacobson / decoherence / Mahaffey Number in scoring code or comments

H_min values appear in card JSONs as anchor values without derivation
context. Card READMEs describe scoring in operational terms (Shannon
entropy against an architectural-class anchor) without explaining why
the anchor takes the value it does. The reference numbers are
publishable; the recipe that produces them is not.

## Cards are added one at a time

Each card sprint follows the canonical seven-internal-files +
eight-external-artifacts workflow. New cards land in this directory
only after Heath's explicit go-ahead at the per-card finalize review
(CHK-5.4).
