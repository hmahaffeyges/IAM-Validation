# Immune Atlas Card v1.0 — Release Notes

**Date:** 2026-06-06
**Card version:** v1.0
**Card type:** universal_baseline_card
**Status:** SKELETON RELEASED — validation evidence (CPG-VAL-015 through CPG-VAL-021) PENDING; full sealing target before June 11 GeoMetric meeting

## What this release contains

`immune-atlas_card_v1_0.json` — 70 KB, 55 top-level keys, 13 stage blocks. The first comprehensive IAMAtlas-native immune card. Replaces the pre-build immune-atlas card v0.3.2 + the pre-build immune card v1.0-draft (both retired to `RETIRED_PREBUILD_REFERENCE/`).

## Major changes from pre-build

### Architectural
1. **Per-stage block architecture** adopted from breast v3.1 (Stage 0 → 10) with explicit module + file references for every chain element
2. **All multi-atlas references replaced** with IAMAtlas REBUILD v0_2 single Bayesian deconvolution architecture
3. **Stage 2 dual deconvolver** (Walther NNLS + NILC needlet) replaces single-atlas single-method approach; cross-method gate mandatory
4. **Stage 3 three explicit foreground axes** (age operational, smoking + sex layers fit but runtime scripts v1.1)
5. **Stage 4.5 bidirectional decomposition** as MANDATORY engine step (per VAL-050/051 doctrine) replaces pooled-only scoring
6. **Stage 4.6 Mollweide + HEALPix** Cosmic Microwave Methylome rendering NEW (no pre-build equivalent)
7. **Stage 5 Mahalanobis** against frozen n=601 HC reference in 115-cell A-score feature space replaces cohort-internal Mahalanobis
8. **Stage 6 cellular age** uses IAMCellularAge β_mean inversion against 80-cell baseline (Recipe §6.3) replaces epigenetic-age regression
9. **Stage 7 6-tier physics-derived** (v1.2) replaces v0 4-tier statistical-percentile system; 1.07 Warburg + 1.10 breach are physics-defined inflection points
10. **Stage 8 three-route architecture** (Route A Mahalanobis + Route B disease matrix + Route C bidirectional)
11. **Report top-of-headline** is immune_age_delta (the inflammaging quantum)
12. **Personal CMM Mollweide PNG** embedded in report

### Doctrinal
1. **10-fingerprint failure-mode heuristic catalog** NOT ported — replaced by measured chain validation (N7 chain-integrity + L9 N1-N8 nulls + SOP CHK-series + Mahalanobis pooled-HC + Stage 4/7 bidirectional flag)
2. **"Stage 1/2/3" pre-build diagnostic-tier language** replaced with "full SOP chain runs Stages 0-10 every time"
3. **"Atlas RUN-everything" sweep doctrine** replaced with single-IAMAtlas chain doctrine
4. **Astro-Genetics framing correction**: framework adapts proven astrophysics/cosmology tools (Planck NILC, HEALPix sphere pixelization, Mollweide projection, virial-theorem-derived Mahaffey number) to methylome architecture

### Preserved verbatim from pre-build
- 19 cell types of interest
- 13 covariate dependencies + full _covariate_notes (1223 chars)
- 9 report_strings (full content per flag state)
- 10 report_vigilance_strings (full content per tier state)
- 19 cell type atlas provenance entries (atlas references will be scrubbed in subsequent commit)
- 20-entry cell-to-page mapping with _doc_note
- 5 grouping_rationale entries (naive_CD4_vs_CD8, B_cells_vs_naive_B_vs_memory_B_vs_plasma, macrophages_vs_microglia_vs_kupffer, etc.)
- Universal alarm vocabulary + bidirectional pattern callout language

## Validation evidence (PENDING)

The card declares 7 PENDING VALs with full deliverables list. As each VAL seals, the card's `validation_evidence_v1_0_set` block updates its status from PENDING to SEALED, with the VAL's headline result added.

| VAL | Cohort | Anchor | Status |
|---|---|---|---|
| CPG-VAL-015 | GSE40279 Hannum n=656 | Pre-build VAL-006 r=0.9999 | PENDING |
| CPG-VAL-016 | Reuse breast + AD + Crohn's | Cross-disease consolidation | PENDING |
| CPG-VAL-017 | Pooled HC ~800 ages 40-90 | Inflammaging literature | PENDING |
| CPG-VAL-018 | GSE51057 HRT field | First-of-its-kind | PENDING |
| CPG-VAL-019 | Reuse breast + AD | VAL-050/051 bidirectional | PENDING |
| CPG-VAL-020 | GSE40279 Hannum | Pre-build VAL-006 reproduction | PENDING (Heath priority for meeting) |
| CPG-VAL-021 | GSE61450 bariatric paired | Weight-loss inflammaging | PENDING (Dr. Escobedo) |

## Outstanding work (12 items)

See `card.outstanding_work_v1_0` for the full list. Highest priority:
1. Run CPG-VAL-015 through CPG-VAL-021 with proper chain modules (replace last session's off-scope exploration)
2. Build immune_atlas_residual_map_chr_annotated.csv + pca_projections.csv + bimodality_map.csv during VAL sealing
3. Scrub the 19 per-cell pages for IAMAtlas-only references
4. Update DISEASE_MATRIX v1_7 → v1_8 with immune card v1.0 rows
5. Build GeoMetric demo report HTML (single-page, four-provider sections) for June 11 meeting

## Honest limitations (10 items)

See `card.honest_limitations` for the full list. Key v1.0 caveats:
- Smoking + sex foreground subtraction at Stage 3 β-level NOT yet built (v1.1) — mitigated via Stage 7 threshold stratification
- L5 / L7 / L8 (correlation structure / Bayesian likelihood / per-card MCMC posterior) EMPTY — deferred to later phases
- Per-card immune residual map not yet built (builds during VAL sealing)
- v1.0 tier thresholds physics-derived but final cohort-specific tuning waits for VAL-015 + VAL-017 sealed pooled-HC distribution
- 19 per-cell pages contain pre-build atlas references that need scrubbing

## Chain-of-custody anchors

- IAMAtlas canonical SHA-256: `41b7c16f043bce96e085a2b8b4e709efd2b862af9de8dbe9a8646e9fb94c32ee`
- Celltype marker artifact SHA-256: `46ea5be1db377f2b8773a02418a7f481a191630e0fa833d3294eab1fd19c47bd`
- VAL-051 directional panel SHA-256 anchor: `52061285fc97bfff871ba7b62f625b14d953bccf25ee24e35f328e15b9827998`
- BUILD_SPEC reference: `walther_clinical_BUILD_SPEC_v1_2.md`
- SOP reference: `CPG_Chain_of_Custody_SOP_v1_3.md`

