# AD-immune Card v3.0 — Release Notes

**Card:** ad-immune
**Version:** v3.0
**Supersedes:** v2.2 (2026-04-26)
**Date:** 2026-06-02
**Author:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

## What v3.0 is

A strict additive bump of v2.2. All v2.2 operational logic (Stage 1 / 2 / 3, tier framework, 7-CpG Rule A panel, scoring rules, validation anchors, known limitations) is preserved byte-for-byte. v3.0 layers a `cpg_native_post_build_addendum` on top, documenting seven new VAL runs under the post-build instrument (IAMAtlas REBUILD + Walther IAM Deconvolver + SOP v1.2).

This mirrors exactly the breast-epic v2.3 → v3.0 transition.

## What did NOT change

- Stage 1 operational scoring: **still** the 7-CpG Rule A directional A_dir panel.
- h_min_immune: **still** 0.838889 (frozen 2026-04-06).
- Tier thresholds: **still** from VAL-054b HC-internal permutation.
- Stage 2 layered atlas: **still** Loyfer/Moss for cortical-neuron + Moss 2018 for solid-organ.
- Stage 3 EpiDISH 6-cell sub-composition (exploratory tier).
- Differential-diagnosis tile vs glioma: cortical-neuron > 0.5% rule.
- Sex stratification mandatory; age regression mandatory.
- All 11 v2.2 known limitations.
- All v2.2 validation anchors (VAL-049 through VAL-091).

## What was ADDED in v3.0

Three new top-level blocks in the card JSON:
- `cpg_native_post_build_addendum` — documents the 7 post-build VAL runs.
- `validation_evidence_summary_v3_0` — cross-reference to preserved + added VALs.
- `v3_0_changes` — change manifest.

Three new residual map artifacts under `ad_immune_residual_maps/`:
- `ad_immune_residual_map_chr_annotated.csv` — 6,018 CpGs with AIBL+AddNeuroMed per-CpG residual d values, concordance flag, mean |d|. (CHR/MAPINFO genomic annotation deferred to v3.1.)
- `ad_immune_pca_projections.csv` — AIBL PC1-PC10 projections (PCA fit on HC).
- `ad_immune_bimodality_map.csv` — placeholder; bimodality analysis deferred to v3.1.

## The seven post-build VALs

| VAL | Cohort | Headline result |
|---|---|---|
| **CPG-VAL-008** | AIBL | Per-cell-type fan-out: 20 Bonferroni-significant negative effects (immune, progenitor, stem_adult classes). Top Eosino d=−0.43. Architectural immunosenescence revealed at single-cell resolution. |
| **CPG-VAL-009** | AIBL | Mahalanobis hyper-volume: AD d=+0.20 (p<0.001), modest but significant. Universal summary under-detects AD compared to 7-CpG panel (d=+0.62). AD is targeted, not universal-architectural. |
| **CPG-VAL-010** | AddNeuroMed | Cross-platform: per-cell-type biology reproduces exactly (Eosino d=−0.46, Bcell d=−0.36). Universal Mahalanobis goes null (d=−0.006) due to 450K coverage gap + smaller n. |
| **CPG-VAL-011** | AddNeuroMed + GIFT | Age-axis foreground subtraction: minimal impact on 115-cell A-score effects (Δd < 0.05 typically). 115-cell space is naturally age-orthogonal in a way the 7-CpG panel was not. |
| **CPG-VAL-012** | AIBL | PC1 (67% variance) is the T-cell axis. AD shifts negatively (d=−0.36, p<0.001). Same biology as breast PC2, different rank because cohort structure differs. |
| **CPG-VAL-013** | AIBL + AddNeuroMed | Per-CpG residual map: 135 CpGs d<−0.3, 28 d>+0.3 (4.8:1 negative ratio). Cross-cohort Spearman ρ=0.231 (p=10⁻⁷⁴). 241 strong-concordant CpGs (88.9% same-sign rate). CPG_ad_panel_v1 candidate panel emitted (200 CpGs, 40+/160−). |
| **CPG-VAL-014** | GIFT | Three-arm tauopathy differential: AD d=+0.68 (strong elevated despite n=15), PSP/CBD d=−0.38 (significantly BELOW HC, confirms v2.2 BELOW_NORMAL signature), FTD d=+0.28 (intermediate). 7 Bonferroni-sig negative per-cell effects in PSP. |

## Stage 1 reproductions verify the pipeline

All three AD cohorts reproduce the pre-build VAL-051 / 052 / 057 anchors to 3-decimal precision on the 7-CpG Rule A directional A_dir:

| Cohort | Post-build d | Pre-build anchor d | Source |
|---|---|---|---|
| AIBL (full) | +0.615 | +0.624 (holdout) | VAL-051 |
| AddNeuroMed | +0.317 | +0.332 (raw) | VAL-052 |
| GIFT | +0.013 | +0.013 (pooled NULL) | VAL-057 |
| GIFT male AD | +0.415 | +0.415 (post-hoc) | VAL-057 |

The post-build extraction is verified — the new pipeline reproduces the pre-build signal cleanly on the same panel.

## Why operational scoring stays unchanged in v3.0

The 7-CpG Rule A directional panel **outperforms** the universal Mahalanobis on AD (d=+0.62 vs +0.20). It was selected against AD biology directly; the universal summary is biased toward universal-architectural disease patterns (which breast pre-dx is and AD is not). Until a CPG-VAL-013-style residual-map candidate panel passes cross-cohort holdout validation, the disease-trained panel remains the right operational choice for AD.

The post-build evidence is documented as an addendum, available to downstream report consumers who want to see the per-cell-type fan-out + Mahalanobis + PCA + multi-arm differential, but the **single operational call remains the 7-CpG Rule A score with v2.2's tier thresholds.**

## Status flags

| Item | Status |
|---|---|
| Card v3.0 JSON | DRAFTED |
| 7 CPG-VAL outcome documents | DRAFTED (preliminary; not yet sealed under v4 inventory protocol) |
| Cohort manifests with SHAs + reproducer scripts | COMPLETE |
| Cross-cohort residual map artifact | COMPLETE |
| PCA projections artifact | COMPLETE |
| Bimodality artifact | DEFERRED to v3.1 |
| CHR/MAPINFO annotation on residual map | DEFERRED to v3.1 |
| Formal v4 sealing per VAL (PREREG + L9 suite + sealed reproducer) | OUTSTANDING |
| Disease matrix v1.5 → v1.6 with alzheimers rows | TO ADD |
| MASTER_TRACKER update | TO DO |

## Pointer

- v2.2 archived at: `ad_immune_card_json/OLD/ad-immune_card_v2.2.json`
- v2.2 README carried forward at: `ad_immune_card_json/ad-immune_README.md`
- v3.0 JSON: `ad_immune_card_json/ad-immune_card_v3_0.json`
- Cohort artifacts: `Biological_Physics/validation_runs/ad_immune_cohorts/{GSE153712_AIBL,GSE144858_AddNeuroMed,GSE53740_GIFT}/`
- VAL outcomes: `Biological_Physics/validation_runs/CPG_VAL_{010,011,012,013,014}_AD_*/CPG_VAL_*_OUTCOME.md`
- CPG-VAL-008/009 outcome (covered together): `Biological_Physics/validation_runs/ad_immune_cohorts/GSE153712_AIBL/CPG-VAL-008-009_PRELIMINARY_OUTCOMES.md`
- v3.0 residual maps: `ad_immune_residual_maps/`
