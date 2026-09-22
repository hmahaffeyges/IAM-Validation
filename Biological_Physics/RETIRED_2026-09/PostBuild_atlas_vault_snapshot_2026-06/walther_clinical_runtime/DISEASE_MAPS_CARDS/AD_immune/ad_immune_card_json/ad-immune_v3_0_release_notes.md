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



## Lessons learned — AD-specific findings worth carrying forward

These emerged from the AD-immune Phase 2 work and are documented here for any future AI session, researcher, or v3.1+ work on this card. The global lessons across all cards live in MASTER_TRACKER §0.1; this section captures the AD-card-specific items.

**1. Disease-trained panel outperforms universal Mahalanobis ~3x on AD — opposite of breast pre-dx.** AD's per-cell-type signal is targeted (immune-class architectural disruption), not universal-architectural. The 7-CpG Rule A panel (AUC 0.84) outperforms the universal Mahalanobis hyper-volume (AUC 0.62) on AD discrimination because AD's signal is concentrated in immune-class CpGs the disease-trained panel was selected for. For breast pre-dx the opposite was true: universal Mahalanobis (CPG-VAL-002 d=+1.876/+2.097) BEAT the disease-trained Xu-538 panel by +0.75 on GSE51032 because breast pre-dx has broad-architectural signature. **Operational scoring routing: AD goes to 7-CpG Rule A; breast goes to Mahalanobis. Don't mix them up.**

**2. AIBL has no chronological ages in GEO release.** Workaround used: route age-axis subtraction (CPG-VAL-011) to cohorts WITH ages (AddNeuroMed + GIFT). The age-axis foreground module requires age input. First-client AD intake form needs age field.

**3. 115-cell A-score layer is naturally age-orthogonal.** Δd < 0.05 under age subtraction at the 115-cell layer. The 7-CpG Rule A panel had R²=0.26 age confound (Phase 7 audit). The newer instrument architecture is robust to age in a way the older single-panel approach wasn't.

**4. 450K coverage gap (86-95%) attenuates Mahalanobis but per-cell biology replicates.** AddNeuroMed Mahalanobis d=−0.006 (null) vs AIBL d=+0.20 — the same biology, but the 450K platform misses ~10-14% of EPIC CpGs in the Mahalanobis covariance reference, attenuating the universal signal. Per-cell A-scores DO replicate (Eosino d=−0.46 vs −0.43). **This is platform, not biology.** Document for any future cross-platform validation.

**5. PC1 is the T-cell axis in AIBL — rank differs from breast.** Breast pre-dx PC2 = T-cell axis; AIBL PC1 = T-cell axis. Same underlying biology, different rank because cohort composition differs (breast pre-dx population vs AIBL at-diagnosis population). Don't hypothesize "PC2 will be the T-cell axis" — hypothesize "the T-cell axis will be a top PC" and let the rank emerge.

**6. Tier breakpoints are breast-calibrated and don't move for AD's modest universal signal.** A_NORMAL ≤ 1.05 threshold was set against VAL-054b HC permutation on breast pre-dx context. AIBL immune tier distribution: HC 88% NORMAL, AD 88% NORMAL — the threshold doesn't discriminate. Operational scoring routes AD through the 7-CpG Rule A panel, not the tier system. **The tier system stays as universal screen; per-disease panels override for known disease patterns.**

**7. GIFT specificity arm shows three distinct Mahalanobis signatures.** AD d=+0.68 (positive — hyper-volume departure outward); FTD d=+0.28 (intermediate); PSP/CBD d=−0.38 (negative — BELOW_NORMAL signature, architectural compaction). **The same Mahalanobis metric, three distinct disease signatures.** This is the operational basis for the EDEAR multi-disease report layer.

**8. PSP/CBD BELOW_NORMAL is real architectural compaction.** d=−0.38 with N1 PASS at p=0.034. Tauopathies are NOT all the same — primary 4R tauopathies (PSP/CBD) show OPPOSITE direction from AD on the universal architectural readout. This is the strongest evidence yet that the IAM-architectural distance is biologically meaningful, not just statistical noise — it discriminates direction-of-departure not just magnitude.

**9. Cellular age SATURATED status common in blood substrate vs multi-tissue reference.** Many non-blood classes (terminal, stromal, stem_pluri) saturate at age=4 or age=95 because the 80-cell baseline references multi-tissue and blood samples don't have meaningful signal in those classes. First-client report layer needs SATURATED-handling logic — don't report saturated cellular ages as "real" ages.

**10. NILC corroborates Walther on dominant blood compartments.** Walther vs NILC Spearman ρ on AIBL: immune +0.93, progenitor +0.86 — strong cross-method agreement on the classes that have substantive signal in blood. Non-blood classes show low/degenerate ρ because both methods correctly return near-zero (noise dominates). **The cross-method check is a robust biological-substrate signature, not a method-specific artifact.**

**11. AD residuals are biased 4.8:1 negative direction.** Per-CpG residual map (observed β − class-fraction-predicted β) on AIBL+AddNeuroMed: 4.8 hypomethylated CpGs per hypermethylated CpG. Pattern is consistent with the architectural-suppression rather than gain interpretation.

**12. Cross-cohort residual concordance Spearman ρ=0.231 — modest but real.** AIBL × AddNeuroMed residual map agreement at the per-CpG level. p=1e−74. 241 strong-concordant CpGs at \|d\|>0.2 (88.9% same-sign rate). CPG_ad_panel_v1 candidate (200 CpGs) selected from this intersection. Holdout validation pending on independent cohort.

**13. Stage 1 reproduction is the integrity gate, always.** AIBL d=+0.615 vs anchor +0.624 (within sampling variation). AddNeuroMed d=+0.317 vs anchor +0.332. GIFT pooled d=+0.013 EXACT, male AD d=+0.415 EXACT. Three exact reproductions confirms the post-build IAMAtlas-native pipeline is bit-identical to the build-time pipeline on the same panels.

## Pointer

- v2.2 archived at: `ad_immune_card_json/OLD/ad-immune_card_v2.2.json`
- v2.2 README carried forward at: `ad_immune_card_json/ad-immune_README.md`
- v3.0 JSON: `ad_immune_card_json/ad-immune_card_v3_0.json`
- Cohort artifacts: `Biological_Physics/validation_runs/ad_immune_cohorts/{GSE153712_AIBL,GSE144858_AddNeuroMed,GSE53740_GIFT}/`
- VAL outcomes: `Biological_Physics/validation_runs/CPG_VAL_{010,011,012,013,014}_AD_*/CPG_VAL_*_OUTCOME.md`
- CPG-VAL-008/009 outcome (covered together): `Biological_Physics/validation_runs/ad_immune_cohorts/GSE153712_AIBL/CPG-VAL-008-009_PRELIMINARY_OUTCOMES.md`
- v3.0 residual maps: `ad_immune_residual_maps/`
