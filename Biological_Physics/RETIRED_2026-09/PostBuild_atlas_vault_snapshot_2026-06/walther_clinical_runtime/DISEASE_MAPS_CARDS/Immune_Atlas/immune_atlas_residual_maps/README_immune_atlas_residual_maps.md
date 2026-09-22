# Immune-Atlas Residual Maps

**Card:** immune-atlas v2.1 (2026-06-07)
**README date:** 2026-06-07

This folder contains the operational artifacts of the immune-atlas card v2.1 Stage 8 Route A residual-map-overlap channel. Unlike the breast-EPIC and AD-immune residual maps (which are derived directly from cohort data), these maps are **derived from the post-build sealed residual maps of the disease cards** — the immune-atlas card is the universal baseline that integrates per-CpG signatures across diseases. The 4-file structure mirrors the disease-card residual map convention.

## Files in this folder

| File | What it is | Stage 8 use | Anchor VAL |
|---|---|---|---|
| `immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.csv` | 6,018 CpGs at intersection of breast-EPIC + AD-immune residual maps with cross-disease firing-pattern classification (5 mutually exclusive buckets) | Stage 8 Route A residual-map-overlap channel with two sub-channels: cross_disease_concordance + bidirectional_universal_alarm | Inherits CPG-VAL-003 (breast) + CPG-VAL-013 (AD); operationalizes CPG-VAL-016 + CPG-VAL-019 at per-CpG resolution |
| `immune_atlas_cross_disease_universal_alarm_bimodality_map.csv` | Per-CpG bimodality data from BOTH source cohorts side-by-side (24 columns) + cross-disease bimodality tags (double_disease_gain / double_disease_loss / opposing_bimodality_pattern) | Stage 8 Route A supporting evidence — bimodality changes in same direction across diseases are a complementary signature to the Cohen's d residual map | Inherits CPG-VAL-004 (breast, RESTATED) + AD bimodality derived from CPG-VAL-013 cohort |
| `immune_atlas_cross_disease_universal_alarm_pca_projections.csv` | Per-sample PCA projections from BOTH source cohorts combined into a single file with cohort/disease/PCA-basis tags (n=1,373: 647 breast + 726 AD) | Stage 8 Route A supporting evidence — cross-cohort PC axis comparison (breast PC2 = T-cell axis, AD PC1 = T-cell axis — same biology, different rank) | Inherits CPG-VAL-005 (breast PCA) + CPG-VAL-012 (AD PCA) |
| `README_immune_atlas_residual_maps.md` | This file | Documentation | — |

## Headline numbers

### Residual map (v0.1)
- **6,018 CpGs** at inner-join of breast-EPIC + AD-immune residual maps (CHR/MAPINFO consistency verified: 0 mismatches)
- Firing-pattern distribution:
  - `fires_neither_background`: 4,641 CpGs (77.12%)
  - `fires_breast_only`: 1,136 CpGs (18.88%)
  - `fires_AD_only`: 212 CpGs (3.52%)
  - `fires_in_both_diseases_same_direction`: 17 CpGs (0.28%) — cross-disease concordance
  - `fires_in_both_diseases_opposing_direction`: **12 CpGs (0.20%) — the VAL-016 bidirectional universal alarm signature**
- Schema: `cpg, CHR, MAPINFO, breast_d_GSE51057, breast_d_GSE51032, breast_d_mean, breast_sign, breast_mean_abs_d, breast_concordant_strong, ad_d_AIBL, ad_d_AddNeuroMed, ad_d_mean, ad_sign, ad_mean_abs_d, ad_concordant_strong, cross_disease_firing_pattern, cross_disease_mean_abs_d`

### Bimodality intersection map (v0.1)
- **6,018 CpGs** at intersection of breast + AD bimodality maps
- **1,592 CpGs gain bimodality in BOTH diseases** (`double_disease_gain_of_bimodality`) — broad cross-disease chromatin-state divergence pattern; both breast pre-dx and AD shift toward gain
- **26 CpGs lose bimodality in BOTH diseases** (`double_disease_loss_of_bimodality`) — narrow cross-disease lost-bimodality pattern
- **850 CpGs show opposing bimodality direction** (`opposing_bimodality_pattern`) — gain in one disease + loss in the other; the bimodality analog of the residual map's opposing-direction firing
- Schema: per-disease bimodality (bc_hc, bc_case, delta_bc, mean_beta_hc, mean_beta_case, sd_beta_hc, sd_beta_case, delta_var, bimodal_in_hc, lost_in_case, loss_of_bimodality, in_residual_concordant) for both breast and AD, plus 3 cross-disease bimodality tags + residual_map_firing_pattern + CHR/MAPINFO

### PCA projections combined (v0.1)
- **1,373 samples** total (647 breast EPIC-Italy + 726 AD AIBL)
- Each sample tagged with `source_cohort`, `source_disease_context`, `pca_basis` (which cohort's PCs were used)
- Breast PCs: fit on EPIC-Italy GSE51057 + GSE51032 HC (PC2 = T-cell suppression axis, d=-0.67/-0.58 cases vs HC)
- AD PCs: fit on AIBL n=471 HC (PC1 = T-cell axis, d=-0.356 cases vs HC; PC3 = secondary axis, d=+0.22)
- **Cross-cohort observation:** the T-cell axis appears as PC2 in breast and PC1 in AD — same biology, different rank, driven by cohort age distribution and case prevalence. The carryforward hypothesis is "T-cell axis will be a top PC" (not "PC2 will be the T-cell axis").

## Relationship to the VAL-051 7-CpG directional panel (Stage 4.5)

The cross-disease universal alarm residual map's 12 opposing-direction CpGs and the VAL-051 7-CpG directional panel are **disjoint instruments by design — zero CpG overlap.** This is a complementarity, not a redundancy:

| Instrument | Stage | CpG universe | Selection method | Operational use |
|---|---|---|---|---|
| VAL-051 7-CpG directional panel | Stage 4.5 bidirectional decomposition | AIBL-only training data | Rule A criterion (\|Δβ\| > 0.015 AND q_FDR < 0.10) on AIBL training split, pre-build methodology | High-precision within-AD directional discriminator (AUC=0.84 AIBL holdout) |
| Cross-disease universal alarm v0_1 (12-CpG opposing subset) | Stage 8 Route A residual-map-overlap channel | Post-build VAL-003 + VAL-013 intersection (6,018 CpGs) | Disease-internal concordance (both cohorts strong + same direction) + opposing-direction filter across diseases | Broader cross-disease Stage 8 Route A overlap channel |

The two operate at different scales — VAL-051 is the high-precision within-disease alarm; v0_1 is the broader cross-disease alarm. A future v0_2 may merge both CpG universes into a unified bidirectional instrument.

## How Stage 8 consumes these maps

Per SOP v1.3 Stage 8 Route A v2.1 (residual-map-overlap channel):

1. Patient's foreground-cleaned β matrix from Stage 3 is the input
2. Pearson ρ between patient's per-CpG departure and the cross-disease residual map's signed Cohen's d is computed, **separately for two sub-channels:**
   - `cross_disease_concordance_channel`: ρ on the 17 `fires_in_both_diseases_same_direction` CpGs
   - `bidirectional_universal_alarm_channel`: ρ on the 12 `fires_in_both_diseases_opposing_direction` CpGs
3. Fisher z-transform 95% CI on ρ for each sub-channel
4. Per-disease d columns (breast_d_mean, ad_d_mean) are AVERAGED when applying — the operational residual signature is the cross-disease consensus on signed magnitude
5. Bimodality map overlap is computed in parallel as supporting evidence
6. Route A fires when EITHER the Mahalanobis channel OR either sub-channel of the residual-map-overlap channel triggers (rho ≥ 0.10, CI lower bound > 0)

The card JSON `immune-atlas_card_v2_1.json` declares matching rules in `stage_8_card_matching.route_A_universal_architectural`.

## Coverage requirements

Per the card's substrate spec, 450K samples require ≥80% CpG coverage of the EPIC superset. AIBL is HM450 native (full source coverage); AddNeuroMed is HM450 with EPIC-CpG superset attenuation documented (CPG-VAL-052 cross-platform d=+0.33 EPIC vs d=+0.624 HM450). EPIC-Italy GSE51057 + GSE51032 are HM450 (anchor platform per breast card v3.1).

## v0_1 lineage caveats

These maps are **derived vault artifacts**, not independently sealed VALs. They inherit the validation status of their source artifacts:
- CPG-VAL-003 (breast cross-cohort residual concordance) — SEALED PASS
- CPG-VAL-013 (AD cross-cohort residual map) — SEALED
- CPG-VAL-004 (breast bimodality — RESTATED to gain-dominant 2.77:1) — SEALED with restatement
- CPG-VAL-005 (breast PCA — PC2 T-cell axis) — SEALED PASS
- CPG-VAL-012 (AD PCA — PC1 T-cell axis) — SEALED PASS

The cross-disease firing-pattern classification, bimodality intersection tags, and PCA combination do NOT introduce new validation claims beyond the source artifacts. v0_2 may include an independent sealed VAL.

## What's pending in v0_2

- Extend to additional disease residual maps as they become available (CRC pre-dx, lung pre-dx, autoimmune classes)
- Optional merge with VAL-051 7-CpG panel CpG universe to create a unified bidirectional instrument
- Joint cross-disease PCA on the 6,018 CpG universe (requires per-sample β matrix access for both cohorts)
- Gene annotation pass on the 12 opposing-direction CpGs (manifest gene/island annotation)
- Stage 8 Route A residual-map-overlap channel threshold calibration on first patient cohort runs through the immune-atlas v2.1 chain
- Independent sealed VAL for the cross-disease universal alarm signature with N1-N8 null suite

---

**Lineage.** These maps were generated 2026-06-07 by inner-joining the sealed post-build residual map artifacts of the breast-EPIC and AD-immune disease cards. They are NOT derived from any pre-build atlas. The 12-CpG opposing-direction subset is the per-CpG operationalization of the VAL-016 cross-disease universal alarm signature (which was demonstrated at directional class level). Pre-build evidence trail for the immune-atlas card lineage is in `immune-atlas_card_v2_1.json` under `pre_build_audit_lineage`.
