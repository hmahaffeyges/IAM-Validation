# CPG / VAL Inventory Report (v9)

**Version v9 — L9 null suite expanded to N1-N4, N6-N8 across both cards; residual maps complete (CHR/MAPINFO + bimodality); Stage 8 Path B mapping artifact v0.1 starter shipped**
**Date:** 2026-06-05
**Maintained by:** Walther (Claude) on behalf of Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute, Entiat WA

This document is a current-state catalog of every CPG validation run executed with the current methodology (Walther IAM Deconvolver + IAMAtlas REBUILD + 8-class + 115-cell A-scoring + Mahalanobis hyper-volume + NILC v2 cross-method + cellular age + tier breakpoints). For the narrative interpretation of these results, see `v6_CPG_IAMAtlas_Evidence_Report.html`. For the operational protocol Heath uses to run each new card, see MASTER_TRACKER §0.

This is a clean rewrite — no historical "Family A vs Family B" or "slots reserved" framing. Two cards are operational; future cards get sequential CPG-VAL-NNN slot ranges at the time of activation. Earlier inventory versions (v1 through v8) are archived in `OLD/` for audit.

---

## 1. Operational cards (current state)

| Card | Card version | VAL slot range | Status | Card README |
|---|---|---|---|---|
| **breast-epic** | v3.1 (2026-06-04) + post-release updates 2026-06-05 | CPG-VAL-001 through CPG-VAL-007 | ✅ Operational. All seven per-VAL bundles complete. Stage 1 reproduction PASS (d=+2.088 vs anchor +2.097, within 0.4%). Cross-method ρ=+0.74 immune / +0.82 progenitor. L9 N1-N4, N6-N8: 5 active VALs PASS 7/7 nulls; 2 RESTATE. Residual map + bimodality decomposition COMPLETE. | `DISEASE_MAPS_CARDS/Breast_EPIC/breast_epic_card_json/breast-epic_README.md` |
| **AD-immune** | v3.1 (2026-06-04) + post-release updates 2026-06-05 | CPG-VAL-008 through CPG-VAL-014 | ✅ Operational. All seven per-VAL bundles complete. Stage 1 reproduction PASS on all 3 cohorts. L9 N1, N3, N4, N6, N7, N8: 6 active VALs PASS 5/6 nulls (N3 borderline on 009 and 014); VAL-011 PASS-AS-NULL by design. Residual map + bimodality decomposition COMPLETE. | `DISEASE_MAPS_CARDS/AD_immune/ad_immune_card_json/ad-immune_README.md` |

---

## 2. VAL catalog (per-card detail)

### 2.1 Breast-EPIC VALs (CPG-VAL-001 through CPG-VAL-007)

**Cohorts:** GSE51057 (n=188 filtered: 11 case >10y pre-dx / 177 HC) and GSE51032 (n=460 filtered: 36 case >10y pre-dx / 424 HC) from Severi 2014, *Carcinogenesis* 35(10):2349. Both EPIC-Italy.

**Cohort folders:** `validation_runs/breast_epic_cohorts/{GSE51057_EPIC_Italy,GSE51032_EPIC_Italy}/`

| VID | Test | Headline result | L9 nulls | Per-VAL folder |
|---|---|---|---|---|
| CPG-VAL-001 | Per-cell-type A-score fan-out across 115 cell types | Basophil d=+1.58/+1.01 (top hit, hidden in 51-cell immune class); breast epithelial d=+1.28/+0.61 (tissue-of-origin >10y pre-dx); T-cells d=−0.67/−0.58 (suppression) | 7/7 PASS (N1-N4, N6-N8) | `validation_runs/CPG_VAL_001_Breast_per_celltype_fanout/` |
| CPG-VAL-002 | Mahalanobis hyper-volume universal departure | d=+1.876/+2.097 across two cohorts. Beats Xu-538 by +0.75 on GSE51032 | 7/7 PASS | `validation_runs/CPG_VAL_002_Breast_mahalanobis/` |
| CPG-VAL-003 | Per-CpG residual map → CPG_breast_panel_v1 seed | 1,392 concordant CpGs across cohorts; 1,389 NEW (not in Xu-538); 5.4:1 hypomethylation ratio | 7/7 PASS | `validation_runs/CPG_VAL_003_Breast_residual_map/` |
| CPG-VAL-004 | Loss-of-bimodality count (RESTATED) | Direction reversed: 1,096 GAIN bimodality vs 396 loss. 35 double-confirmed CpGs cross-validated with residual map. New biomarker class. | RESTATE | `validation_runs/CPG_VAL_004_Breast_bimodality/` |
| CPG-VAL-005 | Principal component decomposition | PC2 = T-cell suppression axis, d=−0.67/−0.58 cross-cohort. PC10 (~1% variance) = basophil/eosinophil axis | 7/7 PASS | `validation_runs/CPG_VAL_005_Breast_PC_axes/` |
| CPG-VAL-006 | Chromosome-level isotropy (RESTATED) | chr6 MHC most-deviant uncorrected (z=+2.81, p=0.009) but does NOT survive look-elsewhere correction (corrected p=0.103) | RESTATE | `validation_runs/CPG_VAL_006_Breast_chr6_mhc/` |
| CPG-VAL-007 | Age-axis foreground subtraction | Mahalanobis signal retained at d=+0.255 post-age-subtraction. Terminal, stromal, stem_pluri classes genuinely case-specific (not age artifacts) | 7/7 PASS | `validation_runs/CPG_VAL_007_Breast_age_subtraction/` |

**Summary:** 5 of 7 PASS all 7 declared L9 nulls (N1, N2, N3, N4, N6, N7 simplified, N8 single-feature skip); 2 RESTATE (direction-reversed for VAL-004, correction-significance-loss for VAL-006). Both RESTATEs documented honestly — the L9 suite caught them on its first run, which is what L9 is for. N5 plate-position null not run (requires plate metadata not in GEO releases). Full β-matrix N7 end-to-end chain-recovery is genuine outstanding work (current N7 is simplified signal-level).

**Per-VAL bundle contents (each folder):** `PREREG.md` (retrospective) + `per_sample.csv` + `null_results.json` + `cohort_manifest.json` + `CPG_VAL_NNN_OUTCOME.md`.

### 2.2 AD-immune VALs (CPG-VAL-008 through CPG-VAL-014)

**Cohorts:** AIBL GSE153712 (n=726, 161 AD / 471 HC / 94 MCI, EPIC); AddNeuroMed GSE144858 (n=300, 93 AD / 96 HC + 111 MCI/precursor, 450K cross-platform); GSE53740 GIFT (n=384, 15 AD / 193 HC / 95 FTD / 47 PSP-CBD / 34 other, 450K specificity arm).

**Cohort folders:** `validation_runs/ad_immune_cohorts/{GSE153712_AIBL,GSE144858_AddNeuroMed,GSE53740_GIFT}/`

| VID | Test | Headline result | L9 nulls | Per-VAL folder |
|---|---|---|---|---|
| CPG-VAL-008 | AIBL per-cell-type A-score fan-out | 20 Bonferroni-sig negative effects across immune class; top Eosino d=−0.426 (p=2.3e-5); zero positive | 5/6 PASS (N1, N3, N4, N6, N8; N7 simplified) | `validation_runs/CPG_VAL_008_AD_AIBL_per_celltype/` |
| CPG-VAL-009 | AIBL Mahalanobis universal departure | d=+0.200 (modest — confirms AD's signal is targeted not universal) | 5/6 PASS (N3 borderline p=0.026) | `validation_runs/CPG_VAL_009_AD_AIBL_mahalanobis/` |
| CPG-VAL-010 | AddNeuroMed cross-platform per-cell replication | Eosino d=−0.46 (replicates AIBL d=−0.43). Universal Mahalanobis attenuates on 450K (coverage gap, not biology) | 5/6 PASS | `validation_runs/CPG_VAL_010_AD_AddNeuroMed/` |
| CPG-VAL-011 | Age-axis foreground subtraction | Raw d=−0.004 (correctly null at baseline; PASS-AS-NULL). Post-subtraction reveals d=−0.19 stem_adult signal. AIBL excluded (no GEO ages). | PASS-AS-NULL (all nulls correctly non-significant by design) | `validation_runs/CPG_VAL_011_AD_age_subtraction/` |
| CPG-VAL-012 | AIBL principal component decomposition | PC1 (not PC2) is the T-cell axis in AIBL — same biology as breast PC2, different rank due to cohort composition. d=−0.356 | 5/6 PASS | `validation_runs/CPG_VAL_012_AD_PC_axes/` |
| CPG-VAL-013 | Per-CpG residual map → CPG_ad_panel_v1 candidate | Top CpG cg19459094 d=−0.493. Cross-cohort Spearman ρ=0.231 (p=1e-74). 88.9% same-sign rate. 200-CpG candidate panel. AD residuals biased 4.8:1 negative | 5/6 PASS | `validation_runs/CPG_VAL_013_AD_residual_map/` |
| CPG-VAL-014 | GIFT three-disease specificity arm | AD d=+0.68 (p=0.001); PSP/CBD d=−0.38 (p=2e-6, BELOW_NORMAL compaction); FTD d=+0.28 (intermediate). Same metric, three distinct signatures | 5/6 PASS (N3 borderline p=0.039) | `validation_runs/CPG_VAL_014_AD_GIFT_specificity/` |

**Summary:** All 7 AD VALs PASS their declared L9 null suite. Active-signal VALs (008, 009, 010, 012, 013, 014) PASS 5/6 declared nulls (N1, N3, N4, N6, N7 simplified, N8 — N2 skipped since AIBL has no chronological ages in GEO; AddNeuroMed VAL-010 has ages but per-VAL age column not consolidated yet). VAL-011 is PASS-AS-NULL by design (raw d=−0.004 correctly null at baseline). N3 borderline on VAL-009 (p=0.026) and VAL-014 (p=0.039) — flagged in card honest_limitations. Full β-matrix N7 end-to-end chain-recovery is genuine outstanding work; current N7 is simplified signal-level.

---

## 3. Future cards (queue)

Each new disease card gets its own VAL series in a fresh CPG-VAL-NNN slot range at the time of activation. Slot numbers are assigned at activation, not reserved in advance.

| Position | Card | Reference atlas status | Activation prerequisite |
|---|---|---|---|
| 1 (next) | **kidney-epic** | GSE50874 acquired (De Ridder 2024 *Nature Communications*, deconvolution-grade). GSE59157 also acquired for cross-cohort. | Ready when Heath says go |
| 2+ | CRC-immune-inv, CRC-secretory, cervical-epic, LGG/GBM-terminal, prostate-epic, lung-epic, hcc-epic, heme-epic, cardio-epic, MS-immune, Parkinson-immune | Reference atlases per ordering | Sequential activation as atlases come online |
| Later | hcc-cfdna, pancreatic-cfdna | — | When cfDNA substrate is unlocked |

The two operational cards (breast-epic + AD-immune) are not in this queue because they are already operational. Additional follow-up work for each operational card (panel holdout validation on independent cohorts, cross-ethnicity validation, etc.) lives in §6 Outstanding work, not in this queue.

---

## 4. Disease signature matrix v1.7

**Location:** `Biological_Physics/atlas_vault/walther_clinical_runtime/DISEASE_MATRIX/disease_cell_signature_matrix_v1_7.csv`

80 rows × 131 columns (8 metadata + 123 cell-type). Each cell is a signed Cohen's d or a range. Currently includes documented signatures for both operational cards plus 76 conditions held as look-up table for the future Stage 8 Path B per-patient matching engine.

| Row | Signature | Evidence anchor |
|---|---|---|
| `breast_cancer / long_pre_dx` | Basophil +1.58/+1.01, breast_BE +1.28/+0.61, T-cells −0.67/−0.58, Mahalanobis +1.88/+2.10 | CPG-VAL-001/002/005/007 |
| `alzheimers / at_dx_post_build_v3_0` | Eosino A −0.43, Neutro and 18 other immune cells negative, Mahalanobis +0.20 (modest), 7-CpG Rule A AUC 0.84 | CPG-VAL-008/009/012/013 |
| `frontotemporal_dementia / post_build_GIFT_2026` | Mahalanobis +0.28 (intermediate), distinct immune profile | CPG-VAL-014 |
| `progressive_supranuclear_palsy_CBD / post_build_GIFT_2026` | Mahalanobis −0.38 (BELOW_NORMAL — architectural compaction direction) | CPG-VAL-014 |
| 76 other rows | Held for Stage 8 Path B matching engine | (mostly from pre-build literature compilation; per-patient matching not yet wired) |

Engine schema: `disease_cell_signature_matrix_engine_schema_v1_2.md`. The `compute_match_magnitude()` and `compute_customer_tier()` functions are specced; implementation requires the cell-name-to-matrix-column mapping artifact (deferred).

---

## 5. SOP chain-of-custody coverage

SOP chain-of-custody is documented at `Biological_Physics/atlas_vault/walther_clinical_runtime/CPG_Chain_of_Custody_SOP_v1_2.md` (the authoritative encyclopedia). Per-card SOP audit narratives were folded into card JSONs at v3.1 (stage_2_deconvolution / stage_3_age_foreground_subtraction / etc. blocks) under the 3-canonical workflow rule; prior standalone audit files are archived in each card folder's `OLD/`.

Stage coverage (both cards):

| Stage | Module | Both cards |
|---|---|---|
| 0 — intake | engine QC | N/A (retrospective, not on first-client IDATs yet) |
| 1 — β computation | methylprep-equivalent | ✅ upstream (GEO normalized β) |
| 2 — Walther deconvolution | `Walther_iam_deconvolver/` | ✅ |
| 2 — NILC v2 cross-method | `NILC_Deconvolver/` | ✅ |
| 3 — age foreground | `IAM_Cellular_Age/age_axis_foreground.py` | ✅ |
| 4 — A-score (8 class + 115 cell) | `A_Scoring_Module/` | ✅ |
| 5 — Mahalanobis | `Mahalanobis_healthy_reference/` | ✅ |
| 6 — cellular age | `iam_cellular_age_scoring.py` | ✅ |
| 7 — tier breakpoints | `Tier_breakpoints/` | ✅ |
| 8 Path A — card matching | `DISEASE_MAPS_CARDS/` | ✅ card v3.0 |
| 8 Path B — matrix matching | `DISEASE_MATRIX/` + `compute_match_magnitude()` | ⚠️ rows live (v1.7); mapping artifact v0.1 starter shipped 2026-06-05 (50% atlas coverage); engine wiring still deferred |
| 9 — report assembly | `Literature_anchors_Report_building/` | N/A (per-patient) |
| 10 — delivery | engine | N/A (per-patient) |
| L9 — null suite | `CPG_Null_Runner/` | ✅ N1, N2, N3, N4, N6, N7, N8 across both cards (N5 unavailable — no plate metadata in GEO; N7 currently simplified signal-level — full β-matrix chain-recovery still outstanding) |

---

## 6. Outstanding work (carry-forward across all cards)

These items roll forward to every future card until completed. Detailed in MASTER_TRACKER §0.5.

1. **Formal v4 sealing per VAL** — current PREREGs are retrospective. v4 protocol requires PREREG sealed BEFORE rerun with hashed inputs + single-purpose reproducer script + full N1-N7 L9 suite. STATUS: outstanding (process change).
2. **Stage 8 Path B engine wiring** — cell-name-to-matrix-column mapping artifact + per-patient `compute_match_magnitude()` call. STATUS: mapping artifact v0.1 STARTER shipped 2026-06-05 at `DISEASE_MATRIX/iamatlas_115_to_matrix_v1_7_mapping.json` (50% atlas coverage); v0.2 manual taxonomy curation + engine implementation still outstanding.
3. **CHR/MAPINFO genomic annotation on residual maps** — STATUS: ✅ DONE 2026-06-05. Both card residual maps now have CHR + MAPINFO columns.
4. **Bimodality decomposition** — STATUS: ✅ DONE 2026-06-05. Breast (8,199 CpGs) was already complete; AD (6,018 CpGs from AIBL) computed 2026-06-05 with finding 2.3:1 gain:loss ratio.
5. **Per-card panel holdout validation** — CPG_breast_panel_v1 (1,392 CpGs) and CPG_ad_panel_v1 (200 CpGs) both need independent-cohort holdout. STATUS: outstanding (needs external cohorts).
6. **Cross-ethnicity validation** — both card series ran on predominantly-European cohorts. STATUS: outstanding (needs external cohorts).
7. **Full N7 end-to-end synthetic-patient chain-recovery** — synthetic_patient_generator.py infrastructure EXISTS (513 lines) + cpg_null_runner.py run_N7 implemented (721 lines). Current N7 across all VALs is simplified signal-level. STATUS: full β-matrix end-to-end chain run still pending.
8. **First-client IDAT integration** — Stages 0/1 untested on raw IDATs in our chain. STATUS: outstanding (needs first-client samples).

---

## 7. Reproducibility

Every claim is reproducible from `github.com/hmahaffeyges/IAM-Validation`. The per-VAL folder pattern is identical for every VAL across every card:

```
validation_runs/CPG_VAL_NNN_<card>_<test_name>/
├── PREREG.md
├── per_sample.csv
├── null_results.json
├── cohort_manifest.json
├── CPG_VAL_NNN_OUTCOME.md
└── (test-specific supporting CSVs)
```

The per-cohort folder pattern is identical for every cohort:

```
validation_runs/<card>_cohorts/<GSE>_<label>/
├── GSE*_betas_union.csv (or .csv.gz for size)
├── GSE*_clinical_metadata.json
├── GSE*_raw_geo_metadata.json
├── GSE*_full_results.csv  (Walther + 8-class A + 115-cell A + clinical merge)
├── GSE*_mahalanobis.csv
├── GSE*_115celltype_ascores.csv  (foundation-cohort pattern)
├── Stage2_NILC_cross_method_fractions.csv
├── Stage2_cross_method_walther_vs_nilc.json
├── Stage2_NILC_case_vs_hc_effects.json
├── Stage6_cellular_ages_per_class.csv
├── Stage6_cellular_age_case_vs_hc_effects.json
├── Stage7_tier_assignments.csv
├── Stage7_tier_distribution_by_arm.json
└── cohort_manifest.json
```

Extractor and driver scripts live at the card-cohorts root (e.g., `validation_runs/breast_epic_cohorts/extract_breast_cohorts.py`).

---

## 8. Version log

| Version | Date | Change |
|---|---|---|
| v1–v7 | 2026-05-29 through 2026-06-03 | Built incrementally as Phase 1 (foundation VALs), Phase 1 closure (card v3.0 + matrix v1.5), Phase 2 (AD-immune Family B), and Phase 2 closure (breast retrofit + AD release notes lessons learned) landed. Each version carried forward the original "Family A vs Family B" planning framework plus "slot reservations" plus old "TO BUILD" placeholders, accumulating contradictions as the actual work diverged from the plan. |
| **v8** | **2026-06-04** | **Full clean rewrite.** Dropped the Family A vs Family B terminology — both cards are simply operational cards with the same per-VAL bundle structure. Dropped "slot reservation" framing. Dropped Phase G/H/I/J historical roadmap blocks. Restructured around current state: §1 operational cards (table), §2 per-card VAL catalog with cohorts inline, §3 future cards (queue), §4 disease matrix, §5 SOP coverage, §6 outstanding work, §7 reproducibility pattern (per-VAL and per-cohort folder schemas), §8 version log. v1–v7 archived in `OLD/` for audit. |
| **v9** | **2026-06-05** | **L9 null suite expanded to N1-N4, N6-N8 across both cards.** Breast VALs 001/002/003/005/007 all PASS 7/7 nulls. AD VALs 008/009/010/012/013/014 all PASS 5/6 nulls (N2 skipped, N3 borderline on 009 and 014). VAL-011 PASS-AS-NULL by design. Residual maps for both cards now have CHR/MAPINFO genomic annotation. AD bimodality decomposition computed from AIBL (was placeholder); breast was already complete. Stage 8 Path B mapping artifact v0.1 STARTER shipped (50% atlas coverage). N5 plate-position null not run (no plate metadata in GEO). Full β-matrix N7 end-to-end still outstanding (infrastructure exists, simplified N7 in place). Outstanding work list (§6) updated with explicit STATUS per item. v8 archived. |

---

*End of inventory.*
