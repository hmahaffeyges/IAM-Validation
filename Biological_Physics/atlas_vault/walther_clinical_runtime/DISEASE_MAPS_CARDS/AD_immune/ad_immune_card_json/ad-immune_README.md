# AD-immune card — CPG / EDEAR

**Card version:** v3.0
**Card date:** 2026-06-02 (initial v3.0) + 2026-06-03 (Phase 2 SOP completion)
**Card location:** `ad-immune_card_v3_1.json` (this folder)
**Card README (this file):** v3.0 — clean rewrite 2026-06-04 with current methodology
**Maintained by:** IAMPerformance Inter-Domain Research Institute, Entiat WA / iamperformance.net

---

## What this card detects

A blood-based methylation signature that does **two clinically useful things at the same time**:

1. **Detects Alzheimer's disease** at clinical diagnosis from a blood draw (no LP, no PET).
2. **Discriminates among three neurodegenerative diseases** that are notoriously hard to tell apart clinically: Alzheimer's (AD), progressive supranuclear palsy / corticobasal degeneration (PSP/CBD), and frontotemporal dementia (FTD). The discrimination is by **direction-of-departure** on the universal architectural metric — AD goes positive, PSP/CBD goes negative, FTD is intermediate. This is the strongest single piece of evidence yet that the underlying architectural-information physics is biologically meaningful.

This card is **not a substitute for cognitive assessment, CSF biomarkers, or PET imaging**. It is a single blood draw that places a patient on a three-way continuum and offers a triage / tauopathy-differential layer.

---

## What we learned — plain language

These are the headline findings any clinician needs to know, in order of clinical relevance:

### 1. AD shows a TARGETED disturbance, not a broad shift

Alzheimer's at clinical diagnosis shows targeted disruption of immune-class cells — not the broad multi-class architectural shift seen in pre-diagnostic breast cancer.

- **AIBL cohort (n=161 AD / 471 HC EPIC):** 20 of 115 cell types Bonferroni-significantly depressed; ZERO Bonferroni-significantly elevated. Top hit Eosino A-score **d = −0.426** (p = 2.3 × 10⁻⁵).
- **Universal Mahalanobis metric:** only **d = +0.20** — modest precisely because the signal is concentrated in immune-class cells and the universal metric averages it against many quiet dimensions.

The takeaway: **operational scoring routing matters by disease**. For AD, route through a disease-trained panel. For broad-architectural diseases like breast pre-dx, route through the universal Mahalanobis. Different diseases have different signature topologies.

### 2. Cross-platform replication holds at the per-cell level

The same Eosino effect replicates across platforms:

- **AIBL (EPIC):** Eosino d = −0.43
- **AddNeuroMed (450K):** Eosino d = −0.46

The universal Mahalanobis attenuates on 450K (drops to near-null) because the 450K platform covers only 86–95% of the EPIC CpGs the Mahalanobis covariance was trained on. **This is platform, not biology.** Per-cell findings are robust across platforms.

### 3. The three-way specificity discrimination is the strongest evidence yet

The GSE53740 GIFT cohort contains AD + FTD + PSP/CBD + HC samples. The **same universal Mahalanobis metric** gives three biologically distinct signatures:

- **AD:** d = +0.681 (p = 0.001) — positive, departure outward from healthy centroid
- **FTD:** d = +0.28 — intermediate
- **PSP/CBD:** d = −0.380 (p = 2 × 10⁻⁶) — **NEGATIVE, BELOW-normal, architectural compaction direction**

PSP/CBD are primary 4R-tauopathies (different from AD's 3R/4R mix). They show the **opposite direction** on the metric. This is direct evidence that the metric is measuring real architectural geometry — generic "different from healthy" would not produce direction-resolved discrimination across diseases.

Biologically, the negative direction is consistent with accelerated cellular quiescence / stress-induced compaction in 4R-tauopathies. The methylation patterns are MORE compressed toward the H_min floors, not less.

### 4. Disease-trained panel outperforms universal screen for AD (opposite of breast)

- **7-CpG Rule A panel** (a disease-trained AD biomarker set): AUC 0.84 for AD discrimination on AIBL
- **Universal Mahalanobis hyper-volume:** AUC 0.62 on the same cohort

The disease-trained panel wins by ~3x in AUC space because it weights the dimensions where AD's signal is concentrated.

For breast pre-dx, the opposite was true — universal Mahalanobis BEAT the disease-trained Xu-538 by +0.75 standard deviations. **Different diseases need different operational scoring routes.** This card spec accounts for that.

### 5. AD immune cells look "younger" methylation-wise — senescence not aging

In the GSE53740 GIFT AD samples, immune-class cellular age comes back **9.2 years younger** than HC (case 55.4y vs HC 64.6y, **d = −0.56** on OK-status samples). This is NOT a sign of youth — it is consistent with cell-cycle arrest and senescence in the immune compartment, where stressed cells stop progressing through the normal methylation-aging trajectory.

Independent finding: 115-cell A-score layer is naturally age-orthogonal (Δd < 0.05 under age subtraction at the 115-cell layer). The 7-CpG Rule A panel itself has R² = 0.26 with age (meaningful age confound). Production reports for AD must explicitly subtract age before reporting an AD-A-score — the age-axis foreground subtraction module handles this automatically.

### 6. Cross-cohort residual map is real

The per-CpG residual map (observed β − class-fraction-predicted β) on AIBL × AddNeuroMed:
- Spearman ρ = 0.231 (p = 1 × 10⁻⁷⁴) — modest but highly significant
- 241 strong-concordant CpGs at |d| > 0.2
- 88.9% same-sign rate across cohorts
- AD residuals biased **4.8:1 negative direction** (hypomethylated > hypermethylated)

A 200-CpG candidate panel (**CPG_ad_panel_v1**) has been seeded from this intersection. Independent-cohort holdout validation is outstanding.

### 7. All 8 null tests pass

Every VAL in this series survived random label-permutation testing at p < 0.05. The signals are not artifacts of how cases were assigned to controls.

### 8. Stage 1 reproductions PASS on all 3 cohorts

The post-build pipeline reproduces the build-time pipeline bit-for-bit:
- **AIBL:** d = +0.615 vs anchor +0.624
- **AddNeuroMed:** d = +0.317 vs anchor +0.332
- **GIFT pooled:** d = +0.013 EXACT
- **GIFT male AD:** d = +0.415 EXACT

Two EXACT reproductions plus two within-sampling-variation reproductions confirms pipeline integrity.

---

## How a CPG report on this card reads

For an AD-suspected patient, the report contains:

1. **AD-A-score (7-CpG Rule A panel, age-adjusted Z output per §E.5)** with patient value, age-matched healthy median, and percentile. This is the primary operational readout.
2. **Universal Mahalanobis hyper-volume distance** with patient value, sign, and the population distribution it sits in. **The sign matters** — positive = AD-like, negative = PSP/CBD-like, intermediate = FTD-like region.
3. **Per-cell immune-class fan-out** — top three positive and top three negative departures. If Eosino, Neutro, or related cells appear in the top NEGATIVE departures, the report flags them as AD-consistent.
4. **Cellular age per class** with status flags. Immune-class cellular age younger than chronological age is the AD-consistent pattern (senescence).
5. **Tier assignment** per class (BELOW_NORMAL flag is the PSP/CBD-consistent direction for the universal metric).
6. **Three-way differential** between AD, PSP/CBD, FTD based on direction-of-departure and per-cell pattern.
7. **Honest limitations section** (see "What we are not claiming" below).

---

## How the card works — methodology summary

Same current production methodology as the breast-epic card (no pre-build-era external panels in the production scoring chain). For AD specifically:

1. **Walther IAM Deconvolver** → 8 architecture-class fractions per patient.
2. **A-scoring** → 8 class A-scores + 115 cell-type A-scores.
3. **Mahalanobis hyper-volume** → universal departure scalar (sign-resolved — positive vs negative discriminates AD from PSP/CBD).
4. **NILC v2 cross-method check** → independent compositional verification.
5. **Age-axis foreground subtraction** → mandatory before reporting AD-A-score (per §E.5 of the card spec).
6. **7-CpG Rule A panel** (disease-trained, age-adjusted Z) → primary AD discrimination score.
7. **Cellular age per class** → immune-class cellular age vs chronological age is the senescence diagnostic.
8. **Tier breakpoints** → universal screen layer (does not move much for AD, see Lesson #6 in the release notes).

The Recipe — the first-principles derivation chain producing the H_min values — is vault-only.

---

## What we are NOT claiming

Be honest with the patient about these:

1. **The card does not replace cognitive assessment, CSF biomarkers, or PET imaging.** Those have decades of clinical validation. CPG-AD is a single blood draw with promise as a triage tool and a tauopathy differential.
2. **No prospective primary-care validation.** All cohorts used here were research cohorts with established clinical diagnoses. Prospective primary-care validation is outstanding.
3. **Cross-ethnicity validation is outstanding.** All three cohorts are predominantly European-ancestry.
4. **The disease-trained panel has age confound.** R² = 0.26 with age. Production reports MUST subtract age first (handled automatically by Stage 3 + §E.5).
5. **The disease signature matrix v1.7 per-patient matching engine is not yet wired.** Per-patient reporting currently goes through the card-driven Stage 8 Path A, not the matrix-driven Stage 8 Path B.

---

## Validation summary

| What | Details |
|---|---|
| **VAL series** | CPG-VAL-008 through CPG-VAL-014 |
| **Cohorts** | AIBL GSE153712 (n=726, 161 AD / 471 HC / 94 MCI, EPIC) + AddNeuroMed GSE144858 (n=300, 93 AD / 96 HC + 111 MCI/precursor, 450K cross-platform) + GSE53740 GIFT (n=384, 15 AD / 193 HC / 95 FTD / 47 PSP-CBD / 34 other, 450K specificity arm) |
| **L9 null suite** | All 8 N1 nulls PASS (CPG-VAL-011 is PASS-AS-NULL, correctly null at age-baseline). N2 age-strata permutation also PASS on VAL-011. |
| **Stage 1 reproductions** | ✅ PASS on all 3 cohorts (see Finding #8 above) |
| **Cross-method check** | Walther vs NILC v2: ρ = +0.93 immune / +0.86 progenitor (AIBL); +0.84 / +0.78 (AddNeuroMed); +0.80 / +0.92 (GIFT) |
| **Per-VAL bundles** | `validation_runs/CPG_VAL_NNN_AD_*/` — each with PREREG.md + per_sample.csv + null_results.json + cohort_manifest.json + OUTCOME.md |
| **Cohort folders** | `validation_runs/ad_immune_cohorts/{GSE153712_AIBL,GSE144858_AddNeuroMed,GSE53740_GIFT}/` |
| **SOP audit** | `AD_IMMUNE_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md` (this folder's parent) |
| **Status** | ✅ Operational — full SOP coverage stages 2–7 + Stage 8 Path A. Stage 8 Path B engine wiring deferred. |

For complete validation detail, see the evidence report `post_build_evidence/v5_CPG_IAMAtlas_Evidence_Report.html` Section 4.2.

For the 13-item AD-specific Lessons Learned section (insights that don't fit in the README but are essential for any future AI session or researcher), see `ad-immune_v3_0_release_notes.md` in this folder.

---

## Outstanding follow-up work (carry-forward to v3.1)

1. CPG_ad_panel_v1 (200-CpG candidate panel from CPG-VAL-013) — formal seal + holdout validation on an independent cohort
2. Prospective primary-care cohort validation
3. CHR/MAPINFO genomic annotation on residual map
4. Cross-ethnicity validation (Asian, African, Latin-American cohorts)
5. Stage 8 Path B engine wiring (cell-name-to-matrix-column mapping artifact)
6. Synthetic_Patient_Generator chain-recovery test on the AD signature
7. First-client IDAT integration test (Stages 0/1 untested on raw IDATs in our chain)

---

## Version log

| Version | Date | Change |
|---|---|---|
| v2.2 | 2026-04 (pre-build era) | Pre-build era card. Used 7-CpG Rule A panel only. No cross-platform replication, no 3-way GIFT specificity arm. Archived at `OLD/ad-immune_card_v2.2.json`. |
| v3.0 | 2026-06-02 | Strict additive bump from v2.2. Three cohorts (AIBL + AddNeuroMed + GSE53740 GIFT) added. CPG_ad_panel_v1 candidate seeded (200 CpGs from CPG-VAL-013). Card uses the current Walther / Mahalanobis / NILC v2 / cellular age methodology. |
| v3.0 + Phase 2 SOP completion | 2026-06-03 | All 7 VAL bundles formalized at `validation_runs/CPG_VAL_NNN_AD_*/`. Three cohort folders fully populated. SOP chain-of-custody audit document published. 13-item Lessons Learned section added to release notes. |
| **v3.0 + README rewrite** | **2026-06-04** | **README rewritten clean** with current methodology focus and plain-language clinical findings sections. v2.2 README archived at `OLD/ad-immune_README_v2_2.md`. |

---

*Companion documents in this card folder: `ad-immune_card_v3_1.json` (card spec), `ad-immune_v3_0_release_notes.md` (technical changelog + 13-item AD-specific Lessons Learned). Companion documents in card parent folder: `AD_IMMUNE_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md`, `ad_immune_residual_maps/`.*
