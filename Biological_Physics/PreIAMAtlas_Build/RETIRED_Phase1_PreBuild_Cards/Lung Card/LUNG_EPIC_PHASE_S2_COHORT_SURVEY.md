# Lung-epic v0.5 — Phase S2 cohort survey

**Author:** Walther
**Date:** 2026-04-28
**Standard applied:** CHK-1.7 (exhaustive cohort survey), CHK-1.5 (substrate scope), CHK-1.6 (access tier), CHK-2.7 (mandatory stratification block)
**Status:** Survey complete across all 6 CHK-1.7 buckets. Heath sign-off requested before any test scoping.

---

## Survey scope and methodology

Every public methylation cohort that touches lung cancer has been classified across six buckets per CHK-1.7. Per-cohort columns capture: GEO accession or biobank ID, n cases / n controls, platform, specimen type, pre-dx vs at-dx with TtD, smoking metadata availability (CCL-009 absolute), age distribution, sex composition, ethnicity, access tier (CHK-1.6), Sample_title verification status (CHK-1.1), substrate scope per saturation matrix (CHK-1.5), and direction expectation per CCL-006/CCL-019/CCL-023.

**Search method:** GEO eUtils eSearch on lung+methylation+blood/tissue/sputum × HM450 (GPL13534) + EPIC (GPL21145), supplemented by literature review of the Battram/Baglietto pre-dx cohort series (NOWAC/MCCS/NSHDS/EPIC-HD/EPIC-Italy), CLUE II Michaud-Kelsey nested case-control series, and tissue/multi-omic deposits with smoking strata.

**Honest deferral language applied per CHK-1.7:** Tier 2/3 cohorts that exist but cannot be reached today are listed with the access path named (PI contact, biobank study ID, application timeline). Cohorts are not silently dropped from the survey because access is gated.

---

## Bucket 1 — Pre-diagnostic blood methylation (TtD-stratified)

The canonical cohorts for "how does the immune system respond at 10/5/2 years pre-dx" — all biobank-gated, none GEO-deposited.

| Cohort | n cases / n ctrl | Platform | Specimen | TtD median | Smoking strata | Age | Sex | Access tier | Action plan |
|---|---|---|---|---|---|---|---|---|---|
| **CLUE II** (Michaud-Kelsey, Tufts/Brown) | 208 / 222 | EPIC 850K | Buffy coat | 14 yr | Available (matched on never/former/current+intensity) | Median 59/57 cases/ctrl | 55% F | **Tier 3** | Contact: ekelsey@brown.edu, dominique.michaud@tufts.edu. Application via Johns Hopkins CLUE II steering committee. 99% white, 11% never-smokers — limits Asian/never-smoker generalization. |
| **NOWAC** (Norwegian Women and Cancer) | 132 / 132 | HM450 | Whole blood | Multi-window | Available (CSI score in publication) | Adult | 100% F | **Tier 3** | UiT-Tromsø biobank application via T.M. Sandanger. Single-sex cohort — opposite-sex caveat absolute. |
| **MCCS** (Melbourne Collaborative Cohort) | 367 / 367 | HM450 | Whole blood | Multi-window | Available (matched on smoking) | Adult | M+F | **Tier 3** | Cancer Council Victoria biobank application. Largest pre-dx HM450 cohort. |
| **NSHDS** (Northern Sweden Health & Disease) | 234 / 234 | HM450 | Whole blood | Multi-window | Available (matched on smoking) | Adult | M+F | **Tier 3** | Umeå University biobank application via Mikael Johansson. |
| **EPIC-Heidelberg** (EPIC HD) | 63 / 63 | HM450 | Whole blood | Multi-window | Available (matched on smoking) | Adult | M+F | **Tier 3** | DKFZ Heidelberg biobank application via Rudolf Kaaks. Smallest of the Battram set. |
| **EPIC-Italy** (Vineis/Polidoro) | 185 / 185 | HM450 | Buffy coat | Multi-window | Available (matched on smoking) | Adult | M+F | **Tier 3** | HuGeF Torino biobank. **Same cohort that anchored VAL-093 breast pre-dx (different nested case-control arm).** |

**Tier 1 in this bucket: ZERO.** No public lung pre-dx blood methylation cohort exists on GEO at any platform.

**Operational consequence.** The "10yr/5yr/2yr immune response" question for lung cannot be answered from publicly-accessible GEO data alone. Reaching `single_cohort_validated` tier on lung pre-dx requires biobank applications. The honest v0.5 lung-epic must state this explicitly. Per heme-LL-011 + cervical-LL precedent: biobank tier limitation is documented honestly, not papered over.

**Recommended action plan.** CLUE II is the highest-value first application because (1) it is on EPIC (matches v1 production platform), (2) it is the only EPIC pre-dx lung cohort, (3) Michaud-Kelsey are reachable academic contacts, (4) it has full smoking strata + age + sex covariates. NOWAC + MCCS rank next as the largest HM450 cohorts.

---

## Bucket 2 — At-diagnosis blood case-control methylation (Tier 1 runnable)

| Cohort | n cases / n ctrl | Platform | Specimen | Smoking strata | Age | Sex | Ethnicity | Access tier | Sample_title verified |
|---|---|---|---|---|---|---|---|---|---|
| **GSE275371** (LUAD PBMC, He 2024) | 35 LUAD / 50 normal | EPIC 850K | PBMC | **NOT visible in series matrix** | Not in series matrix | Sex visible per sample | Likely Asian (no GEO institutional metadata) | **Tier 1** | Yes — LC_N / L_N = LUAD, NC_N = normal control |
| **GSE277078** (NSCLC plasma cfDNA, Kanai 2025) | 185 NSCLC / 0 internal HC | EPIC 850K | Plasma cfDNA | **NOT visible in series matrix** | Age bracket per sample (40s/50s/60s/70s/80s) | Sex visible per sample | Japanese (Keio U, Tokyo) | **Tier 1** | Yes — all samples disease=NSCLC |
| **Asan Medical Center Korean** (Hong 2019) | 150 NSCLC / 150 healthy | EPIC 850K | Whole blood | Available (current/past/never, frequency-matched) | 40-70 | M+F | Korean | **Tier 2 or 3** | Not GEO-deposited; BioResource Center contact ccm@amc.seoul.kr / pulmo2@kangwon.ac.kr; IRB AMC IRB 2011-0883 |

**Critical caveats:**

- **GSE275371 (PBMC LUAD vs normal):** No smoking metadata visible in series matrix. Smoking-CpG handling per CCL-009 cannot be applied without supplementary metadata or contact with submitter. **Same-cohort healthy controls are present (n=50)** — first lung blood case-control cohort with internal HC structure on EPIC, GEO-deposited. Sample size at n=35 cases is small but adequate for proof-of-concept Stage 1.
- **GSE277078 (NSCLC plasma cfDNA):** No internal healthy controls; healthy comparator must come from cross-cohort reference. **Cross-substrate caveat per CCL-010** — plasma cfDNA reads against buffy-coat or PBMC reference is NOT directly comparable. Substrate-scope translation rule (CHK-1.5) applies: v1 methyl-only on plasma cfDNA reads at the cfDNA detection floor (CCL-021 4% rule). Used as a within-cohort substrate-pattern characterization at v1, not a case-vs-control validation.
- **Asan Medical Center:** Has full smoking strata in publication (current/past/never frequency-matched) — **best smoking-strata-complete blood case-control cohort identified** — but not GEO-deposited. Tier 2/3 access path. Hong 2019 cg12169243 (DPH6) + cg25429010 (IMP3) reach genome-wide significance in current smokers only — these CpGs cannot be interpreted as cancer markers without smoking strata.

---

## Bucket 3 — Tumor vs adjacent-normal tissue case-control

Existing cookbook coverage in lung-epic v0.4 anchored on TCGA-LUAD HM450 (VAL-063 paired d=+1.020 ever-smoker, n=22). Survey extends to non-TCGA tissue cohorts.

| Cohort | n / structure | Platform | Smoking | Age | Sex | Ethnicity | Access tier | Status |
|---|---|---|---|---|---|---|---|---|
| **TCGA-LUAD** (existing VAL-063) | n=29 paired tumor/normal HM450 | HM450 | 22 ever / 2 never / 5 unk | Adult | M+F | Mostly Western | Tier 1 | **Already validated VAL-063, paired d=+1.020** |
| **TCGA-LUSC** (partial via VAL-056) | partial coverage | HM450 | Smoker-enriched | Adult | M+F | Mostly Western | Tier 1 | VAL-056 Part 2 partial; full TCGA-LUSC tissue arm has not been run as a properly pre-registered VAL |
| **GSE275371 LUAD-tissue arm** (parallel) | 35+ LUAD tissue | EPIC | Unknown | Unknown | M+F | Asian | Tier 1 | Same publication; check if companion tissue series exists |
| **GSE235414** (driver-stratified LUAD) | 130 LUAD ± EGFR | EPIC | Unknown | Adult | M+F | Asian (China) | Tier 1 | EGFR-mutation vs driver-negative LUAD; smoking metadata likely in supplementary |
| **GSE314841** (East Asian LUAD risk, Feb 2026) | 161 LUAD | 3-platform (HM450 + EPIC + GPL33022) | Unknown | Adult | M+F | East Asian | Tier 1 | **Newest deposit; multi-omic includes expression data; cross-platform within single study — built-in CHK-3.2 anchor** |
| **GSE217732** (high-grade LUAD subtype) | 19 paired | HM450 | Unknown | Adult | M+F | Western | Tier 1 | Small but specific to high-grade subtype |
| **GSE114989** (LUAD spatial heterogeneity) | 40 multi-region | EPIC | Unknown | Adult | M+F | Western | Tier 1 | Methodological; tests spatial heterogeneity within tumors |

**Operational consequence.** TCGA-LUAD VAL-063 stays the anchor at v1. Adding **GSE275371 tissue arm + GSE235414** as second/third independent tissue cohorts strengthens the tier from `single_cohort_validated` to `cross_cohort_validated_two_cohorts` per CCL-011 retroactive minimum standard. Priority order for tissue VAL queue: GSE275371 (paired tumor/PBMC same cohort) > GSE235414 (driver stratification) > GSE314841 (East Asian risk).

---

## Bucket 4 — Region/ethnicity-enriched cohorts (never-smoker stratum)

VAL-063 lifelong non-smoker n=2 was underpowered. The Asian/never-smoker LUAD literature has multiple deposits.

| Cohort | n / structure | Platform | Smoking | Age | Sex | Ethnicity | Access tier | Note |
|---|---|---|---|---|---|---|---|---|
| **GSE256092** (Never-smoker LUAD proteogenomic, Korean) | 141 NSLA tissue | EPIC | **All never-smoker (cohort definition)** | Per sample (range 37-85) | M+F | Korean | **Tier 1** | **Directly addresses VAL-063 never-smoker n=2 underpower.** ALL samples are never-smoker LUAD. No internal healthy comparator — needs cross-cohort matched-tissue reference (e.g. TCGA-LUAD adjacent-normal, with cohort-baseline caveat per CHK-3.2). Stage per sample (1/2/3/4). |
| **GSE314841** (East Asian LUAD risk) | 161 LUAD | 3-platform | Unknown | Adult | M+F | East Asian | Tier 1 | Multi-omic; smoking strata not yet verified in series matrix |
| **GSE235414** (driver-negative + EGFR LUAD) | 130 LUAD + adjacent-normal | EPIC | Unknown — likely available in supplementary | Adult | M+F | Asian (China) | Tier 1 | EGFR-mutation enriched (driver vs non-driver LUAD comparison) |
| **GSE311943** (LUAD multi-omic subtypes) | 13 | EPIC | Unknown | Unknown | Unknown | Likely Asian | Tier 1 | Small sample size — descriptive only |
| **Asan Medical Center Korean** (Bucket 2) | 150+150 | EPIC | Available (frequency-matched) | 40-70 | M+F | Korean | Tier 2/3 | If accessible, doubles as Bucket 2 + Bucket 4 |

**GSE256092 is the highest-value cohort in this bucket.** Per CHK-2.7 stratification rules: this is a single-stratum cohort (all never-smoker), so the analysis pre-registers within-cohort tumor-vs-tumor stratification (stage I/II/III/IV) plus cross-cohort comparison to TCGA-LUAD ever-smoker subset and TCGA-LUAD adjacent-normal — with explicit CCL-025 driver-stratification disclosure (never-smoker LUAD is mechanistically distinct from smoker-driven LUAD).

---

## Bucket 5 — Specialized substrates (sputum / BAL / bronchial brushing)

| Cohort | n / structure | Platform | Specimen | Smoking | Lung-relevant? | Access tier | Note |
|---|---|---|---|---|---|---|---|
| **GSE289379** | 64 | EPIC | Sputum | Unknown | NO — cystic fibrosis, not lung cancer | Tier 1 | Skip for lung-epic; possibly relevant for future cf-epic |
| **GSE268635** | 90 | EPIC | Sputum | Smoke-exposure focus | Partial — COPD biomass smoke; **value as smoking-confounder negative control** | Tier 1 | COPD without lung cancer; fires Stage 1 immune response without firing Stage 2 lung tile? Useful differential cohort. |
| **GSE262656** | 41 | EPIC | BAL | HIV-related senescence | Partial — HIV-positive not lung cancer; differential negative control | Tier 1 | Bronchoalveolar lavage senescence in PLWH |
| **GSE206709** | 72 | EPIC | BAL | Beryllium exposure | NO — chronic beryllium disease | Tier 1 | Skip for lung-epic |
| **GSE250513** | 376 | EPIC | Nasal epithelium | Asthma | NO — asthma not lung cancer | Tier 1 | Possibly relevant for upper-respiratory-tract differential |
| **GSE178809** | 152 | EPIC + RNA-seq | Airway | HIV-COPD | Partial — HIV-COPD differential | Tier 1 | Differential negative control |

**Operational consequence.** No public sputum/BAL/bronchial brushing lung-cancer-specific case-control cohort surfaces on GEO. The "sputum-based lung detection" pathway is not v1-validated — would require commercial cohorts (e.g., Veracyte Percepta) at Tier 3 partnership access.

**Useful for differential negative controls:** GSE268635 (COPD biomass smoke) and GSE262656 (BAL HIV-senescence) can serve as **CCL-021 differential cohorts** — diseases that fire Stage 1 immune response without firing Stage 2 lung-cell-of-origin. Useful for lung-epic specificity arm.

---

## Bucket 6 — Multi-substrate cohorts (L2/L3 framework predictions)

Per CHK-3.7: framework predictions at multi-substrate scope (Issue 002) are L3-deployment claims; v1 readings are methyl-only.

| Cohort | n / structure | Platform | Substrates | Lung-relevant? | Access tier | Note |
|---|---|---|---|---|---|---|
| **DELFI / Cristiano 2019** | varies (cancer types incl. lung) | WGS fragmentation | wps + frag | Partial — lung subset within multi-cancer cohort | Tier 2 (dbGaP) | Authoritative fragmentomic cohort; lung subset n~30; substrate-scope L3 |
| **MESA cancer cohort** (Mathios 2022) | varies | WGS + 5-substrate | All 5 substrates | Yes — has lung subset | Tier 2 (dbGaP) | The reference cohort for the 5-substrate framework calibration |

**Operational consequence.** No GEO-deposited multi-substrate lung cohort exists. L2/L3 framework predictions for lung cannot be tested against a public cohort at v1 deployment. Per CHK-3.7 substrate-scope rule, lung-epic v1 methyl-only readings are not directly comparable to any L3 prediction at absolute magnitude — direction transfers, magnitude does not. The v1 lung-epic stays at L1 lab partnership scope (methyl-only on EPIC arrays).

---

## Summary roll-up

| Bucket | Tier 1 (runnable today) | Tier 2/3 (gated) | Skip with reason |
|---|---|---|---|
| 1 — Pre-dx blood | **0** | 6 (CLUE II + 5 European biobank) | — |
| 2 — At-dx blood case-control | 2 (GSE275371, GSE277078) | 1 (Asan Korean) | — |
| 3 — Tumor vs adjacent-normal | 5 (TCGA-LUAD existing + GSE235414, GSE275371-tissue, GSE314841, GSE217732) | — | — |
| 4 — Asian / never-smoker | 4 (GSE256092, GSE314841, GSE235414, GSE311943) | 1 (Asan Korean) | — |
| 5 — Specialized substrates | 0 lung-cancer | — | 4 cohorts skipped (CF, asthma, beryllium, HIV-non-cancer); 2 retained as differential negative controls |
| 6 — Multi-substrate | 0 | 2 (DELFI/MESA dbGaP) | — |

**Headline gaps for lung-epic v0.5:**

1. **Pre-diagnostic blood methylation: zero Tier 1.** All Battram-set cohorts are biobank-gated. Lung-epic v0.5 cannot reach `single_cohort_validated` on pre-dx detection from public data alone. Honest documentation per heme-LL-011 precedent required.
2. **Smoking metadata gap on Tier 1 at-dx blood cohorts.** GSE275371 + GSE277078 do not have smoking strata visible in series matrix. **Hong 2019 / Baglietto 2017 smoking-CpG handling cannot be applied without contacting submitters** (Yae Kanai for GSE277078, He et al. for GSE275371) to request smoking metadata, or extracting it from supplementary processed matrix files.
3. **Asian / never-smoker stratum is now well-covered at the tissue level** via GSE256092 (n=141 all-never-smoker LUAD) — promotes lung-epic from VAL-063 underpower-on-non-smokers to a properly powered never-smoker stratum.
4. **Multi-substrate L2/L3 validation: no public lung cohort exists.** Substrate-scope translation rule (CHK-3.7) governs all v1 magnitude claims.

---

## Proposed test slate (for Heath sign-off)

Per CHK-1.7 absolute rule: tests are scoped only on cohorts that survived the survey with adequate stratification metadata. Heath signs off on the survey AND the test plan independently.

**Slate is presented in priority order. Each test specifies the cohort, the question, the prereg structure (CHK-2.7), the substrate scope (CHK-3.7), and the expected outcome label per pre-locked criteria.**

### Tier A — Tier 1 runnable, full stratification metadata available

**VAL-064 (proposed): GSE256092 never-smoker LUAD tissue, layered atlas Stage 2 + Stage 3 run-everything**
- Cohort: 141 NSLA tissue + cross-cohort TCGA-LUAD adjacent-normal as healthy reference
- Substrate scope: v1 methyl-only (EPIC)
- Stratification block (CHK-2.7): sex (M+F per sample), age (per sample, range 37-85, decade strata where n permits), stage (I-IV per sample), smoking (single-stratum: all never-smoker — explicit cohort-definition caveat). Driver mutations (TP53/KRAS/STK11/ERBB2/EGFR) per sample.
- Pre-locked outcome label criteria: H_A LOCALIZED (Lung_cells tile d > +0.5, others |d| < 0.3); H_B DISTRIBUTED (≥3 tiles d > +0.3 across cycling+secretory class); H_C NULL (all tiles |d| < 0.2); H_D BELOW_NORMAL (any tile d < −0.5).
- Expected reading: cycling-class lung tile elevation per VAL-063 ever-smoker pattern, magnitude unknown for never-smoker stratum because VAL-063 lifelong non-smoker n=2. **This VAL is the first properly-powered never-smoker LUAD lung-tile reading in the cookbook.**
- Cross-cohort baseline check (CHK-3.2): GSE256092 vs TCGA-LUAD adjacent-normal anchor on Loyfer 25 tiles; flag any tile at >1 anchor-SD.
- Substrate scope translation (CHK-3.7): v1 methyl-only; any comparison to Issue 002 multi-substrate framework prediction states L3-scope translation explicitly.

**VAL-065 (proposed): GSE275371 LUAD PBMC at-dx case-control, run-everything**
- Cohort: 35 LUAD PBMC + 50 normal PBMC, EPIC, paired sex stratification
- Substrate scope: v1 methyl-only
- Stratification block (CHK-2.7): sex per sample (CCL-002 absolute), smoking — **request supplementary metadata from submitter; if unavailable, prereg pre-locks the smoking-stratification-not-available caveat and runs whole-cohort + sex-stratified analyses only**.
- Pre-locked outcome label criteria: standard run-everything 4-question CCL-027 (H_A pooled direction Stage 1 immune, H_B bidirectional, H_C directional fallback, H_D lymphoid/myeloid pattern Stage 2 immune-class tiles).
- Expected reading: this is the first GEO-deposited at-dx LUAD blood case-control with internal HC. Stage 1 immune A-score reading direction tests CCL-023 prior. Per VAL-063 tissue ever-smoker direction is positive; PBMC direction at v1 methyl-only is open hypothesis — could test the orthogonal-priors pattern (CCL-036) versus inverted priors.
- Caveat: small n_cases=35 limits power; report d with 95% CI and underpowered language per CHK-2.7.

### Tier B — Tier 1 runnable, partial stratification metadata

**VAL-066 (proposed): GSE277078 NSCLC plasma cfDNA, run-everything within-cohort substrate characterization**
- Cohort: 185 NSCLC plasma cfDNA + cross-cohort GSE51057 EPIC-Italy buffy-coat as healthy reference
- Substrate scope: v1 methyl-only on plasma cfDNA — **cross-substrate caveat per CCL-010 absolute** (plasma cfDNA vs buffy coat is not directly comparable; cell-type fraction interpretation depends on detection floor per CCL-021 4% rule).
- Stratification block (CHK-2.7): age bracket per sample (40s/50s/60s/70s/80s), sex per sample, smoking — **NOT visible in series matrix; pre-reg pre-locks the smoking-NA caveat**.
- Pre-locked outcome label criteria: cohort distribution characterization only (within-cohort A-score distribution by age + sex). NO direct case-vs-control d reported because cross-substrate (plasma cfDNA vs buffy coat HC reference) violates CCL-010. **This VAL is descriptive, not validation.** O5_DESCRIPTIVE_CROSS_SUBSTRATE label.
- Recommended action: do not run as a full validation VAL until L3 framework deployment. Catalog as an exploratory characterization for the substrate-scope translation literature.

### Tier C — Tissue cohort retroactive expansion

**VAL-067 (proposed): GSE235414 EGFR-mutation LUAD vs driver-negative + adjacent-normal**
- Cohort: 130 LUAD ± EGFR vs adjacent-normal, EPIC tissue
- Substrate scope: v1 methyl-only
- Stratification block: driver mutation (EGFR vs driver-negative), sex, age, stage, smoking — **request supplementary metadata; smoking pre-lock per CHK-2.7**.
- Pre-locked outcome label criteria: tissue paired d expectation tests CCL-019 (panel choice vs class choice vs specimen choice) — does EGFR-mutation status modulate the cycling-class tile signal magnitude?
- Cookbook value: tests whether driver stratification (CCL-006 disease-direction varies by molecular subtype) extends to within-LUAD molecular subtypes.

### Tier D — Specificity arm (differential negative controls)

**VAL-068 (proposed, lower priority): GSE268635 COPD biomass smoke specificity arm**
- Cohort: 90 COPD sputum + biomass smoke exposure
- Use: differential negative control. Tests whether COPD-related smoke exposure fires lung-cell-of-origin tile drift WITHOUT lung cancer present. Per CCL-025 chronic disease-driver field-defect prior.
- Recommended outcome: descriptive characterization only; defines specificity boundaries for v1 commercial deployment.

---

## What the survey does NOT support (honest weakness statement)

Per CHK-2.7 honest underpower language and CCL-032 diagnostic-order-before-claim:

- **Pre-diagnostic 10yr/5yr/2yr immune temporal characterization for lung is NOT runnable from public data.** Lung-epic v0.5 cannot match the breast-epic VAL-093 / VAL-096 window-stratified analysis structure without biobank application. Honest documentation per heme-LL-011 precedent: "long-window pre-dx detection on lung requires CLUE II / NOWAC / MCCS / NSHDS / EPIC-HD / EPIC-Italy biobank applications. Application path: contact Michaud-Kelsey (CLUE II), Sandanger (NOWAC), Cancer Council Victoria (MCCS), Mikael Johansson (NSHDS), Rudolf Kaaks (EPIC-HD), Vineis-Polidoro (EPIC-Italy). Application timeline 6+ months."
- **Sputum-based lung detection is NOT v1-validated.** No public sputum lung-cancer case-control methylation cohort exists.
- **Multi-substrate L2/L3 lung framework predictions are NOT testable against public data.** No public GEO-deposited multi-substrate lung cohort exists. DELFI / MESA at dbGaP Tier 2.
- **Smoking-CpG handling is incomplete on the Tier 1 at-dx cohorts.** GSE275371 + GSE277078 do not surface smoking metadata in the series matrix. Either contact submitters for supplementary metadata or pre-lock the "smoking-NA, treat as confounder-uncontrolled" caveat per CHK-2.7. Hong 2019 cg12169243/cg25429010 cannot be interpreted on these cohorts without smoking strata.
- **Cross-cohort baseline alignment (CHK-3.2) is unknown for every Tier 1 cohort identified.** Each VAL must run CHK-3.2 against the chosen healthy reference before assigning the outcome label per CCL-034 within-cohort-vs-cross-cohort hierarchy.

---

## Heath sign-off requested on:

1. The survey is comprehensive across the six CHK-1.7 buckets — agree or flag missing buckets / cohorts I should add.
2. The honest weakness statement is acceptable to publish as-is in lung-epic v0.5 — agree or modify.
3. The proposed test slate priority order:
   - Tier A: VAL-064 (GSE256092 never-smoker), VAL-065 (GSE275371 PBMC at-dx)
   - Tier B: VAL-066 (GSE277078 plasma cfDNA descriptive only — recommend NOT running as full validation)
   - Tier C: VAL-067 (GSE235414 EGFR-mutation LUAD)
   - Tier D: VAL-068 (GSE268635 COPD specificity arm)
4. Whether to proceed with biobank applications for CLUE II + NOWAC + MCCS + EPIC-Italy (highest priority for pre-dx, longest application timeline, requires Heath-as-PI letter).
5. Whether to contact Yae Kanai (ykanai@keio.jp, Keio U Tokyo) and the GSE275371 submitter for supplementary smoking metadata before running VAL-065 / VAL-066, or pre-lock the smoking-NA caveat.

Per CHK-1.7 absolute rule: no prereg seals until Heath signs off on this survey AND the test plan independently.
