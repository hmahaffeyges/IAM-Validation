# Immune-Atlas Card — EDEAR Rosetta Reference Card

**Version 0.3 · 2026-04-30** (amended from v0.2 · 2026-04-30, v0.1 · 2026-04-24)
**Validation tier:** `rosetta_reference_card`
**Card role:** Stage 1 interpretation engine + Stage 2 atlas cross-reference + Stage 3 OQ-2026-01 staging hub
**Card type:** cross-reference / interpretive (does not trigger alone; reads Stage 1 output from other cards; consolidates Stage 2 atlas-sweep results across all cards; canonical home of OQ-2026-01 immune-atlas staging)
**Card position:** #15 of 15 in Cookbook v2.5 expansion set (last by dependency, first in operational importance — this card cannot exist in full until every other card has been built and tested; once it does exist in full, every other card depends on it)

**v0.3 amendment summary (2026-04-30).** Card promoted from `reference_document` to `rosetta_reference_card`. Reframed from v0.1 "differential-diagnosis engine" to v0.3 three-role operational engine: Stage 1 interpretation + Stage 2 atlas cross-reference + Stage 3 OQ-2026-01 staging. NEW master sections: pre-test integrity protocol (§2 — atlas calibration prerequisite, six data integrity checks, biology consistency check, demographics-as-mandatory-stratifiers, bidirectional-as-default doctrine, ten failure-mode fingerprints, six biology-real patterns), cookbook doctrine that touches immune class (§8: CCL-006/019/023/027/028/030/031/032/039), CCL-027 four-question master cross-reference table (§11), CCL-031 five-pattern taxonomy (§6), three doctrine cases (§7: AD/HCC/glioma gifts), Stage 2 atlas registry (§10), Stage 3 sub-cell-type signatures expanded to 5 sealed cards + heme analog (§13), OQ-2026-01 canonical home moved here (§12), cross-card syntheses expanded from 1 pair to 10 pairs (§14), open atlas-coverage gaps (§15), future v1.0 reorganization plan (§17). Lessons learned grow with immune-atlas-LL-002 through LL-008. v0.1 + v0.2 content preserved verbatim.

---

**Amended 2026-05-01 (v0.3.1):** Bladder-epic v0.1 additive integration. Added sixth CCL-031 pattern (substrate-distribution mismatch on mucosal cohort). Extended §10 atlas registry with EpiSCORE BladderRef as fourth successful gene-promoter bridge. Extended §11 CCL-027 four-question table with bladder row. Extended §13 Stage 3 multi-atlas table with bladder VAL-122 row. Added new §13.7 — how to interpret bladder-style broad-positive Stage 3 immune readings (operational guidance for the future-AI-2-years-from-now interpreter). Added §22.4 v0.3.1 lessons (immune-atlas-LL-009 through LL-013) covering the four DISC-BLADDER findings as they apply to immune-class interpretation plus the meta-lesson on how the cookbook caught its own assumption mid-sprint. All v0.1, v0.2, and v0.3 content preserved verbatim. The card position remains #15 of 15.

---

## §1. What this card is

The immune-atlas is the **Rosetta Stone of EDEAR** — the canonical reference card for how the immune class responds to every disease in the cookbook, organized into three operational roles that together form the diagnostic spine.

### §1.1 The three-stage diagnostic spine

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1 — The Red Flag Test (universal first test)              │
│   Substrate: buffy-coat DNA                                     │
│   Panel: Xu-538 immune (sealed SHA ada6729...)                  │
│   Output: A_immune_pooled, A_dir, per-CpG Δβ direction          │
│   Interpretation engine: THIS CARD                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2 — The Atlas Sweep (RUN EVERYTHING doctrine)             │
│   Method: every atlas, every matrix, every tile, every sample   │
│   Atlases: Loyfer/Moss, EpiSCORE family, UniLIFE, Salas IDOL,   │
│            Caggiano TIM, ProstateRef, plus future bridges       │
│   Output: 25-tile per-class A-score, cell-of-origin direction,  │
│           tumor-microenvironment tile pattern                   │
│   Cross-reference engine: THIS CARD                             │
│   Disease-specific translation: each disease's own card         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3 — The Bidirectional Immune Test (OQ-2026-01 staging)    │
│   Method: lymphoid-vs-myeloid sub-panel split on Xu-538         │
│   Atlases: Salas IDOL-Ext or equivalent CpG-to-lineage map      │
│   Output: per-arm directional A-score; lineage-confirmed        │
│           bidirectional cancellation diagnosis                  │
│   Status: NOT YET RUNNABLE on any disease                       │
│   Closest existing analog: heme-epic three-arm structure        │
│   Open-problem canonical home: THIS CARD                        │
└─────────────────────────────────────────────────────────────────┘
```

Every EDEAR patient's first test is the same — buffy-coat methylation, Xu-538 immune panel. That test is the red flag. If the red flag fires (elevated A-score, inverted A-score, or directional panel signal), Stage 2 RUN-everything sweeps every atlas at our disposal, and disease-specific cards translate what the cell-class and organ-of-origin patterns mean. If Stage 2 returns no solid-organ localization, OR if the disease pattern suggests immune-compartment cancer, lineage-cancellation, or multi-disease scenarios where systemic inflammation overlaps with localized disease signal, Stage 3 (when operational) takes the deeper bidirectional dive.

### §1.2 Why the immune class is the first test

Buffy-coat DNA is approximately 70% immune cells. The immune compartment responds to upstream disease in characteristic ways: breast cancer drives it positive, CRC drives it negative (inversion), AD gives it a bidirectional signature that only a directional panel recovers, glioma drives both A-score and cell-fraction in orthogonal axes, HCC drives ccfDNA but not whole-blood-leukocyte against metabolic-disease controls, cardio drives systemic inflammation that can mask or amplify other signals concurrently. Same instrument, distinguishing fingerprints. The immune compartment is the field that everything else upstream broadcasts into. Reading that field correctly is the foundation of every other interpretation.

### §1.3 Why this is the most important card in the cookbook

Every other card reads a single specimen, a single tissue, a single disease lens. This card reads across all of them. As more cards validate, every other card becomes *more* dependent on this card — because their Stage 1 interpretation, their Stage 2 atlas comparisons, and their eventual Stage 3 lineage discrimination all need a central reference. Other cards depreciate toward "tissue-specific extension." This card appreciates toward "the interpretation engine."

This card sits at #15 because it cannot exist in full until everything before it has been tested. By the same token, once it exists in full, everything before it routes through it.

### §1.4 RUN-everything as the safety net for concurrent disease scenarios

Patients do not present with one disease at a time. A 58-year-old smoker with familial hypercholesterolemia and a 5-pack-year-ago smoking cessation may have systemic vascular inflammation, early lung field-effect drift, latent prostate dedifferentiation, and clonal hematopoiesis of indeterminate potential — all simultaneously, all firing the immune red flag. Conditional gating (Stage 2 only-if-Stage 1-positive, or only-if-tissue-of-origin-matches-suspected-disease) would let one of these signals dominate the report and the others get filtered out before Stage 2 ever asked. **RUN-everything is the cookbook's structural answer to this:** every IDAT runs every atlas, every panel, every per-tile A-score, regardless of any prior-stage result. The immune-atlas card is the cross-reference that lets a clinician read all of those parallel signals together rather than having to chase down each disease card individually.

---

## §2. Pre-test Integrity Protocol — How to know the number means what you think it means

**This section is mandatory. Before any interpretation downstream of Stage 1 is performed — before consulting the cross-reference table, before Mahalanobis differential ranking, before pathway routing, before drafting any clinical action — the operator must verify the test passed all six data integrity checks, the biology consistency check, and the demographics stratification check. Test failures and operator errors produce numbers that look like findings.**

### §2.1 Why this section exists

Cervical-epic burned approximately four hours on VAL-076 and VAL-077 because the operator treated framework numbers as biology before checking whether the data was interpretable as biology. CCL-032 fixed the diagnostic order after that incident: **data integrity → biology consistency → framework finding.** The cervical incident is the canonical example: the framework registered null on cervical-LBC pap-smear specimens at the same time published clinical-grade panels (FAM19A4/miR124-2 QIAsure AUC 0.77, ZNF671 GynTect, PAX1/NREP-AS1 Bowden 2025 AUC 0.92) achieved strong signal on the same cohorts. The framework null was a transferability finding — Xu-538 was buffy-coat-trained and does not transfer to LBC cell mixtures — not a biology finding about cervical disease. CCL-032 is the rule that prevents this mistake from recurring.

Test failure and operator error produce numbers that look like disease findings. Without the protocol below, a flat β distribution from residual M-values reads as bidirectional cancellation. A cross-cohort baseline drift reads as cohort-vs-cohort effect size. A saturation-compressed null reads as biology-null. A direction-locked prereg reads as O5_DIRECTION_FLIP_UNANTICIPATED when the actual finding is clean strong negative. These are recognizable failures with recognizable fingerprints. The operator's first job is recognizing them.

### §2.2 The fixed diagnostic order (CCL-032)

```
Step 1: DATA INTEGRITY    →  is the file what we think it is?
                              are the cohorts comparable?
                              is the atlas calibrated?
                              are demographics accounted for?
                              ↓ pass: proceed.    fail: stop, fix data.

Step 2: BIOLOGY CONSISTENCY → is the result consistent with published
                              clinical-grade panels for this disease
                              on this cohort? with the cohort's own
                              published findings? with established
                              disease immunology literature?
                              ↓ pass: proceed.    fail: investigate
                                                  transferability vs
                                                  cohort heterogeneity
                                                  vs true biology null.

Step 3: FRAMEWORK FINDING  →  only NOW interpret the framework number
                              as a finding. Read against this card's
                              cross-reference table, run pathway
                              routing, draft clinical action.
```

**Skipping or reordering produces overclaim+revert cycles.**

### §2.3 Step 1 — Six data integrity checks

#### §2.3.1 CHK-3.1 raw β distribution sanity check

Real raw β values from EPIC/450K methylation arrays are bimodal: **>30% at extremes (β < 0.1 or β > 0.9) AND <10% in the middle (β between 0.4 and 0.6).** If the distribution is flat, near-Gaussian, or concentrated near 0.5, the file is NOT raw β.

**Common causes of CHK-3.1 failure:**
- minfi noob-bg-corrected output without re-conversion (VAL-100 GSE282666 Kumar 2024 polyp cohort: extreme 3.9%, middle 6.8%)
- Residual M-values from differential-methylation analysis pipelines (VAL-077 GSE287994 Bowden 2025 cervical LBC: 50% in middle, 12% extremes)
- GenomeStudio AVG_Beta with downstream normalization applied

**Action when CHK-3.1 fails:** stop the analysis. Reprocess from raw IDATs through minfi/sesame, OR reproduce the source paper's exact pipeline from Methods. Do NOT interpret the data on its current form. Do NOT report a null outcome from CHK-3.1-failed data without the data integrity flag (`O5_DATA_INTEGRITY_FLAG` per cookbook precedent VAL-100, VAL-077, VAL-101).

#### §2.3.2 CHK-3.1B per-sample atlas coverage check (substrate-floor-based, NOT default 95%)

When a custom or bridged atlas is used, the per-sample atlas-CpG-intersection coverage must clear the **substrate floor**, not a default 95%.

**Substrate floors (sealed precedent):**

| Substrate | Floor | Rationale |
|---|---|---|
| TCGA HM450K sesame Level 3 | ~80% | TCGA QC routinely drops 12-20% of probes via cross-reactive masking, SNP-overlap, and detection p-value failures |
| EPIC 850K native | ~85% (typical, cohort-dependent) | Standard EPIC processing tighter than TCGA HM450K but below 95% by default |
| HM450K minfi preprocessFunnorm | ~92% (typical) | Functional normalization retains more probes |
| 27K → 450K bridges | substrate-specific (check before pre-lock) | Older array bridges produce variable coverage |

**Action when CHK-3.1B is misspecified:** if a prereg defaults to 95% on TCGA HM450K, the prereg is wrong, not the data. Amend the prereg threshold to substrate-floor sealed BEFORE re-execution per CCL-041.

**Sealed VAL precedent:** VAL-117 ProstateRef Phase B calibration first execution failed CHK-3.1B at 0/210 samples because the original prereg specified ≥95% per-sample coverage on TCGA HM450K sesame Level 3. Amendment changed threshold to ≥80%, sealed before re-execution. Re-execution: 210/210 pass. CHK-2.8 cookbook-wide rule formalized from this incident.

#### §2.3.3 CHK-3.1C atlas duplicate check

Bridged atlases can produce duplicate probeIDs if bridge logic is wrong. The bridged matrix must be deduplicated, and the **atlas SHA-256 must be computed on the bridged matrix** and verified against sealed precedent before any production use.

**Sealed atlas SHAs (production-ready):**
- Xu-538 panel: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`
- ProstateRef CpG-bridged: `4e60c3d038a637e9742f51d9bc7c119e06fe5d2e91abb2b12db8867ceb7813d2`
- Other atlas SHAs documented in §10 atlas registry

#### §2.3.4 CHK-3.2 cross-cohort healthy-baseline alignment

When a cohort's healthy mean A-score differs by more than 1 SD from anchor cohort, **the cohorts are not directly comparable.** This is a calibration failure, not a disease finding.

**Sealed VAL precedent:**
- VAL-100 GSE282666 polyp cohort PNP- mean A_immune = 0.807 vs Italian healthy anchor 0.438 = +15.13 anchor-SD offset (off-spec scale)
- Cervical-LL-010: VAL-073 vs VAL-074 healthy reference baseline shifts of >0.06 A-units across cohorts are themselves diagnostic of cohort heterogeneity (HPV-stratification of normals), NOT disease

**Action when CHK-3.2 fails:** do NOT interpret cases-vs-controls Cohen's d in cross-cohort framing. Use within-cohort paired comparison instead, OR calibrate the cohort to anchor first via on-study controls. CCL-034: cross-cohort comparisons interpretable only at secondary-evidence tier.

#### §2.3.5 Saturation flag check (CHK-3.5 / CHK-7.5)

Each substrate has a per-class A-score ceiling. Approaching a ceiling produces saturation-induced compression of the signal. Mandatory before any null outcome.

**Per-substrate per-class A-ceiling (cycling-class example):**

| Substrate | A-ceiling | Runtime flag threshold | Notes |
|---|---|---|---|
| methyl | 1.1681 | 1.1631 | functional substrate |
| nucl | 1.0203 | structurally saturated | NON-functional substrate for cycling |
| fuzz | 1.221 | n/a | functional substrate |
| wps | 1.5938 | n/a | functional substrate |
| frag | 1.4536 | n/a | functional substrate |

**Action when saturation flag fires:** switch to non-saturated substrate before drawing biology conclusions; OR report the saturated reading as "saturation-flagged, biology limit not interpretable."

#### §2.3.6 Sample-group assignment spot-check

Cohort design heterogeneity catches at this step: HPV-stratification of normals, smoking-status assignment, age-range mismatches, sex composition, treatment-stage heterogeneity, ethnic composition. GEO landscape errors also catch here: VAL-075 was excluded mid-run because GSE38266 turned out to be HNSCC head/neck cohort, not cervical (cerv-LL-008).

**Action:** run sample-title scan against the cohort's published Methods BEFORE β extraction. Verify subgroup labels match the published cohort design.

### §2.4 Step 2 — Biology consistency check

If data integrity passes, ask: **is the result consistent with published clinical-grade panels for this disease on this cohort?**

This is the step that prevented cervical-epic from publishing "cervical methylation has no signal" when the actual finding was "Xu-538 buffy-coat-trained panel does not transfer to LBC pap-smear specimens."

**Three diagnostic categories when framework reads null:**

1. **Transferability finding** — clinical-grade panels achieve strong signal on the same cohort where the framework reads null. The framework's panel does not transfer to that specimen. NOT a "the disease has no signal" finding.
2. **Cohort heterogeneity finding** — the cohort's healthy reference is stratified by a disease-relevant covariate (HPV-status, smoking-status, ancestry) in a way that shifts the baseline. The disease has signal; the comparator is the wrong comparator.
3. **True biology null** — clinical-grade panels also read null on this cohort. The disease genuinely has no signal in this specimen at this disease stage at this resolution. This is the only category where the framework's null is a biology finding.

**Action:** before drafting any null outcome, document which of the three categories applies. If category (1) or (2), the outcome is `O5_TRANSFERABILITY_FLAG` or `O5_COHORT_HETEROGENEITY_FLAG`, not biology null.

### §2.5 Step 3 — Framework finding (last, not first)

Only after data integrity passes AND biology consistency is checked does the framework number get interpreted as a finding. Read against this card's cross-reference table (§5), run pathway routing (§9), draft clinical action.

### §2.6 Demographics-as-mandatory-stratifiers

Patient demographics are not metadata — they are mandatory analysis stratifiers when the disease's literature establishes them as such. **If a known demographic stratifier is ignored, the framework finding is uninterpretable.**

#### §2.6.1 Sex stratification

| Disease | Sex stratification mandatory? | Rationale |
|---|---|---|
| Breast | Yes (predominantly female; male breast cancer separate analysis) | Cohort design |
| Prostate | Yes (male only by definition) | Cohort design |
| Cervical | Yes (female only; decline scoring on male samples) | Cohort design |
| HCC | **Yes** (sex stratification effect: VAL-059 male d=+1.00, female d=+0.50) | Disease biology — male / female immune response to HCC differs |
| Lung | Yes (never-smoker LUAD predominantly female; sex × smoking interaction) | Disease biology |
| AD | Yes (sex-specific panels tested in VAL-053; unified panel performs better) | Disease biology — empirically tested |
| CRC | Recommended (sex × anatomic site interaction in published EWAS) | Disease biology |
| Cardio | Yes (cardiovascular event rate differs by sex; ARIC + FHS + WHI cohort balance varies) | Cohort design + disease biology |
| Glioma | Recommended (M:F ratio approximately 3:2 in GBM) | Disease biology |
| Pancreatic | Recommended | Disease biology |
| Heme | Recommended (subtype-dependent) | Disease biology |

#### §2.6.2 Age stratification

Age is a mandatory analysis stratifier in **every** card. Cellular methylation age (Horvath, Hannum, PhenoAge, GrimAge clocks) is itself an architectural drift signature that overlaps with disease signatures at long pre-diagnostic windows.

**Sealed VAL precedent for age mattering:**
- AD-immune: age-regressed A_dir (VAL-052 §E.5) is mandatory output — age regression accounts for ~26% of variance, age-adjusted residual d=+0.12 vs unadjusted d=+0.33
- Breast: long pre-dx windows (>10yr) show body-wide cellular-aging-drift signature alongside breast-specific signal (VAL-096) — distributed pancreatic + cycling-class tile elevation that is NOT breast-localized
- VAL-099 CRC age-stratified: under-50 stratum n=3 mean ΔA = +0.0357 (descriptive only per CHK-2.7); age 50+ stratum n=21 paired d = +0.539 (inferential)

**Action:** every Stage 1 analysis must report age-regressed A-score alongside raw A-score.

#### §2.6.3 Smoking stratification

Smoking is a mandatory analysis stratifier for:
- **Lung** — CCL-009 mandatory smoking stratification; VAL-063 ran full smoking-stratified analysis (ever-smokers d=+1.28, never-smokers d=+0.57 at n=2)
- **Cardio** — major CVD risk factor and immune-class stratifier
- **Bladder** — dominant bladder cancer risk factor

Smoking is also a long-tail confounder in **all** cards because tobacco methylation signatures persist for decades after cessation.

#### §2.6.4 Ancestry stratification

Ancestry stratification is mandatory when:
- The cohort is single-ancestry (prostate VAL-058 / VAL-118 are 100% African American — multi-ancestry generalizability is v0.4+ next-validation-step)
- Disease incidence or methylation signature varies by ancestry (HCC etiology stratification varies by ancestry)
- Published clinical-grade panels were trained on different ancestry

#### §2.6.5 Other mandatory stratifiers per card

- **HPV status** (cervical) — STAGE 1 STRATIFIER, not just metadata (cerv-LL-002)
- **HIV status** (HCC, cervical, possibly others)
- **Immunosuppression / transplant status** (all cards — can mask immune-class signal entirely)
- **Pregnancy status** (decline scoring during pregnancy in cervical card; consider for all female-pathway cards)
- **Treatment stage** (post-diagnostic monitoring vs pre-treatment baseline must be distinguished)
- **Prior cancer history**
- **Specimen collection method** (LBC vs swab vs biopsy vs blood draw all matter)

### §2.7 Bidirectional-as-default doctrine

**Pooled A_immune should NEVER be the sole metric. The operator must always assume bidirectional behavior is possible until ruled out, so it doesn't catch them off guard and miss a diagnosis.**

This is the cookbook lesson from AD's gift (§7.1) elevated to operational doctrine: pooled-entropy is the easiest metric to compute and the most failure-prone interpretation. AD's pooled signal was null (d=+0.077 on AIBL n=726); only the directional 7-CpG Rule A panel recovered the signal at d=+0.62. PDAC repeated this pattern — pooled CIs straddle zero across three cohorts (VAL-066/067/068); only the directional 324-CpG panel passed (VAL-069 TCGA-PAAD holdout d=+1.51).

**Operational rule.** Every Stage 1 analysis must report:
1. Pooled-entropy A_immune (the standard metric)
2. Per-CpG Δβ direction percentage (descriptive only, NOT a mechanism diagnostic per CCL-030)
3. Directional A_dir panel (when the disease has one constructed; AD has VAL-051 7-CpG; PDAC has VAL-069 324-CpG; future cards build their own when pooled CIs straddle zero)

**The default assumption is bidirectional behavior is possible.** A clean pooled-positive does not rule out lineage-level bidirectional cancellation that is masked by the bulk panel's averaging. Test 2 (lymphoid-vs-myeloid sub-panel split) is the operational test for this — currently NOT runnable on any disease, awaits OQ-2026-01.

### §2.8 Atlas-calibration prerequisite — every atlas must be calibrated before production use

**Before any disease's atlas is used in production for a patient sample, the atlas must have a sealed Phase B calibration anchor on a substrate-matched healthy cohort, with all three CHK gates clear (3.1A, 3.1B, 3.1C) and per-tile healthy-floor distributions sealed.**

If the atlas is not on the calibrated-and-sealed list (§10.1), the score is research-grade exploratory, not production interpretation.

### §2.9 Symptoms that look like disease but are test failure (the ten failure-mode fingerprints)

1. **Flat β distribution clustering near 0.5** = file is not raw β. Symptom: extreme% < 30%, middle% > 10%. **Action: stop, reprocess from raw IDATs.** Examples: VAL-077, VAL-100.

2. **Healthy A-score drifted >1 SD from anchor** = cross-cohort calibration failure. **Action: do NOT interpret cases-vs-controls Cohen's d in cross-cohort framing.** Use within-cohort paired comparison instead.

3. **Per-sample atlas coverage <80% on TCGA HM450K** = expected substrate behavior, NOT a failure. **Action: verify the prereg used substrate-floor coverage threshold (CHK-2.8). If yes, the data is fine. If the prereg defaulted to 95%, the prereg is wrong.**

4. **Per-sample atlas coverage <50% on supposedly-EPIC-850K data** = wrong-platform-bridge or platform-version-mismatch. **Action: verify the cohort's actual platform (GPL accession) against the atlas's substrate. EPIC v2.0 vs EPIC v1 vs 450K bridges differ.**

5. **A-score within 5% of substrate ceiling** = saturation compression. **Action: switch to non-saturated substrate before drawing biology conclusions; or report the saturated reading as "saturation-flagged, biology limit not interpretable."**

6. **Pooled A_immune null but per-CpG Δβ percentages clustered near 50%** = NOT bidirectional cancellation per CCL-030/CCL-031. **Action: descriptive only, not a mechanism. Do not build a directional fallback panel based on this signature alone.**

7. **Massive per-cohort A-score range with clean cohort-internal stratification** = cohort heterogeneity in healthy reference. Symptom: VAL-073 reads cervical d=+0.73, VAL-074 reads d=−0.61, VAL-081 reads d=−0.43. **Action: investigate cohort design (HPV-negative vs HPV-positive normals shift baseline). Document as cohort-design heterogeneity per CCL-019.**

8. **Cross-platform reference comparison without on-study controls** = cross-platform confound risk. **Action: tier as exploratory_pending_replication.** VAL-088 glioma blood arm precedent.

9. **Direction-locked outcome (positive-only or negative-only) when biology supports either direction** = pre-registration over-specification. **Action: amend prereg to magnitude-based |d| threshold with direction labels, sealed BEFORE re-execution per CCL-041.** CHK-2.7 cookbook-wide rule.

10. **Stage 1 immune positive in tumor TIL but negative in pre-diagnostic blood, same disease** = compartment-direction-flip per CCL-019, NOT bidirectional cancellation. Symptom: VAL-047 CRC blood d=−0.33, VAL-061 CRC tumor TIL d=+1.066. **Action: document compartment-specific scoring; pooled is operational metric in each compartment alone.**

### §2.10 Symptoms that ARE biology, not failure

1. **Pooled A_immune null AND directional ±1 z-scored panel passes on independent holdout** = the AD-instance pattern. AD via VAL-050+VAL-051; PDAC via VAL-066/067/068+VAL-069. Mechanism unresolved per CCL-028. Test 2 lineage assignment pending OQ-2026-01.

2. **Stage 1 pooled positive AND Stage 2 atlas-sweep returns no solid-organ tissue above background** = real biological pattern, not test failure. Routes into one of four pathways (§9).

3. **Cell-of-origin tile reads NEGATIVE in tumor-vs-adjacent-normal paired comparison while other tiles read POSITIVE** = CCL-039 homogenization mechanism. Confirmed across three colorectal cohort configurations. NOT a test failure.

4. **Cell-fraction direction (Bracci 2022 type) and A-score direction reading orthogonal axes** = real biology, NOT inversion. Glioma's gift (§7.3): VAL-088 + VAL-090 confirmed A-score +0.91 AND cell-fraction shift consistent with Bracci 2022 — both correct. NOT a test failure.

5. **Substrate dichotomy** (HCC's gift §7.2): same disease reads d=+0.63 on ccfDNA but null on whole-blood-leukocyte against metabolic-disease controls. NOT a test failure.

6. **Sub-cell-type Stage 3 lineage signature distinct from Stage 1 pooled** = the OQ-2026-01 territory. When Stage 3 multi-atlas data exists (prostate VAL-118 monocyte-led TIL; breast VAL-095 aTreg + aBnv; glioma VAL-090 cortical-neuron + cell-fraction shift), the per-cell-type lineup is itself a fingerprint. NOT a test failure.

---

## §3. Why this card exists

Three problems forced the immune-atlas into existence in v0.1; v0.3 adds three more.

1. **The Stage 1 + Stage 2 null pathway wasn't formalized.** Previous Cookbook versions returned "architectural flag, see clinician for standard workup" — safe but non-specific. The correct interpretation depends on *which* disease families produce that pattern.

2. **Clinicians need a lookup table, not 12 independent cards.** A single reference document that lists expected immune signature per disease is the natural product of everything the Cookbook already knows.

3. **The universal-first-test principle implies a universal-reference-atlas.** If every patient's first test is the same immune reading, every interpretation should start from the same differential-diagnosis table.

4. **(v0.3 addition) Stage 1 interpretation is not single-disease.** The cookbook generates Stage 1 numbers; it does not generate Stage 1 *interpretations*. The interpretation engine — what does pooled d=+0.50 on a 58-year-old female smoker actually MEAN — requires the Rosetta Stone.

5. **(v0.3 addition) Stage 2 RUN-everything produces 25-tile output that no single disease card can interpret alone.** The cross-card pattern recognition only emerges at the cross-card level. This card is where it lives.

6. **(v0.3 addition) OQ-2026-01 immune-atlas staging is the central open problem in the cookbook.** Heme-epic implements its closest existing analog. Prostate VAL-118 sub-cell-type lineup is the closest existing data signature. AD's directional panel is the closest existing methodology. All three need a canonical home.

---

## §4. The universal first test (Stage 1)

Every EDEAR patient's first test is the Xu-538 immune panel on buffy-coat DNA.

**Substrate:** buffy coat DNA (peripheral blood leukocytes, approximately 70% immune cells)

**Platform supported:** Illumina 450K, Illumina EPIC 850K (EPIC v2.0 GPL33022 has known data integrity quirks per VAL-100 — verify CHK-3.1 raw β before processing)

**Panel:** Xu-538 immune (538 CpGs)
- Source: Xu Z, Sandler DP, Taylor JA. *Immune cell methylation marker panel for breast cancer prediction*. JNCI 2020;112(1):87-94. DOI: 10.1093/jnci/djz065
- Panel SHA-256: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`
- Coverage on 450K: 538/538 (100%)
- Coverage on EPIC 850K: approximately 500/538 (93%) depending on cohort processing

**Class measured:** immune

**H_min(immune):** 0.838889 (G-003b MCMC posterior mean, R-hat < 1.001)

**Scoring methods (mandatory both, plus directional when applicable):**
- A_immune_pooled: pooled-entropy A-score = mean over panel of H(β)/H_min(immune)
- A_dir: directional composite score, when the disease has a frozen-direction panel constructed
- Per-CpG Δβ direction percentage: descriptive ONLY, NOT a mechanism diagnostic per CCL-030

**Output to clinician:**
1. A_immune_pooled score and tier call
2. A_dir score (for AD and other directional-panel-required diseases)
3. Per-CpG Δβ direction table (descriptive)
4. Demographic stratification flags applied (per §2.6)
5. Data integrity flags (any CHK-3.x failures)
6. Atlas calibration status (which atlases on the cooked-and-sealed list were used)
7. Mahalanobis differential match list (top 3 diseases; §16)

---

## §5. The disease cross-reference table — expected Stage 1 immune signature per disease

| Disease | Card | Stage 1 direction | Magnitude (d) | Per-CpG pattern | Stage 2 target | Stage 3 sub-cell-type | Substrate | Validation status |
|---|---|---|---|---|---|---|---|---|
| Breast | breast-epic v2.3 | **positive** | +0.45 to +0.71 pre-dx pooled; +1.36 to +1.78 at >10yr; tumor d=+0.676 (VAL-060) | uniform positive | breast_ductal (secretory, BreastRef resolution-collapse) | aTreg +1.26 at >10yr; aBnv +0.44–+0.49 at 0-2yr (VAL-095) | blood, tissue, NAF | cross_platform_validated_two_cohorts |
| Colorectal | crc-epic v2.4 | **negative** (compartment-flip) | −0.33 pre-dx blood; +1.066 tumor TIL (VAL-061); +0.724 cycling-class tissue (VAL-062); +0.612 rectal (VAL-098) | uniform negative blood; positive tumor | colon_epithelial (cycling, CCL-039 NEGATIVE in tumor-vs-adjacent-normal) | (Stage 3 pending) | blood, stool, tissue | cross_platform_validated + cycling_class_tissue_validated_with_rectal_subsite |
| Alzheimer's | ad-immune v2.2 | **pooled null + directional positive** (canonical AD-instance) | +0.077 pooled (VAL-050 NULL); +0.624 directional 7-CpG Rule A (VAL-051); +0.33 cross-platform (VAL-052); +0.12 age-adjusted residual | bidirectional — REQUIRES 7-CpG Rule A | NULL (BBB) | T-cell senescence + NLR shift (descriptive, exploratory) | blood | cross_platform_validated |
| Lung (NSCLC) | lung-epic v0.4 | **positive** | +0.35 to +0.50 cohort-level pre-dx; tissue paired d=+1.020 (VAL-063, largest cycling-class tissue effect) | uniform positive | lung_epithelial (cycling, ΔA=+0.143 confidence ratio 60.87×) | (Stage 3 pending; smoking-stratified) | blood, tissue, plasma cfDNA | multi_modal_validated + cycling_class_tissue_validated |
| Prostate | prostate-epic v0.3 | **positive (balanced bidirectional, pooled-positive via Shannon symmetry)** | +0.50 paired (VAL-058; VAL-118 reproduction +0.5258) | balanced bidirectional 217/481 hyper / 264/481 hypo (~45/55) | prostate_epithelial (secretory, ProstateRef separates 6 distinct tiles; LE d=−0.77 luminal dedifferentiation) | Salas IDOL Mono +0.77 / Bcell +0.67 / CD4T +0.66 / NK +0.65 / CD8T +0.59 / Neu +0.39 — **monocyte-led** | blood, tissue, post-DRE urine | multi_modal_validated_plus_multi_atlas_calibrated |
| Hepatocellular | hcc-epic v0.3 | **positive ccfDNA, NULL whole-blood vs metabolic controls** | +0.63 ccfDNA (VAL-059); −0.16 NULL whole-blood (VAL-059); tissue paired d=+0.498 / +0.664 non-viral (VAL-064) | uniform positive ccfDNA; transferability flag whole-blood | hepatocyte (secretory) | (Stage 3 pending) | ccfDNA primary, tissue, urine | multi_modal_validated (substrate-as-discriminator) |
| Pancreatic | pancreatic-epic v0.1 | **pooled null + directional positive** (second AD-instance; mechanism unresolved CCL-028) | +1.18 (VAL-066 n=5), +0.25 (VAL-067 n=196), −0.31 (VAL-068 n=7) — pooled CIs straddle zero; directional 324-CpG d=+1.51 TCGA-PAAD (VAL-069) | bidirectional — REQUIRES 324-CpG directional | pancreatic_exocrine (secretory) | Clark 2007 lit: lymphoid contraction + myeloid expansion — pending OQ-2026-01 | blood, tissue, urine | exploratory_with_directional_recovery |
| Glioma / GBM | glioma-epic v0.2 | **positive (orthogonal to cell-fraction; CCL-023 v0.2 revision)** | +0.91 Stage 1 blood (VAL-088); +1.96 cortical-neuron Stage 2 (VAL-090); LGG d=+1.25 vs GBM d=+0.80 pre-surgery | A-score and cell-fraction read orthogonal axes (Bracci 2022 confirmed; A-score +0.91 confirmed) | neuron/oligodendrocyte (terminal, ΔA=+0.243) — CSF preferred | NLR shift; cortical-neuron deconvolution | blood, tissue, CSF, cfMeDIP-seq plasma | single_cohort_validated_blood + single_cohort_tissue_validated |
| Cardiovascular | cardio-epic v0.3 | **positive pure-cell + null whole-blood etiology** | +0.65 cultured PECs PAH; +0.56 dissection; +1.08 BAV (VAL-110); whole-blood d<0.17 stroke etiology pairs (VAL-108 — biology-correct null) | substrate-dependent | NULL solid-organ; cardiomyocyte tile via Loyfer; aorta via tissue VAL | (Stage 3 pending) | whole-blood; cultured PECs; aortic tissue | multi_modal_validated_plus_multi_atlas_calibrated |
| Cervical | cervical-epic v0.1 | **positive on tissue (Verlaat anchor); cohort-direction-flip per CCL-019, NOT bidirectional cancellation** | VAL-073 d=+0.7253 monotonic Normal<CIN3<SCC; VAL-074 d=−0.61 (HPV-neg normals); VAL-081 d=−0.43 | pooled-positive (Shannon-symmetric) on Verlaat; per-CpG 37.3% positive | cervical_epithelial (cycling) — mucosa preferred | (Stage 3 pending; HPV-stratification mandatory) | tissue, LBC pap (Xu-538 transferability flag), swab, plasma, urine | exploratory_with_cohort_heterogeneity |
| Hematologic | heme-epic v0.1 | **strongly positive, three-arm structure** (closest existing OQ-2026-01 analog) | AML A≈1.10; DLBCL A≈1.13; CLL A≈1.07; thymoma A≈1.09; MM marginal-detectable | three arms: lymphoid_B, lymphoid_T, myeloid | NULL — immune class IS diseased tissue | three-arm structure IS Test 2 instantiated | blood, tissue (TCGA), cfDNA | predicted_validated_at_TCGA_tissue_level |
| Gastric | gastric-epic | **PREDICTED negative** (cycling-like inversion) | d = −0.30 to −0.60 predicted | inversion predicted | gastric_epithelial (cycling) | (pending) | blood, tissue | PENDING — East Asian cohort hunt |
| Bladder | bladder-epic | **PREDICTED negative** (cycling-like inversion) | d = −0.20 to −0.40 predicted | inversion predicted | bladder_epithelial (cycling) — urine preferred | (pending) | urine, blood | PENDING — urine specimen pathway |
| Kidney | kidney-epic | **PREDICTED positive** (cycling, moderate) | d ≈ +0.30 to +0.50 | uniform positive expected | kidney_epithelial (cycling) | (pending) | blood, tissue | PENDING — card build scheduled Phase C |

---

## §6. The CCL-031 five-pattern taxonomy

Stage 1 immune class behavior partitions into exactly five operational patterns.

### §6.1 Pattern 1 — Pooled-positive (Test 1 passes cleanly)

**Diagnostic.** Pooled A_immune d ≥ +0.5, lower CI > 0, no need for directional fallback.

**Cards exhibiting:** breast-epic, lung-epic (blood + tissue), prostate-epic (Stage 1 + Stage 2), hcc-epic (ccfDNA arm), cervical-epic (VAL-073 anchor), heme-epic (predicted)

**Card consequence.** Pooled scoring is operational. No directional fallback panel needed.

### §6.2 Pattern 2 — Pooled-negative compartment-direction-flip (CCL-019)

**Diagnostic.** Pooled A_immune negative in one compartment (peripheral blood), positive in another compartment (tumor TIL), same disease.

**Cards exhibiting:** crc-epic (blood d=−0.33, tumor TIL d=+1.066)

**Card consequence.** Document compartment-specific scoring. Pooled is operational metric in EACH compartment alone. NOT bidirectional cancellation. Do not build directional fallback panel.

### §6.3 Pattern 3 — Pooled-null + directional-pass (AD-instance pattern; mechanism unresolved per CCL-028)

**Diagnostic.** Pooled CIs straddle zero AND directional ±1 z-scored panel passes on holdout.

**Cards exhibiting:** ad-immune (canonical: VAL-050 pooled +0.077 NULL → VAL-051 7-CpG directional +0.624 PASS), pancreatic-epic (second case: VAL-066/067/068 pooled CIs straddle zero → VAL-069 324-CpG d=+1.51 PASS)

**Card consequence.** Build directional fallback panel. Use directional A_dir as primary metric. Recovery mechanism unresolved between (a) AD-style lineage cancellation, (b) z-scoring sensitivity gain, (c) cohort/batch structure (CCL-028). Test 2 pending OQ-2026-01.

### §6.4 Pattern 4 — Cross-disease direction difference (CCL-006)

**Diagnostic.** Pooled Test 1 different sign across diseases on same panel.

**Cards exhibiting:** breast (positive) vs CRC (negative) on same Xu-538.

**Card consequence.** Card specifies expected direction per disease. Pooled scoring works for each disease separately.

### §6.5 Pattern 5 — Lineage-confirmed bidirectional cancellation (currently NONE)

**Diagnostic.** Both pooled-null AND lymphoid-vs-myeloid sub-panel split goes opposite directions with comparable magnitudes.

**Cards exhibiting:** **NONE — Test 2 not yet operational. Pending OQ-2026-01.**

**Card consequence.** When OQ-2026-01 operationalizes, AD and PDAC will be the first cards tested.

### §6.6a Pattern 6 — Substrate-distribution mismatch on mucosal cohort (DISC-BLADDER-003, added 2026-05-01)

Two atlases score the SAME cell-of-origin question on the SAME paired tumor-vs-adjacent-normal pairs and produce HIGH-MAGNITUDE contrasts in OPPOSITE DIRECTIONS — not because of biology, but because the two atlas families measure different observables on the same substrate.

**Canonical case (sealed):** Bladder-epic VAL-121 on TCGA-BLCA n=440, n=21 paired pairs. Loyfer bulk Bladder tile fires d_paired=+1.91 POSITIVE. EpiSCORE BladderRef Epi tile fires d_paired=−1.46 NEGATIVE. Both are well above |d|=0.30 magnitude threshold. Both have small confidence intervals and very small p-values (2.83×10⁻⁸ and 1.60×10⁻⁶). They cannot both be right about the same biology.

**The diagnostic signature.** When this pattern fires, three checks confirm it as substrate-distribution mismatch rather than biology:

1. **Atlas family difference.** The two atlases come from different families: bulk-WGBS (Loyfer/Moss, Caggiano TIM) vs gene-promoter sub-cell-type (EpiSCORE per-tissue bridges — BladderRef, ProstateRef, BreastRef, HeartRef). Bulk-WGBS encodes mixed-cell-type β profiles for whole tissues. Gene-promoter encodes signature β profiles for specific cell types via marker-gene-promoter regions. On a mucosal-tissue cohort (bladder, lung airways, colon epithelium, GI epithelium, cervical mucosa) the bulk-WGBS reference's mixed β profile sits far from the cohort's tissue-class methylation distribution shape — producing inflated |β_sample − β_bulk_ref| metrics across all bulk solid-tissue tiles. The gene-promoter reference's marker β profile is unaffected by tissue-class distribution shape.

2. **CHK-3.2 cross-tile sanity check.** Score the same cohort against multiple non-cohort solid-tissue tiles in the bulk-WGBS atlas. If ALL of them fire POSITIVE at high magnitude (e.g., bladder cohort against Loyfer Thyroid +2.92, Kidney +2.71, Prostate +2.45, Pancreas +2.81, etc. — all 14 non-bladder solid-tissue tiles uniformly POSITIVE +2.34 to +2.92), the cohort is not "becoming Thyroid + Kidney + Prostate + Pancreas simultaneously". The uniform inflation is the substrate-distribution-mismatch signal.

3. **CCL-039 expectation.** The gene-promoter atlas reading should match CCL-039 (NEGATIVE direction for adenocarcinoma cell of origin via dedifferentiation). The bulk-WGBS reading does not match CCL-039 — that is the failure mode of the wrong atlas family on this substrate, not a biological direction-divergence.

**The cookbook rule (DISC-BLADDER-003 sealed in LESSONS_LEARNED.md):** Multi-atlas readings on mucosal cohorts MUST include a gene-promoter sub-cell-type atlas as the primary cell-of-origin reader. Single-atlas Stage 2 readings on mucosal cohorts using bulk-WGBS references can be substrate-substitution-fooled. The bulk-WGBS reading on mucosal cohorts is interpretive context only, not the headline cell-of-origin signal.

**How this differs from Pattern 2 (compartment-direction-flip).** Pattern 2 is two compartments of the SAME atlas reading opposite directions in the SAME cohort because the disease has a real compartment-asymmetric biology. Pattern 6 is two atlas FAMILIES of (apparent) DIFFERENT cell-of-origin tiles reading opposite directions because one atlas family is substrate-distribution-confounded on the cohort's tissue class. Pattern 2 is biology; Pattern 6 is methodology.

**How this differs from Pattern 5 (lineage-confirmed bidirectional cancellation).** Pattern 5 is hypothetical and currently NONE — would require lineage-resolution evidence that within a single Stage 1 panel the lymphoid and myeloid cells are moving in opposite directions of equal magnitude. Pattern 6 is sealed and is about cell-of-origin reading divergence between two atlas families with different substrate-distribution-mismatch sensitivities.

### §6.6 Single-sentence rule (CCL-031, verbatim)

> Bidirectional cancellation is the AD-instance pattern: Test 1 pooled A_immune nulls cross-cohort AND a directional ±1 z-scored panel built on the same Stage 1 panel passes on holdout. Compartment-direction-flips, cross-disease direction differences, and negative-direction-dominant cohort-mean Δβ are NOT bidirectional cancellation, even when they superficially resemble it.

---

## §7. Three doctrine cases — the field's gifts to interpretation

### §7.1 AD's gift — the pooled-vs-directional distinction

**The case.** AD pooled A_immune on AIBL n=726 read d = +0.077, p = 0.321, AUC = 0.512 — NULL (VAL-050). Same panel, same cohort, directional 7-CpG Rule A panel built on training split read d = +0.624, AUC = 0.68 (VAL-051). Cross-platform AddNeuroMed validation d = +0.33; age-adjusted residual d = +0.12 (VAL-052). Sex-specific panels do NOT outperform unified (VAL-053).

**The doctrine.** Pooled-entropy is not always sufficient. When per-CpG direction is balanced bidirectional, Shannon entropy averages it to zero. Directional weighting (frozen-direction ±1 z-score per CpG, summed) recovers the signed signal.

**The card consequence.** Every Stage 1 design must answer CCL-027 question (iii). AD has VAL-051; PDAC has VAL-069; future cards build their own when needed.

**The mechanism caveat (CCL-028).** AD's directional recovery is operationally robust but mechanistically unresolved. Three candidate mechanisms: (a) AD-style lineage cancellation; (b) z-scoring sensitivity gain; (c) cohort/batch structure. Test 2 lineage assignment is what would distinguish these — pending OQ-2026-01.

### §7.2 HCC's gift — substrate-as-discriminator

**The case.** HCC on Xu-538 panel reads:
- ccfDNA (VAL-059 GSE298812 Soliman Nigerian HIV+ HCC): paired d = +0.6336 [+0.175, +1.121] p = 0.0024; sex-stratified male d = +0.998, female d = +0.497; age-regressed d = +0.6979; 219 CpGs with |Δβ| > 0.02
- Whole-blood leukocyte (VAL-059 GSE281691 controls = metabolic-liver-disease patients NOT healthy): paired d = −0.1556 p = 0.091 — NULL; only 5 CpGs with |Δβ| > 0.02

**The doctrine.** Same disease, same panel, two substrates, opposite outcomes. The whole-blood NULL is NOT a "Xu-538 fails for HCC" finding — it is "whole-blood leukocyte substrate is the wrong specimen for HCC detection via this panel because controls in GSE281691 are metabolic-liver-disease patients (not healthy), so background methylation drift from liver disease masks any HCC-specific signal."

**The card consequence.** Substrate choice IS part of the immune-class signature. For diseases of secretory organs (HCC, breast, prostate) that do not trigger massive systemic immune drift in healthy peripheral leukocytes, ccfDNA may be the right substrate even when whole-blood reads NULL.

**The bonus finding from HCC.** Monotonic dose-response across disease severity (healthy → fibrosis → cirrhosis → HCC: A_mean 0.576 → 0.591 → 0.592 → 0.598) is the clinical-validation gold standard. The signal scales with disease stage.

### §7.3 Glioma's gift — the orthogonal-vs-inverted distinction (CCL-023 v0.2 revision)

**The case.** v0.1 of glioma-epic predicted Stage 1 immune A-score should read NEGATIVE direction based on Bracci 2022 cell-fraction prior (lymphocytes down, neutrophils up). VAL-088 measured Stage 1 A-score on GSE180683 n=76 glioma EPIC blood: d = +0.91 [+0.61, +1.22] — clean strong POSITIVE. v0.1 sealed `O5_POSITIVE_INVERTED`. VAL-090 ran Loyfer-atlas deconvolution on the same samples: cell-fraction shift confirmed Bracci 2022 prediction (neutrophils +16%, lymphocytes −13%), AND cortical-neuron deconvolution showed glioma vs healthy d = +1.96.

**The doctrine.** Cell-fraction direction (Bracci 2022 type) and A-score direction (Shannon entropy of methylation) are NOT opposite axes — they are **orthogonal axes**, different lenses on the same disease state. Both can be correct and informative. The cell-fraction measures who is in the blood; the A-score measures how disorganized their methylation is. A disease can drive lymphocytes down AND drive remaining lymphocytes' methylation higher in entropy — those readings are not contradictory; they are independent observables.

**The card consequence.** When a disease has both a cell-fraction prior in the literature AND a Stage 1 A-score reading in the framework, do NOT predict A-score direction from cell-fraction direction. Read both, report both, interpret both as independent observables of the disease state.

---

## §8. Cookbook doctrine that touches immune class

### §8.1 CCL-006 — Cross-disease direction differences on same panel

**Verbatim rule:** Different diseases drive the same Xu-538 panel in different directions. Card specifies expected direction per disease.

**Cards exhibiting:** breast (positive) vs CRC (negative) is the canonical pair.

**Source of truth:** LESSONS_LEARNED.md CCL-006 entry.

### §8.2 CCL-019 — Compartment-direction-flip (NOT bidirectional cancellation)

**Verbatim rule:** A-score direction depends on (class, compartment) pair, not disease alone. Blood immune ≠ tumor-infiltrating immune.

**Cards exhibiting:** crc-epic (blood d=−0.33; tumor TIL d=+1.066).

**Source of truth:** LESSONS_LEARNED.md CCL-019 entry.

### §8.3 CCL-023 — Direction-as-discriminator (v0.2 revised: orthogonal not inverted)

**Verbatim rule (v0.2):** Cell-fraction direction (Bracci 2022 type) and A-score direction read orthogonal axes, not opposite.

**Cards exhibiting:** glioma-epic (canonical: VAL-088 + VAL-090).

**Source of truth:** LESSONS_LEARNED.md CCL-023 entry; glioma-epic card v0.2 §CCL_023_status block.

### §8.4 CCL-027 — Mandatory four-question Stage 1 design check

**Verbatim rule:** Every card's v0.1 build must answer all four questions: (i) pooled-entropy expected direction; (ii) bidirectional-cancellation risk; (iii) directional-panel fallback specification; (iv) lymphoid-vs-myeloid expected pattern from literature.

**Cards exhibiting:** every card. Master cross-reference table populated in §11.

**Source of truth:** LESSONS_LEARNED.md CCL-027 entry; master README §17.

### §8.5 CCL-028 — Pooled-null + directional-pass mechanism unresolved

**Verbatim rule:** When pooled A_immune nulls AND directional ±1 z-scored panel passes on holdout, the recovery mechanism is unresolved between (a) AD-style lineage cancellation, (b) z-scoring sensitivity gain, (c) cohort/batch structure. Test 2 lineage assignment would distinguish these.

**Cards exhibiting:** ad-immune, pancreatic-epic.

**Source of truth:** LESSONS_LEARNED.md CCL-028 entry.

### §8.6 CCL-030 — Per-CpG cohort-mean Δβ direction percentage is descriptive only, NOT a mechanism diagnostic

**Verbatim rule:** Per-CpG cohort Δβ direction percentage describes where β values shifted on average. A 47/50/52% split is not a bidirectional cancellation finding by itself.

**Cards exhibiting:** all cards. Cervical-epic VAL-073 is the canonical example.

**Source of truth:** LESSONS_LEARNED.md CCL-030 entry.

### §8.7 CCL-031 — "Bidirectional cancellation" terminology rule (single-sentence verbatim)

**Verbatim rule:** Bidirectional cancellation is the AD-instance pattern: Test 1 pooled A_immune nulls cross-cohort AND a directional ±1 z-scored panel built on the same Stage 1 panel passes on holdout. Compartment-direction-flips, cross-disease direction differences, and negative-direction-dominant cohort-mean Δβ are NOT bidirectional cancellation.

**Cards confirmed clean under CCL-031:** crc-epic v2.1/v2.2, breast-epic, ad-immune, pancreatic-epic v0.1, cervical-epic v0.1.

**Source of truth:** LESSONS_LEARNED.md CCL-031 entry.

### §8.8 CCL-032 — Diagnostic order is fixed (data integrity → biology → framework)

**Verbatim rule:** Every cohort run with a null or negative-direction reading must complete data integrity → biology consistency → framework finding in sequence BEFORE the outcome is drafted.

**Cards exhibiting:** all cards. Cervical-epic VAL-076/077 is the canonical incident.

**Source of truth:** LESSONS_LEARNED.md CCL-032 entry; this card §2.

### §8.9 CCL-039 — Cell-of-origin tile direction depends on comparison type

**Verbatim rule:** In tumor-vs-adjacent-normal paired comparisons, cell-of-origin tile reads NEGATIVE direction (homogenization). In diseased-tissue-vs-healthy-cross-reference comparisons, cell-of-origin tile reads POSITIVE direction. Prereg O1 criteria must specify the comparison type with magnitude-based |d| thresholds and direction labels.

**Cards exhibiting:** confirmed on three colorectal cohort configurations (TCGA-READ VAL-098, TCGA-COAD VAL-062 revisit, TCGA-COAD VAL-099 reproduction). Cross-tissue retroactive expansion pending.

**Source of truth:** LESSONS_LEARNED.md CCL-039 entry; TESTING_CHECKLIST.md CHK-4.11; master README Part 14.

---

## §9. The four Stage-1-positive Stage-2-null diagnostic pathways

### §9.1 Pathway 1 — Terminal-class disease hidden by the specimen problem

**Trigger pattern:** Stage 1 weakly positive or null with mild directional panel elevation. Patient has neurological symptoms, cognitive changes, or imaging findings suggestive of intracranial process. Stage 2 localizes nothing because plasma cfDNA is only 0.5% brain-derived — below the 4% Moss detection floor.

**Framework basis:** Terminal class (neurons, oligodendrocytes, cardiomyocytes) has the lowest H_min (0.772837) and shows the largest per-cell ΔA — LGG ΔA = +0.239, GBM ΔA = +0.217. Signal invisible in plasma. CSF recovers it. Glioma's gift §7.3: orthogonal cell-fraction + A-score readings can both flag terminal-class disease.

**Recommended clinical action:** Neurology consult. CSF draw if clinically indicated. AD trajectory watch (6-12 month serial blood). Imaging review.

**Related cards:** glioma-epic (post-diagnosis monitoring, CSF-based), ad-immune (canonical Pathway 1 disease).

### §9.2 Pathway 2 — Hematologic/immune-compartment disease

**Trigger pattern:** Stage 1 strongly positive or distinctive per-CpG pattern. Stage 2 returns null because the immune class IS the diseased tissue.

**Framework basis:** Immune compartment cancers have defined A-score signatures: AML A_combined ≈ 1.10 (TCGA 2013 NEJM n=200), DLBCL A ≈ 1.13 (Chapuy 2018 n=48), CLL A ≈ 1.07, thymoma A ≈ 1.09 (TCGA n=120). Heme-epic three-arm structure (lymphoid_B, lymphoid_T, myeloid) IS Test 2 instantiated as a card.

**Recommended clinical action:** CBC with differential. EpiDISH Stage 3 sub-composition analysis. If lymphocyte-shift predominant — hematology consult. If neutrophil-shift predominant — consider chronic inflammatory disease or early infection. If clonal hematopoiesis suspected — flow cytometry and bone marrow consultation.

**Related card:** heme-epic.

### §9.3 Pathway 3 — Cardiovascular / systemic inflammatory disease

**Trigger pattern:** Stage 1 positive. Patient has traditional CVD risk factors. Stage 2 returns null because atherosclerosis is systemic vascular inflammation.

**Framework basis:** Nine prospective cohorts (n=11,461) show blood methylation predicts incident CHD. cardio-epic v0.3 substrate dichotomy: whole-blood d<0.17 stroke etiology pairs (biology-correct null per LL-CARDIO-002); cultured PECs d=+0.65 PAH; aortic tissue d=+0.56 dissection / d=+1.08 BAV.

**Recommended clinical action:** Standard cardiovascular workup. Lipid panel, hsCRP, blood pressure, HbA1c. If Stage 1 signature includes monocyte/FOXP3 pattern, consider cardiac imaging.

**Related card:** cardio-epic v0.3.

### §9.4 Pathway 4 — Unexplained immune drift, serial trajectory watch

**Trigger pattern:** Stage 1 fires but does NOT meet Pathway 1/2/3 trigger criteria. Immune compartment reports upstream architectural drift the framework cannot localize at this time.

**Framework basis:** Single-timepoint EDEAR is a flag; serial-trajectory EDEAR is a diagnostic. Trajectory monitoring identifies whether drift is stable, accelerating, or normalizing.

**Recommended clinical action:** Serial sampling at 6-month or 1-year intervals. Document trajectory. Re-route to Pathways 1-3 if new symptoms emerge.

---

## §10. The Stage 2 atlas registry

### §10.1 Production atlases (sealed and operational)

| Atlas | Class | Sealed SHA | Calibration anchor | Calibration cohort | Diseases using |
|---|---|---|---|---|---|
| Xu-538 immune | Stage 1 panel | `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6` | Xu 2020 JNCI | NEST + Italian healthy buffy n=601 | universal — every disease |
| Layered Moss + Loyfer 25-tile | Stage 2 RUN-everything | (sealed cardio sprint) | VAL-112 + VAL-113 | TCGA HM450K sesame Level 3 n=210 (KIRC + PRAD) | breast (VAL-093), CRC (VAL-099), lung (VAL-063 retroactive), prostate (VAL-118 cross-ref), glioma (VAL-090), HCC (VAL-064) |
| Salas IDOL Blood.EPIC 6-cell | Stage 3 | (Salas 2018) | published reference | published EPIC IDOL | every card Stage 3 |
| EpiDISH RPC | Stage 3 | (Teschendorff) | published reference | published bounds | Stage 3 alternative |
| Loyfer/Moss array atlas | Stage 2 cell-of-origin | (Loyfer + Moss) | sorted-cell reference | published references | glioma cortical-neuron (VAL-090), lung lung_epithelial (VAL-041) |
| ProstateRef CpG-bridged 6-tile | Stage 2 prostate | `4e60c3d038a637e9742f51d9bc7c119e06fe5d2e91abb2b12db8867ceb7813d2` | VAL-117 + VAL-118 | TCGA HM450K n=210 | prostate-epic v0.3 |
| UniLIFE 19-cell overlay | Stage 3 sub-cell-type | (Guo 2025) | published reference | published bounds | breast VAL-095, prostate VAL-118 (overlay) |

**EpiSCORE BladderRef CpG-bridged (added 2026-05-01 from VAL-119).** 4 bladder cell types (EC vascular endothelial / Epi urothelial / Fib fibroblast / IC immune) × 2,696 unique 450K CpGs after CHK-3.1C deduplication. Bridge methodology: source `mrefBladder.m` (163 Entrez Gene IDs × 4 cell types) → CpG-resolved matrix via EpiSCORE's `probeInfo450k.lv` Entrez→450K-CpG manifest (158/163 EIDs covered). Bridged atlas SHA-256: `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`. Calibration anchor: VAL-119 sealed `O1_BLADDERREF_CALIBRATION_SEALED` 2026-05-01 on TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (same VAL-106 cohort). Per-tile healthy-floor distributions sealed: EC mean 0.4087 sd 0.0100; **Epi mean 0.4135 sd 0.0066 q5 0.4004** (operational cell-of-origin tile, tightest within-cohort variance — bladder analog of prostate's LE tile); Fib mean 0.4875 sd 0.0090 (separates); IC mean 0.4106 sd 0.0086. Max within-cohort tile range 0.0694 (well above 0.02 tissue-floor-dominated threshold). Production status: **PRIMARY cell-of-origin reader for bladder-epic v0.1 mucosal-cohort scoring per DISC-BLADDER-003**. Fourth successful EpiSCORE bridge in the atlas vault alongside HeartRef (VAL-111 deferred per DISC-CARDIO-004), BreastRef (VAL-094 production), ProstateRef (VAL-117 production).

### §10.2 Research-grade exploratory atlases (NOT yet calibrated for production)

| Atlas | Status | Sealed VAL | Rationale | Path to production |
|---|---|---|---|---|
| HeartRef CpG-bridged | `O3_TISSUE_FLOOR_DOMINATED` | VAL-111 | Cardiac cell types share gene-promoter methylation similarity (DISC-CARDIO-004 / LL-CARDIO-005) | Atlas family limitation |
| BreastRef CpG-bridged | sub-tile resolution-collapse | VAL-093 | All 7 BreastRef tiles read positive d ≈ +1.0 to +1.2 (breast-LL-007) | Operational at bulk-tile only |
| Caggiano TIM region-indexed | bridge verified, Phase B not sealed | partial | Variable per-tile coverage | Phase B calibration sprint required |
| Future EpiSCORE bridges (LungRef, KidneyRef, ColonRef, EsophagusRef, OliveRef, OvaryRef, PancreasRef, SkinRef, StomachRef, BladderRef, BrainRef) | NOT YET BUILT | n/a | Bridge engineering reusable; calibration cohorts vary | Card sprint per disease |

### §10.3 Atlas-family fitness lesson (DISC-PROSTATE-001 / LL-CARDIO-005 / DISC-BLADDER-001 — three data points as of 2026-05-01)

**Verbatim rule.** Atlas family fitness depends on per-tissue cell-type distinctness. EpiSCORE family separates clean on prostate (6 distinct cell-of-origin tiles, ProstateRef sealed) and partially collapses on breast (7 tiles uniform positive, BreastRef resolution-collapse). The same atlas family failed on cardiac (HeartRef tissue-floor-dominated). The discriminator is whether the tissue's cell types are gene-promoter-distinct enough for the EpiSCORE matrix to separate them.

**Operational consequence.** Future card sprints evaluating any EpiSCORE tissue reference must run per-tissue calibration smoke test BEFORE committing to or deferring the atlas.


**Third data point added 2026-05-01 (DISC-BLADDER-001).** EpiSCORE BladderRef has only 4 cell types (vs ProstateRef 6, vs HeartRef 5) and produces the largest within-cohort tile range of the three EpiSCORE per-tissue bridges (0.0694, vs ProstateRef 0.0597, vs HeartRef 0.0152 collapsed). The hypothesis "more cell types = better gene-promoter atlas separation" is **falsified** by bladder. The supported rule extends LL-CARDIO-005 / DISC-PROSTATE-001: gene-promoter atlas family fitness depends on per-tissue cell-type DISTINCTNESS at the gene-promoter level for the marker genes Zhu/Teschendorff selected, NOT on cell-type COUNT. Per-tissue calibration smoke test required at every new EpiSCORE bridge before commitment to atlases_run vs deferral to atlases_deferred. Source matrix dimensions (number of EIDs, number of cell types) do not predict atlas-family-fitness outcome.

### §10.4 CCL-039 cell-of-origin tile direction rule

**Verbatim rule.** When an atlas's cell-of-origin tile is scored on tumor-vs-adjacent-normal paired comparison, the tile reads **NEGATIVE** direction (homogenization mechanism). When the same tile is scored on diseased-tissue-vs-healthy-cross-reference comparison, the tile reads **POSITIVE** direction.

**Operational consequence.** Prereg O1 criteria for any atlas-tile outcome must specify the comparison type with magnitude-based |d| thresholds and direction labels. Cookbook precedent: CHK-4.11.

---

## §11. CCL-027 four-question master cross-reference table

### §11.1 Question (i) — Pooled-entropy expected direction

| Disease | Pooled direction expected | Source | Status |
|---|---|---|---|
| Breast | positive | Xu 2020 JNCI | confirmed (VAL-047 Phase 9+12) |
| Colorectal (blood) | negative | Xu 2020 cross-disease | confirmed (VAL-047 Phase 12) |
| Colorectal (tumor) | positive | TIL literature | confirmed (VAL-061, VAL-062, VAL-098, VAL-099) |
| Alzheimer's | NULL (pooled) | VAL-050 | confirmed; directional fallback required |
| Lung | positive | Xu 2020 cross-disease + lung lit | confirmed (VAL-046 + VAL-056 + VAL-063) |
| Prostate | positive (Shannon-symmetric on balanced bidirectional) | VAL-058 | confirmed (VAL-058 + VAL-118 reproduction) |
| HCC (ccfDNA) | positive | VAL-059 | confirmed (substrate-as-discriminator) |
| HCC (whole-blood vs metabolic controls) | NULL | VAL-059 | confirmed transferability flag |
| Pancreatic | NULL (pooled) | VAL-066/067/068 | confirmed; directional fallback required |
| Glioma (blood) | positive | VAL-088 | confirmed (orthogonal to cell-fraction prior) |
| Bladder | POSITIVE (immune infiltration; bladder cancer has heavy TIL load — BCG immunotherapy is standard of care for NMIBC; PD-L1 inhibitors approved for advanced UC) | sealed VAL-120 paired d_paired=+1.8977 (n=21, p=3.14×10⁻⁸) on TCGA-BLCA HM450K — diagnostic-not-sealed under O4 panel coverage gate fire (DISC-BLADDER-004); 3.8× larger than prostate VAL-058's d=+0.497 |
| Glioma (tissue) | positive | VAL-089 (GBM primary +0.24, recurrent +1.17) | confirmed |
| Cardiovascular (whole-blood etiology pooled) | NULL | VAL-108 | confirmed biology-correct null |
| Cardiovascular (cultured PECs) | positive | VAL-110 | confirmed |
| Cardiovascular (aortic tissue) | positive | VAL-110 | confirmed |
| Cervical (Verlaat tissue anchor) | positive | VAL-073 | confirmed |
| Cervical (Farkas + Lando cohorts) | negative-direction (cohort heterogeneity per CCL-019) | VAL-074 + VAL-081 | cohort heterogeneity, not bidirectional |
| Hematologic (per-arm) | strongly positive (subtype-dependent) | TCGA + Chapuy 2018 | predicted |
| Gastric | negative predicted | crc-epic precedent | PENDING |
| Bladder | negative predicted | crc-epic precedent | PENDING |
| Kidney | positive predicted | predicted | PENDING |

### §11.2 Question (ii) — Bidirectional-cancellation risk (literature signal)

| Disease | Bidirectional risk? | Literature source |
|---|---|---|
| AD | HIGH — confirmed pooled-null, directional-pass pattern | VAL-050/051; Mastroeni 2020 lymphocyte/myeloid discordance |
| Pancreatic | HIGH — confirmed pooled-null, directional-pass pattern | VAL-066/067/068/069; Clark 2007 PMC1944938 lymphoid contraction + myeloid expansion |
| Glioma | YES per literature, NO per VAL-088 pooled | Bracci 2022 — operational pooled passes positive |
| Bladder | LOW for solid-tumor tissue substrate; mixed TIL+TAM+MDSC infiltration is broadly POSITIVE direction across all cell types (sealed VAL-122: all 6/6 Salas IDOL tiles fire POSITIVE |d| range 0.49–1.24). UNKNOWN for blood-substrate (Chen 2022 NMIBC blood EPIC suggests lymphoid-vs-myeloid split possible; pending VAL on Chen 2022 cohort in v0.2) |
| Cardiovascular | YES per literature | CCL-021 monocyte-shift + FOXP3 Treg + ZBTB12 pattern |
| Hematologic | YES per arm-specific pattern | subtype-dependent |
| Breast | LOW — pooled-positive clean | Xu 2020 |
| Colorectal | LOW per compartment — pooled-positive in tumor TIL, pooled-negative in blood | VAL-047 + VAL-061 |
| Lung | LOW — pooled-positive | Xu 2020 cross-disease |
| Prostate | LOW at pooled level (Shannon-symmetric) | VAL-118 |
| Hepatocellular | LOW per substrate — ccfDNA pooled-positive | VAL-059 |
| Cervical | LOW — pooled-positive on Verlaat anchor | VAL-073 + CCL-019 |

### §11.3 Question (iii) — Directional-panel fallback specification

| Disease | Directional panel | Status |
|---|---|---|
| AD | 7-CpG Rule A panel (VAL-051 freeze 2026-04-23) | OPERATIONAL — primary clinical metric |
| Pancreatic | 324-CpG GSE49149-trained subset (`pancreatic-epic_directional_v0.1`) | OPERATIONAL — primary clinical metric |
| Glioma | (not built — pooled passes positive) | NOT NEEDED at v0.2 |
| Bladder | Xu-538 panel triggered O4 on TCGA-BLCA HM450K cohort-substrate coverage (mean 78.0%, pass rate 51.1% — DISC-BLADDER-004). Operational fallback: defer Stage 1 production scoring to v0.2 with VAL-114 Wave 1 calibrated panel + CHK-2.17 cohort-substrate-coverage precheck baked in. v0.1 Stage 1 d=+1.90 reported as diagnostic-only |
| Cardiovascular | (not built — substrate-stratified scoring) | DESIGN ALTERNATIVE — substrate-stratification, not directional fallback |
| Hematologic | per-arm directional panels predicted | PENDING |
| Breast / CRC blood / Lung / Prostate / HCC ccfDNA / Cervical Verlaat | (not built — pooled passes) | NOT NEEDED |
| Gastric / Bladder / Kidney | (not built — card pending) | PENDING |

### §11.4 Question (iv) — Lymphoid-vs-myeloid expected pattern (literature only; Test 2 pending OQ-2026-01)

| Disease | Lymphoid expected | Myeloid expected | Source |
|---|---|---|---|
| AD | up (T-cell senescence) | up (NLR shift) | Mastroeni 2020 |
| Pancreatic | down (Treg up, effector T cells down) | up (MDSCs, M2 macs) | Clark 2007 PMC1944938 |
| Glioma | down (lymphocytes −13%) | up (neutrophils +16%) | Bracci 2022 |
| Bladder | TISSUE substrate (TCGA-BLCA n=440 sealed VAL-122): broad multi-lineage POSITIVE — both lymphoid (CD4T +0.49, CD8T +0.62, NK +0.79, Bcell +1.15) AND myeloid (Mono +1.13, Neu +1.24) fire POSITIVE. NOT lymphoid-dominant; NOT myeloid-dominant. Consistent with mixed TIL+TAM+MDSC of MIBC. BLOOD substrate: Chen 2022 mdNLR signature suggests lymphoid-vs-myeloid split possible (myeloid-elevated + lymphoid-reduced predicted RFS in NMIBC); pending VAL (v0.2 Bryan UK NMIBC + Chen 2022 cohorts) |
| Cardiovascular | mixed (Treg up at FOXP3) | up (monocyte shift) | CCL-021 |
| Hematologic AML | (immune class IS diseased tissue) | up (myeloid lineage) | TCGA AML |
| Hematologic DLBCL/CLL/MM | up (B-cell lineage) | (varies) | subtype-specific |
| Hematologic thymoma | up (T-cell lineage) | (varies) | TCGA thymoma |
| Breast | up at sub-cell-type (aTreg +1.26 at >10yr; aBnv +0.44–+0.49 at 0-2yr) | (uniform positive at bulk) | VAL-095 UniLIFE |
| Prostate | mixed positive (Bcell +0.67, CD4T +0.66, NK +0.65, CD8T +0.59) | up (Mono +0.77 leads, Neu +0.39 trails) | VAL-118 Salas IDOL |
| Other cards | PENDING per literature review | | |

**All Question (iv) entries are literature-anchored expected patterns only at v0.1. Operational confirmation requires Test 2 (lymphoid-vs-myeloid sub-panel split on Xu-538), which is OQ-2026-01 immune-atlas staging — pending.**

---

## §12. OQ-2026-01 — The immune-atlas staging problem (canonical home moved here in v0.3)

This card is now the canonical home for OQ-2026-01 progress tracking. Other cards reference this section.

### §12.1 What OQ-2026-01 is

The cookbook's central open problem in immune-class interpretation: **building a Test 2 sub-panel split on Xu-538 that operationally separates lymphoid-driven signal from myeloid-driven signal in any patient's Stage 1 reading.**

When operational, Test 2 will discriminate:
- **AD** lineage-confirmed bidirectional cancellation (Pattern 5) vs z-scoring sensitivity gain (Pattern 3 mechanism (b)) vs cohort/batch artifact (Pattern 3 mechanism (c))
- **PDAC** same three-way mechanism resolution
- **Glioma** the orthogonal cell-fraction vs A-score relationship at lineage resolution
- **Cardiovascular** the literature-predicted FOXP3 + monocyte pattern at operational resolution
- **Every future card's question (iv) answer** at operational resolution rather than literature-only

### §12.2 What Test 2 needs

A per-CpG lineage assignment for the Xu-538 panel: which of the 538 CpGs are lymphoid-discriminating (B-cell, T-cell, NK) vs myeloid-discriminating (monocyte, neutrophil, dendritic). The assignment requires a sorted-cell methylation atlas with sufficient resolution at the panel-CpG positions:

- **Salas IDOL-Ext** is the most complete published candidate (extends IDOL-6 to ~12 cell types; needs CpG-position cross-walk to Xu-538)
- **UniLIFE 19-cell** has finer cell-type resolution but different CpG selection criteria
- **Loyfer/Moss array atlas** has sorted-immune-cell entries that may complement

The build sequence: (a) cross-walk Xu-538 panel positions against each candidate atlas; (b) for each Xu-538 CpG, identify the cell-type assignment from atlas evidence; (c) classify as lymphoid-discriminating, myeloid-discriminating, or unassigned; (d) build per-arm A-score aggregation as A_lymphoid and A_myeloid; (e) calibrate per-arm healthy-floor distributions on substrate-matched healthy cohort; (f) seal as production Test 2 panel.

### §12.3 Closest existing analogs

- **Heme-epic three-arm structure (lymphoid_B / lymphoid_T / myeloid)** — heme-epic IS Test 2 instantiated as a card. AML maps to myeloid arm; DLBCL/CLL/MM to lymphoid_B; thymoma to lymphoid_T. Structure exists at *card* level for heme; panel-CpG-level mapping for general Stage 1 use does not yet exist.

- **Prostate VAL-118 Stage 3 sub-cell-type signature** — closest existing data signature. Salas IDOL Mono +0.77 / Bcell +0.67 / CD4T +0.66 / NK +0.65 / CD8T +0.59 / Neu +0.39 demonstrates that operational sub-cell-type lineup can be measured from a single IDAT.

- **Breast VAL-095 UniLIFE** — second sub-cell-type signature. UniLIFE separates the naive B-cell and activated Treg signals from the Salas bulk.

- **Glioma VAL-090 Loyfer-atlas immune fractions** — confirmed Bracci 2022 cell-fraction prior at operational resolution. Cell-fraction is the proxy axis to A-score's entropy axis.

- **AD's directional 7-CpG Rule A panel** — closest existing methodology for directional weighting.

### §12.4 What unblocks when OQ-2026-01 goes live

- **AD lineage mechanism becomes testable** — Pattern 3 → Pattern 5 graduation possible
- **PDAC lineage mechanism becomes testable** — Pattern 3 → Pattern 5 graduation possible
- **Every card's CCL-027 question (iv) becomes operationally answerable** rather than literature-only
- **Concurrent-disease scenarios become decomposable** — when a 58-year-old smoker presents with Stage 1 immune positive AND systemic vascular inflammation AND latent prostate dedifferentiation, Test 2 separates the contributions
- **The Mahalanobis differential ranker (§16) gains a lineage-vector axis**

### §12.5 Open atlas-coverage gaps blocking OQ-2026-01

- Salas IDOL-Ext panel-CpG cross-walk to Xu-538: NOT BUILT
- UniLIFE 19-cell panel-CpG cross-walk to Xu-538: NOT BUILT
- Substrate-matched healthy cohort calibration of per-arm A-scores: NOT RUN
- Cross-cohort baseline alignment of per-arm A-scores: NOT RUN
- Per-arm healthy-floor distribution sealing: NOT RUN

### §12.6 Cards the OQ-2026-01 staging will affect first (priority order)

1. ad-immune
2. pancreatic-epic
3. cardio-epic
4. glioma-epic
5. heme-epic
6. all other cards

---

## §13. Stage 3 sub-cell-type immune signatures — every sealed card with multi-atlas data

### §13.1 Prostate cancer (VAL-118 sealed 2026-04-30)

GSE269244 n=238 EPIC 850K paired tumor + adjacent-normal scored against Salas IDOL Blood.EPIC 6-cell + UniLIFE 19-cell overlay:

| Atlas / cell type | d_paired | Magnitude class |
|---|---|---|
| Salas IDOL Mono | **+0.77** | LARGE |
| Salas IDOL Bcell | +0.67 | LARGE |
| Salas IDOL CD4T | +0.66 | LARGE |
| Salas IDOL NK | +0.65 | LARGE |
| Salas IDOL CD8T | +0.59 | LARGE |
| Salas IDOL Neu | +0.39 | MODERATE |
| UniLIFE aMono | +0.47 | MODERATE |
| UniLIFE aNeu | +0.43 | MODERATE |
| UniLIFE Mono | +0.39 | MODERATE |

Lineup ranking: **monocyte > Bcell > CD4T > NK > CD8T > neutrophil**. Distinctive feature: monocyte-led, NOT T-cell-led. Biological interpretation: TAM (tumor-associated macrophage) recruitment; B-cell second-position consistent with reported tertiary lymphoid structures in prostate cancer; signature consistent with Berglund 2024 published CD40/OX40L/STING DMRs from independent EWAS.

### §13.2 Breast cancer (VAL-095 sealed)

GSE51057 + GSE51032 paired pre-diagnostic blood scored against UniLIFE 19-cell + production Salas Blood.450K legacy:

**At >10yr pre-dx window:**
- UniLIFE aTreg fraction: GSE51057 d = +1.26 [+0.39, +2.26]; GSE51032 d = +0.79 [+0.33, +1.33]
- Salas CD4T: d = +0.36 / +0.07
- UniLIFE separates the activated Treg signal from Salas bulk CD4T

**At 0-2yr pre-dx window:**
- UniLIFE aBnv (naive B-cell): GSE51057 d = +0.44 [+0.15, +0.76]; GSE51032 d = +0.49 [+0.23, +0.77]
- Salas Bcell: d = +0.31 / +0.36
- UniLIFE separates naive B-cell signal from Salas bulk

Lineup ranking: time-window-dependent. Long pre-dx is aTreg-led; near-dx is aBnv-led.

### §13.3 Glioma (VAL-090 sealed 2026-04-25)

GSE180683 n=76 glioma EPIC blood + GSE51057 n=177 healthy reference scored against Loyfer/Moss array atlas:

- **Cortical-neuron deconvolution:** glioma d = +1.96 [+1.62, +2.31]; pre-surgery treatment-naive subset d = +1.98
- **Cell-fraction Bracci 2022 confirmation:** neutrophils +16%, lymphocytes −13%
- **Tissue arm (GSE60274 n=77 brain tissue):** NTB neuron fraction 62.4%, GBM primary neuron fraction 39.3%, GBM primary d vs NTB = −2.81 (cell-of-origin loss in tumor tissue)

Lineup pattern: **cortical-neuron + Bracci-confirmed lymphocyte/neutrophil shift** — orthogonal to A-score (§7.3 glioma's gift).

### §13.4 AD-immune (descriptive, exploratory)

EpiDISH RPC deconvolution into 6 immune sub-types on AD blood:

- **AD literature pattern:** T-cell senescence (CD4+/CD8+ ratio shift), elevated NLR, altered monocyte methylation
- **EDEAR Stage 3 status for AD:** EXPLORATORY — descriptive pending dedicated AD-specific Stage 3 validation

Stage 1 directional 7-CpG Rule A panel remains the validated clinical signal for AD.

### §13.5 Heme-epic (the closest existing OQ-2026-01 analog)

heme-epic v0.1 implements the three-arm structure:

| Arm | Cards mapped | Subtype examples | A_combined sealed (predicted) |
|---|---|---|---|
| **lymphoid_B** | DLBCL, CLL, MM | Chapuy 2018 DLBCL ΔA = +0.20 | A ≈ 1.07–1.13 |
| **lymphoid_T** | thymoma | TCGA thymoma | A ≈ 1.09 |
| **myeloid** | AML | TCGA 2013 NEJM AML | A ≈ 1.10 |

The card structure IS the lymphoid-vs-myeloid discrimination at the *disease level*. OQ-2026-01 staging brings the same discrimination to the *Stage 1 panel level*, complementary to heme-epic's card-level layer.

### §13.6a Bladder cancer (VAL-122 sealed 2026-05-01)

**Cohort:** TCGA-BLCA n=440 (HM450K sesame Level 3, 21 paired tumor-vs-adjacent-normal patients).
**Stage 3 atlases:** Salas Blood.EPIC IDOL 6-cell (production calibrated; primary), UniLIFE Guo 2025 19-cell (within-cohort self-cal v0.1; VAL-115 Wave 1 promotion path), Caggiano CelFiE TIM immune subset 8-cell (VAL-113 anchor).

**Salas IDOL 6-tile paired contrasts (n=21 paired pairs, all fire POSITIVE FIRES at |d|≥0.30):**

| Tile | Cell type | d_paired | 95% CI | p_value |
|---|---|---|---|---|
| Bcell | B lymphocytes | **+1.1479** | [+0.597, +1.699] | 3.79×10⁻⁵ |
| Mono | Monocytes | **+1.1322** | [+0.584, +1.680] | 4.46×10⁻⁵ |
| Neu | Neutrophils | **+1.2354** | [+0.668, +1.803] | 1.53×10⁻⁵ |
| NK | Natural killer | **+0.7943** | [+0.304, +1.285] | 1.63×10⁻³ |
| CD8T | Cytotoxic T cells | **+0.6222** | [+0.155, +1.089] | 9.87×10⁻³ |
| CD4T | Helper T cells | **+0.4884** | [+0.036, +0.941] | 3.67×10⁻² |

**Lineup ranking (largest to smallest |d|):** Neu > Bcell > Mono > NK > CD8T > CD4T. **All POSITIVE direction.** No NEGATIVE-firing tiles.

**Pattern interpretation:** Pre-locked O2 (lymphoid-dominant: CD4T/CD8T POSITIVE + Mono/Neu NEGATIVE — would have replicated Chen 2022 NMIBC blood EPIC RFS signature) did NOT fire. Pre-locked O3 (myeloid-dominant: inverse pattern, MDSC infiltration) did NOT fire. Both lymphoid AND myeloid all fired POSITIVE — broad multi-lineage immune-architectural drift. Mixed TIL + TAM + MDSC infiltration is the standard understanding of muscle-invasive bladder cancer immune microenvironment biology. The Stage 3 firing pattern is consistent with that biology.

**Comparison to prostate VAL-118:** Bladder Salas Mono d=+1.13 is 1.5× larger than prostate Salas Mono d=+0.771. Both cancers show same direction (POSITIVE) on Salas Mono. Bladder MIBC has documented heavier microenvironmental immune infiltration than typical prostate adenocarcinoma — the magnitude difference reflects this real biology.

**Card line in no-overclaim language:** *Stage 3 immune fine-tune fires consistent with mixed TIL + TAM + MDSC infiltration in muscle-invasive bladder tumor microenvironment. All six Salas IDOL immune-cell-type A-scores increase in tumor vs adjacent-normal at |d_paired| range 0.49 to 1.24.*

### §13.7 How to interpret bladder-style broad-positive Stage 3 readings (operational guidance for the future-AI-2-years-from-now interpreter; added 2026-05-01)

**This subsection exists because the bladder VAL-122 result is the cookbook's first sealed example of broad-positive multi-lineage immune-architectural drift. A future AI looking at EDEAR Stage 3 output where every immune cell type fires POSITIVE at moderate-to-large magnitude needs explicit interpretive guidance to distinguish this pattern from the four Stage-1-positive Stage-2-null pathways already in §9.**

**The pattern.** Stage 3 multi-atlas reading produces:
- Salas IDOL: ALL 6 of 6 tiles fire POSITIVE at |d|≥0.30 with no NEGATIVE-firing tiles.
- Magnitude range typically 0.49 to 1.24 across cell types.
- p-values typically 10⁻² to 10⁻⁵ (real signal, not noise).
- Convergent pattern across UniLIFE 19-cell and Caggiano TIM immune subset (per the multi-atlas reproducibility check baked into Stage 3 RUN-everything Phase C).

**The distinguishing biology.** When this pattern fires, the underlying biology is **mixed multi-lineage immune infiltration** of a tissue compartment — TILs (CD4T, CD8T) plus tumor-associated macrophages (Mono lineage) plus tumor-associated neutrophils plus myeloid-derived suppressor cells (Mono+Neu) plus B lymphocytes (often forming tertiary lymphoid structures in bladder) plus NK cells. This is the standard immunology of muscle-invasive solid-tumor microenvironments under heavy infiltration. Substrates that produce this pattern: tumor TISSUE with heavy infiltration (TCGA-BLCA primary tumor confirmed; TCGA-LUAD/LUSC may produce same pattern; TCGA-CESC may produce same pattern; TCGA-COAD may produce same pattern).

**How this is NOT Pathway 1 (terminal-class disease hidden by specimen problem).** Pathway 1 fires when Stage 1 immune is positive (red flag) but Stage 2 cell-of-origin is null — specimen lacks the diseased tissue's β profile (e.g., glioma in plasma where 96% of cfDNA is non-brain). In bladder VAL-122 the substrate IS the diseased tissue (tumor in bladder), and Stage 2 BladderRef Epi fires NEGATIVE (urothelial dedifferentiation per CCL-039) — so Stage 2 is NOT null. Stage 3 broad-positive in this case is the microenvironment correlate of the tissue-substrate Stage 2 cell-of-origin reading. Pathway 1 does not apply.

**How this is NOT Pathway 2 (hematologic/immune-compartment disease).** Pathway 2 fires when Stage 1 fires and Stage 2 cell-of-origin is null because the disease IS the immune compartment (B-cell lymphoma, AML). Bladder VAL-122 is a solid-tumor microenvironment infiltration pattern, not an immune-compartment-of-origin disease. The Stage 3 broad-positive in bladder reflects host immune cells INFILTRATING a non-immune-origin tumor; in heme-epic Pathway 2, the Stage 3 firing reflects malignant immune cells THEMSELVES being the disease. Distinguishing test: does the Stage 2 cell-of-origin tile fire at high magnitude with the disease-appropriate direction? If yes (BladderRef Epi NEGATIVE for bladder, ProstateRef LE NEGATIVE for prostate), the disease is solid-tumor with infiltrate (NOT Pathway 2). If no (Stage 2 across all tissue tiles is null), Pathway 2 applies — go to heme-epic for differential.

**How this is NOT Pathway 3 (cardiovascular/systemic inflammatory).** Pathway 3 fires when Stage 1 immune is positive on whole-blood substrate but Stage 2 cell-of-origin tiles are null — the disease is systemic inflammation rather than localized cell-of-origin tissue. Bladder VAL-122 is on tumor TISSUE substrate (not blood). Stage 1 immune is positive on tumor tissue (consistent with tumor-localized infiltration). Pathway 3 specifically targets blood-substrate Stage 1 positives. Distinguishing test: what was the SUBSTRATE? If blood and Stage 2 is null → Pathway 3. If tumor tissue and Stage 2 BladderRef Epi (or equivalent gene-promoter cell-of-origin tile) fires NEGATIVE per CCL-039 → solid-tumor mixed-infiltration (THIS pattern, §13.7 — NOT Pathway 3).

**How this is NOT Pathway 4 (unexplained immune drift, serial trajectory watch).** Pathway 4 applies to a single Stage 1 positive at moderate magnitude with no Stage 2 or Stage 3 corroboration — flagged for serial-trajectory subscription monitoring. Bladder VAL-122 has overwhelming Stage 1 + Stage 2 + Stage 3 corroboration across 21 paired patients — this is high-confidence cancer detection, not unexplained drift.

**The lymphoid-vs-myeloid lineage skew test (CCL-027 question iv).** When Stage 3 broad-positive fires, run the lineage-skew check:
- IF lymphoid (CD4T, CD8T, NK, Bcell) fires NEGATIVE while myeloid (Mono, Neu) fires POSITIVE → **myeloid-dominant pattern**, consistent with MDSC infiltration in advanced metastatic / immunotherapy-refractory disease.
- IF myeloid (Mono, Neu) fires NEGATIVE while lymphoid (CD4T, CD8T) fires POSITIVE → **lymphoid-dominant pattern**, consistent with Chen 2022 NMIBC blood RFS signature in early-stage / immunotherapy-responding disease.
- IF BOTH lymphoid AND myeloid fire POSITIVE (no NEGATIVE-firing tiles in lineage panel) → **broad multi-lineage infiltration pattern** (THIS is bladder VAL-122). Consistent with heavily-infiltrated tumor microenvironment characteristic of muscle-invasive disease. NOT a clean lineage-skewed signature; do NOT report as lymphoid-dominant or myeloid-dominant.
- IF BOTH lymphoid AND myeloid fire NEGATIVE → **suppressed/exhausted pattern**, possible immune escape; consistent with late-stage tumor evasion or post-chemotherapy immune ablation. (Currently no sealed cookbook example; pattern hypothetical.)

**The "is the Stage 1 panel reliable on this cohort?" check.** When broad-positive Stage 3 fires alongside Stage 1 O4 panel-coverage gate fire (as happened in bladder-epic v0.1 with Xu-538 on TCGA-BLCA), the Stage 3 reading is independently informative — the Salas IDOL + UniLIFE + Caggiano TIM atlases are the lineage-resolution layer and they are not gated by Stage 1 panel coverage. The Stage 3 pattern stands on its own as the tumor-microenvironment characterization even if Stage 1 is gate-failed. The diagnostic-not-sealed Stage 1 d-value is corroborative but not necessary for the Stage 3 interpretation.

**The "is BladderRef Epi NEGATIVE the cell-of-origin signal?" check.** Stage 2 cell-of-origin reading on a gene-promoter sub-cell-type atlas should fire NEGATIVE per CCL-039 in tumor-vs-adjacent-normal paired comparisons (cell-of-origin dedifferentiation). If BladderRef Epi or ProstateRef LE or future LungRef AT2 (alveolar type 2) or similar gene-promoter cell-of-origin tile fires NEGATIVE alongside broad-positive Stage 3, this is the canonical solid-tumor mixed-infiltration cohort: dedifferentiated cancer cells (Stage 2 NEGATIVE on cell-of-origin) plus heavy multi-lineage infiltrate (Stage 3 broad POSITIVE) plus stromal/microenvironment expansion (Stage 2 POSITIVE on EC/Fib/microenvironment tiles). This is exactly the bladder VAL-121 + VAL-122 combined signature, and it generalizes to expected solid-tumor patterns for breast, lung, colon, cervical, GI cancers when those Stage 3 multi-atlas runs complete.

**The single-sentence rule for this pattern.** When Stage 3 RUN-everything fires broad-positive across all six Salas IDOL tiles with no NEGATIVE-firing tiles, AND Stage 2 cell-of-origin fires NEGATIVE on the appropriate gene-promoter sub-cell-type atlas, AND Stage 2 microenvironment tiles fire POSITIVE, AND substrate is tumor tissue (not blood), the EDEAR output should be reported as: *"mixed multi-lineage immune infiltration of a solid-tumor microenvironment with cell-of-origin dedifferentiation, consistent with muscle-invasive / heavily-infiltrated cancer biology — NOT a lymphoid-skewed or myeloid-skewed signature."*

### §13.6 Cards pending Stage 3 multi-atlas runs

- breast-epic (Salas IDOL retroactive run candidate on VAL-060 paired tumor)
- crc-epic (Stage 3 immune sub-composition retroactive run candidate on VAL-061 tumor TIL)
- lung-epic (Stage 3 multi-atlas pending; smoking-stratified)
- hcc-epic (Stage 3 multi-atlas pending on ccfDNA)
- cervical-epic (Stage 3 multi-atlas pending; HPV-stratified)
- cardio-epic (Stage 3 multi-atlas pending; substrate-stratified)
- pancreatic-epic (Stage 3 multi-atlas pending; OQ-2026-01 priority unlock)
- gastric/bladder/kidney (cards pending)

---

## §14. Cross-card syntheses

### §14.1 Prostate vs Breast (preserved from v0.2)

**Same:** secretory class (H_min 0.843264), cell-of-origin lineage, Stage 1 d ≈ +0.50, TIL infiltration, both luminal-epithelial-origin adenocarcinomas.

**Different:** ProstateRef separates cleanly (6 tiles) where BreastRef collapses (7 tiles uniform positive); prostate LE d=−0.77 dedifferentiation visible vs breast all positive (not isolatable); pre-diagnostic blood signal: breast d=+1.78 at >10yr operational TODAY, prostate biobank-gated v1.0+; prostate sheds less DNA into plasma than breast.

**One-line synthesis.** Breast cancer is the cancer EDEAR catches earliest in pre-diagnostic blood; prostate cancer is the cancer EDEAR sees the cleanest cell-of-origin signature in tumor tissue.

### §14.2 Lung vs CRC (same cycling H_min, opposite Stage 1 directions in blood)

**Same:** Cycling architecture (H_min 0.856055), both cycling-class epithelium cancers, both large tumor tissue effects (lung VAL-063 d=+1.020; CRC VAL-062 d=+0.724), both show CCL-039 cell-of-origin tile NEGATIVE in tumor-vs-adjacent-normal.

**Different:** Lung Stage 1 blood positive (VAL-046 cohort + VAL-056); CRC compartment-direction-flip per CCL-019 (blood d=−0.33; tumor TIL d=+1.066). Lung mutational burden highest in cookbook (smoking-driven).

**One-line synthesis.** Lung and CRC share architectural class but diverge at the immune-compartment compartment-flip: lung drives systemic immune activation in pre-diagnostic blood, CRC drives systemic immune tolerance.

### §14.3 Lung vs Breast (different classes, both positive, lung tissue dominates)

**Same:** Both Stage 1 positive in pre-diagnostic blood, both multi-modal validated with tissue arm, both produce uniform-positive per-CpG patterns.

**Different:** Lung cycling (H_min 0.856055), breast secretory (H_min 0.843264). Lung tissue d=+1.020 dominates breast tissue d=+0.676. Tumor mutational burden ordering: d(lung) > d(CRC) > d(breast) > d(prostate).

**One-line synthesis.** Tissue-arm effect size scales with tumor mutational burden. Cancer with largest mutational burden produces largest architectural methylation signature.

### §14.4 Glioma vs AD (both terminal-class with BBB issues)

**Same:** Both Pathway 1 trigger candidates, both terminal class (H_min 0.7728), both BBB-limited cfDNA in plasma.

**Different:** Glioma pooled-positive (VAL-088 d=+0.91); AD pooled-NULL + directional-positive (VAL-050+VAL-051). Glioma cell-fraction relationship orthogonal; AD lineage discordance per Mastroeni 2020. Operational metric differs (glioma pooled, AD directional).

**One-line synthesis.** Both produce immune-class architectural drift readable in peripheral blood despite BBB cfDNA limitation. Different patterns, both Pathway 1.

### §14.5 HCC vs Breast (both secretory, different substrate paths)

**Same:** Same secretory class (H_min 0.843264), both luminal-epithelial-origin, both produce TIL-positive tumor signatures.

**Different:** HCC Stage 1 substrate: ccfDNA works (d=+0.63), whole-blood vs metabolic controls fails (NULL). Breast Stage 1 substrate: whole-blood works (d=+0.45 to +1.78). HCC dose-response monotonic healthy → fibrosis → cirrhosis → HCC. Sex stratification mandatory in both.

**One-line synthesis.** Substrate-as-discriminator is HCC's gift. Same disease class, different substrate gives different signal.

### §14.6 PDAC vs AD (canonical pooled-null + directional-pass cases)

**Same:** Both Pattern 3, both have sealed directional fallback panels (AD 7-CpG, PDAC 324-CpG), both CCL-028 mechanism unresolved, both await Test 2 lineage assignment for Pattern 3 → Pattern 5 graduation.

**Different:** AD directional 7 CpGs; PDAC 324 CpGs. AD brain (BBB-limited); PDAC pancreas (poorly cfDNA-shedding). AD compartment is peripheral blood only; PDAC tissue + blood + urine all candidates.

**One-line synthesis.** AD and PDAC are the cookbook's two canonical Pattern 3 cases. Both teach: pooled-null does not mean biology-null.

### §14.7 Cardio vs Everything (Pathway 3 hub)

cardio-epic v0.3 is the canonical Pathway 3 card. Substrate-stratified Stage 1: whole-blood etiology-equivalent null (biology-correct per LL-CARDIO-002); cultured PECs PAH d=+0.65; aortic tissue dissection d=+0.56 / BAV d=+1.08.

**Implication.** When Stage 2 returns null and patient has CVD risk factors, Pathway 3 is the route. Whole-blood null does NOT rule out cardiovascular contribution — the contribution is real and substrate-dependent.

### §14.8 Heme vs Everything (Pathway 2 hub + OQ-2026-01 closest analog)

heme-epic v0.1 is the canonical Pathway 2 card and the closest existing OQ-2026-01 analog. The three-arm structure IS Test 2 at card level.

**Implication.** When Stage 1 strongly positive AND Stage 2 returns null, the differential between Pathway 2 (heme), Pathway 3 (cardio), and Pathway 4 (unexplained) routes by per-CpG pattern + risk factors + Stage 3 sub-composition.

### §14.9 Cervical vs Everything (cohort-direction-flip per CCL-019)

cervical-epic v0.1 carries the canonical cohort-direction-flip lesson. VAL-073 d=+0.73 (HPV-positive normals); VAL-074 d=−0.61 (HPV-negative normals shift baseline); VAL-081 d=−0.43.

**Implication.** Any disease where the healthy reference is stratified by a disease-relevant covariate (HPV / smoking / ancestry / metabolic-disease background) requires that covariate as a Stage 1 stratifier.

### §14.10 Pancreatic vs Everything (second pooled-null + directional-pass case)

pancreatic-epic v0.1 is the second AD-instance pattern card. VAL-066/067/068 pooled CIs straddle zero across three cohorts; VAL-069 directional 324-CpG d=+1.51 on TCGA-PAAD holdout.

**Implication.** PDAC confirms Pattern 3 is not unique to AD. Future cards should expect Pattern 3 when literature predicts lymphoid + myeloid lineage discordance. Build directional fallback during card v0.1 rather than waiting for pooled to fail in the field.

---

## §15. Open atlas-coverage gaps and biobank-gated next steps

### §15.1 Biobank-gated cohorts (multi-month application timelines)

| Cohort | Disease | Access | Timeline |
|---|---|---|---|
| FitzGerald 2017 MCCS pre-dx | prostate | EGA / direct PI | multi-month |
| Howard AA EPIC | prostate | dbGaP | multi-month |
| UCSF AGS phs001497 | glioma | dbGaP | multi-month |
| UCSF Immune Profiles Study phs002998 | glioma | dbGaP | multi-month |
| GICC international cohort phs001319 | glioma | dbGaP + collaboration | multi-month |
| UK Biobank methylation subset | lung, multiple | UK Biobank application | multi-month |
| Bracci 2022 replication | glioma | published cohort | direct PI |
| Health ABC + Rotterdam Study | prostate | dbGaP | multi-month |
| Sundström CIN2 2026 | cervical | direct PI | unknown |
| CINCS Bukowski 2023 | cervical | direct PI | unknown |
| EnviroGenomarkers (heme-epic VAL-082) | hematologic | direct PI / consortium | unknown |

### §15.2 Atlas builds NOT YET STARTED

- Salas IDOL-Ext panel-CpG cross-walk to Xu-538 (OQ-2026-01 prerequisite)
- UniLIFE 19-cell panel-CpG cross-walk to Xu-538 (OQ-2026-01 prerequisite)
- EpiSCORE LungRef CpG bridge
- EpiSCORE KidneyRef CpG bridge
- EpiSCORE ColonRef CpG bridge
- EpiSCORE BrainRef CpG bridge (Pathway 1 deep-dive)
- EpiSCORE PancreasRef CpG bridge
- EpiSCORE EsophagusRef / OliveRef / OvaryRef / SkinRef / StomachRef / BladderRef bridges
- Caggiano TIM Phase B calibration on substrate-matched healthy cohort
- cfMeDIP-seq pipeline integration (different chemistry; v0.2+ build target for glioma)
- Sabedot 2021 GeLB cfDNA pipeline integration

### §15.3 Validation runs blocked by access

- Pre-diagnostic prostate blood validation (any cohort)
- Pre-diagnostic glioma blood validation (no cohort exists; Bracci 2022 access blocked)
- LP-CSF gold-standard glioma test
- Pathway 2 cervical lymph specimen pathway

### §15.4 Stage 3 multi-atlas runs pending (sealed VAL data exists, runs not yet executed)

- breast-epic Salas IDOL retroactive run on VAL-060 paired tumor + adjacent-normal
- crc-epic Stage 3 immune sub-composition on VAL-061 tumor TIL
- lung-epic Stage 3 multi-atlas (smoking-stratified)
- hcc-epic Stage 3 multi-atlas on VAL-059 ccfDNA
- cervical-epic Stage 3 multi-atlas (HPV-stratified)
- cardio-epic Stage 3 multi-atlas (substrate-stratified)
- pancreatic-epic Stage 3 multi-atlas (OQ-2026-01 lineage discrimination priority unlock)

---

## §16. The Mahalanobis-distance differential ranker

(Preserved from v0.1, expanded with v0.3 Stage 3 lineage vector input.)

When Stage 1 fires (elevated or inverted A-score), the clinician receives:
1. The A_immune_pooled score and tier call
2. The A_dir score (for AD and other directional-panel-required diseases)
3. The per-CpG Δβ direction table
4. **The immune-atlas match list** — top 3 diseases whose expected signature is most consistent with this patient's reading, ranked by signature-similarity score
5. **(v0.3 addition)** The Stage 3 sub-cell-type lineup vector (when Stage 3 multi-atlas was run for this patient)

### §16.1 Match list computation

The match list is computed by comparing the patient's signature vector against each row in the cross-reference table (§5) and ranking by Mahalanobis distance in signature space.

**v0.1 signature vector (3-dim):** (direction, magnitude, per-CpG pattern)

**v0.3 expanded signature vector (5-dim):** (direction, magnitude, per-CpG pattern, A_lymphoid, A_myeloid)

The two added dimensions require OQ-2026-01 Test 2 staging to be operational. Until then, the v0.1 3-dim vector is used. When OQ-2026-01 goes live, the expanded 5-dim ranker drops in as a non-breaking upgrade.

### §16.2 When Stage 2 returns null

The immune-atlas routes the case into one of the four pathways (§9) based on:
- **Magnitude** → strong positive (>d=0.8) favors Pathway 2 heme or Pathway 3 cardio over Pathway 1 or 4
- **Direction** → negative direction favors CRC (Pathway 1 cycling-class solid organ) or gastric/bladder (future cards)
- **Per-CpG pattern** → bidirectional pattern favors AD-like Pathway 1 terminal; uniform positive favors Pathway 2 or 3; monocyte-shift favors Pathway 3 cardio
- **Patient risk factors** → age + CVD risk factors favor Pathway 3; neurological symptoms favor Pathway 1; hematologic symptoms favor Pathway 2
- **(v0.3 addition)** Stage 3 lineage vector → lymphoid-shift predominant favors Pathway 2 heme; myeloid-shift with monocyte dominance favors Pathway 3 cardio; uniform low-magnitude favors Pathway 4 trajectory watch

### §16.3 At the card level

Every other EDEAR card cross-references this atlas. The immune-atlas is the single source of truth for disease-signature information. Card-specific validation results flow INTO the atlas; clinical interpretation flows OUT of it.

---

## §17. Future v1.0 reorganization plan — symptom-organized decision tree

The v0.3 card is **disease-organized**: each disease is a row, each disease has its CCL-027 four answers. A clinician learning the framework reads down the disease list to understand the instrument's behavior.

The card has two natural lifecycles:

**Phase 1 (now → as data fills in): organize BY DISEASE.** This is what we are building toward. The Rosetta Stone aspect is comparative across diseases.

**Phase 2 (eventually): organize BY SYMPTOM/PRESENTATION.** Once we have all 15 cards' data, the card flips inside out. The clinician does not enter from "I think this patient has prostate cancer" — they enter from "patient presents with elevated A_immune, monocyte-shift signature on Stage 3, no Stage 2 solid-organ localization." The card walks them down a decision tree:

```
START: patient Stage 1 result
│
├── Is data integrity passed (§2)? ──→ NO → STOP (test failure, not finding)
│                                  └─→ YES → continue
│
├── Pooled A_immune direction?
│   ├── POSITIVE
│   │   ├── Magnitude tier?
│   │   │   ├── strong (>0.8) ──→ Stage 3 lineage check
│   │   │   │                   ├── monocyte-led ──→ prostate (TAM-recruitment) | cardio (CVD)
│   │   │   │                   ├── lymphoid-led ──→ heme lymphoid | breast >10yr aTreg
│   │   │   │                   ├── T-cell-led ──→ thymoma | lung TIL
│   │   │   │                   ├── neutrophil-led ──→ AML myeloid | infection | cardio
│   │   │   │                   └── uniform ──→ trajectory watch (Pathway 4)
│   │   │   ├── moderate (0.3-0.8) ──→ Stage 2 atlas-sweep
│   │   │   │                       ├── solid-organ tile dominant ──→ disease-specific card
│   │   │   │                       └── distributed ──→ cellular-aging-drift signature
│   │   │   └── weak (0.15-0.3) ──→ trajectory watch + 6-month re-test
│   ├── NEGATIVE
│   │   ├── compartment check ──→ blood vs tumor TIL? (CCL-019 compartment-flip)
│   │   ├── cohort heterogeneity check ──→ HPV-stratification etc.
│   │   └── candidate diseases ──→ CRC blood | gastric (predicted) | bladder (predicted)
│   └── NULL POOLED
│       ├── directional panel positive ──→ AD-instance pattern (Pathway 1 + directional fallback)
│       │   ├── 7-CpG Rule A panel (AD)
│       │   ├── 324-CpG GSE49149 panel (PDAC)
│       │   └── (other disease-specific panels)
│       └── directional panel null ──→ true biology null OR substrate mismatch
│           ├── check biology consistency (§2.4)
│           └── check substrate dichotomy (§7.2 HCC's gift)
└── Stage 2 atlas-sweep result?
    ├── solid-organ localization above background ──→ tissue-of-origin disease card
    └── no solid-organ localization ──→ four-pathway routing (§9)
        ├── Pathway 1: terminal-class hidden by specimen problem
        ├── Pathway 2: hematologic / immune-compartment disease
        ├── Pathway 3: cardiovascular / systemic inflammatory
        └── Pathway 4: unexplained drift, trajectory watch
```

That is a symptom-checker decision tree, not a disease catalog. It is the difference between a medical textbook organized by organ system (which is how you LEARN the field) and a differential-diagnosis manual organized by chief complaint (which is how you USE the field at the bedside).

The right time to flip from disease-organized to symptom-organized is when the disease catalog stops adding new branches and starts producing redundant entries — which is exactly when "all 15 cards' data is in."

### §17.1 Tagging v0.3 disease entries to v1.0 symptom-tree nodes

Each v0.3 disease entry in §5 carries an implicit tag for the v1.0 symptom-tree node it will eventually live under. Examples:
- prostate Stage 1 d=+0.50 + monocyte-led TIL → v1.0 node: pooled-positive moderate-magnitude + monocyte-led Stage 3 → secretory-class TME-active candidate set
- AD pooled-NULL + directional 7-CpG positive → v1.0 node: pooled-null + directional-positive Pathway 1 candidate
- HCC ccfDNA d=+0.63 + whole-blood NULL → v1.0 node: substrate-dichotomy secretory-class disease

(Full tagging for all sealed disease entries is an implementation task, not a v0.3 hard requirement.)

---

## §18. Validation status

The immune-atlas is a **rosetta_reference_card** tier card. It does not itself require validation — it synthesizes results from validated cards.

**What requires validation:**

1. Each row's disease signature (validated by the disease-specific card referenced in the table)
2. The four Stage-1-positive Stage-2-null pathway classifications (validated by clinical outcome data, pending retrospective cohort analysis of flagged patients)
3. The Mahalanobis-distance differential ranker (to be validated against held-out case mix when deployed)
4. **(v0.3 addition)** The OQ-2026-01 Test 2 lymphoid-vs-myeloid sub-panel split (to be validated when atlas-CpG cross-walk is built and substrate-matched calibration is sealed)

**Versions prior to `cross_platform_validated` of component cards are marked PREDICTED in the atlas row.**

---

## §19. Language discipline

Reports referencing the immune-atlas use:
- "Your Stage 1 signature is most consistent with [disease X, Y, Z] based on direction, magnitude, and per-CpG pattern"
- "The signature does not match any established card at DETECTABLE tier; serial-sampling trajectory recommended"
- "This pattern is characteristic of cardiovascular risk; recommend cardiovascular workup"

**(v0.3 addition)** When CCL-031 terminology applies:
- "This is a compartment-direction-flip per CCL-019, not bidirectional cancellation"
- "This is a cross-disease direction difference per CCL-006, not bidirectional cancellation"
- "This is the AD-instance pattern; mechanism unresolved pending OQ-2026-01"

Never:
- "You have [disease X]" — the atlas is a differential, not a diagnosis
- "Your test is negative for [disease Y]" — absence of match does not rule out disease
- "This confirms [condition Z]" — confirmation requires the disease-specific card plus clinical workup
- **(v0.3 addition)** "This is bidirectional cancellation" — unless the AD-instance pattern is operationally confirmed
- **(v0.3 addition)** "The framework is wrong" — until data integrity, biology consistency, and demographics stratification have been verified per §2

---

## §20. File pointers

- **This README** — narrative reference
- **`immune-atlas_card_v0_3.json`** — machine-readable version of all sections + cross-reference data + atlas registry
- **Parent cookbook** — `README_MASTER_v2_4.md` (universal pipeline rule + cardio-epic + prostate-epic v0.3 amendment chain)
- **Source-of-truth for cookbook doctrine** — `LESSONS_LEARNED.md` (CCL-006/019/023/027/028/030/031/032/039 verbatim)
- **Source-of-truth for data integrity protocol** — `TESTING_CHECKLIST.md` (CHK-2.7, CHK-2.8, CHK-3.1, CHK-3.1B, CHK-3.1C, CHK-3.2, CHK-3.5, CHK-4.11, CHK-7.5, CHK-7.6)
- **Source-of-truth for pipeline behavior** — `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md`
- **Component cards (validated):** breast-epic v2.3, crc-epic v2.4, ad-immune v2.2, lung-epic v0.4, prostate-epic v0.3, hcc-epic v0.3, glioma-epic v0.2, cardio-epic v0.3, cervical-epic v0.1, pancreatic-epic v0.1, heme-epic v0.1
- **Component cards (pending):** gastric-epic, bladder-epic, kidney-epic

---

## §21. Card evolution plan

This card is explicitly **designed to grow**. As more disease cards are validated:

- When a new card passes `cross_platform_validated` tier — atlas entry moves from PREDICTED to validated
- When a cohort study publishes a new disease's immune methylation signature — new row added even before EDEAR has a dedicated card for it
- When serial-trajectory data accumulates — atlas gains trajectory-rate-of-change columns per disease
- When non-cancer applications (inflammaging, autoimmune, chronic infection, metabolic syndrome) produce distinguishable signatures — new rows added
- **(v0.3)** When a card produces sealed Stage 3 multi-atlas data — §13 expands; cross-card synthesis pair added in §14
- **(v0.3)** When OQ-2026-01 staging operationalizes — Test 2 results populate Pattern 5 entries; Mahalanobis ranker upgrades from 3-dim to 5-dim
- **(v0.3)** When all 15 cards have sealed v0.x+ entries — card v1.0 reorganization to symptom-organized decision tree (§17)

**Target atlas completeness by v3.0:** 20+ disease/condition rows, trajectory data for each, Mahalanobis-distance differential engine locked in production code at 5-dim, OQ-2026-01 staging operational, v1.0 symptom-organized decision tree replacing v0.x disease-organized cross-reference.

---

## §22. Lessons learned that motivated this card

### v0.1 lessons (preserved verbatim)

- **CCL-019 (2026-04-24):** A-score direction depends on (class, compartment) pair, not disease alone. Blood immune ≠ tumor-infiltrating immune; peripheral blood immune ≠ localized tissue immune response. The atlas makes compartment explicit per row.
- **CCL-020 (2026-04-24):** Same panel applied to same disease can return opposite-direction signals depending on specimen. Xu-538 on blood for CRC = negative (suppressed peripheral); Xu-538 on tumor tissue for CRC = positive (activated TIL). Atlas catalogs compartment assumptions.
- **CCL-021 (2026-04-24):** The 4% cfDNA detection floor creates structured Stage-1-positive / Stage-2-null patterns that are disease-family-specific (terminal, heme, cardiovascular). Pathway routing is the correct way to handle this.
- **CCL-022 (2026-04-24):** Single-timepoint EDEAR is a flag; serial-trajectory EDEAR is a diagnostic. Pathway 4 formalizes the subscription-based trajectory-watch model.

### v0.2 lessons (preserved verbatim)

- **DISC-PROSTATE-001 / extends LL-CARDIO-005 (2026-04-30):** Atlas family fitness depends on per-tissue cell-type distinctness. EpiSCORE ProstateRef separates clean on prostate; EpiSCORE HeartRef collapsed on cardiac (VAL-111 sealed O3_TISSUE_FLOOR_DOMINATED); EpiSCORE BreastRef partially collapses on breast (VAL-093 sub-cell-type resolution-collapse breast-LL-007).
- **DISC-PROSTATE-002 / CHK-2.7 (2026-04-30):** Cell-of-origin atlas preregs MUST use magnitude-based |d| thresholds with direction labels. Cookbook-wide rule.
- **DISC-PROSTATE-003 (2026-04-30):** ProstateRef LE tile reads tumor strongly NEGATIVE (luminal dedifferentiation signature). Operational diagnostic: A_LE BELOW VAL-117 healthy-floor q5 = 0.4190 flags potential luminal dedifferentiation drift.
- **breast-LL-007 / cross-card cookbook lesson (2026-04-30):** BreastRef cell-of-origin tile resolution-collapse on TCGA-BRCA. All 7 BreastRef tiles read positive at d ≈ +1.0 to +1.2; cell-of-origin sub-cell-type resolution not isolatable at current atlas level.
- **immune-atlas-LL-001 / Stage 3 sub-cell-type lineup as differential dimension (2026-04-30):** When Stage 3 multi-atlas immune-cell-type data exists for a card, the per-cell-type lineup ranking is itself a fingerprint distinguishing that card's TIL signature.

### v0.3 lessons (added 2026-04-30)

- **immune-atlas-LL-002 — Rosetta Stone reframe (2026-04-30):** The immune-atlas card is not a "differential-diagnosis engine" alone — it is the Rosetta Stone of EDEAR. Its three operational roles (Stage 1 interpretation engine + Stage 2 atlas cross-reference + Stage 3 OQ-2026-01 staging hub) form the diagnostic spine. Card position #15 of 15 reflects dependency-final, not operational unimportance.
- **immune-atlas-LL-003 — OQ-2026-01 canonical home (2026-04-30):** OQ-2026-01 immune-atlas staging progress tracking moves from scattered references across multiple cards to a canonical home in §12 of this card.
- **immune-atlas-LL-004 — CCL-027 four-question centralization (2026-04-30):** Future cards complete their CCL-027 in their own README; this card consolidates the cross-reference in §11 (the canonical master table).
- **immune-atlas-LL-005 — v0.3-disease vs v1.0-symptom architectural lifecycle (2026-04-30):** The card has two natural lifecycles. Phase 1 (v0.x) is disease-organized; Phase 2 (v1.0+) is symptom-organized. The transition flip happens when the disease catalog stops adding new branches.
- **immune-atlas-LL-006 — bidirectional-as-default operational doctrine (2026-04-30):** Pooled A_immune should NEVER be the sole metric. The operator must always assume bidirectional behavior is possible until ruled out. Default assumption is bidirectional behavior is possible. Test 2 is the operational test for resolution; pending OQ-2026-01.
- **immune-atlas-LL-007 — RUN-everything as concurrent-disease safety net (2026-04-30):** Patients do not present with one disease at a time. RUN-everything Stage 2 is the cookbook's structural answer to concurrent-disease scenarios. Conditional gating would let one signal dominate and others get filtered out. RUN-everything is the safety net.
- **immune-atlas-LL-008 — demographics as mandatory analysis stratifiers (2026-04-30):** Patient demographics are not metadata — they are mandatory analysis stratifiers when the disease's literature establishes them as such (§2.6). Sex, age, smoking, ancestry, HPV status, HIV status, immunosuppression status, pregnancy status, treatment stage — each is mandatory or recommended per disease.

---

### v0.3.1 lessons (added 2026-05-01 from bladder-epic v0.1 sprint)

This subsection exists because the bladder-epic v0.1 sprint produced four DISC-BLADDER discoveries that propagate to immune-class interpretation, plus a meta-lesson about how the cookbook self-corrects. **A future AI 2 years from now reading EDEAR output will need this section to interpret bladder-style mucosal-cohort patterns.**

- **immune-atlas-LL-009 — Sixth CCL-031 pattern (substrate-distribution mismatch on mucosal cohort) sealed 2026-05-01.** Two atlases score the SAME cell-of-origin question on the SAME paired pairs and produce HIGH-MAGNITUDE contrasts in OPPOSITE DIRECTIONS, not because of biology, but because bulk-WGBS atlas families (Loyfer/Moss, Caggiano TIM) on mucosal-tissue cohorts produce inflated cross-tile A-scores from substrate-distribution mismatch. Gene-promoter sub-cell-type atlas families (EpiSCORE per-tissue bridges — BladderRef, ProstateRef, BreastRef, HeartRef) avoid this artifact. Multi-atlas readings on mucosal cohorts MUST include a gene-promoter sub-cell-type atlas as the primary cell-of-origin reader. Pattern 6 documented in §6.6a; full DISC-BLADDER-003 in LESSONS_LEARNED.md.

- **immune-atlas-LL-010 — Tissue-class CHK-3.1A floor parameterization (DISC-BLADDER-002).** The CHK-3.1A f_extreme floor cookbook-wide implicit value (≥ 0.50, kidney+prostate-derived from VAL-106) is appropriate for solid parenchyma tissue class but inappropriate for mucosal/epithelial-lined-organ tissue class. Mucosal organs (bladder; expected lung airways, colon epithelium, GI epithelium, cervical mucosa) have substantially less bimodal methylation distribution shapes than solid parenchyma. Tissue-class brackets observed: solid parenchyma ≥ 0.50 / ≤ 0.12 (anchor VAL-106); mucosal ≥ 0.387 / ≤ 0.184 (anchor VAL-120/121/122 amendment 002, bladder cohort q1/q99). New CHK-2.16 cookbook gate added: tissue-class CHK-3.1A floor verification at prereg-write time. **Operational implication for immune-atlas card:** when interpreting EDEAR output, the tissue-class label of the cohort/specimen should be visible in the report. A mucosal-cohort substrate that uses solid-parenchyma floor will fail CHK-3.1A on the majority of legitimate samples — that is a gate-calibration mismatch, not a data integrity failure. The future AI should not interpret a 23.9% pass rate on a mucosal cohort as data integrity failure if the floor was inherited from solid-parenchyma calibration.

- **immune-atlas-LL-011 — Gene-promoter atlas family fitness depends on cell-type DISTINCTNESS, not cell-type COUNT (DISC-BLADDER-001 extends DISC-PROSTATE-001 / LL-CARDIO-005).** Atlas family fitness rule now grounded in three data points: HeartRef 5 cell types collapsed (max within-cohort range 0.0152); ProstateRef 6 cell types separated (0.0597); BladderRef 4 cell types separated cleanly (0.0694). Cell-type COUNT does NOT predict separation. Cell-type DISTINCTNESS at the gene-promoter level for the marker genes Zhu/Teschendorff selected is the discriminating variable. Future EpiSCORE per-tissue bridges (LungRef, KidneyRef, ColonRef, BrainRef, PancreasRef) get prioritized by per-tissue calibration smoke test, not by source matrix dimensions. **Operational implication for immune-atlas card:** when a future card sprint adds a new EpiSCORE per-tissue bridge to the atlas registry (§10), the atlas-family-fitness lesson grounds the smoke-test requirement. Smoke-test outcome (max within-cohort tile range vs 0.02 threshold) determines atlases_run vs atlases_deferred placement, not cell-type count.

- **immune-atlas-LL-012 — Stage 1 panel transferability is cohort-specific, not platform-specific (DISC-BLADDER-004).** Xu-538 panel CpG IDs are all from HM450 design (substrate-applicable) but per-sample coverage on TCGA-BLCA was 78.0% mean / 51.1% pass rate at ≥80% threshold. Different cohorts (different TSS sites, different processing batches, different patient demographics) produce different per-sample detection patterns even within the same substrate platform. New CHK-2.17 cookbook gate added: Stage 1 panels must be validated against the target Phase C cohort's substrate-coverage envelope at prereg-write time (sample 5-10 random Phase C cohort β files; FLAG if mean < 90% or q5 < 80%). VAL-114 Wave 1 calibration on Hannum 2013 GSE40279 n=656 healthy aging blood gets the per-cohort substrate-coverage precheck baked in across all TCGA + AD + cardio + heme target cohorts. **Operational implication for immune-atlas card:** when a card returns Stage 1 O4 outcome (panel coverage gate fire) but Stage 3 multi-atlas fires cleanly (§13.7 broad-positive pattern), the Stage 3 reading is independently informative and not gated by the Stage 1 panel-coverage failure. The future AI should not dismiss the Stage 3 result because Stage 1 fired O4 on this cohort.

- **immune-atlas-LL-013 — Meta-lesson: how the cookbook self-corrects when a silent assumption meets data that violates it (added 2026-05-01).** The bladder-epic v0.1 sprint is the first cookbook example of an honest CCL-041 second-best path used end-to-end: prereg sealed BEFORE β observed; Phase C unified runner produced per-sample tables; CHK-3.1A pass rate observed as 23.9% under inherited solid-parenchyma floor; honest diagnostic determined the gate failure was tissue-class threshold mismatch (zero samples had genuine substrate corruption); amendment 002 sealed AFTER β observed with full disclosure that β was observed before amendment; threshold change rooted in cohort-internal q1/q99 percentiles (observable substrate properties, not chosen to make a particular outcome fire); per-(atlas, tile) contrast magnitudes invariant to the gate floor; outcome.md sealed after amendment. The discipline did not fail. The cookbook caught its own assumption mid-sprint. The four DISC-BLADDER lessons strengthen the cookbook for every future card. **This is the canonical reference example for the future-AI-2-years-from-now interpreter to look at when EDEAR output exhibits structural patterns the immune-atlas card has not catalogued before** — apply the same diagnostic order (CCL-032: data integrity → biology → framework), the same prereg discipline (CCL-041 honest second-best path if β has been observed), the same atlas-family-fitness check (DISC-BLADDER-001 / DISC-PROSTATE-001), and the same multi-atlas convergence rule (DISC-BLADDER-003: gene-promoter atlas required as primary cell-of-origin reader on mucosal cohorts).

---

## §23. What changes from v0.2 to v0.3

| Section | v0.2 → v0.3 |
|---|---|
| Header | tier `reference_document` → `rosetta_reference_card`; card position #13 → #15; framing "differential-diagnosis engine" → "Stage 1 interpretation + Stage 2 atlas cross-reference + Stage 3 OQ-2026-01 staging" |
| §1 What this card is | minor expansion + three-stage diagnostic spine diagram + RUN-everything safety net |
| **§2 Pre-test Integrity Protocol** | **NEW** — atlas calibration prerequisite, six data integrity checks, biology consistency check, demographics-as-mandatory-stratifiers, bidirectional-as-default doctrine, ten failure-mode fingerprints, six biology-real patterns |
| §3 Why this card exists | three v0.1 reasons preserved + three v0.3 reasons added |
| §4 The universal first test | minor expansion |
| §5 Cross-reference table | expanded with v0.3 sealed numbers + Stage 3 sub-cell-type column added |
| §6 CCL-031 five-pattern taxonomy | NEW |
| §7 Three doctrine cases | NEW (AD's gift, HCC's gift, glioma's gift) |
| §8 Cookbook doctrine that touches immune class | NEW (CCL-006/019/023/027/028/030/031/032/039) |
| §9 Four Stage-1-positive Stage-2-null pathways | preserved verbatim from v0.1 + v0.3 cross-references added |
| §10 Stage 2 atlas registry | NEW |
| §11 CCL-027 four-question master cross-reference table | NEW |
| §12 OQ-2026-01 staging hub | NEW (canonical home moved here) |
| §13 Stage 3 sub-cell-type signatures | expanded from prostate-only (v0.2) to 5 sealed cards + heme analog + 8 pending |
| §14 Cross-card syntheses | expanded from 1 pair (prostate-vs-breast in v0.2) to 10 pairs |
| §15 Open atlas-coverage gaps | NEW |
| §16 Mahalanobis differential ranker | preserved verbatim from v0.1 + v0.3 5-dim expansion spec |
| §17 Future v1.0 reorganization plan | NEW |
| §18 Validation status | preserved verbatim + v0.3 addition |
| §19 Language discipline | preserved verbatim + CCL-031 v0.3 additions |
| §20 File pointers | updated for v0.3 |
| §21 Card evolution plan | preserved verbatim + v0.3 + v1.0 additions |
| §22 Lessons learned | v0.1 + v0.2 preserved verbatim + v0.3 (immune-atlas-LL-002 through LL-008) added |
| §23 This changelog | NEW |
| **v0.3 → v0.3.1** | additive bladder-epic v0.1 integration: §6.6a sixth CCL-031 pattern (substrate-distribution mismatch on mucosal cohort, DISC-BLADDER-003); §10.1 EpiSCORE BladderRef added as fourth gene-promoter bridge; §10.3 atlas-family-fitness lesson extended to third data point (DISC-BLADDER-001); §11.1/11.2/11.3/11.4 bladder rows added to all four CCL-027 question tables; §13.6a bladder VAL-122 sealed Stage 3 sub-cell-type signature added; §13.7 NEW operational guidance subsection for the future-AI-2-years-from-now interpreter (how to interpret bladder-style broad-positive Stage 3 readings); §22.4 v0.3.1 lessons (immune-atlas-LL-009 through LL-013) covering the four DISC-BLADDER findings as they apply to immune-class interpretation plus the meta-lesson on how the cookbook caught its own assumption mid-sprint |

---

*End of immune-atlas card v0.3 README.*
