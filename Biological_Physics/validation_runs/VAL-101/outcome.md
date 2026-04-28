# VAL-101 Outcome — hcc-epic Run-Everything 25-Tile Per-Class A-Score with Full Etiology Stratification on TCGA-LIHC HM450

**Date:** 2026-04-28
**Card:** hcc-epic v0.2 (no change to card; biological readouts do NOT propagate from this VAL)
**Cohort:** TCGA-LIHC HM450 paired tumor/adjacent-normal — 50 candidate paired pairs, 46 QC-passed (mirrors VAL-064 sealed cohort)
**Pre-registration SHA:** `fa366bf00316597bb65032b747029133acb5f1bbb40f6251094b563732185512`
**Sealed at:** 2026-04-28T19:53:19.249263+00:00
**RNG seed:** 20260428
**Outcome label:** **`O5_DATA_INTEGRITY_FLAG`**
**Runtime:** 49.8 s

---

## Why this VAL outcome is O5_DATA_INTEGRITY_FLAG

The pre-locked CHK-3.1 beta distribution check tripped. Pre-registered thresholds: extreme [<0.05 or >0.95] >30% AND middle [0.4-0.6] <10%. Observed: extreme 26.6%, middle 9.1%. The middle threshold passes (9.1% < 10%); the extreme threshold misses (26.6% < 30%). Bimodal raw β signature: pre-registered criterion not met.

Per the pre-locked outcome decision matrix, this triggers `O5_DATA_INTEGRITY_FLAG`. Per CCL-032 diagnostic order (data integrity → biology → framework), the biological readouts produced under this VAL do NOT get interpreted as biology and do NOT propagate to the hcc-epic card or any cookbook reference document.

This is the right call because:

1. **The pre-registration was sealed before β-access.** SHA `fa366bf00316597bb65032b747029133acb5f1bbb40f6251094b563732185512` at 2026-04-28T19:53:19 UTC. The threshold was a pre-locked criterion. It tripped. Prereg discipline requires honoring sealed criteria; relaxing them after a desirable biological readout is the exact failure mode that prereg discipline exists to prevent.

2. **CHK-4.8 honest-revision does NOT apply here.** CHK-4.8 covers structurally degenerate pre-locked criteria — criteria that cannot logically discriminate the outcomes they were intended to discriminate (cf. VAL-097 where the auto-assigned O2_CYCLING_DISTRIBUTED criterion was computed identically to its own falsification, making the criterion structurally degenerate). CHK-3.1 in VAL-101 is not structurally degenerate. It tripped because the threshold was misspecified for the platform. Misspecification is a prereg-design error, not a structural-degeneracy condition. The cookbook protocol does not authorize post-hoc threshold relaxation for misspecification.

3. **The cookbook precedent argument cuts the wrong way.** It is true that the same TCGA HM450 sesame Level 3 substrate that anchors VAL-062 / VAL-098 / VAL-099 reads extreme ~24-27% / middle ~9% on the same check (verified post-hoc on cached COAD files used by VAL-099). It is also true that those VALs produced clean biological signal at framework-predicted effect sizes. But those VALs never explicitly ran CHK-3.1. The fact that they would have read similar values does NOT retroactively justify relaxing the pre-locked threshold in VAL-101. It justifies tightening prereg discipline going forward (by setting platform-specific thresholds) and re-running the present VAL under a properly-tuned threshold. The cookbook precedent does not authorize me to say "previous VALs would also have failed this check if they had run it, so I'll waive it for this one too."

4. **Clean biology makes this temptation more dangerous, not less.** The biological readouts in `results.json` look strong: Hepatocytes tile pooled paired d = −1.521, CCL-039 cross-tissue confirmation, viral-vs-non-viral mechanism refinement, Marcus-analog stratum documented. None of that is permitted to influence a sealed data-integrity decision. If sealed criteria are bent when the resulting biology is exciting, the cookbook has no real prereg discipline.

The outcome stands as O5_DATA_INTEGRITY_FLAG. The biological readouts in results.json are descriptive-only supplementary documentation and do not propagate.

---

## What happens next — honest path forward, not a same-day re-seal

The proper path to recover the biological-propagation status of these readouts is NOT to immediately seal a new VAL with a threshold derived from the data that just tripped. That is post-hoc threshold accommodation with a SHA stamp on it; it is not pre-registration. An attempt at this (VAL-102, sealed and voided 2026-04-28T20:35Z within minutes; audit trail preserved at `Biological_Physics/validation_runs/VAL-102/VOIDED_BEFORE_EXECUTION.md`) has been logged as a methodological self-correction.

The proper paths are either:

1. **Calibration-cohort path.** Set a CHK-3.1 platform threshold for TCGA HM450 sesame Level 3 using a calibration cohort that is structurally separated from any active hcc-epic test cohort. Candidates include TCGA samples from a tissue NOT currently under hcc-epic test (e.g., TCGA-KIRC kidney adjacent-normal, TCGA-PRAD prostate adjacent-normal). Measure the bimodality distribution on the calibration cohort. Set the platform threshold from that distribution. Seal it. THEN run a future hcc-epic VAL on the TCGA-LIHC test cohort under the calibration-derived threshold.

2. **CCL-040 deferral path.** Process the TCGA-LIHC .idat files through sesame from raw IDAT input (the standard TCGA pipeline produces .idat → sesame Level 3 .txt; we have the Level 3 product, but the upstream .idat files would let us verify the pipeline output and rule out any subtle normalization choice that lifted middle from 7-8% to 9.1%). Confirm bimodality at the standard pipeline output. Re-run hcc-epic test under reprocessed betas.

Both paths are multi-VAL workstreams, not same-session work. Both are honest. The first path is the proper extension of the cookbook to a new platform calibration; it produces a generally-applicable threshold for any future TCGA HM450 sesame Level 3 cohort. The second path follows the CCL-040 precedent already established for VAL-100; it is more specific to this cohort but more directly comparable to how VAL-100 was deferred.

VAL-101 stays as O5_DATA_INTEGRITY_FLAG until one of these paths is executed and produces a properly-pre-registered re-run. CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION is logged as the cookbook lesson; it documents the need for platform-specific thresholds going forward, but it does NOT retroactively rescue VAL-101's biological interpretation.

---

## Cohort + method

50 candidate paired patient pairs from TCGA-LIHC HM450 (the same patient list as VAL-064). 100 .txt files re-downloaded fresh from NIH GDC public API at run time. 46 pairs QC-passed (≥400,000 valid β per sample). Methodology: run-everything Loyfer 25-tile per-class A-score, top-100 marker CpGs per tile, paired Cohen's d on (tumor − normal) per tile, bootstrap 10000 iterations, RNG seed 20260428. Mirrors VAL-098 / VAL-099 methodology exactly.

Patient stratum assignments (n=46 QC-passed):

| Stratum | n |
|---|---|
| HBV+ alone | 19 |
| HCV+ alone | 3 (descriptive-only per CHK-2.7) |
| HBV+HCV co-infection | 2 (descriptive-only per CHK-2.7) |
| Alcohol+ | 6 |
| NAFLD+ | 1 |
| Other (Tobacco-only, Diabetes-only, Schistosoma, Hemochromatosis, Unknown viral) | 5 |
| No_documented_risk | 10 |
| All_viral (HBV+HCV+co-infection) | 24 |
| All_non_viral (Alcohol+NAFLD+Other+No_documented_risk) | 22 |

---

## CHK-3.1 result — sealed outcome trigger

Pre-locked thresholds: extreme [<0.05 or >0.95] >30% AND middle [0.4-0.6] <10%.

Observed: extreme **26.6%**, middle **9.1%**.

Pass criterion: BOTH conditions required. Middle passes (9.1% < 10%). Extreme misses (26.6% < 30%). Bimodal raw β signature: criterion not met. Outcome trigger: O5_DATA_INTEGRITY_FLAG per pre-locked decision matrix.

---

## Biological readouts (descriptive supplementary documentation only — DO NOT propagate)

Per CCL-032 diagnostic order (data integrity → biology → framework), the following readouts are descriptive supplementary documentation. They do NOT propagate to the hcc-epic card, CCL catalog, README_MASTER, TESTING_CHECKLIST, EDEAR_PIPELINE_OFFICIAL_REFERENCE, or any other cookbook document. The numbers are reported here for completeness of the run record, not for biological interpretation. Their proper validation pathway is VAL-102 with platform-tuned CHK-3.1.

### Pooled 25-tile output (n=46) — descriptive supplementary

| Rank | Tile | Class | Paired d | 95% CI | p |
|---|---|---|---|---|---|
| 1 | Colon_epithelial_cells | cycling | +1.807 | [+1.486, +2.267] | <0.0001 |
| 2 | Head_and_neck_larynx | cycling | +1.585 | [+1.254, +2.117] | <0.0001 |
| 3 | Hepatocytes | secretory | −1.521 | [−2.192, −1.182] | <0.0001 |
| 4 | Bladder | cycling | +1.091 | [+0.819, +1.463] | <0.0001 |
| 5 | Lung_cells | cycling | +0.951 | [+0.659, +1.358] | <0.0001 |

### Stratified Hepatocytes-tile readouts — descriptive supplementary

| Stratum | n | Hepatocytes paired d | 95% CI |
|---|---|---|---|
| All_viral | 24 | −1.726 | [−3.025, −1.117] |
| All_non_viral | 22 | −1.393 | [−2.301, −1.064] |
| HBV+ alone | 19 | −2.036 | [−3.185, −1.499] |
| Alcohol+ | 6 | −1.681 | [−7.273, −0.774] |
| Other | 5 | −1.682 | [−12.519, −1.397] |
| No_documented_risk | 10 | −1.141 | [−6.157, −0.847] |

(Strata with n<5 omitted per CHK-2.7; full table available in stratified.json.)

### Marcus-analog stratum (n=10) — descriptive supplementary

The no_documented_risk stratum readouts are documented in stratified.json. The numbers are descriptive-only per CHK-2.7 (n=10 below the threshold for inferential claims) AND per CCL-032 (data integrity flag prevents biological interpretation regardless of sample size). Their cross-validation pathway is VAL-102.

---

## CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION (new cookbook lesson, logged from VAL-101)

**Source:** VAL-101 prereg-threshold trip + post-hoc verification against VAL-099 cached TCGA-COAD HM450 sesame Level 3 data (extreme 24.4%, middle 9.7% on the same check methodology).

**Lesson:** CHK-3.1 beta distribution check thresholds must be platform-specific. The original cookbook threshold (extreme >30% AND middle <10%) was tuned against raw EPIC β in VAL-100 prereg. TCGA HM450 sesame Level 3 — the cookbook's standard public tissue-validation substrate, used in VAL-058 / VAL-060 / VAL-062 / VAL-063 / VAL-064 / VAL-098 / VAL-099 — reads extreme ~24-27% / middle ~9% on the same check. This is bimodal raw β with a slightly less extreme bimodality than raw EPIC due to dye bias correction in the standard TCGA pipeline. The β values are not normalized residuals or batch-corrected output; the sesame Level 3 product is bimodal raw β.

**Distinction from CCL-040.** CCL-040 covers PROCESSED OUTPUT (residual M-values, batch+chip+age+HPV-corrected, noob-bg-corrected with additional normalization) — the kind that loses bimodal raw β signature entirely (extreme 3.9% / middle 6.8% in VAL-100 GSE282666; extreme low and middle high). CCL-041 is about raw β bimodality manifesting at slightly different threshold values across raw-β platforms (sharper on raw EPIC, softer on sesame-corrected HM450). Two distinct concerns; CCL-041 does NOT generalize CCL-040's deferral pathway.

**Operational rule going forward.** CHK-3.1 thresholds are platform-specific. The thresholds for any new platform must be set by a calibration VAL on a structurally-separate cohort (NOT by retroactive accommodation of the data that triggered the discovery of platform mismatch):

| Platform | extreme threshold | middle threshold | Status |
|---|---|---|---|
| Raw EPIC β / EPIC v2.0 β (un-normalized) | > 30% | < 10% | Established (VAL-100) |
| TCGA HM450 sesame Level 3 β | TBD | < 10% | **Calibration VAL needed** — must be done on a cohort structurally separated from any active hcc-epic test cohort |
| Other platforms | TBD | TBD | Document at first calibration VAL on platform |

**Why a calibration VAL is required, not a retroactive threshold.** The post-hoc verification documented above (TCGA-COAD VAL-099 cohort reads extreme 24.4% / middle 9.7% on the same check) is informative but cannot be the basis for setting the TCGA HM450 platform threshold. Setting the threshold from data that is also being interpreted under the threshold is circular. The proper calibration VAL would use TCGA samples from a tissue NOT under active hcc-epic test (TCGA-KIRC, TCGA-PRAD adjacent-normal, etc.), measure the bimodality distribution there, set the threshold from THAT distribution, seal it, and apply it to future hcc-epic test cohorts as a pre-registered platform criterion.

**Application to VAL-101.** VAL-101's pre-locked threshold (extreme >30%, middle <10%) was the raw-EPIC default. It was misspecified for the TCGA HM450 sesame Level 3 substrate. CCL-041 documents this lesson going forward; it does NOT retroactively rescue VAL-101's outcome. VAL-101 stands as O5_DATA_INTEGRITY_FLAG. The biological-propagation pathway requires a properly pre-registered platform threshold derived from a structurally-separate calibration cohort (or, alternatively, the CCL-040 raw-IDAT deferral pathway). See "What happens next — honest path forward" below.

**Lesson logged in cookbook:** add CCL-041 to LESSONS_LEARNED.md. Update CHK-3.1 in TESTING_CHECKLIST.md to reference platform-specific thresholds. The platform threshold for TCGA HM450 sesame Level 3 is NOT set by VAL-101; it requires a future calibration VAL on a structurally-separate cohort.

---

## EDEAR commercial deployment unaffected

Per CCL-037, VAL-101 is retrospective cookbook validation with no impact on EDEAR commercial deployment. Sealed-outcome flag has no deployment consequence.

---

## Reproducibility triple (CHK-7.6)

### Source code

`Biological_Physics/validation_runs/VAL-101/val_101.py`. Python 3 stdlib only. 19 KB.

### Inputs

- TCGA-LIHC HM450 paired tumor/adjacent-normal .txt files via NIH GDC public API per `LIHC_matched_manifest.json`. 100 files, ~1.3 GB. Public access, no dbGaP application required.
- `LIHC_clinical.json` from VAL-064 sealed cohort metadata pull (committed to GitHub repo).
- Loyfer 25-tile reference atlas at `Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv`.

### Environment

- Python 3.12 + stdlib only (math, csv, gzip, json, hashlib, os, random, re, time)
- Expected runtime: ~50 s on a modern laptop after the 100 .txt files are downloaded
- Expected memory: < 4 GB (stream-process per sample, release β dictionaries after scoring)

### Expected headline outputs

```
CHK-3.1 sealed-threshold check:    extreme 26.6%, middle 9.1% — FAILED (extreme < 30% threshold)
Sealed outcome label:              O5_DATA_INTEGRITY_FLAG
Pre-reg seal:                      SHA fa366bf00316597b...
RNG seed:                          20260428
Runtime:                           ~50 seconds

Biological readouts (DESCRIPTIVE ONLY, do not propagate):
  Pooled Hepatocytes tile d:       −1.521 [−2.192, −1.182]
  All_viral Hepatocytes (n=24):    −1.726 [−3.025, −1.117]
  All_non_viral (n=22):            −1.393 [−2.301, −1.064]
  No_documented_risk (n=10):       −1.141 [−6.157, −0.847]

CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION logged: TCGA HM450 sesame Level 3
  threshold tuning needed (extreme >20%, not >30%). Apply in VAL-102 prereg.
```

---

## Files in this VAL bundle

| File | Size | Purpose |
|---|---|---|
| `prereg.md` | 13 KB | Pre-registration document |
| `PREREG_SEAL.txt` | ~200 B | Prereg seal with SHA-256 |
| `val_101.py` | 19 KB | Reproducible Python script |
| `results.json` | ~13 KB | Pooled + 25-tile + stratified + Marcus-analog readouts (descriptive supplementary) + sealed_outcome_decision block |
| `stratified.json` | ~5 KB | Stratified-only summary (descriptive supplementary) |
| `per_sample.csv` | ~10 KB | Per-sample per-tile ΔA values |
| `outcome.md` | this file | Outcome write-up — O5_DATA_INTEGRITY_FLAG sealed; CCL-041 logged |

---

## Lessons logged

- **CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION (new cookbook lesson).** TCGA HM450 sesame Level 3 betas read extreme ~24-27% / middle ~9% on the CHK-3.1 bimodal raw β check (verified post-hoc on VAL-099 cached COAD files). The original raw-EPIC threshold (>30% extreme) is misspecified for this platform. Platform-specific thresholds are required going forward — but the TCGA HM450 platform threshold value itself MUST be set by a future calibration VAL on a structurally-separate cohort, NOT by retroactive accommodation of the values that triggered the discovery.
- **Prereg discipline reinforced.** A sealed CHK-3.1 threshold that trips on a substrate-with-clean-biology is still a triggered O5 flag. CHK-4.8 honest-revision is reserved for structural degeneracy, not threshold misspecification. The biology being exciting does not authorize post-hoc threshold relaxation.
- **Self-correction logged: VAL-102 voided before execution.** A VAL-102 prereg was sealed at 2026-04-28T20:31:23Z with a TCGA HM450 platform threshold (extreme >20%) derived from the very data VAL-102 was scheduled to interpret. This was identified as post-hoc threshold accommodation and voided at 2026-04-28T20:35Z within minutes of seal, before any execution. Audit trail preserved at `Biological_Physics/validation_runs/VAL-102/VOIDED_BEFORE_EXECUTION.md` with the original SHA `2b77ad9d3b69554a0658260756db0f08722e2be3fa96eb48aad9213974f4717c`. The cookbook does not delete sealed records; it marks them and explains.
- **Biological readouts in VAL-101 results.json** (Pooled Hepatocytes tile d = −1.521, viral d = −1.726, non-viral d = −1.393, No_documented_risk d = −1.141, CCL-039 cross-tissue cross-cohort pattern, etc.) remain DESCRIPTIVE SUPPLEMENTARY documentation only. They do NOT propagate. Their proper inferential pathway requires either (a) a calibration VAL on a structurally-separate cohort that establishes the TCGA HM450 platform threshold, then a re-run of TCGA-LIHC test under that pre-registered threshold, OR (b) a CCL-040 raw-IDAT deferral re-processing the TCGA-LIHC .idat files through sesame and re-running with verified pipeline output. Both are multi-VAL workstreams.
