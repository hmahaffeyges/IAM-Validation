# VAL-100 Pre-Registration — crc-epic Under-50 Buffy Coat Polyp Stage 1 Immune A-Score on GSE282666

**Sealed:** 2026-04-28 (sealed before β-value scoring)
**Card:** crc-epic v2.4 (early-onset rectal subsection — under-50 polyp arm)
**Cohort:** GSE282666 (Kumar/Brown/Yow, University of Miami) — n=51 buffy coat EPIC 850K, all patients under age 50, undergoing colonoscopy with PNP+/PNP- status
**Platform:** Illumina Infinium MethylationEPIC v2.0 (GPL33022 — note v2 platform)
**RNG seed:** 20260428

---

## Purpose

VAL-098 confirmed cycling-class architectural drift in the rectal subsite (TCGA-READ paired d = +0.612). VAL-099 reproduced the colon anchor (TCGA-COAD pooled d = +0.7241) and provided descriptive direction signal in the under-50 colon stratum (n=3, ΔA = +0.0357). The under-50 evidence chain needs proper-power confirmation in a buffy-coat blood pre-neoplastic cohort.

GSE282666 is a published n=51 buffy coat EPIC 850K cohort of patients ALL UNDER AGE 50 undergoing colonoscopy, with pre-neoplastic polyp (PNP) status recorded. PNP+ = tubular adenomas + sessile serrated adenomas (the two pre-neoplastic lesion types that progress to early-onset CRC). PNP- = clean colonoscopy. n=16 PNP+ / n=35 PNP-.

VAL-100 tests whether the crc-epic Stage 1 universal Xu-538 immune A-score extends backward in disease trajectory from pre-diagnostic invasive CRC (VAL-047 d = −0.33 on EPIC-Italy GSE51032 secretory-class breast comparator cohort) to pre-neoplastic polyps. The crc-epic Stage 1 expected direction is NEGATIVE (CCL-019 compartment-flip: blood immune A-score depressed, tumor cycling A-score elevated). Direction expectation for PNP+ vs PNP-: same negative direction as pre-diagnostic invasive CRC, with magnitude expected smaller (polyps are pre-neoplastic, not invasive).

This is a within-cohort comparison (CHK-3.8 condition 1 satisfied) — no cross-cohort calibration problem. Single cohort, single pipeline (minfi v1.40.0 noob-bg-corrected), single demographic stratum (Miami clinic under-50 patients), PNP+ vs PNP- contrast.

---

## Cohort

**GSE282666 — University of Miami Gastroenterology, Kumar et al. 2024.**

- n=51 buffy coat samples on Illumina EPIC v2.0 (GPL33022)
- All patients under age 50 (cohort design constraint per Kumar 2024)
- 16 PNP+ (with pre-neoplastic polyps: tubular adenomas + sessile serrated adenomas)
- 35 PNP- (clean colonoscopy)
- minfi v1.40.0 noob-bg-corrected β values, supplementary file `GSE282666_Betas.csv.gz` (235 MB compressed, 936,991 CpG rows × 51 samples)
- Sentrix-position column headers mapped to GSM IDs via IDAT URL parsing in series matrix (51/51 columns mapped, mapping saved to `column_mapping.json`)

**Patient labels** extracted from !Sample_title field of series matrix (patterns "Buffy coat from patient with PNPs" vs "Buffy coat from patient without PNPs"). Saved to `clinical_metadata.json`. PNP+ count: 16. PNP- count: 35. No UNKNOWN status.

**Important platform note (CHK-3.8 / CHK-3.1 risk).** GSE282666 is **EPIC v2.0** (GPL33022), not EPIC 850K v1.0 (GPL21145). EPIC v2 has ~935K probes vs EPIC v1 ~865K, with overlapping but not identical probe sets. The Xu-538 panel was originally designed against HM450 / EPIC v1; coverage on EPIC v2 must be checked at run time. Pre-locked: report Xu-538 / EPIC-v2 coverage in results.json. Coverage drop of >10% triggers CHK-3.1 panel-transferability flag (analogous to VAL-076 cervical-LBC finding).

---

## Method

1. **Load Xu-538 panel** (538 CpGs) from `xu538_panel.json`.
2. **Stream-parse `GSE282666_Betas.csv.gz`** keeping only Xu-538 panel CpGs. Note: probe ID format in this file is `cg00008800_BC11`-style (with EPIC-v2 suffix). Match by `cgXXXXXXX` prefix.
3. **Score every sample** with pooled A_immune: `A_immune(sample) = mean over Xu-538 CpGs (in coverage) of [ H(β) / H_min(immune) ]` where H_min(immune) = 0.838889 (G-002 MCMC posterior; panc-LL-007 universal Stage 1 H_min rule applies — Xu-538 always scores against H_min_immune regardless of disease).
4. **Compute case-control Cohen's d** (PNP+ vs PNP-) on A_immune.
5. **Bootstrap 95% CI** with 10,000 iterations.
6. **Welch's t** and approximate p-value.
7. **Within-cohort baseline check (CHK-3.2):** Compare PNP- mean ± SD on Xu-538 against the EPIC-Italy GSE51032 healthy buffy coat baseline (Cohort A in VAL-047 / VAL-093/094/095/096). If PNP- baseline differs from EPIC-Italy by > 1 anchor-SD, flag as healthy-baseline-cohort-heterogeneity. Within-cohort comparison remains valid; cross-cohort interpretation requires platform-stratified threshold.
8. **Pre-locked outcome decision matrix** (see below).
9. **Beta distribution check (CHK-3.1):** Confirm raw β distribution on Xu-538 panel — β > 30% extremes [<0.05 or >0.95] AND <10% middle [0.4-0.6] = bimodal raw β signature. Failure (uniform β with high middle fraction) flags residual-M-values data integrity issue.

RNG seed 20260428.

---

## Pre-registered outcomes

**O1_PNP_NEGATIVE_DIRECTION_DETECTED.** PNP+ vs PNP- Cohen's d ≤ −0.30 with 95% CI upper bound < 0. Direction matches CCL-019 crc-epic blood immune compartment-flip prediction. Magnitude consistent with smaller-than-VAL-047 (polyps are pre-neoplastic, smaller signal expected than invasive pre-dx CRC at d = −0.33). Beta distribution check passes (CHK-3.1).

**O2_PNP_DETECTABLE_DIRECTION_PARTIAL.** PNP+ vs PNP- Cohen's d in negative direction, but |d| ≤ 0.30 OR 95% CI crosses zero. Direction signal but magnitude underpowered at n=51 / 16 vs 35. Sub-stratification (e.g. multiple polyps vs single polyp; advanced adenoma vs hyperplastic only) not in this VAL's scope but flagged as v0.2 next step.

**O3_PNP_NULL.** PNP+ vs PNP- Cohen's d in [−0.20, +0.20] with 95% CI crossing zero. No detectable per-patient pre-neoplastic immune A-score signal at this cohort design at this n.

**O4_PNP_INVERTED_POSITIVE.** PNP+ vs PNP- Cohen's d ≥ +0.30 with 95% CI lower bound > 0. Inverted direction from CCL-019 crc-epic blood immune compartment-flip prediction. Convene with Heath; investigate cohort design (e.g. stress / inflammation contamination at colonoscopy procedural visit).

**O5_DATA_INTEGRITY_FLAG.** Beta distribution check fails (CHK-3.1). Residual-M-values or processed-betas-not-raw issue suspected. Report numbers descriptively; do NOT take card direction; defer to v0.2+ raw IDAT processing through minfi/sesame.

**O6_PANEL_TRANSFERABILITY_FLAG.** Xu-538 / EPIC-v2 coverage drops > 10% AND Cohen's d magnitude is at the small-effect boundary. Raises panel-transferability question (analogous to VAL-076 cervical-LBC LBC vs buffy-coat substrate transferability). Re-design panel for EPIC-v2 platform or extend coverage check.

---

## Why this VAL completes the under-50 evidence chain

The early-onset rectal subsection of crc-epic v2.4 has three evidence sources:

| VAL | Cohort | n | Stratum | Method | Pre-locked direction expectation |
|---|---|---|---|---|---|
| VAL-098 | TCGA-READ paired tumor/normal | 7 | mostly 50+ (1 under 50) | tissue cycling-class | Positive (extends VAL-062) |
| VAL-099 | TCGA-COAD paired tumor/normal | 26 | 3 under 50, 21 50+ | tissue cycling-class | Positive (reproduces VAL-062) |
| VAL-100 | GSE282666 buffy coat polyp | 51 | ALL under 50 | blood immune Xu-538 | Negative (CCL-019 compartment-flip) |

VAL-100 is the proper-power under-50 confirmation arm — single cohort, single pipeline, all under 50, n=51. The first two VALs anchor the tissue arm (positive direction); VAL-100 anchors the blood arm under-50 stratum (negative direction by CCL-019 compartment-flip). The two arms together form the complete under-50 evidence base for crc-epic v2.4.

If VAL-100 fires O1 or O2 in the negative direction, the under-50 evidence chain is complete: tissue arm direction confirmed (VAL-098 + VAL-099) AND blood arm under-50 direction confirmed (VAL-100). If VAL-100 fires O3 (null) or O4 (inverted positive), the under-50 blood arm is unresolved and requires either a follow-up validation or a card commentary describing the open question.

---

## EDEAR commercial deployment unaffected

Per CCL-037, VAL-100 is retrospective cookbook validation with no impact on EDEAR commercial deployment. The single-pipeline patient-vs-internal-reference architecture is structurally insulated from cookbook-validation cohort coverage limitations.

---

## Reproducibility

- **Source:** GSE282666 series matrix + supplementary betas file from NCBI GEO public FTP. URL: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE282nnn/GSE282666/`. No dbGaP application, no biobank gating, fully Tier 1 public.
- **Files downloaded at run time:** `GSE282666_series_matrix.txt.gz` (4.3 KB, metadata) + `GSE282666_Betas.csv.gz` (235 MB, β matrix).
- **Reference:** Xu-538 panel (frozen since v0.1, `xu538_panel.json` in card resources, 538 CpGs, panel SHA recorded in results.json).
- **RNG seed:** 20260428.
- **Environment:** Python 3 stdlib + (math, gzip, json, csv, urllib).
- **Healthy comparator** for CHK-3.2 cross-cohort baseline check: GSE51057 EPIC-Italy menarche cohort buffy coat cancer-free subset (the Cohort A reference used in VAL-047 / VAL-093). Mean A_immune anchor ≈ 0.4384 ± 0.0244 (from VAL-082 reported Italian healthy comparator).

---

## Pre-locked notes

- This is the FIRST VAL run on EPIC v2.0 (GPL33022) in the entire cookbook. EPIC-v1 panels may have reduced coverage on EPIC v2 due to probe set differences (some probes dropped, some new probes added, some renamed with EPIC-v2 suffixes). Coverage check is mandatory.
- Patient demographics (age within 18-50, sex, race, ethnicity) are not deposited in series matrix !Sample_characteristics_ch1 fields — only "tissue: Buffy Coat" was recorded. Sex and race stratification cannot be performed from public metadata for this VAL. Kumar 2024 paper Table 1 reports cohort-level demographics; per-sample mapping is not in the public deposit. Sex stratification is documented as a v0.2+ follow-up requiring corresponding-author contact.
- The Kumar 2024 paper reports OR=1.17 per 1-year GrimAge acceleration (logistic regression). VAL-100 does NOT replicate the GrimAge analysis; VAL-100 tests the crc-epic Stage 1 universal Xu-538 panel A-score, which is a different metric than GrimAge.
