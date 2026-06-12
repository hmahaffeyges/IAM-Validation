# VAL-101 Pre-Registration — hcc-epic Run-Everything 25-Tile Per-Class A-Score with Full Etiology Stratification on TCGA-LIHC HM450

**Sealed:** 2026-04-28 (sealed before run-everything 25-tile β-value scoring; the pooled cycling-class result on this cohort is already published as VAL-064)
**Card:** hcc-epic v0.3 candidate (tissue arm 25-tile observation + Marcus-analog stratum analysis)
**Cohort:** TCGA-LIHC HM450 paired tumor/adjacent-normal — 50 candidate paired pairs (4 expected QC dropouts, n=46 carried forward consistent with VAL-064 sealed cohort)
**RNG seed:** 20260428

---

## Purpose

VAL-064 established hcc-epic tissue arm at pooled paired d = +0.498 (full HM450 secretory-class scoring against H_min=0.843264) with risk-factor stratification revealing:
- Non-viral HCC (n=34) paired d = +0.664
- Viral hepatitis (n=12) paired d = +0.023 (NULL, blunted by Villanueva 2015 adjacent-normal field defect)

VAL-101 applies the run-everything 25-tile per-class methodology (validated cookbook-wide via VAL-093/094/095/096 + CCL-039 confirmed at three colorectal cohorts) to the same TCGA-LIHC paired cohort. Three pre-locked questions:

1. **CCL-039 cross-tissue generalization test.** Does the Hepatocytes tile in HCC tumor-vs-adjacent-normal paired comparisons read strongly negative the same way the Colon_epithelial_cells tile reads negative in colorectal paired tumor-vs-normal? If yes, CCL-039 upgrades from "robustly-confirmed colorectal observation" to "framework-level rule across at least two cancer types." If no, CCL-039 may be colorectal-specific or may depend on tissue-architecture properties we haven't characterized yet.

2. **Viral-vs-non-viral blunting at the per-tile level.** Does the viral-hepatitis blunting persist at the per-tile Hepatocytes level, or only at the pooled-cycling-class level? If the Hepatocytes tile shows clear negative direction even in viral HCC where pooled cycling-class was null, that means tumor architecture IS disrupted in viral HCC — the chronic-infection field defect blunts the global pooled contrast, not the tile-specific cell-of-origin contrast. This refines the v0.2 mechanism story.

3. **Marcus-analog stratum characterization.** What does the run-everything 25-tile pattern look like in the "no documented risk" stratum (n=10 in VAL-064 cohort, patients who developed HCC without HBV / HCV / alcohol / NAFLD documented in their TCGA clinical record)? This is the closest available public analog to aggressive HCC arising without a chronic-driver risk factor. Stratum is descriptive-only at n=10 per CHK-2.7 (n<5 threshold for inferential claim is satisfied at n=10, but n=10 is small; magnitude precision will be limited by bootstrap CI width). The analytical purpose is not to claim a finding but to document the tile pattern (which tiles co-fire, which direction, what the cell-of-origin tile reads) so the pattern is on record for cross-validation against future no-risk-factor HCC cohorts.

---

## Cohort

**TCGA-LIHC HM450 paired tumor/adjacent-normal — 50 candidate pairs.** Same patient list as VAL-064. Files re-downloaded fresh from NIH GDC public API at run time per `LIHC_matched_manifest.json` (manifest SHA `760bf65a213da5a86cdd7ecde6ff6d46dad04777eafc00eb14f56356b0088371`). Files cached locally at `/home/claude/edear_working/VAL-101/lihc_downloads/` (100 files, ~1.3 GB). For independent reproduction: download via GDC public API per the manifest. Public access, no dbGaP application required.

**QC dropouts:** Pre-locked per VAL-064 — patients with <400,000 valid β values per sample drop out at the QC step. VAL-064 reported 4 dropouts → n=46 carried forward. Pre-locked: VAL-101 mirrors VAL-064's QC threshold exactly. n_carried_forward expected = 46.

**Clinical metadata** loaded from `LIHC_clinical.json` (already in iam_repo, derived from VAL-064 sealed cohort metadata pull). Risk factor classification (extracted at pre-seal time from clinical metadata; counts confirmed):

| Stratum | n (candidates) | Notes |
|---|---|---|
| HBV+ alone | 21 | Largest stratum; Asian-population prevalence pattern |
| HCV+ alone | 3 | |
| HBV+HCV co-infection | 2 | |
| Alcohol+ | 7 | |
| NAFLD+ | 1 | n<5 — descriptive-only per CHK-2.7 |
| Other (Tobacco-only, Diabetes-only, Schistosoma, Hemochromatosis, Unknown viral) | 6 | |
| **No_documented_risk** | **10** | **Marcus-analog stratum — patients with HCC and no HBV/HCV/alcohol/NAFLD documented in TCGA chart** |

Cross-stratum derived buckets for the analysis:
- **All_viral** = HBV+ ∪ HCV+ ∪ HBV+HCV = n=26
- **All_non_viral** = Alcohol+ ∪ NAFLD+ ∪ Other ∪ No_documented_risk = n=24

Honest framing on the no-documented-risk stratum: TCGA records what is in the patient's clinical chart at time of enrollment, not what was actually present biologically. The n=10 may include subclinical NAFLD, occult HBV not tested for in that era, environmental exposures (aflatoxin, hepatotoxin), or genetic predisposition (familial HCC, hereditary hemochromatosis) that wasn't documented. The stratum is the closest available public analog to aggressive HCC arising without one of the four canonical drivers; it is not a pure "healthy-baseline cohort."

---

## Method

1. **Pair files into patient-level (tumor, normal) tuples.** 50 patient pairs from manifest.
2. **QC threshold:** ≥400,000 valid β values per sample. Mirrors VAL-064 exactly.
3. **Load Loyfer 25-tile reference atlas** from `Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv` (atlas SHA recorded in results.json).
4. **Per tile, select top-100 marker CpGs** by |ref_β − mean(ref_β across other 24 tiles)|. Same methodology as VAL-098 + VAL-099.
5. **Per sample, score the run-everything 25-tile per-class A-score:** `A_tile(sample) = mean over top-100 marker CpGs of [ H(β) / H_min(class_of_tile) ]` where H_min depends on the class assignment of each tile (cycling = 0.856055, secretory = 0.843264, terminal = 0.7728, stromal = 0.862950, progenitor = 0.852216, immune = 0.838889).
6. **For every tile, compute paired Cohen's d** (A_tumor − A_normal) on QC-passed pairs. Bootstrap 10,000-iteration BCa-equivalent CI per tile.
7. **Stratified analysis** on the Hepatocytes tile (the cell-of-origin tile for HCC) and on the full 25-tile output:
   - by_etiology: All_viral (n≈26) vs All_non_viral (n≈24)
   - by_specific_etiology: HBV-alone, HCV-alone, HBV+HCV, Alcohol-alone, NAFLD-alone (n=1 — descriptive-only), Other, No_documented_risk
   - **Marcus-analog stratum:** No_documented_risk (n≈10) — descriptive-only; report point estimates + bootstrap CI + tile rank pattern; do NOT claim inferential pass/fail at this n.

Bootstrap RNG seed 20260428.

---

## Pre-registered outcomes

**O1_HEPATOCYTES_TILE_NEGATIVE_DIRECTION_CONFIRMED.** Hepatocytes tile (the cell-of-origin tile for HCC) reads paired d ≤ −0.5 in the pooled cohort with 95% CI upper bound < 0. Direction matches CCL-039 prediction for tumor-vs-adjacent-normal paired comparisons (cell-of-origin tile fidelity loss in tumor → negative d at the cell-of-origin tile). CCL-039 upgrades to cross-tissue confirmed pattern across at least two cancer types (colorectal + HCC).

**O2_HEPATOCYTES_TILE_NEGATIVE_PARTIAL.** Hepatocytes tile reads paired d in [−0.5, 0] with 95% CI upper bound at or near 0. Direction consistent with CCL-039 prediction but magnitude attenuated relative to the colorectal anchor (TCGA-COAD VAL-099 Colon_epithelial_cells tile d = −1.603). Hypothesis: HCC tile signal is attenuated by viral-hepatitis adjacent-normal field defect (consistent with VAL-064 pooled-cycling-class blunting), and the per-tile signal will be stronger in the all_non_viral stratum than in the all_viral stratum.

**O3_HEPATOCYTES_TILE_NULL.** Hepatocytes tile reads paired d in [−0.2, +0.2] with 95% CI crossing zero. Cell-of-origin tile fidelity-loss pattern does NOT generalize from colorectal to HCC at the pooled level. Convene with Heath; investigate whether the viral arm dominates the pooled signal and whether the non-viral arm (n≈24) shows the negative direction independently.

**O4_HEPATOCYTES_TILE_INVERTED_POSITIVE.** Hepatocytes tile reads paired d ≥ +0.5 with 95% CI lower bound > 0. Direction inverted from CCL-039 prediction. Convene with Heath; investigate whether HCC has a fundamentally different per-tile architecture than colorectal cancer (hepatocyte methylation may not de-differentiate the same way colon epithelial methylation does, or the Loyfer Hepatocytes tile may not behave as a clean cell-of-origin marker for HCC the way the Colon_epithelial_cells tile does for CRC).

**O5_DATA_INTEGRITY_FLAG.** Beta distribution check (CHK-3.1) fails OR cross-cohort baseline check (CHK-3.2) flags scale issue. Halt outcome interpretation pending data integrity resolution.

---

## Pre-locked stratum-level expectations

**Viral-vs-non-viral blunting at the per-tile level.** Pre-locked hypothesis based on VAL-064 pooled-cycling-class result: the all_non_viral stratum (n≈24) will show a stronger Hepatocytes tile negative direction than the all_viral stratum (n≈26). Specifically:
- All_non_viral Hepatocytes tile d expected ≤ −0.5 with CI upper bound < 0 (paralleling VAL-064 non-viral pooled-cycling-class d = +0.664).
- All_viral Hepatocytes tile d expected closer to 0 (paralleling VAL-064 viral pooled-cycling-class d = +0.023).

If the per-tile pattern matches this expectation: confirms that viral-hepatitis blunting operates on global cycling-class signal AND on per-tile cell-of-origin signal. If the per-tile Hepatocytes shows clear negative direction in BOTH viral and non-viral strata even though pooled cycling-class was null in viral: refines the mechanism — viral-hepatitis field defect inflates the global entropy reading (because adjacent-normal liver is already inflamed and methylation-disordered) but does NOT mask the local cell-of-origin tile fidelity loss (because the tumor cells still de-differentiate at the Hepatocytes-discriminating CpGs regardless of the field defect). That would be a meaningful refinement of the v0.2 mechanism story.

**Marcus-analog stratum (no_documented_risk, n≈10).** Pre-locked: descriptive-only at this n per CHK-2.7. Report:
- Hepatocytes tile paired d + bootstrap 95% CI (wide, expected)
- Top 5 tiles by |paired d| with magnitudes + directions
- Cell-of-origin tile rank within the 25-tile panel (is Hepatocytes the most-negative tile, or is it not in the top 3?)
- Comparison of mean ΔA pattern to the all_non_viral pooled stratum (does the no_documented_risk pattern look like the alcohol+NAFLD subset, or does it look different?)
- Hepatocytes tile direction concordance with CCL-039 prediction (yes/no/partial)

The analytical purpose at this stratum is documentation, not inference. The descriptive readout becomes a reference pattern for cross-validation against any future no-risk-factor HCC cohort that becomes available (publication search ongoing; UK Biobank dbGaP application would be a v0.2+ task).

---

## CHK-4.11 application (CCL-039 prereg-design rule)

This prereg uses pattern-aware O1 criterion language as required by CHK-4.11:
- **Acceptable** (and used here): "Hepatocytes tile reads paired d ≤ −0.5 in the pooled cohort with 95% CI upper bound < 0" — direction-and-magnitude criterion, comparison-type-explicit (tumor-vs-adjacent-normal-paired).
- **NOT used** (would violate CHK-4.11): "Hepatocytes tile shows positive d" or "Hepatocytes tile is largest |d|".

The expected direction is NEGATIVE because this is a tumor-vs-adjacent-normal-paired comparison, where cell-of-origin tile fidelity is expected to degrade in tumor.

---

## Reproducibility

- **Source data:** TCGA-LIHC HM450 paired tumor/adjacent-normal .txt files via NIH GDC public API per `LIHC_matched_manifest.json` (50 patients × 2 samples = 100 files, ~1.3 GB total). Public access, no dbGaP application required.
- **Clinical metadata:** Pre-existing `LIHC_clinical.json` from VAL-064 sealed cohort metadata pull (committed to GitHub repo).
- **Reference atlas:** Loyfer 25-tile array atlas at `Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv`. Atlas SHA recorded in results.json.
- **Methodology mirror:** Same per-tile A-score methodology as VAL-098 + VAL-099 (top-100 marker CpGs per tile, A-score = mean(H(β)/H_min(class)) over markers).
- **RNG seed:** 20260428.
- **Environment:** Python 3 stdlib + numpy + pandas + scipy + matplotlib (or stdlib-only equivalent matching VAL-099 dependency-free style).

---

## EDEAR commercial deployment unaffected

Per CCL-037, VAL-101 is retrospective cookbook validation with no impact on EDEAR commercial deployment. The single-pipeline patient-vs-internal-reference architecture is structurally insulated from any cohort-coverage or stratum-power limitations documented in this analysis.

---

## What this VAL is NOT

- It is NOT a new tissue-arm validation — VAL-064 already validated hcc-epic tissue arm at pooled d = +0.498 with full risk-factor stratification.
- It is NOT a claim that the no_documented_risk stratum signal is established — n=10 is descriptive-only.
- It is NOT a falsification test for hcc-epic — VAL-101 is an analytical extension to the per-tile level, not a replacement.
- It is NOT a clinical-deployment-relevant finding — the 25-tile output is currently a methodology-development tool for the cookbook; deployment uses the full Stage 2 Moss NNLS deconvolution per the existing card spec.
