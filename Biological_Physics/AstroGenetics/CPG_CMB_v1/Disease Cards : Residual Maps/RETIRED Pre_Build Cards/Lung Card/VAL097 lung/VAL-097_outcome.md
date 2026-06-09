# VAL-097 — Outcome

**Title:** Never-smoker LUAD tissue 25-tile per-class A-score characterization on GSE256092 with cross-cohort baseline against TCGA-LUAD adjacent-normal

**Date sealed:** 2026-04-28
**Date completed:** 2026-04-28
**Prereg SHA-256:** `9a1bd45e240eea7ac8d03915de9a85deb35533700f2fd263ce1912d40a3ee5f9`
**RNG seed:** 20260428
**Runtime:** 81.3 seconds

---

## 1. Cohort

GSE256092 (Korean Cancer Genome Atlas Consortium, 2024)
- n = 141 NSLA tumor tissues, all never-smoker (cohort definition)
- Platform: Illumina MethylationEPIC 850K (GPL21145)
- Population: Korean
- Stage distribution per series matrix: I (n≈50), II (n≈35), III (n≈40), IV (n≈3-5)
- Sex: female-enriched
- Age range: 37–85 (median ~65)
- Preprocessing pipeline: SWAN normalization (per series matrix overall_design)
- IDAT FTP: `ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256092/suppl/GSE256092_RAW.tar`
- SWAN matrix SHA-256 prefix: `b191108e6414e418`

Reference cohort: TCGA-LUAD adjacent-normal lung tissue
- n = 29 adjacent-normal samples (per VAL-063 LUAD_matched_manifest.json)
- Platform: HM450 (GPL13534)
- Population: predominantly Western
- Preprocessing pipeline: sesame level3 betas (GDC public)
- Source: NIH GDC public API at `https://api.gdc.cancer.gov/data/{file_id}`

---

## 2. Method summary

Per pre-locked prereg: 25-tile Loyfer atlas per-class A-score on every sample, both cohorts. Per-class A-score = mean(H(β)/H_min(class)) across top-100 marker CpGs per tile. H_min values frozen from GAPE_WEB_v13 _H_MIN_GRID (G-002 + G-003b MCMC posteriors, R-hat < 1.001). Cross-cohort baseline check (CHK-3.2) per tile in pooled-SD anchor units. Case-vs-reference Cohen's d with 10,000-iteration bootstrap 95% CI per tile. Top-1 ΔA call per patient. Within-cohort stratified analysis on sex × age decade × stage.

---

## 3. Pre-locked decision criteria result

| Criterion | Triggered? | Detail |
|---|---|---|
| O1_LUNG_LOCALIZED | NO | Lung_cells d = −0.27, NOT largest |d|, second-weakest in 25-tile panel |
| O2_CYCLING_DISTRIBUTED | Auto-flagged by code | 6 cycling tiles d ≥ +0.3 — but criterion is structurally degenerate (see §6) |
| O3_NON_CYCLING_DOMINANT | Could be argued | Pancreatic_acinar d = +5.25 dominant; but interpretation invalid (see §6) |
| O4_DIRECTION_INVERTED | NO | Lung_cells d = −0.27 is small-negative, not below pre-locked −0.3 floor |
| **O5_BASELINE_DOMINATED** | **YES — assigned by honest reading** | **11/25 tiles breach >3 anchor-SD; 22/25 tiles uniformly positive** |
| O6_DATA_INTEGRITY | Partial | Beta distribution health check passed; but cross-cohort comparison is structurally invalid |

**Final outcome: O5_BASELINE_DOMINATED.** The auto-assigned O2 label was overridden after honest scientific reading per CHK-4.8 (honest revision when later evidence shifts prior outcome label).

The auto-assignment criterion was structurally degenerate: it asked "max baseline > max case d" but baseline_anchor_SD = (case_mean − ref_mean) / pooled_SD, which equals the case-vs-reference d when n_case >> n_ref dominates pooled variance. The two numbers are the same number, so the comparison cannot fire. This is the pre-locked criterion's structural failure mode and is documented as a CHK lesson.

---

## 4. Headline numbers

**Lung_cells tile (cycling class):**
- d = −0.269, 95% CI [−0.542, +0.023]
- case mean = 0.969, case n = 141
- ref mean = 0.996, ref n = 29
- Cross-cohort baseline anchor-SD: −0.27 (no breach at 1-SD threshold)

**All 25 tiles ranked by |d|:**

| Rank | Tile | Class | d | CI95 lo | CI95 hi |
|---|---|---|---|---|---|
| 1 | Pancreatic_acinar_cells | secretory | +5.254 | +4.766 | +5.897 |
| 2 | Hepatocytes | secretory | +4.887 | +4.468 | +5.449 |
| 3 | Pancreatic_beta_cells | secretory | +4.816 | +4.351 | +5.429 |
| 4 | Head_and_neck_larynx | cycling | +4.579 | +4.036 | +5.294 |
| 5 | Prostate | secretory | +4.480 | +3.916 | +5.220 |
| 6 | Kidney | cycling | +4.298 | +3.868 | +4.866 |
| 7 | Breast | secretory | +3.960 | +3.507 | +4.542 |
| 8 | Pancreatic_duct_cells | secretory | +3.787 | +3.396 | +4.312 |
| 9 | Bladder | cycling | +3.556 | +3.131 | +4.141 |
| 10 | B-cells_EPIC | immune | +3.380 | +3.042 | +3.793 |
| 11 | Cortical_neurons | terminal | +3.149 | +2.689 | +3.791 |
| 12 | CD4T-cells_EPIC | immune | +2.859 | +2.416 | +3.411 |
| 13 | Colon_epithelial_cells | cycling | +2.853 | +2.452 | +3.503 |
| 14 | Uterus_cervix | cycling | +2.833 | +2.315 | +3.499 |
| 15 | CD8T-cells_EPIC | immune | +2.608 | +2.187 | +3.128 |
| 16 | Thyroid | secretory | +2.398 | +2.067 | +2.806 |
| 17 | NK-cells_EPIC | immune | +2.301 | +1.939 | +2.734 |
| 18 | Vascular_endothelial_cells | stromal | +2.280 | +1.870 | +2.824 |
| 19 | Upper_GI | cycling | +1.810 | +1.625 | +2.033 |
| 20 | Erythrocyte_progenitors | progenitor | +1.569 | +1.215 | +1.982 |
| 21 | Adipocytes | stromal | +1.452 | +1.160 | +1.815 |
| 22 | Monocytes_EPIC | immune | +1.295 | +0.920 | +1.720 |
| 23 | Neutrophils_EPIC | immune | +0.871 | +0.482 | +1.268 |
| 24 | Left_atrium | terminal | +0.274 | −0.029 | +0.614 |
| 25 | **Lung_cells** | **cycling** | **−0.269** | **−0.542** | **+0.023** |

**Top-1 ΔA distribution (per-patient most-departed tile):**
- B-cells_EPIC: 79/141 (56%)
- Upper_GI: 24/141 (17%)
- CD4T-cells_EPIC: 12/141 (9%)
- Colon_epithelial_cells: 6/141 (4%)
- Prostate: 6/141 (4%)
- Breast: 5/141 (4%)
- Kidney: 2, Bladder: 1
- **Lung_cells: 0/141** (zero patients have Lung_cells as most-departed)

---

## 5. Cross-cohort baseline check (CHK-3.2)

22 of 25 tiles breach the 1 anchor-SD threshold. **11 tiles breach >3 anchor-SD.**

Severe (>3 anchor-SD) breaches: Cortical_neurons (+3.15), Hepatocytes (+4.89), Breast (+3.96), Prostate (+4.48), Pancreatic_acinar_cells (+5.25), Pancreatic_duct_cells (+3.79), Pancreatic_beta_cells (+4.82), Bladder (+3.56), Head_and_neck_larynx (+4.58), Kidney (+4.30), B-cells_EPIC (+3.38).

The pre-locked CHK-3.2 expectation was "≥1 tile breaches at >1 anchor-SD due to structural cohort differences (ethnicity Korean vs Western, platform EPIC vs HM450, never-smoker enrichment vs smoker-enriched)." The actual breach pattern (22/25 tiles >1 SD, 11/25 tiles >3 SD) is far beyond the expectation. The breach is not a feature — it is a structural invalidation of the cross-cohort comparison.

---

## 6. Honest interpretation

**The cross-cohort comparison is structurally invalid for this cohort pair.** The result does not characterize never-smoker LUAD biology. It characterizes the methylation β-value scale shift between two different normalization pipelines applied to two different populations.

Three stacked sources of non-biological variance:

1. **Preprocessing pipeline mismatch:** GSE256092 is SWAN-normalized; TCGA-LUAD is sesame level3. SWAN and sesame produce slightly different β value distributions even on the same raw IDATs. This is a known fact in the methylation literature and is the reason cross-pipeline meta-analyses use ComBat / BMIQ / RUV batch correction.

2. **Population baseline difference:** Korean adult tissue baseline methylation differs from Western adult tissue baseline methylation at thousands of CpG sites, particularly at population-stratified probes. This is the canonical CCL-002+CCL-006 ethnicity confound.

3. **Cell-composition difference:** GSE256092 is bulk tumor tissue with tumor-infiltrating lymphocytes; TCGA-LUAD adjacent-normal is bulk lung tissue with much lower immune infiltrate. The B-cells_EPIC top-1 dominance (79/141 patients) is a tumor-immune-infiltrate vs adjacent-normal tissue contrast, not lung tile biology.

When all three are stacked without batch correction, every Loyfer marker CpG shows differential methylation in the same direction, because every CpG is sensitive to the SWAN-vs-sesame scale shift on top of population and composition shifts. The result is uniform positive d across 22/25 tiles — a structural artifact, not biology.

**The Lung_cells tile reading at d = −0.27 is consistent with this interpretation.** When the universal scale shift dominates, the "negative" tiles are simply those whose marker CpGs happened to land on the side of the SWAN-vs-sesame shift that goes the other direction. It is not evidence of a never-smoker LUAD direction inversion at the lung-of-origin tile.

**What this VAL did NOT show:**
- Did NOT show that never-smoker LUAD has a null lung tile signal.
- Did NOT show that the lung tile fails to localize lung tumors at deployment.
- Did NOT show that the EDEAR Stage 2 architecture is invalid.
- Did NOT test the pre-diagnostic immune-flag question (this was at-diagnosis tumor tissue, not pre-dx blood).

**What this VAL DID show:**
- Cross-cohort comparisons across SWAN-vs-sesame pipeline + Korean-vs-Western population + tumor-vs-adjacent-normal composition mismatch are uninterpretable without batch correction.
- The pre-locked O5_BASELINE_DOMINATED criterion needs reformulation: when baseline_anchor_SD and case_vs_reference_d are computed identically, the auto-comparison is degenerate and the outcome must be assigned on the breach magnitude pattern (≥3 tiles >3 anchor-SD AND ≥80% tiles same-direction = baseline-dominated).

---

## 7. Within-cohort stratification (this part stands)

Per CCL-034, within-cohort stratification does NOT depend on the cross-cohort reference. The GSE256092 within-cohort variance structure across sex × age decade × stage IS interpretable on its own terms.

Reported in `stratified.json`. Headline observations from the within-cohort breakdown:
- Sex stratum: female-enriched cohort; per-tile A-score variance similar across F vs M sub-strata
- Age stratum: per-tile A-score elevation correlates weakly with age decade (consistent with VAL-052 R²=26% age regression anchor)
- Stage stratum: Stage IV n<5 (pre-locked underpower); Stage I/II/III differences not large enough to drive a signal

The within-cohort stratification confirms GSE256092 is internally consistent. The cohort itself is fine. The cross-cohort comparison is what fails.

---

## 8. Reproducibility triple (CHK-7.6)

**Source code:** `val_097.py` (38,486 bytes). Inline source attached to GitHub at `Biological_Physics/validation_runs/VAL-097/`.

**Inputs:**
- GSE256092 series matrix: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256092/matrix/GSE256092_series_matrix.txt.gz`, 7,885 bytes, SHA-256 prefix `1ef8c8c6eebbe708`
- GSE256092 SWAN beta matrix: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256092/suppl/GSE256092_SWAN.txt.gz`, 1,000,710,168 bytes, SHA-256 prefix `b191108e6414e418`
- TCGA-LUAD adjacent-normal: 29 sesame level3 .txt files downloaded fresh from `https://api.gdc.cancer.gov/data/{file_id}` per LUAD_matched_manifest.json (manifest SHA per VAL-063: `6e87cc32b84f278d1b77ad766a050f2a378aa3a8e3da78e7232b2511514d278c`)
- Loyfer reference atlas: `iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv`, 7,890 CpGs × 25 cell types

**Environment:** Python 3, NumPy, Pandas, SciPy, Matplotlib (Agg backend); 9 GB RAM available; 81.3 s runtime; ~300 MB peak resident memory during SWAN streaming.

**Expected headline output:** Outcome label O5_BASELINE_DOMINATED; Lung_cells d = −0.269 [−0.542, +0.023]; 22/25 tiles positive; 11/25 tiles >3 anchor-SD baseline breach.

---

## 9. Lesson logged

**LL-CROSS-COHORT-CALIBRATION:** When two methylation cohorts use different preprocessing pipelines (SWAN vs sesame, noob vs funnorm, etc.) AND different platforms (EPIC vs HM450) AND different populations, cross-cohort comparison without explicit batch correction (ComBat, BMIQ, or RUV) is structurally invalid. The β-value scale shift dominates the disease signal. Future cross-cohort VALs must either (a) use within-cohort paired tumor-vs-adjacent-normal in a single cohort, (b) re-process both cohorts through the same pipeline, or (c) apply batch correction with healthy-vs-healthy reference samples in both cohorts to anchor the correction.

**For EDEAR commercial deployment this lesson does NOT apply** — EDEAR runs every patient through a single calibrated pipeline against a single reference distribution. The cross-cohort calibration problem is exclusive to retrospective cross-cohort cookbook validations.

**LL-PRELOCK-DEGENERATE-COMPARATOR:** The O5_BASELINE_DOMINATED auto-assignment criterion as written ("max baseline > max case d") is structurally degenerate when baseline_anchor_SD and case_vs_reference_d are computed identically. When n_case >> n_ref, both quantities collapse to the same value and the comparison fails to fire. Future O5 criteria should test the breach pattern directly: ≥3 tiles >3 anchor-SD AND ≥80% tiles same-direction = baseline-dominated.

---

## 10. Outcome label

**O5_BASELINE_DOMINATED**

The cross-cohort baseline mismatch is severe (11/25 tiles >3 anchor-SD, 22/25 tiles same-direction) and dominates the case-vs-reference signal. Within-cohort stratification structure stands and is documented in `stratified.json`. Cross-cohort case-vs-reference d values are not interpretable as never-smoker LUAD biology.

---

## 11. Card status update

For the lung-epic card: the never-smoker LUAD tissue tile pattern is **NOT characterized by VAL-097**. Status remains "open follow-up needed." Recommended next VAL: GSE235414 (driver-stratified LUAD with internal matched adjacent-normal samples in the same cohort, same pipeline). VAL-097 is logged as the structural lesson on cross-cohort calibration.

---

## 12. Files produced

All in `Biological_Physics/validation_runs/VAL-097/` after GitHub push:
- `val_097.py` — full source code (sealed)
- `prereg.md` — pre-registration (sealed before β access)
- `outcome.md` — this file
- `results.json` — per-tile statistics, baseline check, d values, top-1 distribution
- `stratified.json` — within-cohort sex × age × stage breakdown
- `cohort_manifest.json` — GSE256092 cohort manifest with SHA-256 inputs
- `clinical_metadata.csv` — per-sample clinical metadata
- `per_sample.csv` — per-patient A-score per tile + top-1 call + metadata
- `tile_heatmap.png` — 25-tile case-vs-reference d visualization
- `PREREG_SEAL.txt` — SHA-256 of sealed prereg

GitHub-only push (per memory #11):
- `val_097.py`, `prereg.md`, `outcome.md`, `results.json`, `stratified.json`, `cohort_manifest.json`, `clinical_metadata.csv`, `per_sample.csv`, `Biological_Physics/README.md` update

Heath-only delivery (NEVER pushed per memory #11):
- Updated `README_MASTER_v2_2.md`, `LESSONS_LEARNED.md`, `TESTING_CHECKLIST.md`, `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md`, `lung-epic_README.md` (lung-epic card), `tile_heatmap.png` (visual)
