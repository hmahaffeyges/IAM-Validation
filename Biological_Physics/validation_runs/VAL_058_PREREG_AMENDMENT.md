# VAL-058 Pre-Registration Amendment — 2026-04-24 06:55 UTC

## Amendment justification

Pre-registration VAL_058_PREREG.md (SHA `48abe394ad009020...`) sealed 2026-04-24 06:50:36 UTC specified use of the "Moss 2018 prostate_epithelial reference CpG subset" for Stage 2 validation (metrics M2 and M3). Upon attempting to implement the analysis, the per-CpG Moss 2018 reference panels were confirmed to be part of the proprietary calibration layer (see VAL-041 script line: "The per-class and per-substrate H_min posteriors are part of the proprietary calibration layer covered under US Provisional Patents 64/012,720 and 64/014,568"). These per-CpG panels are NOT in the publicly-exposable corpus and cannot be used in a GitHub-pushed analysis script.

**Protocol amendment: M2 and M3 removed from pre-reg. Stage 2 validation will proceed through Xu-538 panel tissue-level case-control instead.**

This amendment is made BEFORE any β-value access to GSE269244. It is published as an amendment (not a revision that replaces the original) so the timeline is transparent: original pre-reg sealed 06:50:36 UTC with M1/M2/M3, amendment issued 06:55 UTC removing M2/M3 before data access, final analysis runs with amended protocol. The original seal SHA-256 `48abe394ad009020d4bafeeb262439ee02fc910df6d79a96ed56d235a0608316` is unchanged and remains valid as timeline evidence. This amendment file receives its own SHA seal.

---

## Amended analytical protocol

**Metrics (amended):**
- **M1: pooled_Ximmune** — Xu-538 immune panel A-score on prostate tissue. Tests whether Xu-538 (blood-derived panel) separates prostate tumor from adjacent-normal tissue via immune-class architectural drift in the tissue's infiltrating/resident immune cells. PRIMARY metric.
- **M1_direction: per-CpG Δβ direction** — For each Xu-538 CpG, compute Δβ(tumor − normal) and report direction-preservation rate relative to published Xu 2020 breast cancer direction pattern (case > control for hypermethylated CpGs per Sister Study).
- **M2 and M3 REMOVED from pre-reg.** Moss per-CpG prostate reference is NDA-gated; not usable in a public-facing script.

**Outcome decision matrix (amended):**

### O1: Xu-538 PROSTATE TISSUE VALIDATED
- d(M1, tumor vs adj-normal) > 0.3 (unpaired) AND paired-difference d > 0.3 (tumor − adj-normal per patient).
- Card enters Cookbook at `stage_2_only_validated` tier. Clinical rationale: Xu-538 separates prostate tumor from adjacent-normal tissue. A Moss NNLS firing on prostate_epithelial in a patient's Stage 2 output is backed by demonstrated panel-level sensitivity on prostate tumor tissue.

### O2: DIRECTIONAL-ONLY VALIDATION
- d(M1) < 0.3 but per-CpG direction preservation rate > 4.5/7 (binomial p < 0.05 at n=538 CpGs, threshold ≥ ~290/538 preserved).
- Card enters at `stage_2_exploratory` tier. Panel direction is informative but pooled-entropy does not separate at d > 0.3. Clinical firing requires supplementary evidence.

### O3: NULL
- d(M1) < 0.3 AND no directional preservation.
- Card NOT deployed. Enters Cookbook at `null_documented` tier with explicit non-validation note. Roadmap: ADNI-equivalent blood methylation cohort required.

### O4: UNEXPECTED
- Any pattern not fitting O1-O3 (e.g., d(M1) > 0.3 in unexpected direction — tumor LOWER A than adj-normal).
- Report numbers honestly; card deferred.

**All other elements of VAL_058_PREREG.md (cohort, panel freeze, RNG seed, paired analysis, age regression, sex N/A, per-CpG direction check, cohort batch offset, no cherry-picking rule) remain unchanged.**

---

## Amendment seal

This amendment will be SHA-sealed and the combined (original prereg + amendment) timeline documented in VAL_058_SEAL.txt before any β access.
