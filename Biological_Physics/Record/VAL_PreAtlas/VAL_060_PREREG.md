# VAL-060 — breast-epic retroactive tissue validation on TCGA-BRCA HM450

## Pre-registration and seal

**Pre-registration timestamp:** 2026-04-24 (to be sealed before any TCGA-BRCA β access)
**Sealed before any TCGA-BRCA β matrix or file download:** YES
**Sealed before any β extraction or computation:** YES
**Author:** Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute
**Panel:** Xu-538 immune panel (full panel) — per universal_reference_block
**H_min(immune):** 0.838889 (G-003b MCMC posterior, frozen)
**H_min(secretory):** 0.843264 (G-003b MCMC posterior, frozen) — for context, breast_ductal is secretory class
**RNG seed:** 20260420

---

## What this run fills

The breast-epic card was built at `cross_platform_validated_two_cohorts` tier based on per-patient pre-diagnostic BLOOD methylation (GSE51057 Italian pre-dx and GSE51032 EPIC-Italy). What has NEVER been run at per-card resolution is the tissue-level tumor vs adjacent-normal case-control using the Xu-538 panel specifically — the equivalent of what VAL-058 did for prostate-epic.

VAL-060 retroactively adds that tissue arm to breast-epic, establishing the template for tissue re-validation as a standard per-card requirement going forward (per CCL-011). This is the first retroactive per-card tissue run; VAL-061 (CRC), VAL-062 (AD), VAL-063 (lung) will follow.

**Framework-level BRCA tissue validation already exists:** VAL-001 (TCGA pan-cancer n=90 BRCA tumor-vs-adjacent-normal, secretory-class ΔA = +0.21422, FLOOR BREACH), VAL-009 (TCGA matched tumor-normal 4,304 pairs, framework confirmed), VAL-039 (Teschendorff 2016 breast adjacent-normal field defect). These establish that BRCA tissue shows architectural drift at the secretory-class level. VAL-060 is distinct — it tests the specific Xu-538 IMMUNE-CLASS panel that the breast-epic card's Stage 1 uses. Whether Xu-538 separates BRCA tumor tissue from adjacent-normal tissue at the immune panel specifically, and what the tissue-level effect size is, has not been measured.

---

## Question

Does the Xu-538 immune panel produce a measurable pooled-entropy A-score elevation in breast tumor tissue vs adjacent-normal breast tissue, and if yes, what is the tissue-level effect size? Comparison anchor: VAL-058 prostate paired d = +0.497 on the same panel, same H_min, same methodology, different disease and different tissue.

Secondary question: does the paired-analysis pattern observed in prostate (paired d > unpaired d, reflecting field-effect suppression in the unpaired normal group) replicate in breast?

---

## Cohort

**TCGA-BRCA HM450** (The Cancer Genome Atlas, breast invasive carcinoma project, Illumina HumanMethylation450 platform).

Public access via NIH Genomic Data Commons (GDC). No dbGaP application required for Level 3 β value files. Access: `https://api.gdc.cancer.gov/files` with filter `cases.project.project_id = TCGA-BRCA`, `data_type = Methylation Beta Value`, `platform = illumina human methylation 450`.

**Available samples (pre-lock inventory, confirmed via GDC API on 2026-04-24 before pre-reg seal):**
- Total HM450 methylation files: 895
- Primary Tumor: 793
- Solid Tissue Normal: 97  
- Metastatic: 5
- Unique cases: 791
- Cases with BOTH Primary Tumor AND Solid Tissue Normal: **91 matched pairs**
- Gender: 783 female, 9 male, 1 unknown (among Primary Tumor)

**Matched pair structure:** 91 complete tumor + adjacent-normal pairs from 91 patients. Analogous to GSE269244's 118 prostate pairs (VAL-058). Primary analysis.

**Unmatched tumor pool:** 702 Primary Tumor samples from cases without adjacent-normal. Random stratified subsample of n=200 drawn for secondary analysis (stratification on available clinical variables: estrogen receptor status, PAM50 subtype where available). Full 793 primary tumor pool used only if secondary analysis is inconclusive.

**Independence:** TCGA-BRCA has not been used for Xu-538 panel derivation (panel came from Sister Study + EPIC-Italy). No overlap with breast-epic's validation cohorts (GSE51057, GSE51032 — both Italian pre-dx blood, not TCGA). Fresh external test of the Xu-538 panel on a completely independent tissue cohort.

---

## Frozen instruments

**Stage 1 panel: Xu-538 immune** — full panel per `universal_reference_block.universal_stage_1_pipeline`.
- Panel SHA-256: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`
- Expected HM450 coverage: 538/538 (Xu-538 was derived from HM450+EPIC overlap; full coverage expected on HM450)
- H_min(immune): 0.838889 (frozen)

**Scoring method:**
M1 = pooled-entropy A-score on Xu-538 = mean(H(β)/H_min_immune) across available panel CpGs per sample.

**z-standardization:** within-TCGA-BRCA Solid Tissue Normal mean/SD per CpG. Per CCL-004, the 80-cell immune blood baseline does NOT apply to tissue; TCGA adjacent-normal is the appropriate tissue-level control. Documented explicitly.

**Platform bridge notes:** HM450 is the predecessor of EPIC 850K. Xu-538 panel was originally derived on EPIC but all 538 CpGs have HM450 probe equivalents (the panel was selected to work across both). Probe coverage fallback: if any Xu-538 CpG is not present on HM450, exclude from the analysis and report coverage fraction. Do NOT substitute nearby probes.

---

## Pre-specified outcomes (locked before any data access)

The analysis produces M1 per sample. Comparisons:

- **M1_unpaired:** d(TCGA-BRCA Primary Tumor vs Solid Tissue Normal), n=793 vs n=97. Bootstrap 95% CI + permutation p.
- **M1_paired:** paired difference (Tumor − Normal per patient), 91 matched pairs. Paired d + sign-flip permutation p.
- **M1_secondary:** d(unmatched_random_200 Primary Tumor vs Solid Tissue Normal), n=200 vs n=97. Secondary balance check.
- **Per-CpG direction:** Δβ(tumor − normal) per Xu-538 CpG, direction-preserved rate vs published Xu 2020 Sister Study direction (case > control in breast blood). HM450 coverage reported.
- **Sex stratification:** analysis repeated on female-only subset (n ≈ 783 tumor, ≈ 97 normal — almost all BRCA is female, but explicitly excluded male BRCA to isolate ductal secretory-class signal). Male BRCA (n=9) reported separately as n-constrained side observation, not a pre-registered primary comparison.
- **Age regression:** M1 regressed on age_at_diagnosis in the Solid Tissue Normal group (the adjacent-normal "baseline"), residualized M1 reported.

**Outcome decision matrix — locked:**

### O1: BREAST-EPIC TISSUE VALIDATED (full)
d(M1_unpaired) > 0.3 AND d(M1_paired) > 0.3. Direction POSITIVE (tumor > normal in A-score). This matches the VAL-047 blood pre-dx finding (breast is a positive-direction disease, opposite to CRC) and matches the VAL-058 prostate tissue pattern.

**Card update:** breast-epic v2.2 — adds "tissue arm" validation entry to the card's `validation_evidence_summary`. Tier label upgrades to `cross_platform_validated_two_cohorts_plus_tissue_arm`. Clinical deployment unchanged; evidence base deepened.

### O2: TISSUE POSITIVE BUT BELOW THRESHOLD
0 < d(M1_unpaired) < 0.3 OR 0 < d(M1_paired) < 0.3, but direction consistently positive. Per-CpG direction preservation > 292/538 (binomial p<0.05 at n=538).

**Interpretation:** Xu-538 tissue-level separation of breast tumor from adjacent-normal is positive-directional but smaller than prostate's d = +0.497. Possible cause: breast adjacent-normal has a larger field effect than prostate (VAL-037 found breast adjacent-normal at +0.036 ΔA; VAL-039 Teschendorff 2016 breast field defect). The unpaired-vs-paired gap may be larger than in prostate.

**Card update:** breast-epic v2.2 with tissue-arm noted as "directionally consistent but below threshold" tier. Card does not change primary validation tier; tissue arm documented as partial support.

### O3: NULL OR OPPOSITE DIRECTION
d(M1_unpaired) and d(M1_paired) both < 0.3 with no consistent positive direction. OR: direction is opposite (tumor < normal in A-score), contradicting VAL-047 blood pre-dx breast positive direction.

**Interpretation:** Either (a) Xu-538 operates differently on tissue vs blood for breast (the blood panel signal is immune-class field effect that doesn't manifest in tumor tissue itself), OR (b) TCGA-BRCA adjacent-normal is so field-effect-contaminated that tumor-vs-adjacent-normal comparison loses signal.

**Card update:** breast-epic v2.2 documents tissue null with explicit caveat. Blood validation tier is preserved; tissue arm flagged as "tissue-level signal not detectable at Xu-538 panel — architectural drift measurable at secretory-class level per VAL-001 (ΔA = +0.21422) but not at immune-class Xu-538 panel." Informative result: confirms that Xu-538 is a blood-signature-specific panel, not a universal pan-substrate breast cancer panel. This is itself a useful clarification for the card.

### O4: UNEXPECTED
Any pattern not fitting O1-O3 (e.g., paired d positive but unpaired d negative, or vice versa, or specific to one platform/ancestry subset). Report honestly; interpret based on pattern.

---

## Analytical protocol (locked before data access)

1. **Download** all 895 TCGA-BRCA HM450 methylation Level 3 β files from GDC API. Record file-level SHA-256 for each file. Verify total file count = 895.
   - Matched-pair subset (186 files, 91 pairs + full normal arm): priority 1
   - Random stratified subsample of 200 unmatched tumors: priority 2
   - Full 793 tumor pool: priority 3 (only if needed)

2. **Build unified β matrix** from the 895 single-sample files. Each file contains ~485K HM450 CpGs in long format; reshape into samples × CpGs matrix.

3. **Extract Xu-538 CpGs** from the unified matrix. Log HM450 coverage fraction.

4. **Compute M1 per sample** as mean(H(β)/H_min_immune) across covered CpGs (≥300 CpG coverage threshold per VAL-058 convention).

5. **Unpaired analysis:** d, CI, permutation p for Primary Tumor vs Solid Tissue Normal.

6. **Paired analysis:** identify 91 matched pairs by case_submitter_id. Compute paired differences, paired d, sign-flip permutation p.

7. **Sex stratification:** female-only (n ≈ 783 vs 97). Primary sex-stratified result.

8. **Age regression:** slope/intercept from adjacent-normal age_at_diagnosis vs M1. Report residualized d.

9. **Per-CpG Δβ direction:** tumor mean vs normal mean per Xu-538 CpG. Direction-preserved count.

10. **Secondary subsample:** random stratified n=200 from unmatched tumor pool. Repeat unpaired d as secondary balance check.

11. **Outcome assignment per locked matrix.**

12. **SHA-lock results JSON.**

---

## What gets published

- `val060_breast_epic_tcga_brca_tissue.py` — analysis script (GitHub)
- `VAL060_breast_epic_tcga_brca_results.json` — SHA-locked results (GitHub)
- `VAL_060_PREREG.md` + `VAL_060_SEAL.txt` — pre-registration artifacts (GitHub)
- Evidence Report §VAL-060 section (vault + GitHub summary)
- breast-epic card v2.2 (vault only) with added tissue-arm validation entry
- Master README update: validated cards section, tier definitions

---

## What does NOT happen under any outcome

- Panel re-training. Xu-538 stays frozen at SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`.
- CpG subset cherry-picking.
- Substitute probes for Xu-538 CpGs absent from HM450 — exclude and report coverage.
- Cross-analysis contamination: each outcome is decided on its own pre-registered comparison, not on cherry-picked subanalyses.
- Retroactive adjustment if results disappoint.

---

## Seal

Pre-registration file `VAL_060_PREREG.md` will be SHA-256 hashed and recorded in `VAL_060_SEAL.txt` before any TCGA-BRCA file download beyond the GDC metadata query already performed (which returned only file counts, sample types, and case/sample submitter IDs — NO β values). Pre-reg SHA and seal timestamp committed to GitHub at commit time.
