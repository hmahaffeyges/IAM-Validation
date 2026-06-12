# VAL-058 — prostate-epic Stage 2 tissue-methylation validation on GSE269244

## Pre-registration and seal

**Pre-registration timestamp:** 2026-04-24 (to be sealed before any GSE269244 β access)
**Sealed before any GSE269244 series matrix download:** YES
**Sealed before any β extraction or computation:** YES
**Author:** Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute
**Panel:** Xu-538 immune panel (full panel) — per universal_reference_block
**H_min(secretory):** 0.843264 (G-003b MCMC posterior, frozen)
**RNG seed:** 20260420

---

## Clinical gap this card fills

An EDEAR patient whose Stage 1 buffy-coat immune A-score flags at DETECTABLE tier or higher, and whose Stage 2 Moss NNLS deconvolution returns `prostate_epithelial` as top-1 tissue localization, currently has no matching card in the Cookbook. Without a prostate card, the EDEAR report returns "architectural flag, tissue localization: prostate_epithelial, no card available — see clinician for standard workup." Functional but clinically incomplete.

VAL-058 builds the minimum-viable prostate-epic card at **`stage_2_only_validated` tier** — the tier that supports Stage 2 tissue-localization firing without claiming per-patient blood pre-diagnostic detection. The card does NOT serve as a blood-based screening test for prostate cancer; no public per-patient blood pre-diagnostic prostate methylation cohort exists at the Phase 9/12 scale we achieved for breast (GSE51057/51032) or CRC (GSE51032). Health ABC and Rotterdam are dbGaP-gated and remain on the roadmap for future upgrade.

What `stage_2_only_validated` means for this card:
- Moss NNLS top-1 prostate_epithelial localization fires the card.
- The card provides tissue β expectations (tumor-typical methylation patterns for prostate adenocarcinoma) sourced from a published EPIC 850K tumor vs adjacent-normal cohort.
- The card outputs a clinical action path: elevated prostate-specific Stage 2 signal → PSA + DRE + multi-parametric MRI + urology consultation per standard of care.
- The card explicitly does NOT claim pre-diagnostic blood detection. Any patient whose Stage 1 immune flags are below DETECTABLE is not flagged as "prostate cancer risk" by this card.

---

## Question

Does Illumina EPIC 850K tumor methylation from prostate adenocarcinoma produce an architectural signature that (a) separates from adjacent-normal prostate tissue at group level with d > 0.3 on pooled-entropy immune-class A-score using the Xu-538 panel, and (b) on the specific Moss 2018 prostate_epithelial reference CpG subset, shows β values departing from the Moss healthy reference β = 0.743 in a direction consistent with VAL-041 Moss 2018 Fig 4d prostate-cancer case signature (β ≈ 0.635, Δβ ≈ −0.108)?

---

## Cohort

**GSE269244** (Epigenome-wide association study of Prostate Cancer in African American Men, differentially methylated genes in tumor vs adjacent-normal and aggressive vs indolent).

- Platform: Illumina HumanMethylationEPIC 850K
- Specimen: fresh-frozen prostate tissue (tumor + adjacent-normal pairs)
- Total n: 238 (121 tumor + adjacent-normal from 121 African-American men, per GEO summary)
- Population: African American men — high-priority underserved population with highest PCa incidence and mortality globally
- Access: public GEO, no application required

**Independence from existing cards:** First EDEAR validation on prostate. No prior validation runs on this cohort. GSE269244 is a fresh, unprocessed external target.

---

## Frozen instruments

**Stage 1 panel: Xu-538 immune** — full panel per `universal_reference_block.universal_stage_1_pipeline`.
- Panel SHA-256: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`
- EPIC coverage: ~500/538 (93%) expected
- H_min(immune): 0.838889 (frozen)

**Stage 2 reference: Moss 2018 prostate_epithelial CpG subset** — as used in VAL-041.
- Moss healthy β reference: 0.743 (per `universal_reference_block.universal_stage_2_moss_deconvolution.healthy_reference_beta_by_tissue.prostate_epithelial`)
- Prostate class: secretory, H_min = 0.843264

**Scoring methods:**
1. Pooled-entropy A-score on Xu-538 panel. Primary Stage 1 metric. A = mean(H(β)/H_min_immune) across the ~500 EPIC-covered CpGs.
2. Pooled-entropy A-score on Moss 2018 prostate_epithelial reference CpG subset. Primary Stage 2 metric. A = mean(H(β)/H_min_secretory) across Moss prostate reference CpGs.
3. Mean β on Moss prostate_epithelial reference CpG subset. Direct comparison against Moss healthy β = 0.743 and expected cancer β ≈ 0.635 from VAL-041.

**z-standardization:** to GSE269244 adjacent-normal tissue mean/SD per CpG (not to Cookbook 80-cell baseline, per CCL-004: tissue cohorts cannot use the blood-derived 80-cell baseline).

---

## Pre-specified outcomes (locked before any data access)

The analysis produces three primary metrics per sample, compared between paired tumor (case) and adjacent-normal (ctrl) groups:

- **M1: pooled_Ximmune** — Xu-538 immune panel A-score
- **M2: pooled_Mossprostate** — Moss prostate-reference CpG A-score (secretory class)
- **M3: meanbeta_Mossprostate** — mean β across Moss prostate reference CpGs

Each produces a Cohen's d (tumor vs adjacent-normal) with bootstrap 95% CI and permutation p.

Plus age-regressed versions of M1-M3 (using GSE269244 adjacent-normal as baseline).

**Outcome decision matrix — locked:**

### Outcome 1 (O1): FULL STAGE 2 VALIDATION
**Pattern:** d(M1) > 0.3 AND d(M2) > 0.3 AND mean β on Moss prostate CpGs in tumor group ∈ [0.55, 0.70] (consistent with VAL-041 Δβ direction and magnitude).

**Interpretation:** Prostate tumor tissue shows elevated immune-class architectural drift (matches breast-epic pattern: cancer → immune field effect) AND Moss prostate-reference CpGs depart from healthy β = 0.743 toward the cancer-typical 0.635 range. Full validation of prostate_epithelial Stage 2 localization on EPIC tissue data.

**Card update:** prostate-epic card v0.1 at `stage_2_only_validated` tier. Moss prostate localization fires the card. Tissue β expectations documented.

### Outcome 2 (O2): STAGE 2 PARTIAL VALIDATION
**Pattern:** d(M1) < 0.3 OR d(M2) < 0.3, but mean β on Moss prostate CpGs is consistent with VAL-041 (β ∈ [0.55, 0.70] in tumor group).

**Interpretation:** Moss prostate reference CpGs behave as expected at the β level (Stage 2 mechanism confirmed), but pooled-entropy aggregation does not separate tumor from adjacent-normal at d > 0.3. Possible causes: (a) adjacent-normal already has field-effect methylation drift (well-documented; VAL-037 quantified +0.036 mean ΔA across 24 TCGA types); (b) pooled-entropy is too aggregated a metric for tissue case-control; per-CpG or directional scoring may recover the signal.

**Card update:** card v0.1 at `stage_2_only_validated` tier with explicit field-effect caveat. Mean β on Moss prostate CpGs documented as primary Stage 2 firing criterion (not pooled-entropy).

### Outcome 3 (O3): STAGE 2 FAILURE
**Pattern:** mean β on Moss prostate CpGs in tumor group is OUTSIDE [0.55, 0.70] AND does not show the expected VAL-041 Δβ ≈ −0.108 direction.

**Interpretation:** GSE269244 tumors do not match the VAL-041 Moss prostate-cancer signature. Possibilities: (a) GSE269244 population-specific biology (African-American PCa has documented different genomic profile, may extend to methylation); (b) EPIC 850K platform artifacts vs 450K; (c) Moss 2018 reference CpG set doesn't generalize to EPIC prostate measurements.

**Card update:** card NOT deployed at `stage_2_only_validated`. Enters Cookbook at `stage_2_exploratory` tier with explicit non-validation note. Future work: cross-population validation on a European-ancestry prostate tissue cohort to distinguish (a) from (b)/(c).

### Outcome 4 (O4): UNEXPECTED PATTERN
Any pattern not fitting O1/O2/O3. Reported honestly; card update deferred pending additional analysis.

---

## Analytical protocol (locked before data access)

1. **Download** GSE269244 series matrix from `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269244/matrix/` using curl. Verify SHA-256.

2. **Parse metadata** from `!Sample_characteristics_ch1`. Map each sample to {tumor, adjacent-normal, other}. Extract age where available; sex is all male in this cohort. Record sample pair structure (tumor-adjacent pairs by patient ID).

3. **Extract Xu-538 panel β** using streaming one-pass matrix read (per VAL-047 protocol). Log coverage (expected ~500/538 on EPIC).

4. **Extract Moss 2018 prostate_epithelial reference CpG β** — CpG list to be specified from the Moss 2018 supplementary data (embedded in GAPE_WEB_v13.py Moss atlas). Log 100% coverage if EPIC transfer is clean.

5. **Compute M1, M2, M3** per sample using the scoring formulas specified above.

6. **Cohen's d** tumor vs adjacent-normal for each metric with bootstrap 95% CI (10,000 iterations) and permutation p (10,000 perms).

7. **Paired analysis:** because the cohort is matched pairs, compute paired-difference Cohen's d (tumor − adjacent-normal, per patient) as additional metric with its own CI. Paired analysis has higher power than unpaired.

8. **Sex stratification:** not applicable — all samples are male (prostate tissue). Noted for transparency.

9. **Per-CpG directional check on Xu-538:** for each of the ~500 EPIC-covered Xu panel CpGs, compute Δβ (tumor − adjacent-normal mean). Count direction-preserved vs Xu panel's published tumor direction (if available) or against a naive "tumor hypomethylated" prior.

10. **80-cell age anchor:** NOT applicable to tissue data. The 80-cell baseline is blood-derived per CCL-004. Tissue β values cannot be compared to the 80-cell immune A-score reference. Documented in results.

11. **Cohort batch offset check:** compare GSE269244 adjacent-normal β distribution for Moss prostate CpGs against Moss 2018 Table S1 healthy prostate β = 0.743. An offset > 0.05 would indicate preprocessing systematic difference; interpret M3 accordingly.

12. **Outcome assignment** per locked decision matrix. Report computed numbers BEFORE assigning outcome.

13. **SHA-lock** results JSON.

---

## What gets published

- `val058_prostate_epic_gse269244.py` — analysis script (GitHub, validation_runs/)
- `VAL058_prostate_epic_gse269244_results.json` — SHA-locked results (GitHub)
- `VAL_058_PREREG.md` + `VAL_058_SEAL.txt` — pre-registration artifacts (GitHub, provenance)
- Evidence Report §VAL-058 section (vault only)
- prostate-epic card v0.1 (vault only) with full universal_reference + lessons_learned blocks
- prostate-epic README (vault only)
- Master README update (vault only) adding prostate-epic to the validated-cards section

---

## What does NOT happen under any outcome

- Panel re-training on GSE269244. Xu-538 stays frozen.
- CpG subset cherry-picking. Full Xu-538 + full Moss prostate reference set, no filtering for "best-performing" CpGs.
- Retroactive threshold adjustment. If the data lands O2 or O3, the card ships at the tier the pre-reg specifies for that outcome.
- Claiming per-patient blood pre-diagnostic prostate detection. This card is tissue-only. Any future upgrade to `cohort_screening_validated` requires a blood methylation cohort.

---

## Seal

Pre-registration file `VAL_058_PREREG.md` will be SHA-256 hashed and recorded in `VAL_058_SEAL.txt` before any GSE269244 matrix download. Both files committed to GitHub at commit time as timeline evidence that this specification preceded data access.
