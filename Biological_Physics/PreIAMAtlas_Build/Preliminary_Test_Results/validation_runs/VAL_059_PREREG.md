# VAL-059 — hcc-epic blood methylation cross-cohort validation on GSE281691 + GSE298812

## Pre-registration and seal

**Pre-registration timestamp:** 2026-04-24 (to be sealed before any β access)
**Sealed before any GSE281691 or GSE298812 matrix download:** YES
**Sealed before any β extraction or computation:** YES
**Author:** Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute
**Panel:** Xu-538 immune panel (full panel) — per universal_reference_block
**H_min(immune):** 0.838889 (G-003b MCMC posterior, frozen)
**H_min(secretory):** 0.843264 (G-003b MCMC posterior, frozen)
**RNG seed:** 20260420

---

## Clinical gap this card fills

Hepatocellular carcinoma (HCC) is the most common primary liver cancer — the sixth most common cancer globally and the third-leading cause of cancer mortality. HCC typically arises on a background of chronic liver disease (hepatitis B, hepatitis C, alcoholic cirrhosis, or metabolic liver disease/NAFLD-NASH). Early detection is hampered by the low sensitivity of current biomarkers (alpha-fetoprotein) and by the fact that cirrhosis itself produces methylation drift that can mask the cancer-specific signal.

An EDEAR patient whose Stage 1 buffy-coat immune A-score flags at DETECTABLE tier or higher, and whose Stage 2 Moss NNLS deconvolution returns `hepatocyte` as top-1 tissue localization, currently has no matching card. VAL-059 builds the hcc-epic card.

Unlike prostate, HCC has TWO large, public, EPIC 850K peripheral-blood methylation cohorts: GSE281691 (metabolic HCC multicenter, n=481, whole-blood leukocyte) and GSE298812 (Nigerian HIV+ HCC, n=245, circulating cell-free DNA). This allows pre-dx blood Phase 9/12-equivalent validation AND cross-cohort replication in a single pre-registered sprint. Target tier: `cross_platform_validated` if both cohorts replicate, otherwise whatever tier the data supports.

---

## Questions

1. **Primary question.** Does the Xu-538 immune panel produce an elevated pooled-entropy A-score in HCC cases vs controls on peripheral blood methylation data at effect size d > 0.3 on each of GSE281691 (whole-blood leukocyte) and GSE298812 (cfDNA)?

2. **Stage 2 question.** On each cohort, does the Moss 2018 hepatocyte-reference CpG subset show β departure from Moss healthy β = 0.742 toward the VAL-041 HCC-case β ≈ 0.598 (Δβ ≈ −0.144) in HCC cases vs controls?

3. **Cross-cohort question.** If d > 0.3 in both cohorts, does direction match? Effect magnitude similar within 2× ratio?

---

## Cohorts

### GSE281691 — Primary validation cohort
- Ref: Metabolic HCC international multicenter study
- Platform: Illumina HumanMethylationEPIC 850K (GPL21145)
- Specimen: whole-blood leukocyte DNA
- Composition per GEO summary: 272 metabolic HCC patients + 316 controls with metabolic liver disease (non-HCC cirrhosis/fibrosis) = 588 total cases/controls, n=481 samples in GEO matrix
- Design: case-control, multi-site international
- Published classifier: 55-CpG panel AUC 0.79, sens 0.77, spec 0.74
- Access: public GEO, no application required

### GSE298812 — Cross-cohort replication
- Ref: Nigerian HIV-positive HCC cohort, Soliman et al.
- Platform: Illumina HumanMethylationEPIC 850K
- Specimen: circulating cell-free DNA (ccfDNA plasma)
- Composition: n=245 spanning HCC, cirrhosis, fibrosis, and HCC-free groups (all HIV-positive)
- Design: disease-spectrum cross-sectional
- Published classifier: ccfDNAmRF random forest AUC 92-97%, combined with AFP AUC up to 98.5%
- Access: public GEO, no application required

**Independence:** Both cohorts are independent of any prior EDEAR validation. Neither was used to derive the Xu-538 panel (that was Sister Study breast cancer, Xu 2020). Neither was used for any prior Cookbook card. True external validations.

**Cohort difference (NOT a blocker, explicitly acknowledged):** GSE281691 is whole-blood leukocyte genomic DNA; GSE298812 is ccfDNA from plasma. These are different substrates. Whole-blood reflects the blood immune compartment directly; ccfDNA reflects the circulating tumor + tissue-of-origin shed DNA plus background blood cell turnover. Both are valid EDEAR substrates (Stage 1 uses leukocyte DNA for immune class scoring; Stage 2 Moss NNLS is ccfDNA-native per Moss 2018). The cross-cohort comparison must respect this difference — see analysis protocol below for per-cohort separate analyses before cross-cohort comparison.

---

## Frozen instruments

**Stage 1 panel: Xu-538 immune** — full panel per `universal_reference_block.universal_stage_1_pipeline`.
- Panel SHA-256: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`
- EPIC coverage: ~500/538 expected on both cohorts
- H_min(immune): 0.838889 (frozen)

**Stage 2 reference: Moss 2018 hepatocyte CpG subset** — as used in VAL-041.
- Moss healthy β reference: 0.742
- Hepatocyte class: secretory, H_min = 0.843264

**Scoring methods** (same across both cohorts):
1. Pooled-entropy A-score on Xu-538 panel. Primary Stage 1 metric.
2. Pooled-entropy A-score on Moss hepatocyte reference CpG subset. Primary Stage 2 metric.
3. Mean β on Moss hepatocyte reference CpGs. Direct comparison against Moss healthy β = 0.742 and VAL-041 HCC case β ≈ 0.598.

**z-standardization:** per-cohort, to cohort-specific control group mean/SD per CpG. Per CCL-004, the 80-cell baseline is not universal across preprocessing pipelines; cross-cohort normalization risk is addressed by per-cohort re-anchoring.

---

## Pre-specified outcomes (locked before any data access)

The analysis produces three primary metrics per sample, compared case vs control within each cohort separately, then cross-cohort:

- **M1: pooled_Ximmune** — Xu-538 immune panel A-score
- **M2: pooled_Mosshepatocyte** — Moss hepatocyte-reference CpG A-score (secretory class)
- **M3: meanbeta_Mosshepatocyte** — mean β across Moss hepatocyte reference CpGs

Each produces Cohen's d with bootstrap 95% CI and permutation p (10,000 perms / 10,000 boot).

Plus age-regressed versions. Plus sex-stratified versions.

**Outcome decision matrix — locked:**

### Outcome 1 (O1): CROSS-PLATFORM VALIDATED
**Pattern:** d(M1) > 0.3 AND d(M2) > 0.3 on GSE281691 (primary), AND d(M1) > 0.3 OR d(M2) > 0.3 on GSE298812 (replication, with ratio of magnitudes within 2×).

**Interpretation:** Stage 1 Xu-538 immune flagging works for HCC in both whole-blood leukocyte (GSE281691) and ccfDNA (GSE298812) substrates. Moss hepatocyte Stage 2 localization confirmed in both. Cross-cohort, cross-substrate replication — the strongest possible tier for a first-card validation.

**Card update:** hcc-epic v0.1 at `cross_platform_validated` tier. Both cohorts documented. Substrate-specific thresholds (whole-blood vs ccfDNA) noted.

### Outcome 2 (O2): SINGLE-COHORT VALIDATED
**Pattern:** d(M1) > 0.3 AND d(M2) > 0.3 on GSE281691 only (primary passes), GSE298812 fails to replicate with d < 0.3.

**Interpretation:** Whole-blood leukocyte Stage 1 works. ccfDNA Stage 1 does not (Xu-538 is a blood-cell-derived panel; tumor-derived cfDNA may not carry the same signature). This is a substrate-specific finding, not a failure of HCC detection.

**Card update:** hcc-epic v0.1 at `cohort_screening_validated` tier (whole-blood leukocyte specimen only). ccfDNA explicitly listed as not-validated in known_limitations. Clinical action path restricted to whole-blood specimen input.

### Outcome 3 (O3): STAGE 2 ONLY
**Pattern:** d(M1) < 0.3 on both cohorts, but d(M2) > 0.3 on at least one cohort (Moss hepatocyte reference specifically fires).

**Interpretation:** Xu-538 immune panel does not generalize to HCC, but Moss hepatocyte reference CpGs do fire on HCC cases. This would be analogous to the prostate-epic `stage_2_only_validated` scenario — the card exists for Stage 2 localization firing, not Stage 1 screening.

**Card update:** hcc-epic v0.1 at `stage_2_only_validated` tier, analogous to prostate-epic. Clinical action: Moss hepatocyte top-1 localization fires the card; no pre-diagnostic screening claim.

### Outcome 4 (O4): NULL ON BOTH COHORTS
**Pattern:** d(M1) < 0.3 AND d(M2) < 0.3 on both cohorts.

**Interpretation:** Neither Stage 1 nor Stage 2 separates HCC from controls on public blood methylation data. Possibilities: (a) Xu-538 panel doesn't transfer to liver cancer (breast-derived panel); (b) Moss hepatocyte reference CpGs don't behave as expected; (c) controls in these cohorts are too-similar to cases (GSE281691 controls are metabolic-liver-disease patients, not healthy — substantial methylation drift expected).

**Card update:** hcc-epic NOT DEPLOYED. Enters Cookbook at `null_documented` tier. Card specification documents what was tested and why it didn't work. Future work: apply a hepatocyte-specific directional panel (analogous to VAL-051 AD 7-CpG Rule A for HCC) if the pooled-entropy null is a Directional-Score-Principle scenario.

### Outcome 5 (O5): UNEXPECTED PATTERN
Any pattern not fitting O1-O4. Reported honestly; card update deferred.

---

## Analytical protocol (locked before data access)

1. **Download both matrices** from GEO FTP. Record SHA-256 for each.
   - GSE281691: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE281nnn/GSE281691/matrix/`
   - GSE298812: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE298nnn/GSE298812/matrix/`

2. **Parse metadata per cohort.** For GSE281691: map HCC case vs metabolic-liver-disease control. For GSE298812: map HCC vs cirrhosis vs fibrosis vs HCC-free (primary comparison: HCC vs HCC-free; secondary: HCC vs cirrhosis).

3. **Extract Xu-538 panel and Moss hepatocyte reference β** per cohort using streaming one-pass.

4. **Compute M1, M2, M3** per sample per cohort.

5. **Per-cohort Cohen's d** with bootstrap 95% CI and permutation p.

6. **Age regression** per cohort (VAL-052 protocol).

7. **Sex stratification** per cohort. HCC is male-predominant (~3:1 male:female ratio globally); sex-specific effect sizes documented. Per CCL-002.

8. **Per-CpG directional check** on Xu-538 panel — compute Δβ(case − control) per CpG, report direction-preserved count relative to Xu panel's published direction.

9. **80-cell age anchor check** on GSE281691 specifically (whole-blood leukocyte, directly applicable) per CCL-004. Report cohort mean A_age_z vs 80-cell baseline. Expected: some offset due to metabolic-disease-state controls, which are not truly healthy. Document explicitly. For GSE298812 (ccfDNA), the 80-cell immune baseline does NOT apply (substrate mismatch) — noted per CCL-004.

10. **Cohort batch offset** per cohort — compare control group β distributions against Moss 2018 Table S1 healthy reference β. Offsets > 0.05 documented.

11. **Cross-cohort comparison** — if both cohorts have d > 0.3 on M1 or M2, compute cross-cohort consistency score = min(d_1, d_2) / max(d_1, d_2) as magnitude-preservation metric. Report direction match.

12. **Outcome assignment** per locked decision matrix. Report numbers BEFORE assigning.

13. **SHA-lock** both results JSONs (one per cohort, plus a consolidated cross-cohort JSON).

---

## What gets published

- `val059_hcc_epic_gse281691.py` — primary cohort analysis script (GitHub)
- `val059_hcc_epic_gse298812.py` — replication cohort analysis script (GitHub)
- `val059_hcc_epic_cross_cohort.py` — cross-cohort synthesis script (GitHub)
- `VAL059_hcc_epic_gse281691_results.json` — SHA-locked (GitHub)
- `VAL059_hcc_epic_gse298812_results.json` — SHA-locked (GitHub)
- `VAL059_hcc_epic_cross_cohort_results.json` — SHA-locked cross-cohort synthesis (GitHub)
- `VAL_059_PREREG.md` + `VAL_059_SEAL.txt` — pre-registration artifacts (GitHub)
- Evidence Report §VAL-059 section (vault only)
- hcc-epic card v0.1 (vault only) with full universal_reference + lessons_learned blocks
- hcc-epic README (vault only)
- Master README update (vault only) adding hcc-epic to the validated-cards section

---

## What does NOT happen under any outcome

- Panel re-training. Xu-538 stays frozen.
- CpG subset cherry-picking.
- Cohort merging before per-cohort analysis completes — each cohort gets its own analysis first; cross-cohort synthesis follows.
- Retroactive threshold adjustment.
- Collapsing GSE298812 across HCC / cirrhosis / fibrosis / HCC-free without separately reporting disease-spectrum effects.
- Claiming HCC screening detection without per-cohort substrate-specific validation support.

---

## Seal

Pre-registration file `VAL_059_PREREG.md` will be SHA-256 hashed and recorded in `VAL_059_SEAL.txt` before any matrix download. Both files committed to GitHub at commit time as timeline evidence that this specification preceded data access.
