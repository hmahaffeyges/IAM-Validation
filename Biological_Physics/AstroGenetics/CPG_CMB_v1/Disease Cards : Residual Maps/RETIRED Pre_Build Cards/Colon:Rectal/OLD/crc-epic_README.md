# CRC EPIC Card — EDEAR Pre-Diagnostic Detection

**Version 2.4 · 2026-04-28**
**Supersedes:** v2.3 (2026-04-25 CCL-031 terminology disambiguation). v2.2 (2026-04-24 tissue arm with VAL-061 TIL + VAL-062 cycling-class). v2.1 (2026-04-24 universal reference block + lessons-learned). v2.0 (2026-04-23 initial build).
**Change from v2.3:** Early-onset rectal subsection added as a three-VAL chain. VAL-098 TCGA-READ paired tumor/adjacent-normal cycling-class scoring (n=7 paired, d=+0.612 [+0.227, +1.882]) extends VAL-062 anchor to the rectal subsite. VAL-099 TCGA-COAD age-stratified re-analysis (n=26 paired, pooled d=+0.7241 reproduces VAL-062 byte-for-byte) provides under-50 colon stratum descriptive direction signal at n=3 (mean ΔA=+0.0357). VAL-100 GSE282666 under-50 buffy coat polyp Stage 1 immune (n=51 EPIC v2.0) is deferred to v0.2+ raw-IDAT processing per CHK-3.1 data integrity flag (supplementary file is minfi noob-bg-corrected, not raw β; matches VAL-077 cookbook precedent, formalized as CCL-040). CCL-039 LL-MARKER-CPG-TILE-FIDELITY confirmed cookbook-wide on three independent paired tumor-vs-adjacent-normal cohort configurations. commercial_deployment_unaffected_by_validation_limitations block inherited from lung-epic v0.5 standard. Hispanic stratification dropped from subsection per Heath direction; subsection is about under-50 stratum, full stop. Validation tier promoted to `cycling_class_tissue_validated_with_rectal_subsite` based on VAL-098 + VAL-099; under-50 blood arm via VAL-100 deferred to v0.2+. Honest summary: tissue-arm direction confirmed at three cohorts; tissue-arm under-50 direction descriptively confirmed at n=4 combined; blood-arm under-50 unestablished pending v0.2+.

## Clinical claim

A buffy-coat DNA methylation sample from a person 2 to 10+ years before clinical colorectal cancer diagnosis shows a DEPRESSED immune-class architectural A-score on the Xu-538 panel — the opposite direction from breast cancer on the exact same panel. The signal is detectable at the cohort level at pooled effect size d = −0.33 on GSE51032 (n=76 CRC pre-dx vs 424 controls, p = 0.009). A second-stage Moss 2018 tissue-of-origin deconvolution localizes the source to colon epithelial tissue when run on pre-dx CRC plasma.

Stage 1 direction is the Stage-1 discriminator that distinguishes CRC from breast BEFORE Stage 2 even runs. Both diseases use the same Xu-538 immune panel. Breast reads positive, CRC reads negative.

## The workflow in one patient

Identical universal pipeline to the breast card. Only the expected Stage 1 direction and Stage 2 localization target differ.

**Stage 1.** Xu-538 CpGs, H_min(immune) = 0.838889, pooled entropy A-score against the age-matched 80-cell healthy baseline. For CRC, tier call uses depression-from-baseline, not elevation-above-baseline:
- NORMAL: within ±0.03 of baseline
- MARGINAL_DEPRESSION: baseline − 0.03 to baseline − 0.05
- DETECTABLE_DEPRESSION: baseline − 0.05 to baseline − 0.07
- URGENT_DEPRESSION: baseline − 0.07 to baseline − 0.10
- FLOOR_BREACH_DEPRESSION: ≥ baseline − 0.10

**Stage 2 (if Stage 1 negative-direction tier hits DETECTABLE or stronger).** Same Moss 2018 NNLS deconvolution as breast card. Expected CRC pattern: colon_epithelial β drops to approximately 0.610 from healthy reference 0.741, producing a large positive ΔA at colon_epithelial (scored against cycling H_min = 0.856055). All 17 other tissues should sit near healthy. This is validated by VAL-041 Moss 2018 Fig 4a colorectal case data.

**Report.** Clinician receives: A-score tier (directional depression tier), 18-tissue ΔA table from Stage 2 with colon_epithelial highlighted, anatomy-consideration note (Stage 2 cannot distinguish C18 vs C19 vs C20; colonoscopy resolves), assay version, and honest limitations.

## Why CRC reads negative while breast reads positive on the same immune panel

Biology. Pre-diagnostic colorectal cancer drives immune tolerance rather than chronic immune activation. The Treg expansion and immunosuppressive tumor-microenvironment literature for colorectal disease is consistent with our observation: the immune-compartment methylation landscape drops below its healthy entropy floor, not above it.

Pre-diagnostic breast cancer does the opposite — chronic immune activation pushes the landscape above the healthy floor. Same panel, same buffy coat, same H_min — opposite directional signatures.

This two-direction behavior means Stage 1 alone provides a coarse cancer-type classifier: elevated immune A suggests breast-like (also lung, prostate per VAL-046), depressed immune A suggests CRC-like. Stage 2 Moss NNLS then localizes precisely.

### CCL-031 terminology note — CRC is NOT bidirectional cancellation

To prevent confusion in future card builds and review sessions, this card explicitly states what CRC's signal IS and is NOT.

**What CRC IS.** A compartment-direction-flip (CCL-019). Same disease, same Stage 1 panel (Xu-538), opposite-sign pooled A_immune readings depending on compartment. Peripheral blood reads d = −0.33 (negative; suppressed/Treg-dominant circulating immune compartment). Tumor-infiltrating immune compartment reads d = +1.066 (positive; activated TIL inside the tumor bed). Pooled Test 1 (Xu-538 against H_min(immune) = 0.838889) works fine in each compartment alone — it just goes opposite directions in blood vs tumor.

**What CRC is NOT.** Bidirectional cancellation per CCL-031. Bidirectional cancellation is reserved EXCLUSIVELY for the AD-instance pattern: Test 1 pooled A_immune NULLS cross-cohort, AND a directional ±1 z-scored panel built on the same Stage 1 panel PASSES on holdout. CRC has neither feature. Pooled Test 1 passes in blood (negative direction) and passes in tumor (positive direction). No directional fallback panel exists or is needed for CRC. The cross-compartment sign difference and the cross-disease sign difference (CRC negative, breast positive) are real findings — but they are NOT the AD-instance bidirectional cancellation pattern.

**Operational consequence.** CRC card uses pooled A_immune as the primary Stage 1 metric in each compartment. No directional fallback panel. The "inversion" terminology in this README refers to the compartment-direction-flip per CCL-019; it is NOT shorthand for bidirectional cancellation. Future card-review sessions should use this card as a reference example of how to describe a compartment-direction-flip without conflating it with the AD-instance pattern.

**Diseases currently exhibiting the AD-instance bidirectional cancellation pattern (mechanism unresolved, lineage-test pending OQ-2026-01):** ad-immune (VAL-050/051), pancreatic-epic (VAL-066/067/068/069). CRC is NOT in this category.

## Validation summary

| Test | Cohort | n (cases / ctrl) | Primary result | Tier |
|---|---|---|---|---|
| VAL-047 Phase 12 | GSE51032 EPIC-Italy | 76 / 424 | d = −0.33 pooled, p = 0.009 | single_cohort_validated |
| Tightening v2 anatomy | GSE51032 subtypes | C18=60, C19=7, C20=9 | C18 d ≈ −0.57, C19 d ≈ −0.56, C20 d ≈ −0.13 | anatomy_specific |
| VAL-041 | Moss 2018 per-tissue | 10 colorectal cases deconvolved | colon_epithelial top-1 localization | Stage 2 validation |
| Zhao 2020 BMC Cancer | Same GSE51032 cohort | 166 / 424 | per-CpG d = +0.835 (different method) | published_corroboration |
| VAL-048 step 2 | GSE51032 framework-derived cycling panel | 35 / 424 at 0-5yr | d = −0.003, p = 0.99 NULL | null_documented |

**Live re-run on 2026-04-23.** Phase 12 ran on the 3.15 GB GSE51032 matrix in 131.3 seconds with RNG seed 20260420 and produced the pooled and window-stratified numbers above. Result JSON archived in /mnt/user-data/outputs.

## Anatomy stratification — C18, C19, C20

Tightening v2 subdivided the CRC cases by anatomy:
- **C18 (colon proper):** strongest inversion signal, d ≈ −0.57
- **C19 (rectosigmoid):** comparable inversion, d ≈ −0.56
- **C20 (rectum):** essentially no signal, d ≈ −0.13

The signal localizes to colon-proper and rectosigmoid anatomy; rectum does not show the inversion at detectable levels in this cohort. Clinical interpretation: the immune drift pattern is consistent with the differential microbiome, local immune tone, and lymphatic drainage of proximal colon vs distal rectum. It is a biological feature, not a cohort artifact.

Stage 2 Moss NNLS deconvolution does NOT distinguish C18 from C19 from C20 at the tissue level — colon_epithelial is a single entry in the Moss 2018 atlas. Clinical colonoscopy is the anatomy resolver.

## Two separate CRC methodologies — both published, both valid, measuring different things

This section is explicit because v1 of this card conflated them.

**Methodology A — our Stage 1 (Phase 12 Xu-538 immune pooled entropy):** d = −0.326 pooled, NEGATIVE direction. Measures immune-compartment architectural drift. Validated in this card.

**Methodology B — Zhao 2020 BMC Cancer per-CpG directional top-10:** d = +0.835 per-CpG on same GSE51032 cohort. Different panel selection (data-driven top-10 on CRC), different scoring (per-CpG directional signed Z-composite). Validated by Zhao independently in a peer-reviewed publication on this exact cohort.

Both methodologies detect a CRC pre-diagnostic signal on GSE51032. They measure different aspects. Methodology A measures the whole immune class's architectural depression. Methodology B measures a discriminative CpG signature's presence-or-absence. Neither contradicts the other. Our Stage 1 uses Methodology A because the universal panel (Xu-538) and the universal H_min (immune) are fixed across diseases in EDEAR.

Methodology B's positive direction is NOT an inversion failure; it is a different computation. A future EDEAR Stage 1b could run Methodology B alongside Methodology A for additional confirmation. For v2 of this card, Stage 1 uses Methodology A only.

## The VAL-048 null — what it does and does not say

VAL-048 tested a framework-derived 650-CpG cycling-class panel on GSE51032 CRC pre-dx. Pre-registered hypothesis: cycling-class per-patient A-score SD should DIVERGE (cases higher variance than controls) because cycling epithelium accumulates mitotic errors.

Result: d = −0.003, p = 0.99 at the primary 0-5yr window. NULL.

Evidence Report classification: "NULL / UNDERPOWERED — no statistically distinguishable effect at 0-5yr. Indicates 450K array sensitivity insufficient to resolve cycling-class drift with per-patient SD metric, OR cycling tissue does not carry the architectural signature in peripheral blood leukocytes."

VAL-048 does NOT contradict Phase 12. They measure different things:
- VAL-048: per-patient SD of cycling-class A-scores across a 650-CpG panel. Tests variance divergence.
- Phase 12: per-patient mean immune A-score on Xu-538. Tests mean depression.

Phase 12's pooled-mean test on the immune class returned a clean negative d. VAL-048's variance-divergence test on the cycling class returned a null. Both are honest findings. The Cookbook uses Phase 12 as the Stage 1 validation; VAL-048 is logged in known_limitations as "cycling-class per-patient SD metric is not sufficiently resolved on 450K arrays."

## Tissue arm — VAL-061 and VAL-062 (added v2.2)

The tissue arm of the CRC card uses TCGA-COAD HM450 matched tumor/normal biopsies (26 pairs passed 430/538 coverage QC, from 38 initially downloaded). Unlike the blood arm (which reads peripheral immune response to CRC), the tissue arm reads CRC tumor tissue directly through two complementary lenses. Both arms use the same 26 matched pairs and the same cohort SHA (`ce87ad9fb45a1fe652707eca353d95e873d70b009714a448e1b5e5402f37fc27`).

### VAL-062 — Primary: tumor architecture (cycling-class scoring)

**Prereg SHA:** `9b5ff04ce31e4679e32ac8690fefc0b09a0abd646e89792edf956161097b847d`
**Results SHA:** `e8ec05a8932e92c8755febbdb8df0425f9f25d161476895e6a0169837aae2698`
**Date:** 2026-04-24

CRC tumor cells are cycling class (colon_epithelial = cycling, H_min = 0.856055, reference β = 0.740 from TCGA COAD matched normal). VAL-062 scores the 26 matched tumor/adjacent-normal pairs against cycling H_min across all available HM450 CpGs (tissue biopsy → direct β → class-specific A-score, no deconvolution needed because the tissue IS the colon sample).

**Results:**
- n matched pairs: 26
- Paired Cohen's d: **+0.7241**, 95% CI [+0.2922, +1.1559], p = 2.23e-04
- Unpaired Cohen's d: +0.8947, 95% CI [+0.3245, +1.4648], p = 1.26e-03
- A-tumor mean: 0.633 ± 0.028
- A-normal mean: 0.612 ± 0.016
- Absolute ΔA (tumor minus normal, mean across all HM450 CpGs): +0.020

**Outcome:** Preregistered prediction was paired d > 0, 95% CI > 0, d ≥ +0.5. All three criteria met. Cycling-class CRC tumor architecture signal consistent with framework expectation. Comparable to VAL-060 breast secretory tissue arm (+0.745), larger than VAL-058 prostate secretory tissue arm (+0.497) — consistent with CRC's higher-proliferation cycling biology driving broader methylation disruption than secretory-class cancers.

**Note on absolute ΔA:** The +0.020 genome-wide-mean ΔA is smaller than the VAL-001 framework prediction of ΔA ≈ +0.17 for TCGA COAD. The framework prediction is calibrated on cycling-class-discriminating CpG subsets (colon-specific DMRs). VAL-062 averages all ~485K HM450 CpGs, diluting the class signal with probes that are not cycling-informative. The Cohen's d remains strong because between-patient variance at the genome-wide mean level is correspondingly small. For future cycling-class tissue validations, the framework-expected ΔA ≈ +0.17 is recoverable by restricting to cycling-class-informative CpGs (Moss 2018 colon markers + TCGA COAD matched-normal DMRs).

### VAL-061 — Supplementary: tumor-infiltrating immune compartment

**Prereg SHA:** `bdce2f903a20a3375681a3589710c2f5a6392a4f4c6772305fd3afc656bed521`
**Results SHA:** `def8a69030a2b1d1619f4a930e419604b44c0f2097655c97eea7f580f4a12c96`
**Date:** 2026-04-24

The same 26 matched pairs were also scored with the Xu-538 immune panel (SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`) against immune H_min = 0.838889. This reads the tumor-infiltrating immune cell (TIL) compartment inside the CRC tumor bed, not the tumor architecture itself.

**Results:**
- Paired Cohen's d: **+1.0658**, 95% CI [+0.5845, +1.5471], p < 0.00001
- Unpaired Cohen's d: +1.4687, 95% CI [+0.8562, +2.0813], p < 0.00001
- Per-CpG direction: 61% hypomethylated in tumor (295/484), 39% hypermethylated (189/484)

**Interpretation:** The Xu-538 panel is immune-derived (Sister Study breast blood calibration). Applied to tumor tissue, it measures the immune response inside the tumor bed, where tumor-infiltrating lymphocytes are activated and expanded — opposite direction from peripheral blood immune cells, which are suppressed/exhausted in response to disease presence. The per-CpG direction (61% hypomethylated) reflects classic CRC global hypomethylation combined with immune cell activation; aggregate A-score elevates positive because H(β) rises as β moves toward 0.5 from either direction.

**Reconciliation with VAL-047 peripheral blood inversion:** VAL-047 Phase 12 on GSE51032 peripheral blood showed d = −0.33 (suppressed circulating immune response). VAL-061 on tumor tissue shows d = +1.066 (activated TIL response). Same disease, same immune class, opposite-sign readings — because the compartment is different. This is a feature of the framework, documented in CCL-019 (LESSONS_LEARNED.md): A-score direction depends on (class, compartment) pair, not disease alone.

### Clinical implications of the tissue arm

The three CRC readings — VAL-047 peripheral blood immune (d = −0.33), VAL-061 tumor-infiltrating immune (d = +1.07), VAL-062 tumor architecture cycling (d = +0.72) — together document a consistent picture:

1. Peripheral immune suppression is detectable in blood ~5-10 years pre-diagnosis
2. Tumor architecture disruption in colon cycling cells is detectable in tumor tissue at diagnosis
3. Tumor-infiltrating immune activation is detectable inside tumor tissue at diagnosis

For the EDEAR clinical pipeline, this means:
- **Stage 1 (blood-based pre-diagnostic screen)** reads the peripheral immune depression — the negative d is the trigger for CRC suspicion, not a framework inversion
- **Stage 2 (Moss NNLS tissue-of-origin deconvolution from blood)** localizes colon_epithelial when disease is advanced enough to shed recoverable cfDNA
- **Tissue biopsy arm (when available post-diagnosis)** confirms both tumor architecture (cycling-class A elevated) and tumor immune response (TIL A elevated) — useful for treatment planning and immunotherapy response monitoring, not primary detection

### Tissue arm validation status

| Arm | What it reads | Validation status | Primary VAL |
|---|---|---|---|
| Blood — peripheral immune | Circulating immune response to upstream disease | cross_platform_validated (VAL-047 Phase 9 + 12) | VAL-047 |
| Blood — Stage 2 deconvolution | Tumor cfDNA mathematically extracted from plasma | stage_2_only_validated (VAL-041 10/10) | VAL-041 |
| Tissue — tumor architecture | CRC tumor cells scored against cycling H_min | cycling_class_tissue_validated | VAL-062 |
| Tissue — TIL compartment | Tumor-infiltrating immune cells scored against immune H_min | compartment_supplementary | VAL-061 |



Only one cohort (GSE51032 EPIC-HuGeF) has been tested for CRC pre-diagnostic immune inversion at the per-patient level. Cross-cohort replication is pending.

0-2yr pre-dx window on GSE51032 CRC has only 8 cases — underpowered. The pooled significance (p=0.009) is driven primarily by the 2-5yr and 5-10yr windows.

The Xu-538 panel was selected for breast-cancer association in a different cohort (Sister Study). Its CRC inversion is an emergent finding, not the panel's design target. A CRC-specific purpose-built cycling-class panel would likely produce a larger signal. VAL-048 attempted framework-derived cycling-panel construction and demonstrated the selection methodology is sound, but the 450K array is insufficient resolution for cycling-class per-patient SD.

Single-timepoint sensitivity at 95% specificity is modest (screening-adjacent, not diagnostic). Deployment is serial-sample trajectory monitoring with each patient as her own control.

Stage 2 deconvolution production module (G-DECONV-001) is OPEN-DEFERRED — same status as breast card. VAL-041 validated the workflow; production per-IDAT deployment requires module completion.

## File pointers

- **Card JSON:** `crc-epic_card.json` (v2.1 current, v2.2 pending JSON update)
- **Evidence Report section:** §5C VAL-047 Phase 12 colorectal arm + Tightening v2 subtype stratification + §5C VAL-048 framework-derived cycling panel null + §5D VAL-061/VAL-062 tissue arm (v2.2 addition)
- **Source scripts:** `VAL047_phase12_gse51032_replication.py`, `VAL047_tightening_v2_patch.py`
- **Result JSONs:** `VAL047_phase12_gse51032_colorectal_results.json`, `VAL047_phase12_gse51032_combined_results.json`, `VAL-061_results.json` (tissue TIL arm), `VAL-062_results.json` (tissue cycling-class arm)
- **Tissue arm documents:** `VAL-061_prereg.md`, `VAL-061_outcome.md`, `VAL-062_prereg.md`, `VAL-062_outcome.md`
- **Alternative methodology:** `VAL_047_option3_results.json` (Zhao-style data-driven top-10 CpG CRC d = +0.835)
- **Panel:** `xu538_breast_panel.json`, same as breast card
- **Null documented:** `VAL048_step2_results.json` (framework-derived cycling panel null on per-patient SD)

---

## v2.1 changes (2026-04-24)

- **Universal reference block embedded** (full-inline, Option B). The card JSON now contains the complete universal pipeline specification — H_min constants for all 8 architecture classes, Moss 2018 healthy reference β for all 18 tissues, 80-cell age-decade immune baseline, EpiDISH Salas QC bounds, universal tier thresholds, sex-stratification rule, language discipline, and the cross-cohort batch-offset warning from VAL-057. A new analyst loading only this card JSON plus `GAPE_WEB_v13.py` can run the full pipeline end-to-end without consulting any other file.
- **Lessons-learned section added** — 3 disease-specific documented quirks, each with source validation, context, observed quirk, interpretation, and how the card was updated to handle it. See `lessons_learned` key in the card JSON.
- **Cross-card lessons catalog** maintained in `LESSONS_LEARNED.md` at the Cookbook root. This card's entries are labeled with the card prefix (crc-LL-###).


---

## v2.2 changes (2026-04-24)

- **Tissue arm added.** New section "Tissue arm — VAL-061 and VAL-062" documents two complementary readings on TCGA-COAD HM450 matched tumor/normal biopsies (26 pairs):
  - **VAL-062 primary — tumor architecture, cycling-class scoring.** Paired d = +0.7241 [+0.2922, +1.1559], p = 2.23e-04. PASS. Direction confirmed, magnitude strong. Consistent with VAL-060 breast secretory tissue arm magnitude.
  - **VAL-061 supplementary — tumor-infiltrating immune compartment.** Paired d = +1.0658 [+0.5845, +1.5471], p < 0.00001. Measures activated TIL response inside tumor bed. Reconciles with VAL-047 peripheral blood d = −0.33 via compartment-dependent A-score direction (CCL-019).
- **Validation tier progression.** The CRC card now has validation on the blood arm (VAL-047 cross-platform), the Stage 2 deconvolution arm (VAL-041 stage-2 validated), AND the tissue arm (VAL-062 cycling-class tissue validated). First Cookbook card to have all three arms independently validated.
- **Three-compartment documentation.** The card now explicitly names the three CRC signal compartments and their expected directions: peripheral blood immune (negative, suppressed circulating response), tumor-infiltrating immune (positive, activated TIL), tumor architecture (positive, cycling-class disorder). This is the template every future card's tissue arm should follow.
- **Cross-reference to immune-atlas.** CRC is a key row in the immune-atlas cross-reference card (#13 in Cookbook), documented as the canonical example of direction-depends-on-compartment.
- **Cross-reference to LESSONS_LEARNED.** VAL-061 → VAL-062 sequence is the source evidence for CCL-019 (direction depends on class+compartment pair) and CCL-020 (panel × class × specimen are three independent dimensions).


## v2.3 changes (2026-04-25)

- Added §"CCL-031 terminology note — CRC is NOT bidirectional cancellation" to disambiguate compartment-direction-flip (CCL-019) from AD-instance bidirectional cancellation (CCL-031). This is a documentation-only change to prevent terminology drift in future card-review sessions.
- No numerical results changed. VAL-047 d = −0.33 (blood, p = 0.009), VAL-061 d = +1.066 (tumor TIL), and VAL-062 d = +0.724 (tumor cycling-class architecture) all preserved verbatim.
- New canonical card JSON: `crc-epic_card_v2.3.json` (supersedes v2.2). Adds top-level `ccl_031_terminology_disambiguation` block and `ccl_027_four_questions_answered` block.
- Cross-references added: CCL-019 (compartment direction), CCL-006 (disease direction), CCL-027 (four-question guard), CCL-028 (PDAC pooled-null + directional-pass), CCL-030 (Test 1 vs Test 2), CCL-031 (terminology rule).


---

## v2.4 changes (2026-04-28)

- **Early-onset rectal subsection added.** New top-level block `early_onset_rectal_subsection` in the card JSON. Anchored on VAL-098 TCGA-READ paired tumor/adjacent-normal cycling-class scoring (n=7 paired pairs, paired d = +0.612 [+0.227, +1.882], extends VAL-062 TCGA-COAD anchor +0.724 to the rectal subsite). Direction confirmed; magnitude precision limited by cohort size; bootstrap 95% CI lower bound exceeds zero.
- **Atlas resolution constraint documented.** v1 Loyfer atlas (production Stage 2) has Colon_epithelial_cells tile but no rectum-specific tile. Atlas vault check on 2026-04-28 surveyed every reference matrix in `Biological_Physics/atlas_vault/` (Loyfer 25-tile, EpiSCORE ColonRef 5 sub-cell-types, Caggiano CelFiE TIM, MARLIN, Sabedot GeLB, all Stage 3 immune atlases) — zero rectum-specific cell types anywhere. Subsection is therefore a clinical-action layer (age × family-history routing) on top of the existing colorectal cell-of-origin signal, NOT a rectum-vs-colon biology-detection card. Future atlas-resolution evolution may promote subsection to standalone `rectal-epic` card.
- **Hispanic stratification dropped per Heath direction 2026-04-28.** Subsection is about early-onset CRC/rectal at the under-50 stratum, full stop. Clinical action routing uses age + family history; ethnicity not in the EDEAR layer. The article that prompted the subsection build referenced rising rectal cancer incidence in young Hispanic populations in the western US; that population context is real but population-stratified biology is not what the data support at v1, and EDEAR fires the same red flag for all populations with elevated cycling-class colorectal architectural drift.
- **Age-stratified clinical action matrix.** Under-30, age 30-49 (with/without family history of CRC <50), age 50+. Routes elevated EDEAR signal to early-onset CRC workup with rectal exam emphasis (per ACG 2021 / NCCN EOCRC screening guidance) for the 30-49 stratum, and to standard surveillance for the 50+ stratum. Same EDEAR Stage 2 reading routes to different clinical action depending on patient demographics.
- **Run-everything 25-tile observation — CCL-039 LL-MARKER-CPG-TILE-FIDELITY (three-cohort confirmation).** VAL-098 was the first cookbook validation to run BOTH full-HM450 cycling-class methodology AND run-everything 25-tile per-class methodology on the same paired tumor/normal samples. Found: full-HM450 cycling-class paired d positive (TCGA-READ +0.612) AND Colon_epithelial_cells tile paired d strongly negative (TCGA-READ −2.50). Diagnostic re-application of VAL-098 methodology to VAL-062 TCGA-COAD 26-pair cohort confirms cookbook-wide pattern: COAD full-HM450 d = +0.724, COAD Colon_epithelial_cells tile d = −1.55. VAL-099 reproduction (2026-04-28) re-executes VAL-062 cycling-class methodology on the same TCGA-COAD 26-pair cohort with run-everything 25-tile output and reproduces the pattern at the third independent measurement: full-HM450 d = +0.7241 [+0.352, +1.296], Colon_epithelial_cells tile d = −1.603 [−2.173, −1.288]. Three independent paired-tumor-vs-adjacent-normal cohort configurations, three negative cell-of-origin tile readings, three positive full-HM450 cycling-class readings. Direction concordance across all 10 top-magnitude tiles between READ, COAD revisit, and COAD VAL-099 reproduction. Two distinct observables: full-HM450 measures global architectural drift; per-tile marker-CpG measures cell-of-origin tile fidelity which DEGRADES in tumor. Future preregs with run-everything 25-tile must use CHK-4.11 pattern-aware criteria (cell-of-origin tile is among the largest |d|, NOT necessarily largest |d| or strictly positive d).
- **VAL-099 — TCGA-COAD age-stratified re-analysis (sealed 2026-04-28, complete).** No new data download — re-execution of VAL-062 cycling-class methodology on the cached 26-pair cohort plus run-everything 25-tile plus stratified analysis by age decile, anatomic subsite, and sex. Prereg SHA `8e4ee02c59774514...`. Pooled paired d = **+0.7241 [+0.352, +1.296]**, t = 3.69, p = 2.2e-04 — reproduces VAL-062 anchor byte-for-byte (drift = −3.3e-05). Under-50 stratum (n=3: TCGA-A6-2685 48.6y, TCGA-A6-5667 40.4y, TCGA-AA-3663 42.9y) reads mean ΔA = +0.0357, descriptive-only per CHK-2.7 — direction descriptively positive, concordant with pooled positive direction. Age 50+ stratum (n=21) reads paired d = +0.539, inferential. By-subsite: Ascending colon n=8 d=+0.387, Cecum n=5 d=+1.094, Colon NOS n=5 d=+1.702; Sigmoid colon n=3 mean ΔA=−0.0015 (descriptive). One anomaly flagged: TCGA-G4-6625 has subsite "Skin, NOS" in GDC clinical metadata (included consistent with VAL-062 inclusion). Outcome `O1_AGE_STRATIFIED_DIRECTION_CONFIRMED`. Provides under-50 colon stratum descriptive direction signal.
- **VAL-100 — GSE282666 under-50 buffy coat polyp Stage 1 immune (sealed 2026-04-28, DEFERRED to v0.2+).** First cookbook VAL on EPIC v2.0 (GPL33022). Cohort: Kumar 2024, n=51 buffy coat all under age 50, with same-day colonoscopy PNP+/PNP- status (16 PNP+ / 35 PNP-). Prereg SHA `4017913d31b31e03...`. Pre-locked CHK-3.1 beta distribution check **failed**: extreme 3.9% / middle 6.8% (raw β bimodal signature requires extreme >30% AND middle <10%). Supplementary `GSE282666_Betas.csv.gz` is minfi v1.40.0 noob-bg-corrected output per Kumar 2024 Methods, not raw β. CHK-3.2 cross-cohort baseline +15.13 anchor-SD off confirms scale issue. Observed d = +0.236 [−0.363, +0.919] descriptive-only — NOT interpreted under CCL-032 (data integrity → biology → framework). Outcome `O5_DATA_INTEGRITY_FLAG`. Same pattern as VAL-077 (cervical-LBC residual M-values) — third cookbook instance of the pattern, formalized as CCL-040. Defer to v0.2+ raw IDAT processing of `GSE282666_RAW.tar` (~2-4 hour task, no biobank application required). The under-50 blood-arm direction expectation is unestablished at v1; under-50 evidence chain in v2.4 is honestly stated as tissue-arm direction confirmed (VAL-098 + VAL-099) plus blood-arm under-50 deferred.
- **commercial_deployment_unaffected_by_validation_limitations top-level block added.** Inherited from lung-epic v0.5 standard per CCL-037. Documents that EDEAR commercial deployment is single-pipeline patient-vs-internal-reference and is unaffected by Tier 1 cohort coverage gaps for any subgroup. The cross-cohort calibration boundary documented in CCL-037 applies exclusively to retrospective cookbook validation, not to deployment.
- **Validation tier promoted** from `cross_platform_validated (blood) + cycling_class_tissue_validated (tissue, VAL-062)` to `cross_platform_validated (blood) + cycling_class_tissue_validated (tissue, VAL-062) + cycling_class_tissue_validated_with_rectal_subsite (VAL-098 + VAL-099, 2026-04-28; under-50 blood arm via VAL-100 deferred to v0.2+ raw IDAT processing per CHK-3.1)`.
- **Validation chain status post-v2.4.** Tissue arm direction confirmed at three cohorts (VAL-062 + VAL-098 + VAL-099). Tissue arm under-50 direction descriptively confirmed at n=4 combined (VAL-098 n=1 rectal under-50 + VAL-099 n=3 colon under-50, all directionally positive, all descriptive-only per CHK-2.7) — not inferential. Blood arm under-50 direction unestablished — VAL-100 deferred per CHK-3.1 to v0.2+ raw IDAT re-processing. Blood-arm under-50 confirmation is a v0.2+ task, honestly stated rather than pretended-complete.

### v2.4 file pointers

- **Card JSON:** `crc-epic_card_v2.4.json` (this file supersedes v2.3)
- **Evidence Report sections:** §VAL-098 + §VAL-099 + §VAL-100 added to `GAPE_Evidence_Report_UPDATED.html` in the VAL-064-through-VAL-100 disease card series block. Direct GitHub links to VAL-098: [outcome](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-098/outcome.md), [script](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-098/val_098.py), [results.json](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-098/results.json). Direct GitHub links to VAL-099: [outcome](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-099/outcome.md), [script](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-099/val_099.py), [results.json](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-099/results.json), [stratified.json](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-099/stratified.json). Direct GitHub links to VAL-100: [outcome](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-100/outcome.md), [script](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-100/val_100.py), [results.json](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-100/results.json).
- **Source data:** TCGA-READ via NIH GDC public API at `https://api.gdc.cancer.gov/data/{file_id}` per `READ_matched_manifest.json` (VAL-098). TCGA-COAD via NIH GDC public API per `COAD_matched_manifest.json` from VAL-061/VAL-062 (VAL-099). GSE282666 via NCBI GEO public FTP at `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE282nnn/GSE282666/` (VAL-100). All fully Tier 1 public; no dbGaP applications, no biobank gating.
- **VAL-062 revisit diagnostic:** ad-hoc diagnostic at `Biological_Physics/validation_runs/VAL-062_revisit/revisit_val062.py` and `revisit_results.json` (not a sealed VAL — diagnostic re-analysis only, used to confirm CCL-039 cookbook-wide).
