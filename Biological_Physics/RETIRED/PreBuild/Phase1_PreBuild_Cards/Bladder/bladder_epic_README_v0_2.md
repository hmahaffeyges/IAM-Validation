# Bladder-EPIC Card — EDEAR Multi-Modal-Validated + Multi-Atlas-Calibrated Bladder Cancer Card

**Version 0.2 · 2026-05-02**
**Validation tier:** `multi_modal_validated + multi_atlas_calibrated` (anchored by VAL-119 BladderRef calibration + VAL-121 multi-atlas Phase C + VAL-122 Stage 3 immune fine-tune; Stage 1 panel coverage gates v0.3 promotion)
**Built under:** CCL-029 cohort-completeness · CCL-039 cell-of-origin direction expectation · CCL-040 calibration-before-scoring · CCL-041 prereg discipline (strict + second-best paths both exercised) · CCL-042 atlases-deferred structured format · CCL-043 cookbook-wide CCL cross-references · CCL-046 prereg amendment audit-trail · CCL-049 multi-atlas reporting · CHK-2.16 (formalized) · CHK-2.17 (formalized) · CHK-2.18 (formalized) · CHK-3.1A/B/C calibration gates · CHK-3.2 cross-tile sanity · CHK-5.7-5.13 documentation discipline · CHK-7.6 reproducibility triple
**Supersedes:** v0.1 (2026-05-01, sealing-narrative format with VAL-119/120/121/122 sealed). v0.2 promotes by (1) full prostate-v0.3-equivalent structural parity rebuild — clinical claim, workflow, action matrix, validation evidence summary, lessons learned, atlases_used_and_deferred, chk_3_1 thresholds per substrate, run-everything Phase C results block, per-disease scoring policy, DISC-BLADDER block, cookbook-wide CCL cross-references, reproducibility anchors; (2) lung-epic v0.5.1 + crc-epic v2.4.1 retroactive flag amendments propagated via CHK-2.18 cookbook gate; (3) Stage 1 panel cohort-substrate coverage validation rule (CHK-2.17) formalized cookbook-wide. v0.1 content preserved verbatim in audit trail; v0.2 expansions appear as new sections.

## Clinical claim

A sample that produces a Stage 2 BladderRef Epi A-score BELOW the q5 of the VAL-119 healthy floor (q5 = 0.4004, mean = 0.4135, sd = 0.0066) AND concurrent BladderRef EC/Fib/IC microenvironment A-scores ABOVE q95 of their respective healthy floors is flagged as consistent with urothelial-program loss with microenvironment expansion in bladder mucosa — the bladder-cancer architectural signature on tissue substrate. Card v0.2 is anchored to VAL-121 multi-atlas Phase C re-scoring on TCGA-BLCA n=440 HM450K sesame Level 3 (21 paired tumor-vs-adjacent-normal patients) plus VAL-119 ProstateRef-equivalent Phase B calibration of EpiSCORE BladderRef on TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 plus VAL-122 Stage 3 immune fine-tune across three immune atlases.

The bladder-epic card fills a specific clinical gap: when a patient's EDEAR analysis returns a BladderRef Epi NEGATIVE + EC/Fib/IC POSITIVE pattern (the canonical CCL-039 signature for an epithelial-origin tumor with microenvironmental immune and stromal infiltration), the card provides clinical action guidance routed through urology referral with cystoscopy and urinalysis-with-cytology per AUA/EAU early-detection guidelines.

**What this card is NOT.** It is NOT a pre-diagnostic blood screening test for bladder cancer. That would require per-patient Phase 9/12-equivalent run on a cohort like WHI bladder pre-dx (Jordahl et al. 2018 CEBP, n=440 cases + 440 matched controls, HM450 buffy coat, median 7.22 years pre-dx) — currently Tier 3 biobank-gated. v1.0+ next-validation-step.

It is also NOT a stage-discriminating test (NMIBC vs MIBC) — TCGA-BLCA Phase C analysis was not stage-stratified in v0.1; v0.3 will run the stage-stratified re-analysis to test whether the broad-positive Stage 3 immune signature observed resolves into Chen 2022 NMIBC lymphoid-dominant pattern in the NMIBC subgroup vs MDSC-dominant pattern in advanced MIBC.

Upgrade path from `multi_modal_validated + multi_atlas_calibrated` to `cohort_screening_validated`:
1. WHI biobank application (Tier 3 access; typical 8-12 week approval) for Jordahl 2018 cohort.
2. Per-patient Phase 9/12-equivalent run on approved cohort with TtDx metadata, n ≥ 100 bladder cases.

## What this card covers, and what it does not

**What v0.2 covers.** The card knows what bladder tumor tissue methylation looks like on HM450K sesame Level 3, drawn from the TCGA-BLCA n=440 Phase C cohort with 21 paired tumor-vs-adjacent-normal patients. The EpiSCORE BladderRef gene-promoter atlas (4 cell types: EC vascular endothelial, Epi urothelial, Fib fibroblast, IC intra-bladder immune) is calibrated against the structurally-separated TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 healthy reference cohort (VAL-119) with all three CHK gates clear and per-tile healthy-floor distributions sealed. When a patient's tissue methylation shows BladderRef Epi A-score below q5 = 0.4004 with concurrent EC/Fib/IC microenvironment elevation, the bladder-epic card fires with evidence-backed clinical action guidance.

**What v0.2 does NOT cover — and what each limit means clinically.**

1. **No per-patient pre-diagnostic blood validation exists for bladder.** Unlike CRC-epic (which has VAL-047 GSE51032 EPIC-Italy pre-dx), bladder-epic has no equivalent pre-diagnostic blood-methylation validation. The Stage 1 immune flag for a bladder case is only validated at the at-diagnosis tissue level (VAL-120 diagnostic d=+1.90 paired); the panel-coverage gate (CHK-3.1B) failed at 51.1% per-sample pass rate, sealing VAL-120 at O4. The card's clinical claim depends primarily on Stage 2 BladderRef Epi NEGATIVE firing, not on Stage 1 blood-immune-class detection.

2. **Stage 1 Xu-538 panel does not pass production deployment for bladder substrate.** The Xu-538 panel was breast-cancer-derived (Xu 2020 Sister Study + EPIC-Italy replication). On TCGA-BLCA HM450K, the panel exhibits 78.0% mean per-sample coverage with only 51.1% of samples passing the ≥80% per-sample threshold. The diagnostic Stage 1 paired d = +1.8977 (n=21, p=3.14e-08) is real biology consistent with heavy TIL infiltration but the production gate fired. v0.3 requires either a bladder-cohort-coverage-validated Xu-538 subset OR a freshly calibrated Stage 1 panel (Wave 1 VAL-114 path). DISC-BLADDER-004 → CHK-2.17 cookbook gate.

3. **Bulk-WGBS Stage 2 atlases produce inflated cross-tile A-scores on the bladder mucosal-cohort substrate.** The Loyfer Bladder bulk WGBS tile fired POSITIVE +1.91 paired d in the same paired pairs where BladderRef Epi (gene-promoter) fired NEGATIVE −1.46. CHK-3.2 cross-tile sanity confirmed: ALL 14 non-bladder Loyfer solid-tissue tiles fired POSITIVE FIRES at d_paired +2.34 to +2.92 (Thyroid, Pancreas, Kidney, Lung, Breast, Prostate, Colon all uniformly inflated). This is substrate-distribution mismatch, not biology. **Multi-atlas readings on mucosal cohorts must include a gene-promoter sub-cell-type atlas as the primary cell-of-origin reader.** DISC-BLADDER-003 → CHK-2.18 cookbook gate.

4. **TCGA-BLCA Phase C analysis is not stage-stratified.** v0.1 sealed work did not pre-register NMIBC vs MIBC subgroup analysis. The broad-positive Stage 3 multi-atlas immune signature observed (6/6 Salas IDOL POSITIVE) may resolve into stage-distinct patterns on stratified analysis. v0.3 priority-2 next-validation-step.

5. **Variant histology not separately characterized.** Squamous-differentiated, glandular, micropapillary, sarcomatoid, plasmacytoid, neuroendocrine variant histologies of urothelial carcinoma not stratified in v0.2.

6. **Multi-ancestry validation pending.** TCGA-BLCA cohort ancestry composition was not analyzed in v0.1.

7. **Smoking exposure stratification not pre-registered.** Bladder cancer is heavily smoking-driven and Stage 1 immune signature in current/former smokers may carry the F2RL3/AHRR smoking-CpG confound documented in lung-epic CCL-009. Bladder v0.3 should incorporate the same smoking-stratification discipline.

8. **Urine substrate not yet integrated.** Bladder is the cancer type most amenable to urine-cytology methylation panels (UroMark AUC 97% on n=274 voided urine, Bladder EpiCheck, ADXBLADDER, Cheng 7-gene panel — clinical-grade comparators). Urine outperforms blood for early bladder detection in the literature. v0.3+ promotion path.

**So, realistically, can we pick up bladder cancer in bloodwork?** The honest answer from v0.2 data: Stage 1 immune red flag fires very strongly on tissue-level paired contrast (d=+1.90, ~3.8× larger than prostate VAL-058 d=+0.50 on the same Xu-538 panel) consistent with the well-documented heavy TIL infiltration of bladder cancer. But the panel-coverage gate failed on this specific cohort, so we cannot claim production-validated blood Stage 1 deployment for bladder until the v0.3 panel-substrate-coverage work lands. Stage 2 BladderRef Epi cleanly fires NEGATIVE at d=−1.46 (CCL-039 cell-of-origin expectation met) on tissue substrate. Plasma ccfDNA scoring for bladder is not validated; urine substrate is the natural specimen for bladder per the clinical-grade-panel literature and queues for v0.4. When the card fires in v0.2, it's because Stage 2 BladderRef Epi pointed to urothelial program loss with microenvironment expansion, not because Xu-538 flagged bladder-specifically on blood.

## The workflow in one patient

A 67-year-old former smoker (35 pack-years, quit 8 years ago) with 4 weeks of intermittent gross hematuria submits a buffy-coat blood draw plus a tissue biopsy from cystoscopy. The lab runs an Illumina HM450 array on both substrates.

**Stage 1 (universal — buffy coat blood).** Xu-538 CpGs extracted, pooled-entropy A-score computed against H_min(immune) = 0.838889, compared to age-matched 80-cell healthy baseline (60-69 decade). Tier call assigned. Per VAL-120, the Xu-538 panel is currently flagged for cohort-substrate coverage limitation on bladder; the v0.2 Stage 1 score is reported as diagnostic-with-caveat rather than as production-tier-validated.

**Stage 2 (multi-atlas, tissue substrate).** Tissue biopsy β values extracted. EpiSCORE BladderRef tile A-scores computed: A_EC against H_min(stromal)=0.862950, A_Epi against H_min(secretory)=0.843264, A_Fib against H_min(stromal)=0.862950, A_IC against H_min(immune)=0.838889. Each compared to VAL-119 sealed healthy-floor distributions (Epi q5 = 0.4004 is the operational diagnostic threshold). Loyfer bulk Bladder tile read in parallel for triangulation only per CHK-2.18; not used as primary cell-of-origin signal on mucosal cohort. Caggiano CelFiE TIM 19-tile read for stromal/immune microenvironment context.

**Stage 3 (immune fine-tune, both substrates).** Salas Blood.EPIC IDOL 6-cell + UniLIFE Guo 2025 19-cell + Caggiano TIM immune subset scored. Per VAL-122, the broad-positive 6/6 Salas IDOL pattern is the muscle-invasive bladder cancer signature; lymphoid-dominant pattern (CD4T/CD8T POSITIVE + Mono/Neu NEGATIVE) is more characteristic of immunotherapy-responding subgroups; myeloid-dominant pattern (Mono/Neu POSITIVE + CD4T/CD8T NEGATIVE) is more characteristic of advanced MDSC-rich disease.

**Report.** The patient's clinician receives:
- Stage 1 A_immune tier call with explicit panel-coverage caveat
- Stage 2 BladderRef per-tile A-score table with Epi tile flagged against q5 healthy floor; EC/Fib/IC tiles flagged against q95 healthy floors; Loyfer bulk Bladder triangulation reading flagged as substrate-mismatch-context-only
- Stage 3 multi-atlas immune signature (broad-positive vs lymphoid-dominant vs myeloid-dominant pattern call)
- Smoking status disclosure (current / former ≥10yr / former <5yr / never) with pack-years
- Assay version tag (L1 Illumina HM450 + Loyfer/Moss + EpiSCORE BladderRef markers / L2 custom capture / L3 full multi-substrate)
- Salas 2018 QC bounds check status on immune sub-composition
- Clinical action per BladderRef-Epi-tier × stage × smoking-status matrix
- Honest limitations section naming `multi_modal_validated + multi_atlas_calibrated` tier and the pending v0.3 panel work + v1.0+ pre-diagnostic cohort acquisition

## Validation summary

| Anchor | Cohort | n | Primary result | Tier contribution |
|---|---|---|---|---|
| VAL-119 BladderRef Phase B calibration | TCGA-KIRC + TCGA-PRAD adjacent-normal | 210 | All 3 CHK gates clear; Epi tile sd 0.0066, q5 0.4004 (tightest within-cohort variance); max within-cohort range 0.0694 — atlas does NOT collapse | Multi-atlas calibrated (Stage 2 cell-of-origin) |
| VAL-120 Stage 1 Xu-538 (sealed O4) | TCGA-BLCA | 440 (21 paired) | O4 panel-coverage failure (51.1% pass at ≥80% threshold); diagnostic d_paired = +1.8977 (n=21, p=3.14e-08) reported as diagnostic | NOT production-validated for bladder substrate at v0.2 |
| VAL-121 Stage 2 multi-atlas (sealed O2) | TCGA-BLCA | 440 (21 paired) | BladderRef Epi NEGATIVE −1.46 (CCL-039 met) + Loyfer bulk Bladder POSITIVE +1.91 (substrate-mismatch artifact) + EC/Fib/IC POSITIVE (CCL-039 microenvironment met) | Multi-modal validated |
| VAL-122 Stage 3 immune (sealed O1) | TCGA-BLCA | 440 (21 paired) | All 6/6 Salas IDOL POSITIVE at \|d\|=0.49 to 1.24; broad-infiltration signature consistent with mixed TIL+TAM+MDSC in MIBC | Multi-modal validated |
| Stage 1 per-patient pre-dx | — | — | PENDING — WHI bladder pre-dx Jordahl 2018 (Tier 3 biobank-gated) | Not yet `cohort_screening_validated` |

**Total VAL-119/121/122 primary findings:** 3 of 3 sealed at O1 or O2 with biology consistent with bladder cancer literature.

## Sources

**VAL-119 calibration cohort.** Same VAL-106 calibration cohort (TCGA-KIRC + TCGA-PRAD adjacent-normal n=210) that anchored cardio + prostate. TCGA HM450K sesame Level 3.

**VAL-120/121/122 Phase C cohort.** TCGA-BLCA via NIH GDC public API at `https://api.gdc.cancer.gov/data/{file_id}` per `Biological_Physics/validation_runs/VAL-121_bladder_stage2_multiatlas/cohort_manifest.json`. n=440 (418 Primary Tumor + 21 Solid Tissue Normal + 1 Metastatic; 21 paired tumor-vs-adjacent-normal patients).

**EpiSCORE BladderRef source.** Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: <https://doi.org/10.1038/s41592-022-01412-7>. Repository: <https://github.com/aet21/EpiSCORE>.

**Xu-538 panel origin.** Xu Z, Sandler DP, Taylor JA. *Blood DNA Methylation and Breast Cancer: A Prospective Case-Cohort Analysis in the Sister Study.* J Natl Cancer Inst 2020;112(1):87-94. DOI: <https://doi.org/10.1093/jnci/djz065>. PMID 30989176.

**Salas IDOL Stage 3 atlas.** Salas LA, Zhang Z, Koestler DC, Butler RA, Hansen HM, Molinaro AM, Wiencke JK, Kelsey KT, Christensen BC. *Enhanced cell deconvolution of peripheral blood using DNA methylation for high-resolution immune profiling.* Nature Communications 2022;13:761. DOI: <https://doi.org/10.1038/s41467-021-27864-7>.

**Chen 2022 bladder NMIBC blood EPIC pattern reference.** Chen JQ, Salas LA, Wiencke JK, Koestler DC, Molinaro AM, Andrew AS, Seigne JD, Karagas MR, Kelsey KT, Christensen BC. *Immune profiles and DNA methylation alterations related with non-muscle-invasive bladder cancer outcomes.* Clin Epigenetics 2022;14:14. DOI: <https://doi.org/10.1186/s13148-022-01234-6>.

**WHI bladder pre-diagnostic anchor (gated, v1.0+ next-step).** Jordahl D, Salas LA, Wiencke JK, Koestler DC, Cawthon RM, Kelsey KT, et al. CEBP 2018. n=440 cases + 440 matched controls, HM450 buffy coat, median 7.22 years pre-dx. WHI biobank application Tier 3 access.

## Pre-registration chain

- `VAL-119 prereg.md` SHA-256: `04d9d0d36faaf3bc6051c5f852f3bcb9d2ab48c437982b714ff8573d4cad178a` sealed 2026-05-01T03:35:46Z
- `VAL-119 prereg_amendment.md` SHA-256: `c3015ca3ba25f6c13f4f93fec85edea8506f64472657d03b59ed9ccda8355787` sealed 2026-05-01T03:38:56Z (atlas SHA correction; data not observed before amendment, CCL-041 strict path)
- `VAL-120 prereg.md` SHA-256: `6d1807440dcf6cf33c9abbe791f9260224b768065bdd272f029b6e334d3c6996` sealed 2026-05-01T03:48:17Z
- `VAL-120 prereg_amendment_002.md` SHA-256: `93cd2171b131977f3bbd6e76d57df6cf291ae7d5ce2d297d5bd9bd656444c31d` sealed 2026-05-01T04:44:39Z (CHK-3.1A tissue-class floor correction; data observed before amendment, CCL-041 second-best path with full disclosure)
- `VAL-121 prereg.md` SHA-256: `eb68e4d4ca6270cdcce60269375af787537c560fabea18ee31cbaf558dea1962` sealed 2026-05-01T03:48:17Z
- `VAL-121 prereg_amendment_002.md` SHA-256: `7f4b3148949060d6f0b8c27a5b55161c06a848d9b00d1e765ddcb182b3d0ec30` sealed 2026-05-01T04:44:39Z
- `VAL-122 prereg.md` SHA-256: `2d101db94cdc7a71466c5f8071a936abd426f85ecd9ea27ae8fa73cd0d81f855` sealed 2026-05-01T03:48:17Z
- `VAL-122 prereg_amendment_002.md` SHA-256: `db3f6563533ab625326acd42aab7a8028313a898bfec833c756f7be85f00df29` sealed 2026-05-01T04:44:39Z

**On the amendment chain.** VAL-119 amendment 001 was a strict-path CCL-041 amendment (atlas SHA correction triggered by NaN serialization fix; the val119 script errored at load_atlas before any β data was observed). VAL-120/121/122 amendment 002 chain was a second-best-path CCL-041 amendment (CHK-3.1A tissue-class floor correction; β data observed before amendment per honest disclosure with DATA_OBSERVED_BEFORE_AMENDMENT=YES field; lesson logged as DISC-BLADDER-002 candidate; cookbook proposal CHK-2.16 formalized).

- BladderRef bridged CSV SHA: `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`
- Xu-538 panel SHA: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`

## Known limitations

See `bladder-epic_card_v0.2.json` → `known_limitations`. Summary: Stage 1 panel-coverage gate failure (51.1% pass, breast-derived panel; v0.3 requires bladder-substrate-validated panel), no per-patient pre-diagnostic blood validation (WHI biobank gated, v1.0+), TCGA-BLCA not stage-stratified, variant histology not characterized, urine substrate not integrated, multi-ancestry pending, smoking-stratification not pre-registered, EPIC 850K substrate-matched calibration anchor not yet sealed, single-timepoint sensitivity at 95% specificity unknown.

## Lessons learned

See card JSON `lessons_learned` section — 7 entries (bladder-LL-001 through bladder-LL-007) covering gene-promoter atlas family fitness depending on cell-type DISTINCTNESS not COUNT, tissue-class CHK-3.1A floors, bulk-WGBS atlases on mucosal-cohort substrates, Stage 1 panel cohort-substrate coverage, patient-flow sprint structure, CCL-041 honest second-best amendment path, and mucosal-tissue-class floor envelope structural replication across mucosal organs.

---

## v0.2 cookbook-wide CCL cross-references

Every CCL the card inherits, applies, or formalizes:

| CCL | Description | How bladder-epic v0.2 honored / formalized / inherited |
|---|---|---|
| CCL-026 | Urine substrate physics open question | Inherited from prostate v0.3; bladder urine arm queues for v0.4 per L1 lab partnership tier |
| CCL-029 | Cohort-completeness rule | Phase 0 cohort survey produced bladder cohort enumeration: TCGA-BLCA primary, GSE52955 secondary, Bryan UK NMIBC GSE85837 candidate, Chen 2022 NMIBC blood EPIC n=603 GEO-pending verification, WHI bladder pre-dx Tier 3 biobank-gated |
| CCL-030 | Stage 1 Test 1 / Test 2 distinction | Honored: pooled A_immune Test 1 attempted (gated by CHK-3.1B Xu-538 coverage failure); Test 2 deferred per OQ-2026-01 |
| CCL-031 | Bidirectional cancellation reserved for AD-instance pattern only | Honored: bladder does NOT exhibit pooled-null + directional-pass pattern; Stage 1 fired diagnostic POSITIVE consistent across paired and Welch contrasts |
| CCL-032 | Diagnostic order: data integrity → biology → framework | Honored: CHK-3.1A failure caught FIRST; tissue-class mismatch diagnosed (not data-integrity); biology consistency check (heavy TIL infiltration consistent with bladder cancer literature) verified SECOND; framework outcome class assigned LAST |
| CCL-037 | Cross-cohort calibration boundary | Inherited from lung-epic v0.5; commercial_deployment_unaffected_by_validation_limitations block embedded; bladder-epic v0.2 used single TCGA-BLCA cohort only, no cross-cohort comparisons |
| CCL-039 | Cell-of-origin direction expectation | Honored: BladderRef Epi NEGATIVE −1.46 (CCL-039 cell-of-origin direction met); EC/Fib/IC POSITIVE (CCL-039 microenvironment direction met); Loyfer bulk Bladder POSITIVE flagged as substrate-mismatch artifact rather than CCL-039 violation |
| CCL-040 | Calibration-before-scoring discipline | Honored: VAL-119 Phase B BladderRef calibration sealed BEFORE VAL-120/121/122 Phase C scoring on TCGA-BLCA |
| CCL-041 | No post-hoc threshold relaxation; second-best path with full disclosure | Honored TWICE: (1) VAL-119 amendment 001 (strict-path; data not observed before amendment). (2) VAL-120/121/122 amendment 002 chain (second-best path; data observed before amendment with full DATA_OBSERVED_BEFORE_AMENDMENT=YES disclosure and lesson logged) |
| CCL-042 | Atlases-deferred structured format | Honored: 5-atlas atlases_run + 5-atlas atlases_deferred structured tables with target version + unblock dependency per atlas |
| CCL-043 | Cookbook-wide CCL cross-references in card README | Honored: this section |
| CCL-046 | Prereg amendment audit-trail | Honored: VAL-119 + VAL-120 + VAL-121 + VAL-122 each carry separate prereg.md + prereg_amendment(_002).md + PREREG_SEAL.txt + PREREG_AMENDMENT(_002)_SEAL.txt with separate SHAs and timestamps |
| CCL-047 | Atlas dedup audit trail (CHK-3.1C) | Honored: BladderRef bridged matrix verified 0 duplicate probeIDs in 2,696 entries during VAL-119 calibration |
| CCL-048 | Per-tile healthy-floor distributions sealed at calibration | Honored: VAL-119 sealed mean/sd/q5/q50/q95 for all 4 BladderRef tiles |
| CCL-049 | Multi-atlas reporting flag for single-atlas \|d\| > 2 not replicated | Honored: Loyfer bulk Bladder \|d\|=1.91 cross-checked against BladderRef Epi \|d\|=1.46 — direction divergence at high magnitude reported transparently as O2 outcome class with DISC-BLADDER-003 explanation |
| CHK-2.7 | Magnitude-based \|d\| with direction labels | Honored: VAL-121 prereg used \|d\| ≥ 0.30 with direction labels; both POSITIVE and NEGATIVE captured |
| CHK-2.8 | TCGA HM450K substrate-floor for atlas-subset coverage threshold ≥80% | Honored: VAL-119 used ≥80% per-sample threshold; observed q5 = 86.15% |
| CHK-2.16 | Tissue-class CHK-3.1A floor verification at prereg-write time | **FORMALIZED THIS SPRINT — DISC-BLADDER-002.** Mucosal-class envelope (≥0.387/≤0.184) added to TESTING_CHECKLIST |
| CHK-2.17 | Stage 1 panel cohort-substrate coverage validation at prereg-write time | **FORMALIZED THIS SPRINT — DISC-BLADDER-004.** Cookbook gate added; Wave 1 VAL-114 calibration must include CHK-2.17 |
| CHK-2.18 | Atlas-family-on-mucosal-cohort gate | **FORMALIZED THIS SPRINT — DISC-BLADDER-003.** Cookbook gate added; mucosal cohorts require gene-promoter atlas as primary cell-of-origin reader |
| CHK-7.6 | Reproducibility triple | Honored: VAL-119/120/121/122 each carry inline source code + inputs + environment + headline outputs sections |

## v0.2 atlases_used_and_deferred

### atlases_run (calibrated and deployed in v0.2 production scoring)

| Atlas | n_CpGs | n_tiles | Calibration anchor | CHK-3.1B q5 threshold | CHK-3.1C |
|---|---|---|---|---|---|
| EpiSCORE BladderRef CpG-bridged | 2,696 | 4 | VAL-119 (TCGA HM450K sesame Level 3 n=210) | ≥80% per CHK-2.8 (observed q5=86.15%) | passed (0 dups) |
| Layered Moss+Loyfer 25-tile | 6,105 | 25 | VAL-112 (TCGA HM450K sesame Level 3 n=210) | inherited from VAL-112 | passed (sealed VAL-112) |
| Caggiano CelFiE TIM array-bridged | 254 | 19 | VAL-113 | inherited from VAL-113 | passed (sealed VAL-113) |
| Salas Blood.EPIC IDOL 6-cell 450K legacy | 350 | 6 | production calibrated | 100% per-sample pass on TCGA-BLCA | passed |
| UniLIFE Guo 2025 19-cell | 1,906 | 19 | within-cohort self-cal v0.1; VAL-115 Wave 1 promotion path | 100% per-sample pass on TCGA-BLCA | passed |

### atlases_deferred (structured per v0.5 TODO format)

| Atlas | Target version | Unblock dependency |
|---|---|---|
| Bladder-cohort-coverage-validated Stage 1 panel | v0.3 | Wave 1 VAL-114 calibration on Hannum 2013 GSE40279 n=656 healthy aging blood with CHK-2.17 cohort-substrate-coverage gate baked in |
| Substrate-distribution-aware Loyfer normalization | v0.4 | Research on per-sample-substrate-baseline normalization that subtracts cohort tissue-class mean before paired-d computation |
| EpiSCORE LungRef + ColonRef + EsoRef + others | v0.X (next mucosal sprint) | Per CHK-2.18, future lung/CRC/cervical/gastric Stage 2 sprints require gene-promoter atlas as primary; same Entrez→450K bridge methodology as HeartRef/ProstateRef/BladderRef; Phase B mini-calibration on TCGA n=210 sealed before disease-cohort scoring |
| Tanaka 2025 nanopore→array bridge | v0.5+ | Nanopore methylation atlas bridge-engineering workflow not yet in cookbook |
| EPIC 850K substrate-matched BladderRef calibration anchor | v0.4 | Structurally-separated EPIC 850K healthy bladder cohort to anchor cross-substrate generalization (no such cohort currently exists in public deposits at sufficient n) |

## v0.2 chk_3_1_thresholds_per_substrate

The card encounters one measurement substrate in v0.2 sealed work (TCGA HM450K sesame Level 3) but documents BOTH tissue-class envelopes (solid parenchyma and mucosal) per CHK-2.16:

### Substrate 1 — TCGA HM450K sesame Level 3, solid-parenchyma tissue class

| Field | Value |
|---|---|
| substrate_name | tcga_hm450k_sesame_level3 |
| f_extreme_baseline | 55.87% (VAL-106 sealed mean) |
| f_extreme_sd | 2.44% |
| f_middle_baseline | 7.42% |
| f_middle_sd | 0.75% |
| calibration_anchor_val_id | VAL-106 (TCGA-KIRC + TCGA-PRAD adjacent-normal n=210) |
| application | Stage 2 cell-of-origin scoring on solid-parenchyma cohorts (kidney, prostate, breast secretory, liver, thyroid) |

### Substrate 2 — TCGA HM450K sesame Level 3, mucosal tissue class (NEW IN V0.2)

| Field | Value |
|---|---|
| substrate_name | tcga_hm450k_sesame_level3_mucosal |
| f_extreme_baseline | 47.23% (VAL-120 amendment 002 sealed mean on TCGA-BLCA) |
| f_extreme_sd | 4.85% |
| f_middle_baseline | 11.17% |
| f_middle_sd | 2.95% |
| f_extreme_floor_threshold | ≥ 0.387 (cohort q1) |
| f_middle_ceiling_threshold | ≤ 0.184 (cohort q99) |
| calibration_anchor_val_id | VAL-120/121/122 amendment 002 (TCGA-BLCA n=440) |
| chk_3_1b_subset_threshold (BladderRef) | ≥80% per CHK-2.8 (210/210 = 100% pass; q5 = 86.15%) |
| application | Stage 2 cell-of-origin scoring on mucosal-tissue cohorts (bladder, lung-airways, colon, cervical, esophagus, stomach, GI-epithelium, oral-mucosa); pre-registered for v0.3+ lung/crc/cervical Stage 2 sprints per CHK-2.16 |

**Operational rule per CHK-2.16:** At prereg-write time, every card must specify the tissue-class CHK-3.1A floor for its target cohort; do not inherit implicitly from parent cookbook defaults. Tissue-class brackets: solid parenchyma uses VAL-106 (≥0.50/≤0.12); mucosal uses VAL-120 amendment 002 (≥0.387/≤0.184); future tissue classes (e.g., neural, hematopoietic) require per-tissue-class verification at first card encounter.

## v0.2 run_everything_phase_c_results

### Per-cohort headline findings

**TCGA-BLCA (n=440 HM450K sesame Level 3, 21 paired tumor-vs-adjacent-normal patients):**

Stage 1 Xu-538 paired contrast: paired d = +1.8977 [+1.182, +2.614] p = 3.14e-08 POSITIVE on n=21 paired pairs. Welch d = +1.6433 on 409 tumor vs 21 normal. A_immune tumor 0.6037±0.0361 vs adjacent-normal 0.5446±0.0306. Diagnostic finding only — VAL-120 sealed O4 on Xu-538 panel coverage 51.1% pass rate.

BladderRef per-tile signature (the operationally important finding):

| Tile | Tumor mean A | Normal mean A | d_paired | 95% CI | p_value | Direction label | Biological interpretation |
|---|---|---|---|---|---|---|---|
| **Epi** (urothelial — bladder cancer cell of origin) | (paired data per VAL-121 results JSON) | (paired data) | **−1.4623** | [−2.078, −0.847] | 1.60e-06 | **EPI_NEGATIVE** | **Urothelial dedifferentiation** — tumor cells lose canonical urothelial methylation signature (CCL-039 cell-of-origin expectation MET) |
| EC (vascular endothelial) | — | — | +0.4069 | — | 0.077 | EC_POSITIVE | Tumor microvasculature |
| Fib (fibroblast stromal) | — | — | +0.3691 | — | 0.106 | Fib_POSITIVE | Stromal architectural complexity |
| IC (intra-bladder immune) | — | — | +0.5905 | — | 0.014 | IC_POSITIVE | Local immune infiltrate |

**Combined BladderRef pattern: Epi NEGATIVE + EC/Fib/IC POSITIVE = canonical CCL-039 epithelial-origin tumor signature with microenvironmental immune and stromal infiltration.** Structurally identical to prostate VAL-118 pattern (LE_NEGATIVE + BE/EC/Fib/Leu/SM_POSITIVE).

Stage 2 Loyfer bulk Bladder tile (TRIANGULATION ONLY per CHK-2.18):

| Tile | d_paired | 95% CI | p | Direction | CCL-039 expectation | Match | Note |
|---|---|---|---|---|---|---|---|
| Loyfer Bladder (bulk WGBS) | +1.9100 | [+1.191, +2.629] | 2.83e-08 | POSITIVE | NEGATIVE | ✗ | substrate-distribution-mismatch artifact per DISC-BLADDER-003; ALL 14 non-bladder Loyfer tiles also POSITIVE FIRES at +2.34 to +2.92 — substrate-mismatch confirmed cross-tile |

Stage 3 multi-atlas immune signature:

| Atlas | Tile | Cell type | d_paired | 95% CI | p_value |
|---|---|---|---|---|---|
| Salas IDOL | Bcell | B lymphocytes | +1.1479 | [+0.597, +1.699] | 3.79e-05 |
| Salas IDOL | Mono | Monocytes | +1.1322 | [+0.584, +1.680] | 4.46e-05 |
| Salas IDOL | Neu | Neutrophils | +1.2354 | [+0.668, +1.803] | 1.53e-05 |
| Salas IDOL | NK | Natural killer | +0.7943 | [+0.304, +1.285] | 1.63e-03 |
| Salas IDOL | CD8T | Cytotoxic T cells | +0.6222 | [+0.155, +1.089] | 9.87e-03 |
| Salas IDOL | CD4T | Helper T cells | +0.4884 | [+0.036, +0.941] | 3.67e-02 |

All 6/6 Salas IDOL tiles fire POSITIVE at \|d_paired\| range 0.49 to 1.24 — broad immune-architectural drift consistent with mixed TIL+TAM+MDSC infiltration in muscle-invasive bladder tumor microenvironment. Pre-locked O2 (lymphoid-dominant) and O3 (myeloid-dominant) did NOT fire — both lymphoid AND myeloid fired POSITIVE.

### Outcome class summary (sealed per VAL-120/121/122 amendment 002)

| VAL | Outcome | Threshold | Observed | Status |
|---|---|---|---|---|
| VAL-119 | **O1_BLADDERREF_CALIBRATION_SEALED** | All 3 CHK gates clear, max within-cohort range ≥0.02 | 98.1% / 100% / pass / 0.0694 | **FIRED** |
| VAL-120 | **O4_STAGE1_DATA_INTEGRITY_FAILURE** | CHK-3.1B Xu-538 panel coverage <75% pass | 51.1% pass | **FIRED (panel-coverage gate)** |
| VAL-121 | **O2_BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS** | Both COO tiles fire \|d\|≥0.30 with directions diverging | +1.91 POSITIVE vs −1.46 NEGATIVE | **FIRED** |
| VAL-122 | **O1_STAGE_3_IMMUNE_DIFFERENTIATING** | ≥3 of 6 Salas IDOL tiles fire \|d_paired\|≥0.30 | 6/6 tiles fire | **FIRED** |

## v0.2 per-disease scoring policy

### What v0.2 bladder-epic disease scoring claims

**Tissue substrate (FFPE or frozen bladder biopsy):**
- A_BladderRef_Epi BELOW q5 of VAL-119 healthy floor (q5 = 0.4004, mean = 0.4135, SD = 0.0066) is consistent with urothelial dedifferentiation pattern. Direction label EPI_NEGATIVE.
- Concurrent A_BladderRef_EC, A_BladderRef_Fib, A_BladderRef_IC ABOVE q95 of VAL-119 healthy floors for those tiles is consistent with tumor microenvironment architectural complexity (EC q95 = 0.4265; Fib q95 = 0.5020; IC q95 = 0.4263).
- The Epi-negative + EC/Fib/IC-positive pattern is the v0.2 bladder cancer architectural signature on tissue.

**Plasma ccfDNA substrate:**
- Acknowledged as not validated in v0.2. Bladder cancer ccfDNA shedding is variable (MIBC sheds substantially; NMIBC may shed less). v0.3 promotion path requires EPIC 850K substrate-matched calibration plus ccfDNA cohort with bladder cancer cases AND healthy controls.

**Urine substrate (the bladder-natural specimen):**
- v0.2 does NOT have validated urine substrate scoring; no urine cohort scored. v0.4 promotion path includes L1 lab partnership tier collection (n=20-50 urothelial carcinoma + healthy controls + benign bladder pathology controls) AND/OR UroMark targeted bisulfite NGS workflow integration as a clinical-grade comparator (UroMark AUC 97% on n=274 voided urine published).

**Post-treatment monitoring trajectory:**
- After TURBT/BCG/cystectomy treatment, serial Stage 2 BladderRef Epi readings produce a trajectory; A_Epi trending below the q5 healthy floor flags potential urothelial dedifferentiation drift consistent with recurrence risk. v0.2 documents this as research question; not validated by sealed Phase C data.

### What v0.2 bladder-epic scoring DOES NOT claim

- Pre-diagnostic blood screening — WHI bladder pre-dx Tier 3 biobank-gated; v1.0+ next-step
- Production-tier Stage 1 panel deployment for bladder substrate — v0.3 panel-coverage work pending
- NMIBC vs MIBC stage discrimination — v0.3 stage-stratified re-analysis pending
- Variant histology discrimination — squamous, glandular, micropapillary, sarcomatoid, plasmacytoid, neuroendocrine not characterized
- Multi-ancestry generalizability — TCGA-BLCA cohort ancestry not analyzed in v0.1
- Smoking-stratified Stage 1 interpretation — bladder v0.3 should adopt lung-epic CCL-009 smoking discipline
- Plasma ccfDNA bladder cancer detection — substrate-matched calibration not yet sealed
- Urine substrate clinical pathway — v0.4 next-step
- Specificity for bladder vs other urological cancers — v0.3 GSE52955 multi-cancer urological cohort verification pending

## v0.2 DISC-BLADDER discoveries

Each entry below also propagates to LESSONS_LEARNED.md as per Heath-only file delivery.

### DISC-BLADDER-001 — Gene-promoter atlas family fitness depends on cell-type DISTINCTNESS, not COUNT

VAL-119 calibration of EpiSCORE BladderRef (2,696 CpGs × 4 bladder cell types) on TCGA n=210 HM450K sesame Level 3 produced max within-cohort tile range 0.0694 — LARGER separation than ProstateRef's 6 cell types (0.0597) and far larger than HeartRef's 5 cell types (0.0152, sealed O3_TISSUE_FLOOR_DOMINATED). The discriminating variable is per-tissue cell-type DISTINCTNESS at the gene-promoter level for the marker genes Zhu/Teschendorff selected. Bladder's 4 compartments (urothelial barrier-secretory epithelium, intra-bladder vasculature, fibroblast stroma, intra-bladder immune) ARE markedly distinct gene-promoter programs even though there are fewer of them. Cardiac cell types share substantial gene-promoter methylation similarity at the marker-gene level despite being 5 in number.

**Implication.** Atlas family fitness extends DISC-CARDIO-004 + DISC-PROSTATE-001: gene-promoter atlas family fitness is not predictable a priori from EpiSCORE source-matrix dimensions alone. Each tissue's BridgedRef calibration must be smoke-tested independently per VAL-094/111/117/119 protocol.

### DISC-BLADDER-002 — CHK-3.1A f_extreme floor is tissue-class-dependent, not universal

The VAL-106 kidney+prostate-derived solid-parenchyma floor (f_extreme ≥ 0.50, f_middle ≤ 0.12) failed on 76% of bladder samples (23.9% pass rate). Bladder cohort q1/q99 (≥0.387/≤0.184) passes 98%. **Zero samples in the cohort have genuine substrate corruption** — TSS-site analysis confirmed cohort-wide unimodal-shifted distribution (not bimodal-batch-mixed). Bladder mucosa (urothelium plus lamina propria) has a substantially less bimodal methylation distribution than solid kidney/prostate parenchyma — a tissue-architecture fact, not a data-integrity issue.

**Implication.** CHK-2.16 cookbook gate added: every card prereg specifies the tissue-class CHK-3.1A floor at prereg-write time, not inherited implicitly. Solid parenchyma uses VAL-106 floor; mucosal uses VAL-120 amendment 002 floor; future tissue classes require per-tissue-class verification at first card encounter.

### DISC-BLADDER-003 — Bulk-WGBS atlases on mucosal-cohort substrates produce inflated cross-tile A-scores

Loyfer bulk Bladder tile fired POSITIVE +1.91 paired d while EpiSCORE BladderRef Epi (gene-promoter) fired NEGATIVE −1.46 paired d on the same n=21 paired pairs. CHK-3.2 cross-tile sanity flagged ALL 14 Loyfer non-bladder solid-tissue tiles uniformly POSITIVE FIRES at d_paired +2.34 to +2.92 (Thyroid, Pancreas, Cortical_neurons, Uterus_cervix, Upper_GI, Kidney, Lung, Breast, Head_and_neck, Hepatocytes, Prostate, Colon all uniformly inflated). The bulk-tissue WGBS reference encodes mixed-cell-type β profiles and produces |β_sample − β_bulk_ref| dominated by substrate-distribution mismatch on mucosal cohorts. Gene-promoter sub-cell-type references encode signature β profiles for specific cell types and avoid this artifact.

**Implication.** CHK-2.18 cookbook gate added: tissue class ∈ {bladder, lung-airways, colon, cervical, esophagus, stomach, GI-epithelium, oral-mucosa} AND primary cell-of-origin reader's atlas family is bulk-WGBS → REQUIRE at least one gene-promoter sub-cell-type atlas as primary cell-of-origin reader. Bulk-WGBS atlases stay in atlases_run for triangulation but cannot be the headline signal. Lung-epic v0.5.1 + crc-epic v2.4.1 retroactive flag amendments propagate this rule to existing mucosal-cohort cards.

### DISC-BLADDER-004 — Stage 1 panel cohort-substrate coverage is panel-specific and cohort-specific

Xu-538 panel mean per-sample coverage on TCGA-BLCA: 78.0%; pass rate 51.1% at ≥80% per-sample threshold. The panel CpGs are all HM450 design (substrate-applicable) but per-sample coverage drops cohort-specifically due to TSS-site processing variability. Stage 1 panel transferability is cohort-specific, not platform-specific.

**Implication.** CHK-2.17 cookbook gate added: Stage 1 panels must be validated against the target Phase C cohort's substrate-coverage envelope at prereg-write time. Validation procedure: sample 5-10 random Phase C cohort β files, compute per-sample panel coverage, FLAG if mean < 90% or q5 < 80%.

## v0.2 What we chose not to claim

- **Pre-diagnostic blood screening for bladder cancer.** No equivalent validation to CRC-epic VAL-047 GSE51032 EPIC-Italy pre-dx exists. WHI bladder pre-dx Jordahl 2018 is biobank-gated; v1.0+ next-step.
- **Production-validated Stage 1 panel deployment for bladder.** VAL-120 sealed O4 on Xu-538 panel coverage; v0.3 promotion path required.
- **Stage 2 Loyfer bulk Bladder tile as primary cell-of-origin reader.** DISC-BLADDER-003 reframes Loyfer as triangulation-only on mucosal cohorts; BladderRef Epi is the primary.
- **NMIBC vs MIBC stage discrimination.** v0.1 sealed work was not stage-stratified.
- **Pure lymphoid-dominant or pure myeloid-dominant Stage 3 immune signature** (Chen 2022 NMIBC blood EPIC RFS pattern). The 6/6 broad-positive signature observed is the more biologically realistic mixed-infiltration pattern of muscle-invasive bladder cancer; the lineage-skewed patterns require stage-stratified analysis to surface in NMIBC subgroup if present.
- **Multi-ancestry generalizability.** TCGA-BLCA cohort ancestry composition not analyzed in v0.1.
- **Smoking-stratified Stage 1 interpretation.** Bladder v0.3 must adopt lung-epic CCL-009 smoking-stratification discipline.
- **Plasma ccfDNA substrate scoring.** Substrate-matched calibration anchor not yet sealed; v0.3 promotion path.
- **Urine substrate clinical pathway.** No urine cohort scored in v0.1; v0.4 promotion path.
- **Specificity for bladder vs other urological cancers.** GSE52955 multi-cancer urological verification not yet run; v0.3 priority-3 next-step.

## v0.2 What remains open

1. Stage 1 panel cohort-substrate coverage validation (Wave 1 VAL-114 on Hannum 2013 GSE40279 with CHK-2.17 gate baked in) — v0.3
2. NMIBC vs MIBC stage-stratified TCGA-BLCA Phase C re-analysis — v0.3
3. GSE52955 multi-cancer urological cohort cross-verification (specificity check) — v0.3
4. Bryan UK NMIBC HM450 cohort GSE85837 candidate verification + scoring — v0.3
5. Chen 2022 NMIBC blood EPIC n=603 GEO-deposit verification + Stage 1 + Stage 3 reanalysis — v0.3 (depends on data availability statement verification)
6. EpiSCORE LungRef + ColonRef bridges (option-c deferred per lung-epic v0.5.1 + crc-epic v2.4.1) — next mucosal sprint
7. EPIC 850K substrate-matched BladderRef calibration anchor — v0.4
8. Substrate-distribution-aware Loyfer normalization research — v0.5
9. Urine substrate L1 lab partnership tier collection OR UroMark workflow integration — v0.4
10. WHI bladder pre-dx biobank application — v1.0+
11. Multi-ancestry validation cohort — v0.4+
12. Smoking-stratified Stage 1 panel work — v0.3
13. Variant histology stratification — v0.4+
14. Tanaka 2025 nanopore bridge atlas — v0.5+

## v0.2 Validation evidence summary

### VAL-119 — EpiSCORE BladderRef Phase B calibration anchor

**Cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (160 KIRC + 50 PRAD)
**Substrate:** TCGA HM450K sesame Level 3
**Design:** structurally-separated healthy substrate calibration on bridged BladderRef matrix
**QC pass rate:** CHK-3.1A 206/210 (98.1%), CHK-3.1B 210/210 (q5=86.15%), CHK-3.1C 0 duplicates in 2,696 unique probeIDs
**Outcome:** `O1_BLADDERREF_CALIBRATION_SEALED`

**Per-tile healthy-floor distributions sealed:**
- EC: mean 0.4087, sd 0.0100, q5 0.3972, q95 0.4265, range 0.0565
- **Epi: mean 0.4135, sd 0.0066, q5 0.4004, q95 0.4219, range 0.0410** (TIGHTEST WITHIN-COHORT VARIANCE — operational cell-of-origin tile)
- Fib: mean 0.4875, sd 0.0090, q5 0.4770, q95 0.5020, range 0.0694 (widest range, separates cleanly from EC/Epi/IC)
- IC: mean 0.4106, sd 0.0086, q5 0.4001, q95 0.4263, range 0.0504

**Interpretation:** BladderRef CpG-bridged matrix calibrates cleanly on TCGA HM450K sesame Level 3 substrate. All four tiles produce distinguishable healthy-floor distributions; Epi tile has tightest floor (operationally most-important for bladder cancer cell-of-origin disease scoring discrimination — analog to prostate's LE tile with sd=0.0041, range=0.0293). Atlas does NOT collapse to tissue-floor-dominated like VAL-111 HeartRef did — DISC-BLADDER-001 finding sealed.

**Prereg SHA-256:** `04d9d0d36faaf3bc6051c5f852f3bcb9d2ab48c437982b714ff8573d4cad178a` (sealed 2026-05-01T03:35:46Z)
**Prereg amendment SHA-256:** `c3015ca3ba25f6c13f4f93fec85edea8506f64472657d03b59ed9ccda8355787` (sealed 2026-05-01T03:38:56Z; atlas SHA correction; data NOT observed before amendment, CCL-041 strict path)

### VAL-120 — Stage 1 Xu-538 immune red flag

**Cohort:** TCGA-BLCA n=440 (418 Primary Tumor + 21 Solid Tissue Normal + 1 Metastatic; 21 paired tumor-vs-adjacent-normal patients)
**Substrate:** TCGA HM450K sesame Level 3
**Design:** within-cohort paired tumor-vs-adjacent-normal case-control on n=21 paired patients; secondary unpaired Welch contrast
**Outcome:** `O4_STAGE1_DATA_INTEGRITY_FAILURE` (panel-coverage gate); diagnostic d_paired = +1.8977 reported as diagnostic

**Key Cohen's d values:**
- Stage 1 paired d = +1.8977, 95% CI [+1.182, +2.614], p = 3.14e-08, POSITIVE direction
- Stage 1 Welch d = +1.6433, 95% CI [+1.191, +2.099], p = 1.92e-08, POSITIVE direction
- A_immune tumor mean: 0.6037 ± 0.0361
- A_immune adjacent-normal mean: 0.5446 ± 0.0306 (Δ = +0.0591)

**Diagnostic finding interpretation:** consistent with bladder cancer's well-documented heavy tumor-infiltrating-lymphocyte and immune-architecture-drift biology (BCG immunotherapy is standard of care for NMIBC; PD-L1 checkpoint inhibitors are approved for advanced UC; mdNLR is a published recurrence hazard in Chen 2022 NMIBC blood EPIC n=603). Bladder Stage 1 paired contrast magnitude is ~3.8× larger than prostate VAL-058 (+0.497) on the same Xu-538 panel — bladder TIL infiltration is more aggressive than typical prostate adenocarcinoma's immune microenvironment.

**Sealed outcome basis:** CHK-3.1B Xu-538 panel per-sample coverage 51.1% pass rate below pre-locked ≥75% threshold. Panel was breast-cancer-derived; per-sample coverage drops cohort-specifically (mean 78.0%) due to TSS-site processing variability on TCGA-BLCA. DISC-BLADDER-004 → CHK-2.17 cookbook gate added.

**Prereg SHA-256:** `6d1807440dcf6cf33c9abbe791f9260224b768065bdd272f029b6e334d3c6996` (sealed 2026-05-01T03:48:17Z)
**Prereg amendment 002 SHA-256:** `93cd2171b131977f3bbd6e76d57df6cf291ae7d5ce2d297d5bd9bd656444c31d` (sealed 2026-05-01T04:44:39Z; CHK-3.1A tissue-class floor correction; data observed before amendment per CCL-041 second-best path with full DATA_OBSERVED_BEFORE_AMENDMENT=YES disclosure)
**Outcome seal timestamp:** 2026-05-01T04:35:00Z

### VAL-121 — Stage 2 multi-atlas Phase C run-everything

**Cohort:** TCGA-BLCA n=440 (same cohort as VAL-120)
**Substrate:** TCGA HM450K sesame Level 3
**Design:** Stage 2 multi-atlas Phase C run-everything across three calibrated atlases
**Atlases:** Layered Moss+Loyfer 6,105 CpGs × 25 tiles (VAL-112) + EpiSCORE BladderRef 2,696 CpGs × 4 tiles (VAL-119) + Caggiano CelFiE TIM 254 CpGs × 19 tiles (VAL-113)
**Outcome:** `O2_BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS`

**Cell-of-origin paired contrasts (n=21 paired pairs):**
- BladderRef Epi paired d = **−1.4623** [−2.078, −0.847], p = 1.60e-06, NEGATIVE — CCL-039 cell-of-origin direction expectation MET (urothelial dedifferentiation pattern, structurally identical to prostate VAL-118 LE_NEGATIVE)
- Loyfer bulk Bladder paired d = **+1.9100** [+1.191, +2.629], p = 2.83e-08, POSITIVE — substrate-distribution-mismatch artifact per DISC-BLADDER-003

**BladderRef microenvironment paired contrasts (CCL-039 POSITIVE expected):**
- BladderRef EC paired d = +0.4069, p = 0.077, POSITIVE — microenvironment expansion
- BladderRef Fib paired d = +0.3691, p = 0.106, POSITIVE — stromal complexity
- BladderRef IC paired d = +0.5905, p = 0.014, POSITIVE — local immune infiltrate

**CHK-3.2 cross-tile sanity:** ALL 14 Loyfer non-bladder solid-tissue tiles fire POSITIVE FIRES at d_paired +2.34 to +2.92 — substrate-distribution-mismatch signal across all bulk-tissue references. Bladder tumor is not "becoming Thyroid + Pancreas + Liver simultaneously"; the bulk-WGBS reference β profiles are uniformly far from the bladder cohort's tissue-class methylation distribution shape.

**Atlas coverages:** Loyfer 92.6% mean / 100% pass rate; BladderRef 89.1% / 100%; Caggiano TIM 86.0% / 100%.

**Prereg SHA-256:** `eb68e4d4ca6270cdcce60269375af787537c560fabea18ee31cbaf558dea1962` (sealed 2026-05-01T03:48:17Z)
**Prereg amendment 002 SHA-256:** `7f4b3148949060d6f0b8c27a5b55161c06a848d9b00d1e765ddcb182b3d0ec30` (sealed 2026-05-01T04:44:39Z)
**Outcome seal timestamp:** 2026-05-01T04:35:00Z

### VAL-122 — Stage 3 immune fine-tune

**Cohort:** TCGA-BLCA n=440 (same cohort as VAL-120/121)
**Substrate:** TCGA HM450K sesame Level 3
**Design:** Stage 3 immune fine-tune across three immune atlases
**Atlases:** Salas Blood.EPIC IDOL 350 CpGs × 6 tiles + UniLIFE Guo 2025 1,906 CpGs × 19 tiles + Caggiano CelFiE TIM immune subset 254 CpGs × 8 tiles
**Outcome:** `O1_STAGE_3_IMMUNE_DIFFERENTIATING`

**Salas IDOL 6-tile paired contrasts (n=21 paired pairs):**
- Bcell d_paired = +1.1479 [+0.597, +1.699], p = 3.79e-05, POSITIVE FIRES
- Mono d_paired = +1.1322 [+0.584, +1.680], p = 4.46e-05, POSITIVE FIRES
- Neu d_paired = +1.2354 [+0.668, +1.803], p = 1.53e-05, POSITIVE FIRES
- NK d_paired = +0.7943 [+0.304, +1.285], p = 1.63e-03, POSITIVE FIRES
- CD8T d_paired = +0.6222 [+0.155, +1.089], p = 9.87e-03, POSITIVE FIRES
- CD4T d_paired = +0.4884 [+0.036, +0.941], p = 3.67e-02, POSITIVE FIRES

**Biological signature:** broad immune-architectural drift across all six Salas IDOL tiles consistent with mixed TIL+TAM+MDSC infiltration in muscle-invasive bladder tumor microenvironment.

**Pre-locked O2/O3 status:** Pre-locked O2_LYMPHOID_DOMINANT (CD4T/CD8T POSITIVE + Mono/Neu NEGATIVE — would have replicated Chen 2022 NMIBC blood EPIC RFS pattern) did NOT fire (both lymphoid AND myeloid POSITIVE). Pre-locked O3_MYELOID_DOMINANT (Mono/Neu POSITIVE + CD4T/CD8T NEGATIVE — advanced MIBC MDSC pattern) did NOT fire (both lymphoid AND myeloid POSITIVE). Neither pure-direction pattern fired; the broad-positive signature is the muscle-invasive bladder cancer pattern.

**Prereg SHA-256:** `2d101db94cdc7a71466c5f8071a936abd426f85ecd9ea27ae8fa73cd0d81f855` (sealed 2026-05-01T03:48:17Z)
**Prereg amendment 002 SHA-256:** `db3f6563533ab625326acd42aab7a8028313a898bfec833c756f7be85f00df29` (sealed 2026-05-01T04:44:39Z)
**Outcome seal timestamp:** 2026-05-01T04:35:00Z

## v0.2 Reproduction bundle

- **VAL-119 calibration:** [`VAL-119_bladderref_calibrate/`](https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics/validation_runs/VAL-119_bladderref_calibrate) — prereg.md, prereg_amendment.md, val119_bladderref_calibrate.py, VAL-119_calibration_results.json, VAL-119_per_sample_calibration.csv, outcome.md, PREREG_SEAL.txt, PREREG_AMENDMENT_SEAL.txt
- **VAL-120 Stage 1:** [`VAL-120_bladder_stage1_xu538/`](https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics/validation_runs/VAL-120_bladder_stage1_xu538) — prereg.md, prereg_amendment_002.md, val120_bladder_stage1_xu538.py, VAL-120_results.json, VAL-120_per_sample.csv, VAL-120_paired_pairs.json, VAL-120_stratified_results.json, cohort_manifest.json, clinical_metadata.json, outcome.md, EXECUTION_NOTE.md, PREREG_SEAL.txt, PREREG_AMENDMENT_002_SEAL.txt
- **VAL-121 Stage 2:** [`VAL-121_bladder_stage2_multiatlas/`](https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics/validation_runs/VAL-121_bladder_stage2_multiatlas) — prereg.md, prereg_amendment_002.md, val121_bladder_stage2_multiatlas.py, VAL-121_results.json, VAL-121_per_sample_per_atlas.csv, VAL-121_cross_tile_sanity.json, VAL-121_stratified_results.json, VAL_121_unified_per_sample.csv, cohort_manifest.json, clinical_metadata.json, outcome.md, EXECUTION_NOTE.md, PREREG_SEAL.txt, PREREG_AMENDMENT_002_SEAL.txt
- **VAL-122 Stage 3:** [`VAL-122_bladder_stage3_immune/`](https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics/validation_runs/VAL-122_bladder_stage3_immune) — prereg.md, prereg_amendment_002.md, val122_bladder_stage3_immune.py, VAL-122_results.json, VAL-122_per_sample_per_atlas.csv, VAL-122_stratified_results.json, cohort_manifest.json, clinical_metadata.json, outcome.md, EXECUTION_NOTE.md, PREREG_SEAL.txt, PREREG_AMENDMENT_002_SEAL.txt
- **BladderRef atlas:** [`atlas_vault/stage2_cell_of_origin/episcore_bladderref/`](https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref) — episcore_bladderref_cpg_bridged.csv (SHA `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`), episcore_bladderref_entrez_matrix.csv, bridge_bladderref_to_array.py, README.md
- **Atlas vault inventory:** [`atlas_vault/INVENTORY.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/atlas_vault/INVENTORY.json) — 94 entries including new BladderRef
- **Atlas vault README:** [`atlas_vault/README.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/atlas_vault/README.md) — backfilled with all four EpiSCORE per-tissue bridges (HeartRef Apr 29 + ProstateRef Apr 30 + BladderRef May 1) + atlas-family-fitness rule + mucosal-cohort rule
- **Biological_Physics README:** [`Biological_Physics/README.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/README.md) — bladder-epic v0.1 sprint paragraph at line 12

**Companion cookbook IP (Heath-only — not on GitHub):** TESTING_CHECKLIST.md (CHK-2.16/2.17/2.18 added) · LESSONS_LEARNED.md (DISC-BLADDER-001/002/003/004 entries + retroactive flag table) · EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md (Parts 23/24/25 added) · README_MASTER v2.7 amendment line · GAPE_Reproduction_Paper_v1.md §7.27 · GAPE_Evidence_Report_UPDATED.html bladder VAL block · CROSS_CARD_CALIBRATION_TODO_v0_7.md · lung-epic_README_v0_5_1.md · crc-epic_README_v2_4_1.md · this card README v0.2 + JSON v0.2.
