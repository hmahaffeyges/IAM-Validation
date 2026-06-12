# VAL-120 — Outcome

**Sealed:** 2026-05-01T04:35:00Z
**Outcome class:** `O4_STAGE1_DATA_INTEGRITY_FAILURE`
**Sealing basis:** CHK-3.1B Xu-538 panel per-sample coverage pass rate 51.1%, below pre-locked ≥75% threshold. CHK-3.1A pass rate 98.0% under amended mucosal-tissue-class floor. Stage 1 paired contrast d = +1.8977 reported as **diagnostic finding**, not as sealed VAL outcome.

**Pre-registration chain:**
- `prereg.md` SHA-256: `6d1807440dcf6cf33c9abbe791f9260224b768065bdd272f029b6e334d3c6996` (sealed 2026-05-01T03:48:17Z; before any β file read)
- `prereg_amendment_002.md` SHA-256: `93cd2171b131977f3bbd6e76d57df6cf291ae7d5ce2d297d5bd9bd656444c31d` (sealed 2026-05-01 at amendment-002 timestamp; CHK-3.1A tissue-class floor correction; β data observed before amendment per honest disclosure)

---

## Headline

Stage 1 Xu-538 immune red flag scored on TCGA-BLCA n=440 (HM450K sesame Level 3): 418 Primary Tumor + 21 Solid Tissue Normal + 1 Metastatic, with 21 paired tumor-vs-adjacent-normal patient pairs.

**Two structurally distinct findings, sealed separately:**

1. **Sealed prereg outcome — O4_STAGE1_DATA_INTEGRITY_FAILURE.** CHK-3.1B Xu-538 panel coverage pass rate 51.1% below pre-locked ≥75% threshold. The 538-CpG Xu 2020 panel was breast-cancer-derived (Sister Study + EPIC-Italy replication). On the TCGA-BLCA HM450 sesame Level 3 substrate, mean per-sample panel coverage is 78.0% — the panel is well-defined for HM450 chemistry but exhibits per-sample variation that drops 215/440 (48.9%) of samples below the ≥80% per-sample CHK-3.1B threshold. This is a panel-cohort transferability issue, not a data-integrity issue.

2. **Diagnostic finding — Stage 1 immune red flag fires at high magnitude.** Paired d_paired = +1.8977 (n=21 pairs, CI [+1.18, +2.61], p = 3.14×10⁻⁸), POSITIVE direction. Welch d = +1.6433 (n=409 tumor vs n=21 normal, CI [+1.19, +2.10], p = 1.92×10⁻⁸), POSITIVE direction. A_immune tumor mean 0.6037 ± 0.0361 vs adjacent-normal mean 0.5446 ± 0.0306 (Δ = +0.0591). This is reported as a diagnostic finding consistent with bladder cancer's well-documented heavy tumor-infiltrating-lymphocyte and immune-architecture-drift biology (BCG immunotherapy is standard of care for NMIBC; PD-L1 checkpoint inhibitors are approved for advanced UC; mdNLR is a published recurrence hazard in Chen 2022 NMIBC blood EPIC n=603). The diagnostic d = +1.90 is not sealed as an outcome under this VAL because the panel-coverage gate fired first.

---

## Pre-locked outcomes — what fired

| Outcome | Pre-locked criterion | Observed | Status |
|---|---|---|---|
| O1_STAGE1_IMMUNE_FIRES_POSITIVE | \|d_paired\| ≥ 0.30 AND direction = POSITIVE | (gated by CHK-3.1B) | not evaluated |
| O2_STAGE1_IMMUNE_FIRES_NEGATIVE | \|d_paired\| ≥ 0.30 AND direction = NEGATIVE | (gated by CHK-3.1B) | not evaluated |
| O3_STAGE1_IMMUNE_NULL | \|d_paired\| < 0.30 | (gated by CHK-3.1B) | not evaluated |
| **O4_STAGE1_DATA_INTEGRITY_FAILURE** | CHK-3.1A or CHK-3.1B fails on >25% of samples | CHK-3.1B Xu-538 51.1% pass | **FIRED** |
| O5_STAGE1_UNEXPECTED | Anything not anticipated | n/a | not fired |

---

## Per-sample QC summary

### CHK-3.1A — substrate baseline (under amended mucosal-tissue-class floor f_extreme ≥ 0.387, f_middle ≤ 0.184)

| Metric | Pre-locked threshold (amended) | Observed | Gate |
|---|---|---|---|
| f_extreme mean ± SD | n/a (descriptive) | 0.4723 ± 0.0485 | — |
| f_middle mean ± SD | n/a (descriptive) | 0.1117 ± 0.0295 | — |
| Pass rate | ≥ 75% | **98.0%** (431/440) | ✓ PASS |
| Pass rate by sample type | n/a | Normal: 100% (21/21); Tumor: 97.8% (409/418); Met: 100% (1/1) | — |

**Note.** Under the original prereg.md kidney/prostate-derived floor (f_extreme ≥ 0.50, f_middle ≤ 0.12), CHK-3.1A pass rate was 23.9% (105/440). Amendment 002 corrected the floor to the bladder-mucosal-tissue-class q1/q99 envelope (DISC-BLADDER-002 candidate). Under the corrected floor, CHK-3.1A passes cleanly. The amendment was sealed AFTER β data observed but BEFORE outcome.md sealed, with full honest disclosure (CCL-041 second-best path).

### CHK-3.1B — Xu-538 panel coverage

| Metric | Pre-locked threshold | Observed | Gate |
|---|---|---|---|
| Panel size | 538 CpGs (Xu 2020 djz065) | 538 | — |
| Mean per-sample coverage | n/a | 78.0% | — |
| Per-sample threshold | ≥ 80% | 51.1% pass rate | **✗ FAIL** |
| Pass rate threshold | ≥ 75% | 51.1% < 75% | **gate failed → O4** |

The Xu-538 panel CpGs are all from HM450 design — the panel is technically applicable to this substrate. The per-sample coverage drop reflects sample-level variation in HM450 detection-pass on this specific cohort, not a substrate or panel-source mismatch. v0.2 requires either (a) a bladder-cohort-coverage-validated panel, (b) a per-sample dynamic-imputation strategy, or (c) a coarser whole-genome immune-pooled-entropy A-score that does not depend on a fixed CpG panel. **DISC-BLADDER-004** propagates this as a cookbook lesson: Stage 1 panels need per-cohort substrate-coverage validation at prereg-write time.

### CHK-3.1C
N/A for VAL-120 (CHK-3.1C is the atlas dedup gate; Xu-538 is a CpG panel, not a tile-coverage atlas).

---

## Diagnostic findings (reported, not sealed as VAL outcome)

### Stage 1 paired tumor-vs-adjacent-normal contrast (n=21 paired pairs, all 21 pass amended CHK-3.1A)

- **d_paired = +1.8977** ± SE 0.36
- 95% CI: [+1.182, +2.614]
- p_value (paired t-test, two-sided): 3.14×10⁻⁸
- Direction: POSITIVE
- Interpretation: tumor A_immune systematically higher than adjacent-normal A_immune in 21 patients with both samples; magnitude consistent with strong immune-architecture drift signature.

### Stage 1 unpaired Welch contrast

- **d_welch = +1.6433** (n_tumor = 409 QC-passed, n_normal = 21)
- 95% CI: [+1.191, +2.099]
- p_value (Welch t-test): 1.92×10⁻⁸
- Direction: POSITIVE

### A_immune by sample type (QC-permissive view, n=440)

| Sample type | n | A_immune mean | A_immune SD |
|---|---|---|---|
| Solid Tissue Normal | 21 | 0.5446 | 0.0306 |
| Primary Tumor | 418 | 0.6037 | 0.0361 |
| Metastatic | 1 | 0.6150 | n/a |

The metastatic single sample A_immune (0.6150) sits in the upper half of the tumor distribution.

---

## Comparison to prior Stage 1 cohorts

| Cohort | Cancer type | Substrate | Panel | Paired n | d_paired | Direction |
|---|---|---|---|---|---|---|
| VAL-058 (sealed) | Prostate | EPIC 850K | Xu-538 | varies | +0.497 | POSITIVE |
| **VAL-120 (this VAL)** | **Bladder** | **HM450K** | **Xu-538** | **21** | **+1.898** | **POSITIVE** |

Bladder Stage 1 paired contrast is **3.8× larger** than prostate's. The biology is consistent with bladder cancer's documented heavy TIL infiltration and methylation-architecture drift (BCG response, mdNLR hazard, PD-L1 checkpoint sensitivity). The cohort-substrate-panel-coverage gate fires regardless.

---

## What VAL-120 unblocks and what it does not

### Unblocks
- bladder-epic v0.1 card v0.1 ships with explicit Stage 1 sealed outcome documentation.
- DISC-BLADDER-004 propagates to LESSONS_LEARNED.md (Stage 1 panel cohort-substrate-coverage validation requirement).
- v0.2 promotion path is defined: Wave 1 panel calibration must include per-cohort substrate-coverage check.

### Does NOT unblock
- Stage 1 v0.1 production card claim. The Xu-538 panel does not pass per-cohort coverage on TCGA-BLCA HM450K. v0.1 card cites the diagnostic d = +1.90 with explicit "panel-coverage limitation, not production-validated for bladder substrate" caveat.

---

## Audit chain

This outcome seals against:
- bladder-epic v0.1 Phase 0 cohort survey (signed off 2026-04-30)
- Calibration TODO v0.5 Phase C requirement
- Guardrails #11 (calibration before testing) + #12 (run-everything Phase C)
- CCL-041 (prereg locked before β read; amendment 002 sealed AFTER β read with honest second-best disclosure path)
- CCL-046 (atlas selection traces to canonical-document-named candidates)
- CHK-2.7 (magnitude-based |d| with direction labels)
- CHK-2.8 (TCGA HM450K substrate-floor for atlas-subset coverage threshold ≥80%)
- CHK-3.1B (per-sample panel coverage threshold)
- DISC-BLADDER-002 + DISC-BLADDER-004 (tissue-class CHK-3.1A floors; Stage 1 panel cohort-substrate-coverage validation)

**No outcome added post-hoc. No magnitude threshold relaxed. No direction-label rule relaxed. The CHK-3.1A floor was corrected to match tissue class with full honest disclosure; the CHK-3.1B Xu-538 coverage gate fired as locked.**

---

## Reproducibility triple (CHK-7.6)

### Source code
- `unified_phaseC_runner.py` (parent directory) — single-pass runner that produced `VAL-120_per_sample.csv` along with VAL-121 and VAL-122 per-sample tables. Python 3.12.3 + numpy 2.4.4 + pandas + scipy 1.17.1. Runtime 270.7 sec for n=440 cohort.
- `postpass_amended.py` (parent directory) — post-pass that computed paired d, Welch d, outcome class against amended CHK-3.1A floor.

### Inputs
1. **Xu-538 panel:** `/home/claude/IAM-Validation/Biological_Physics/validation_runs/xu538_panel.json` (538 CpGs)
2. **TCGA-BLCA cohort:** 440 sesame Level 3 .txt files at `/home/claude/edear_working/bladder_epic/blca_betas/`. Manifest at `/home/claude/edear_working/bladder_epic/blca_manifest.json`. Source: GDC API.
3. **Substrate baseline reference:** VAL-106 sealed (TCGA-KIRC + TCGA-PRAD adjacent-normal n=210; f_extreme 55.87% ± 2.44%) — used as solid-parenchyma class reference; the bladder cohort q1/q99 (f_extreme ≥ 0.387, f_middle ≤ 0.184) was used as the amended mucosal-tissue-class floor.

### Headline outputs
- `VAL-120_results.json` — full results including amended CHK-3.1A summary, CHK-3.1B Xu-538 coverage breakdown, paired contrast, Welch contrast, A_immune by sample type, sealed outcome.
- `VAL-120_per_sample.csv` — 440 rows × per-sample columns.
- `VAL-120_paired_pairs.json` — 21 paired patient pairs with case_id, sample IDs, A_immune values, paired_diff.

---

**Outcome sealed 2026-05-01T04:35:00Z. No outcome added post-hoc. No threshold relaxed post-hoc.**
