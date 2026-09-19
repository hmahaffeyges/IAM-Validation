# VAL-072 — cervical-epic Tissue Arm on TCGA-CESC HM450 Matched Tumor/Normal

**Pre-registration date:** 2026-04-25
**Card:** cervical-epic v0.1
**Status:** SEALED before any β-value access. Manifest retrieval (file IDs only, no β values) completed via NIH GDC public API.

---

## 1. Hypothesis and rationale

Cervical squamous cell carcinoma and adenocarcinoma arise from the cervical transformation zone (squamocolumnar junction). Cervical_epithelial sits in the GAPE cycling class. H_min(cycling) = 0.856100 from G-002 MCMC posterior. The cycling-class tissue-biopsy A-score elevation pattern has been validated in three prior cards (VAL-062 colon d=+0.724, VAL-063 lung d=+1.020, plus the VAL-058 prostate secretory analog).

**Primary hypothesis (H1).** TCGA-CESC matched tumor-vs-adjacent-normal cervical tissue HM450 IDATs, scored on the Xu-538 immune-class panel against H_min(immune) = 0.838889, will produce a positive paired Cohen's d at the panel level. Cycling-class tissue-arm precedent (VAL-062 +0.724, VAL-063 +1.020) suggests d ≥ +0.5 is plausible. Sample size limits inference: TCGA-CESC has only n=3 matched pairs (the smallest available across all Cookbook tissue arms). VAL-072 is therefore an exploratory anchor, not a powered test.

**Secondary hypothesis (H2 — per CCL-027 mandatory bidirectional cancellation guard).** Per-CpG cohort-level direction split of (β_tumor − β_normal). The four CCL-027 questions for cervical-epic:
- (i) Pooled-entropy expected direction: positive expected (cycling-class precedent).
- (ii) Bidirectional-cancellation risk: literature suggests HPV-driven cervical cancer recruits both lymphoid (CD8+ TILs in regressors, suppressed in progressors) and myeloid (M2 macs, MDSCs in advanced disease) compartments. **Risk classification: MODERATE.** Lower than PDAC because cervical cancer is not as stromal-dense, but higher than breast/colon because of HPV-immunoevasion biology.
- (iii) Directional-panel fallback specification: if pooled-entropy nulls or per-CpG split clusters at 50%, build cervical-specific directional panel as VAL-080 using GSE143752 or GSE287994 LBC training set.
- (iv) Lymphoid-vs-myeloid expected pattern: Clarke 2020, Stanley 2010 — HPV-driven progression involves lymphoid suppression (MHC-I downregulation, TCR-restricted Treg expansion) plus myeloid expansion (MDSC, M2). Pattern parallels PDAC structurally.

**Tertiary hypothesis (H3).** Per-CpG positive-direction percentage vs the bidirectional-cancellation threshold. If positive-direction % falls in 60-70% range, consistent with VAL-062/063 cycling cohorts (unidirectional). If 45-55%, consistent with PDAC bidirectional cancellation, triggering VAL-080 directional fallback build.

---

## 2. Cohort, manifest, and access

**Source.** NIH GDC public access. TCGA-CESC project. Level 3 sesame β values. No dbGaP gating required.

**Cohort composition (manifest-verified pre-seal):**

TCGA-CESC has only n=3 patients with adjacent-normal HM450 methylation:
- TCGA-MY-A5BF (Primary Tumor + Solid Tissue Normal)
- TCGA-HM-A3JJ (Primary Tumor + Solid Tissue Normal)
- TCGA-FU-A3EO (Primary Tumor + Solid Tissue Normal)

This is the entire publicly-accessible TCGA-CESC matched-pair pool for HM450. Per CCL-029 cohort-completeness rule: the entire pool is run, even at n=3.

**File IDs (manifest retrieved 2026-04-25 pre-seal):**

| Patient | Sample type | File ID | Size |
|---|---|---|---|
| TCGA-MY-A5BF | Primary Tumor | 7383dcc9-889f-4797-bf0a-9da5ae2e811d | 12.5 MB |
| TCGA-MY-A5BF | Solid Tissue Normal | f292b6bb-c235-420f-8e5d-0fcefaba99c5 | 12.6 MB |
| TCGA-HM-A3JJ | Primary Tumor | 64c27b2f-9c04-469b-b6fc-69156e027c98 | 12.5 MB |
| TCGA-HM-A3JJ | Solid Tissue Normal | 16b792bd-d268-4b39-80d3-23ab3b980879 | 12.5 MB |
| TCGA-FU-A3EO | Primary Tumor | f14a3d2b-4533-4387-a593-2b956b40e82d | 12.5 MB |
| TCGA-FU-A3EO | Solid Tissue Normal | 7d44a9ef-e300-4127-8698-0f9c17fbb123 | 12.6 MB |

6 files total, ~75 MB combined.

**Manifest SHA-256:** computed and locked at file PAAD_matched_manifest.json equivalent (CESC_matched_manifest.json) at the time of seal. The manifest contains only patient IDs, sample types, and file UUIDs — no β values.

---

## 3. Pre-specified analysis pipeline

**Stage 1 — Xu-538 immune-class A-score (per universal pipeline).**

- Panel: Xu-538 (538 CpGs, Xu 2020 JNCI). Panel SHA-256: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6` (file-bytes verified at runtime).
- H_min: 0.838889 (immune class, G-003b MCMC posterior, frozen). **Stage 1 always uses H_min(immune) regardless of disease — panc-LL-007 universal rule.**
- Score: `A_pooled = mean over Xu-538 CpGs present of [ H(β) / H_min(immune) ]` where H(β) = −β log₂(β) − (1−β) log₂(1−β).
- QC threshold: ≥400 valid Xu-538 CpGs per sample. Samples below threshold excluded.
- RNG seed: 20260425 (deterministic; no random sampling in this analysis).

**Primary statistical test.** Paired Cohen's d on per-patient ΔA = A_tumor − A_normal. Paired t-test for p-value. 95% CI on d via standard SE = sqrt(1/n + d²/(2n)). Hedges' correction reported alongside.

**Per-CpG direction analysis (M5).** For each Xu-538 CpG present in all 6 samples, compute cohort-level Δβ = mean(β_tumor) − mean(β_normal). Count CpGs with Δβ > 0 (positive direction) and Δβ < 0 (negative direction). Report positive-direction percentage.

**Bidirectional decomposition (CCL-027 mandatory).** Split Xu-538 CpGs into positive-arm (cohort Δβ > 0) and negative-arm (cohort Δβ < 0). Score each arm independently per patient against H_min(immune). Report per-arm paired Cohen's d separately.

**No deconvolution.** TCGA-CESC IDATs are tissue biopsies. Stage 2 Moss NNLS is conceptually inapplicable to bulk-tissue β values. Direct read against H_min(cycling) = 0.856100 is the Stage 2 conceptual analog but the framework's Stage 2 production module does NOT score cervical_epithelial because cervical_epithelial is not in the Moss 2018 25-tissue reference. **VAL-072 reports Stage 1 A_immune only. Stage 2 cervical_epithelial scoring is documented as a v0.2+ engineering deliverable in card §10 known limitations.**

---

## 4. Pre-locked outcome decision criteria

| Outcome ID | Criterion | Card consequence |
|---|---|---|
| **O1_PASS_STRONG** | Paired d ≥ +0.5 AND lower CI > 0 AND per-CpG positive-direction % ≥ 60% | Tissue arm anchored at single_cohort_validated tier; pooled-entropy is primary Stage 1 metric |
| **O2_PASS_BIDIRECTIONAL_RISK** | Paired d ≥ +0.5 AND lower CI > 0 BUT per-CpG positive-direction % in 45-55% | Pooled-entropy works at THIS cohort but bidirectional risk flagged; trigger directional panel build (VAL-080) for cross-cohort robustness |
| **O3_TISSUE_NULL** | Paired d 95% CI straddles zero | Document cohort as exploratory; rely on VAL-073/074/077 for tissue arm anchor; trigger directional panel build (VAL-080) |
| **O4_NEGATIVE** | Paired d ≤ −0.5 with lower CI > 0 in negative direction | Document inversion finding; cervical-epic tissue arm direction reversed from cycling-class precedent — major framework finding |
| **O5_UNEXPECTED** | Pooled d in opposite direction from per-CpG split, OR n_qc < 2, OR any other inconsistent pattern | Convene with Heath before card update; flag as exploratory; do not lock card direction |

**Note on n=3.** This cohort is too small for definitive inference. ALL outcomes from VAL-072 are exploratory at the n=3 level. The card-level Stage 1 anchor will come from VAL-076/077 LBC cohorts or VAL-073/074 larger tissue cohorts, not from VAL-072 alone. VAL-072 serves as an **internal-consistency check** against the larger validations — if the n=3 TCGA-CESC pattern matches the n=68 GSE99511 pattern, the framework is internally consistent.

---

## 5. Pre-registration seal

This pre-registration is sealed at the time the SHA below is computed. After sealing, no β values may be accessed until VAL-072 outcome is written and SHA-locked.

**Pre-reg file:** VAL-072_prereg.md
**Pre-reg SHA-256:** computed and recorded in VAL-072_PREREG_SEAL.txt at seal time.
**Manifest SHA-256:** computed and recorded in CESC_matched_manifest.json at seal time.
**Xu-538 panel SHA-256:** ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6 (verified at runtime).
**RNG seed:** 20260425.

After this seal, the workflow is:
1. Download 6 β files via GDC API
2. Verify file SHAs match manifest
3. Run val072_cervical_epic_tcga_cesc.py
4. Write VAL-072_outcome.md against pre-locked criteria above
5. Lock VAL-072_results.json SHA
6. Push to GitHub IAM-Validation/Biological_Physics/validation_runs/

No deviations from this prereg without an amendment file with its own SHA seal.
