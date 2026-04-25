# VAL-073 — cervical-epic Tissue Arm on GSE99511 (Verlaat 2018) HM450

**Pre-registration date:** 2026-04-25
**Card:** cervical-epic v0.1
**Status:** SEALED before any β-value access. Series matrix downloaded but unparsed; sample header inspected only to confirm composition.

---

## 1. Hypothesis and rationale

GSE99511 (Verlaat et al. 2018, Amsterdam UMC / Steenbergen group) is the only HM450 cervical cohort that includes adjacent-normal cervical tissue alongside CIN3 lesions and SCC tumors at meaningful sample size (n=68 total: 28 normal + 36 CIN3 + 4 SCC). It is the proper-power tissue-arm anchor for cervical-epic v0.1, replacing the n=3 TCGA-CESC anchor (VAL-072) as the primary tissue-arm reference.

**Primary hypothesis (H1).** Pooled-entropy A_immune (Xu-538 against H_min(immune) = 0.838889) on Normal vs CIN3 vs SCC produces a monotonic progression: A_normal < A_CIN3 < A_SCC. Cycling-class precedent (VAL-062 colon, VAL-063 lung) supports this monotonic pattern. **Pre-locked tests:**
- H1a: Normal (n=28) vs CIN3 (n=36) unpaired d ≥ +0.5 with lower CI > 0.
- H1b: Normal (n=28) vs SCC (n=4) unpaired d ≥ +0.5 with lower CI > 0 (n=4 limits CI precision).
- H1c: CIN3 (n=36) vs SCC (n=4) unpaired d ≥ 0 (any positive direction acceptable; n=4 SCC is small).

**Secondary hypothesis (H2 — CCL-027 mandatory).** Per-CpG cohort-level Δβ direction split for Normal-vs-Tumor (Normal vs SCC + CIN3 combined as "lesion"). VAL-072 TCGA-CESC at n=3 produced 47.9% positive — bidirectional cancellation signature. VAL-073 at n=68 will confirm or refute this signature with proper power.
- If positive-direction % is 45-55%: cervical-epic confirmed bidirectional-cancellation prone, VAL-080 directional fallback build mandatory.
- If positive-direction % is 60-70%: cervical-epic operates as standard cycling-class card; pooled-entropy is primary metric.
- If positive-direction % is below 45%: anomalous; VAL-080 directional fallback uses negative-direction-dominant panel.

**Tertiary hypothesis (H3).** Bidirectional decomposition per CCL-027. Both arms scored independently against H_min(immune). Report per-arm unpaired d.

**Quaternary hypothesis (H4 — CIN3 progression detection).** Does the framework detect CIN3 lesions (pre-cancerous, treatable) at the same magnitude as established cancer? CIN3 is the screening-relevant target for cervical-epic clinical deployment — detection of CIN3 prevents progression to invasive cancer.
- H4a: Normal vs CIN3 d compared against Normal vs SCC d. If Normal-vs-CIN3 d is ≥ 50% of Normal-vs-SCC d, the framework supports CIN3 detection.
- H4b: per-CpG positive-direction % differs between (CIN3 vs Normal) and (SCC vs Normal). If CIN3 shows 50/50 split but SCC shows 60+%, suggests bidirectional cancellation is a CIN3-specific phenomenon (HPV-driven immune dysregulation in pre-cancer) that resolves into uniform direction in invasive cancer.

---

## 2. Cohort, manifest, and access

**Source.** GEO public access. GSE99511 (Verlaat et al. 2018). DOI 10.18632/oncotarget.20454.
**Series matrix file:** `GSE99511_series_matrix.txt` (286 MB unzipped).
**Series matrix SHA-256:** computed and recorded in seal at the time of analysis.
**Platform:** GPL13534 (Illumina HumanMethylation450 BeadChip).

**Cohort composition (header-verified pre-seal):**

| Group | n | GSM IDs |
|---|---|---|
| Normal cervical tissue | 28 | GSM2644971-GSM2644998 |
| CIN3 lesion | 36 | GSM2644999-GSM2645034 |
| SCC tumor | 4 | GSM2645035-GSM2645038 |
| **Total** | **68** |  |

**No paired-sample structure** — Normal, CIN3, and SCC are separate cohorts of different patients. Analysis is unpaired between groups.

---

## 3. Pre-specified analysis pipeline

**Stage 1 — Xu-538 immune-class A-score.**

- Panel: Xu-538 (538 CpGs). Panel SHA-256: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6` (file-bytes verified at runtime).
- H_min: 0.838889 (immune class).
- Score: `A_pooled = mean over Xu-538 CpGs present of [ H(β) / H_min(immune) ]`.
- QC threshold: ≥400 valid Xu-538 CpGs per sample.
- RNG seed: 20260425.

**Primary statistical tests.**
- Normal vs CIN3 unpaired Cohen's d, 95% CI via standard SE, Welch t-test for p-value.
- Normal vs SCC unpaired d (n=4 SCC limits inference; reported with explicit n caveat).
- CIN3 vs SCC unpaired d (n=4 SCC).
- Normal vs (CIN3+SCC pooled as "lesion") unpaired d.

**Per-CpG direction analysis (M5).**
- For each Xu-538 CpG present in ≥80% of samples in each arm: compute cohort-level Δβ_lesion = mean(β_lesion) − mean(β_normal) using lesion = CIN3 + SCC pooled.
- Count positive (Δβ > 0) and negative (Δβ < 0) CpGs.
- Report positive-direction % vs the 45-55% bidirectional band.

**Bidirectional decomposition (CCL-027 mandatory).**
- Split CpGs by sign of cohort Δβ.
- Score each arm independently against H_min(immune).
- Report per-arm unpaired d (Normal vs lesion).

**No deconvolution.** GSE99511 IDATs are tissue biopsies. Stage 2 cervical_epithelial scoring is documented as v0.2+ engineering deliverable.

---

## 4. Pre-locked outcome decision criteria

| Outcome ID | Criterion | Card consequence |
|---|---|---|
| **O1_PASS_PROGRESSION** | Monotonic A_normal < A_CIN3 < A_SCC AND H1a d ≥ +0.5 lower CI > 0 AND per-CpG positive % ≥ 60 | Cycling-class unidirectional pattern; cervical-epic uses pooled-entropy as primary; VAL-080 directional fallback NOT required |
| **O2_PASS_BIDIRECTIONAL_RISK** | H1a d ≥ +0.5 BUT per-CpG positive % in 45-55 | Pooled works at this n but bidirectional risk confirmed; VAL-080 directional fallback build is mandatory |
| **O3_TISSUE_NULL** | H1a d 95% CI straddles zero | Cervical-epic tissue arm at this cohort is null at pooled-entropy; VAL-080 directional fallback mandatory; rely on VAL-074/077 for tissue arm anchor |
| **O4_CIN3_NULL_SCC_PASS** | H1a (Normal vs CIN3) null but H1b (Normal vs SCC) PASS | Framework detects established cancer but not CIN3 — major clinical limitation; cervical-epic v0.1 cannot claim screening-tier deployment |
| **O5_NEGATIVE** | H1a d ≤ −0.5 with upper CI < 0 | Inversion finding; cervical cancer drives Xu-538 in negative direction at pooled level — major framework finding |
| **O6_UNEXPECTED** | Any other inconsistent pattern | Convene before card update |

**Confirmation criterion for VAL-072 cross-cohort match.** If GSE99511 per-CpG positive-direction % falls within ±5 percentage points of TCGA-CESC's 47.9%, cervical-epic's bidirectional-cancellation status is empirically confirmed at proper sample size. This becomes the third independent confirmation (after AD and PDAC) and triggers CCL-028 expansion.

---

## 5. Pre-registration seal

This pre-registration is sealed at the time the SHA below is computed.

**Pre-reg SHA-256:** computed at seal time; recorded in VAL-073_PREREG_SEAL.txt.
**Series matrix SHA-256:** computed at seal time.
**Xu-538 panel SHA-256:** ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6.
**RNG seed:** 20260425.

After this seal, no β values may be parsed from the series matrix until VAL-073_outcome.md is written and SHA-locked.
