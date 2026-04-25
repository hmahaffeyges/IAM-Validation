# VAL-072 — TCGA-CESC tissue arm OUTCOME

**Card:** cervical-epic v0.1
**Date:** 2026-04-25
**Pre-reg SHA:** 5a72e1ec4f3379f1406c747457b00a74952e27c57c598622612ddb43c35a5aaf
**Manifest SHA:** 434c9f2b10570bfc1d92ae2ea0b83cce3218ed9b82898909d7b3f0625d0dd6d9
**Results JSON SHA:** b2ea81a380f38284a7809ed65d200c9b854b496d08ba32508a257d7a959a4476

---

## Outcome: O3_TISSUE_NULL (with strong bidirectional-cancellation signal)

The pre-locked decision criteria from VAL-072_prereg.md §4:
- Paired d 95% CI straddled zero → **O3_TISSUE_NULL**.
- Per-CpG positive-direction percentage = 47.9% — within the 45-55% bidirectional-cancellation band.

## Numerical results

| Metric | Value |
|---|---|
| n_QC | 3 / 3 |
| Per-patient ΔA | TCGA-MY-A5BF +0.011, TCGA-HM-A3JJ +0.065, TCGA-FU-A3EO +0.029 |
| Mean ΔA | +0.0348 |
| Paired Cohen's d | +1.26 |
| 95% CI | [−0.26, +2.78] |
| Paired t | +2.18 |
| Paired p | 0.029 |
| Per-CpG positive-direction % | **47.9%** (182 / 380 evaluated) |

## Bidirectional decomposition (CCL-027 mandatory)

| Arm | n CpGs | Per-pair ΔA | Paired d | 95% CI |
|---|---|---|---|---|
| Positive-direction (cohort Δβ > 0) | 182 | −0.018, +0.066, +0.033 | +0.63 | [−0.61, +1.87] |
| Negative-direction (cohort Δβ < 0) | 198 | +0.064, +0.068, +0.042 | +4.10 | [+0.63, +7.58] |

## Interpretation

**Cervical-epic is the third candidate bidirectional-cancellation disease, after AD and PDAC.** The TCGA-CESC n=3 cohort shows the same diagnostic signature: per-CpG split clustered at 50/50 (47.9% positive), with both directional arms producing positive per-pair entropy elevation when scored independently. The pooled-entropy paired d of +1.26 is suggestive but the wide CI at n=3 prevents independent inference.

**Comparison to other Cookbook tissue arms:**
- **Bidirectional-cancellation prone (per-CpG % around 50):**
  - PDAC: VAL-066 46.9%, VAL-067 50.4%, VAL-068 52.9%
  - **Cervical: VAL-072 47.9%** (this study)
- **Unidirectional (per-CpG % above 60):**
  - Prostate (VAL-058) ~62%, Breast (VAL-060) ~64%, CRC (VAL-062) ~67%, Lung (VAL-063) ~70%, HCC (VAL-064) ~63%

The negative-arm d of +4.10 (CI lower bound +0.63 above zero) is the cleanest signal in this cohort — the "negative-direction" CpGs (those whose mean β goes DOWN from normal to tumor at the cohort level) actually carry MORE per-pair entropy elevation than the "positive-direction" CpGs. This is the bidirectional-cancellation signature: when you stratify CpGs by cohort-level direction and re-score each arm at the per-pair level, both arms can be positive — the pooled metric only nulls because the pooled mean Δβ values cancel, not because the per-patient entropy elevations cancel.

## Caveats and limits of inference

1. **n=3 is the entire publicly-accessible TCGA-CESC matched-pair pool for HM450.** Inference at n=3 is exploratory by definition. The wide 95% CI on the pooled d ([−0.26, +2.78]) is the honest representation.

2. **TCGA-CESC has only 3 normal-tissue HM450 samples vs 307 tumor samples.** This is the smallest matched-pair pool of any Cookbook tissue-arm cohort. The cohort-completeness rule (CCL-029) was honored — all 3 pairs were run — but the result is not a card-level anchor by itself.

3. **The bidirectional-cancellation pattern requires confirmation at larger n.** VAL-073 (GSE99511 Verlaat n=68) and VAL-074 (GSE46306 n=43) will provide the proper-power tissue-arm anchors. If those cohorts show the same 45-55% per-CpG split, cervical-epic is confirmed as the third bidirectional-cancellation disease, and VAL-080 builds the directional fallback panel using the GSE143752 or GSE287994 LBC training set.

4. **No HPV stratification possible at n=3.** TCGA-CESC HPV status is available but with only 3 patients no meaningful HPV-stratified inference is supportable. VAL-075 (GSE38266) will quantify the HPV-stratification effect at larger n.

5. **Cohort generalization gap.** TCGA-CESC is biased toward advanced-stage cervical cancer (the tumor cohort is predominantly Stage IB–III). Cervical-epic v0.1's primary clinical use case is screening (LBC pap smear), which means VAL-076/077 LBC cohorts are more clinically relevant than VAL-072. VAL-072 is structural confirmation, not clinical anchor.

## Card consequences

VAL-072 alone does NOT anchor the cervical-epic tissue arm. It serves as an **internal-consistency check**: if VAL-073/074/077 produce the same 45-55% per-CpG split, cervical-epic is empirically confirmed as bidirectional-cancellation prone and the directional fallback build (VAL-080) is mandatory.

VAL-072 does NOT change the cervical-epic tier target. The card-level tier will be determined by VAL-076/077 LBC-pathway results (the primary specimen) and VAL-073/074 tissue-arm large-n results, not by this n=3 anchor.

## Reproduction

- Script: `val072_cervical_epic_tcga_cesc.py` (to be written + pushed to GitHub)
- Pre-reg: VAL-072_prereg.md (SHA 5a72e1ec...)
- Seal: VAL-072_PREREG_SEAL.txt
- Manifest: CESC_matched_manifest.json (SHA 434c9f2b...)
- Results: VAL-072_results.json (SHA b2ea81a3...)
- RNG seed: 20260425
- Panel: Xu-538 (SHA ada672960...)
- Source data: NIH GDC public access, TCGA-CESC project, 6 sesame Level 3 β files
