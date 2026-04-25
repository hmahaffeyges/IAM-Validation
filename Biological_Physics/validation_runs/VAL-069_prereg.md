# VAL-069 Pre-Registration — pancreatic-epic Directional Panel (CCL-027 fallback)

**Sealed:** 2026-04-25 UTC
**Card:** pancreatic-epic
**Card version target:** v0.1 (directional-panel fallback per CCL-027 mandate)

## Background

VAL-066 (TCGA-PAAD HM450, n=5 paired), VAL-067 (GSE49149 HM450, n=196 case-control), and VAL-068 (GSE74071 HM450, n=7 paired + multi-substrate) collectively show that the Xu-538 immune-class **pooled-entropy** A-score does NOT produce a robust uniform-direction architectural signal for PDAC tumor vs adjacent-normal. The three cohorts span pooled d from +1.18 (VAL-066) to +0.25 (VAL-067) to −0.52 (VAL-068), with all 95% CIs spanning zero. Per-CpG positive-direction percentages are 46.9% (VAL-066), 50.4% (VAL-067), and 52.5% (VAL-068) — all near 50/50.

This is exactly the bidirectional-cancellation pattern CCL-027 mandates flagging. The VAL-051 AD precedent established the operational fix: assign each panel CpG a frozen direction (+1 / −1) from training-cohort Δβ signs, then score new samples as `mean(direction × z(β))` instead of `mean(H(β)/H_min)`. That recovered AD signal from null pooled (d = +0.077) to positive directional (d = +0.624 on holdout).

VAL-069 builds the same directional fallback for pancreatic-epic, using GSE49149 (n=196, the largest available cohort) as training and TCGA-PAAD (VAL-066, n=5) and GSE74071 (VAL-068, n=21 tumor+normal samples that pass QC) as separate holdouts.

## Pre-registered hypotheses (sealed before any directional-score computation)

**H1 (training cohort, GSE49149):** Per-CpG direction assignment (+1 if mean β_tumor > mean β_normal across all GSE49149 samples; −1 if mean β_tumor < mean β_normal) followed by `A_dir = mean(direction × z_score(β))` will exceed pooled-entropy d on the same training cohort. Trivially true by construction since direction is fitted on this cohort; this is the calibration step, not an inferential test.

**H2 (TCGA-PAAD holdout, n=5 from VAL-066):** Applying the GSE49149-trained directional Xu-538 subset to TCGA-PAAD, paired Cohen's d on `A_dir(tumor) − A_dir(normal)` exceeds the VAL-066 pooled-entropy paired d of +1.18 (or, more meaningfully, has a tighter 95% CI lower bound above zero). This is the cross-cohort generalization test.

**H3 (GSE74071 holdout, n=7 paired):** Applying the GSE49149-trained directional Xu-538 subset to GSE74071 produces paired d > +0.3 with lower 95% CI > 0. The VAL-068 pooled-entropy paired d was −0.31; recovering positive signal here would be a strong validation of directional-fallback over pooled-entropy for PDAC.

## Outcome thresholds

- **O1: DIRECTIONAL_RECOVERY_VALIDATED** — H2 PASS AND H3 PASS. Directional Xu-538 subset is the correct Stage 1 metric for pancreatic-epic. Replaces (or supplements) pooled-entropy in v0.1 card.
- **O2: PARTIAL_RECOVERY** — Either H2 OR H3 passes, not both. Direction-fitting helps in some cohorts but does not generalize uniformly. Documented as exploratory; v0.1 retains pooled-entropy with directional-fallback as supplementary report.
- **O3: NULL_DIRECTIONAL** — Neither H2 nor H3 passes. The PDAC signal at Xu-538 is genuinely heterogeneous and cannot be rescued by direction fitting on this panel. v0.1 documents the panel as exploratory for PDAC; future work needs a PDAC-specific CpG selection (not from the Xu-538 panel scope).

## Methods

- Training cohort: GSE49149 (167 tumor + 29 normal, all on HM450, all QC pass per VAL-067).
- Holdout 1: TCGA-PAAD (5 tumor + 5 normal QC-pass per VAL-066).
- Holdout 2: GSE74071 (7 paired tumor + 7 paired normal per VAL-068; 12 unpaired tumor + 8 unpaired normal also reported separately).
- Direction assignment: for each Xu-538 CpG present in GSE49149, sign of (mean β_tumor − mean β_normal) across full GSE49149 cohort. CpGs with |Δβ| < 0.005 or measured in fewer than 80% of GSE49149 samples are excluded from the directional subset.
- Z-score: per-CpG z = (β − μ_normal) / σ_normal, where μ_normal and σ_normal are computed from the GSE49149 normal arm (n=29). Same z-score normalization is applied to all three cohorts (training and both holdouts) using the GSE49149-trained means and standard deviations.
- A_dir = mean across panel of (direction_cpg × z_cpg). Sample-level score.
- Paired Cohen's d on A_dir tumor minus A_dir normal per patient. 95% CI standard formula. Hedges correction reported.
- RNG seed 20260425.

## Reproducibility anchors

- Pre-registration SHA-256: (computed at seal)
- GSE49149 panel-direction subset: SHA computed from sorted (cpg, direction) tuples after build
- Z-score normalization parameters: SHA computed from sorted (cpg, μ, σ) tuples after build
- Holdout results SHA: (computed at run)

## Deliverables

1. `val069_pancreatic_epic_directional.py` — reproducible Python 3 stdlib script
2. `VAL-069_prereg.md` — this document
3. `VAL-069_outcome.md`
4. `VAL-069_results.json` — directional panel composition + holdout results
5. `pancreatic_directional_panel.json` — the per-CpG direction + z-score normalization parameters (this becomes part of the pancreatic-epic card v0.1 JSON `directional_fallback_panel` block)
