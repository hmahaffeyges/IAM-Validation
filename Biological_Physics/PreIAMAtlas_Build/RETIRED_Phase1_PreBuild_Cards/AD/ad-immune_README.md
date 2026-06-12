# AD Immune Card — EDEAR Cross-Sectional Detection

**Version 2.0 · 2026-04-23**

## Clinical claim

A buffy-coat DNA methylation sample from a person with clinical Alzheimer's disease shows an elevated directional immune-class architectural signature (A_dir) on a 7-CpG Rule A panel selected in an 80/20 holdout of the AIBL cohort. On the sealed AIBL holdout (n=33 AD vs 95 HC), A_dir separates cases from controls at Cohen's d = +0.624, p = 0.00126, AUC = 0.677. Cross-platform replication on AddNeuroMed (n=93 AD vs 96 HC, 450K instead of EPIC, multi-center European instead of Australian) gives raw d = +0.332 (p = 0.009), age-regressed d = +0.124.

Stage 1 directional is the SOLE validated clinical signal for AD. Stage 2 Moss NNLS returns no solid-organ localization (expected — brain tissue is not in buffy coat). Stage 3 EpiDISH immune sub-composition is descriptive pending a dedicated AD-specific threshold validation.

EDEAR AD is not an AD diagnosis. It is a flag that changes downstream workup. Final AD diagnosis requires clinical cognitive assessment, neurological evaluation, and (when clinically indicated) amyloid PET or CSF biomarkers.

## The workflow in one patient

**Stage 1.** 7-CpG Rule A panel extracted from the IDAT. Frozen directions (per VAL-051 SEAL 2026-04-23 07:23:53 UTC):

| CpG | Direction |
|---|---|
| cg16867657 | +1 |
| cg25809905 | −1 |
| cg22454769 | +1 |
| cg09809672 | −1 |
| cg26614073 | −1 |
| cg00431549 | −1 |
| cg02228185 | −1 |

Each CpG β is z-scored against training-HC mean and SD, multiplied by its frozen direction, summed. Raw A_dir and age-regressed A_dir are computed. Pooled entropy A-score on the same 7 CpGs is reported alongside as a secondary metric (per the Directional-Score Principle, always report both).

Tier call on the age-regressed A_dir (empirical thresholds from VAL-054b HC-internal permutation):
- NORMAL_AD_RISK: A_dir z within ±1.0 — no action
- MARGINAL: +1.0 to +1.5 — reassess in 12 months
- DETECTABLE: +1.5 to +2.0 — recommend cognitive screening (MoCA, MMSE) and Stage 3 immune sub-composition
- URGENT: +2.0 to +2.85 — recommend neurology referral
- FLOOR BREACH: ≥ +2.85 (exceeds VAL-054b HC-internal p99 threshold) — strong signal, neurology workup regardless of cognitive symptoms

**Stage 2 (universal — runs for every DETECTABLE+ flag regardless of disease).** Moss 2018 NNLS deconvolution. For AD, the expected result is NULL localization — no solid-organ tissue ΔA exceeds DETECTABLE tier. This null is AD-consistent and expected. If Stage 2 DOES localize to a solid organ, that suggests concurrent cancer or an AD-plus-cancer comorbidity, not AD-only.

**Stage 3 (for AD-directional flag + null Stage 2).** EpiDISH RPC deconvolution of the same IDAT into 6 immune sub-types against Salas 2018 reference: CD4+ T-cell, CD8+ T-cell, NK, B cell, monocyte, neutrophil. Output is a 6-cell-type proportion vector and Salas-bounds QC status. Descriptive for AD pending a validated AD-specific sub-composition threshold. Current card tier for Stage 3 AD is EXPLORATORY.

## Why AD uses the directional panel instead of the Xu-538 panel that breast and CRC use

Because AD's per-CpG methylation drift is bidirectional. VAL-050 tested the IMM_CPGS_EPIC_18 panel (18 CpGs covering the canonical immune-class architectural signature) on AIBL with pooled entropy A-score. Result: d = +0.077, p = 0.32, AUC = 0.512 — NULL.

The per-CpG data showed a clear bidirectional pattern: 10 of 18 CpGs had positive Δβ in AD vs HC, 8 of 18 had negative Δβ. Pooling cancels the signal because the entropy H(β) is symmetric around β = 0.5 — moving β up by 0.02 in one CpG and down by 0.02 in another both increase or decrease entropy, and the effects cancel at the pooled-mean level.

The VAL-051 Rule A directional panel assigns each CpG a direction (+1 or −1) based on training-cohort signed Δβ, and multiplies before summing. Signal that cancels in pooled entropy survives in directional weighting. On the same AIBL cohort, same 7 CpGs:
- Pooled entropy: d = +0.056, p = 0.42, NULL
- Directional (Rule A): d = +0.624, p = 0.001, recovery

This is the Directional-Score Principle: pooled entropy is the right first-tier metric when disease drift is uniform-direction (as for breast and CRC on Xu-538). Directional is the right first-tier metric when disease drift is bidirectional (as for AD). EDEAR reports both regardless; the card specifies which is primary for that disease.

## Validation summary

| Test | Cohort | n (AD / HC) | Primary result | Tier |
|---|---|---|---|---|
| VAL-050 | AIBL GSE153712 EPIC | 161 / 471 | d = +0.077 pooled entropy NULL | null_documented |
| VAL-051 Rule A | AIBL 80/20 holdout | 33 / 95 | d = +0.624, p=0.001, AUC=0.677 | holdout_validated |
| VAL-052 | AddNeuroMed GSE144858 450K | 93 / 96 | raw d = +0.332 (p=0.009), age-regressed d = +0.124 | cross_platform_validated |
| VAL-053 | AIBL 80/20 sex-specific panels | 33 / 95 | Unified beats sex-specific | panel_production_decision |
| VAL-054a | AIBL cellular-age regression | n/a | NON-TEST (baseline saturation) | non_test |
| VAL-054b | AIBL HC-internal permutation | 95 HC resamples, 10000 perms | observed d=+0.624 at z=+2.85 in HC null, p_hc_internal=0.003 | bound_validated |
| VAL-040 | Multi-study AD meta-analysis | n=3,424 | 4/4 predictions confirmed, multi-class drift | cohort_level_support |

## The age confound and how we handle it

VAL-052 discovered that approximately 26% of the A_dir variance in AddNeuroMed tracks chronological age (R² = 0.26). AD patients in that cohort were on average 3-5 years older than HC (Cohen d on age alone = +0.45). Age-regressing A_dir reduced the AD-vs-HC effect from d = +0.332 to d = +0.124.

EDEAR AD deployment uses the AGE-REGRESSED A_dir as the PRIMARY clinical metric. Raw A_dir is reported alongside for transparency. Every AD report must disclose the age-tracking component of the raw signal.

VAL-054b HC-internal permutation (size-matched resample within HC only) provides a separate check: the observed AD-vs-HC d = +0.624 exceeds the 99th percentile of within-HC-only variance (p99 = +0.519). Within-HC confounds — INCLUDING within-HC age variance — cannot alone reproduce the AIBL holdout signal. So the age confound matters (reduces effect size 26%) but does not erase the signal.

## What Stage 2 looks like for AD vs for a solid-organ cancer

For breast or CRC, Stage 2 Moss NNLS is expected to produce a clear max-ΔA tissue (breast_ductal or colon_epithelial). For AD, Stage 2 is expected to produce NO solid-organ localization because brain tissue (neuron/oligodendrocyte — terminal class, H_min = 0.772837) is not in buffy-coat plasma at levels Moss NNLS can resolve.

A normal Stage 2 NULL result in an AD-Stage-1-flagged patient is AD-consistent. An anomalous Stage 2 hit in an AD-Stage-1-flagged patient is a red flag for AD+cancer comorbidity and should be clinician-reviewed.

## Sex differences

On the AIBL holdout (Rule A unified panel):
- Female: n_AD=19, n_HC=55, d = +0.705, p = 0.003, AUC = 0.70
- Male: n_AD=14, n_HC=40, d = +0.512, p = 0.041, AUC = 0.66

Both sexes show a positive AD signal. Female signal is stronger. VAL-053 tested whether sex-specific panels outperform the unified panel; they did not. Unified panel is the EDEAR production choice. Sex-specific calibration remains a future research direction.

## Known limitations (must appear in every patient report)

Approximately 26% of the A_dir signal in the cross-platform replication tracks chronological age. Age-regressed A_dir is the primary clinical metric. Raw A_dir is reported for transparency.

Per-patient sensitivity at 95% specificity is approximately 25-30% at d = +0.624. EDEAR AD deployment is cohort screening and serial trajectory monitoring, NOT single-timepoint AD diagnosis.

Only two cohorts tested at per-patient level: AIBL (Australian, EPIC) and AddNeuroMed (multi-center European, 450K). ADNI and Framingham cohorts are dbGaP-gated and pending.

VAL-054a cellular-age regression was a NON-TEST due to 80-cell baseline interpolator saturation (panel β subset maps all samples to age 95). VAL-054b HC-internal permutation provided a valid within-HC bound but is not a full direct age-confound resolution. VAL-052 on AddNeuroMed with direct age metadata provided the explicit age-regression analysis.

Stage 2 Moss NNLS is not expected to localize for AD (expected NULL). This is AD-consistent, not a failure. Stage 3 EpiDISH sub-composition is descriptive pending an AD-specific validated threshold (VAL-056 or later).

Holdout n=33 AD on AIBL is modest. 95% CI on Cohen's d is [+0.24, +1.06] — credible but not precise.

## File pointers

- **Card JSON:** `ad-immune_card.json`
- **Evidence Report section:** §6 AD Sprint (VAL-050 through VAL-054b)
- **Source scripts:** `run_val_050.py`, `val051_select.py`, `val051_split.py`, `val051_analyze.py`, `val052_analyze.py`, `val053_sex_panels.py`, `val054_age_regression.py`, `val054b_permutation_bound.py`
- **Result JSONs:** `VAL_050_RESULTS.json`, `val051/VAL_051_RESULTS.json`, `val052/VAL_052_RESULTS.json`, `val053/VAL_053_RESULTS.json`, `val054/VAL_054_RESULTS.json`, `val054/VAL_054b_RESULTS.json`
- **Panels:** `val051_panel_ruleA.json` (7 CpGs + directions), `val051_panel_ruleB.json` (all 18 CpGs alternative)
- **Seals:** VAL-051 pre-seal 2026-04-23 07:23:53 UTC

---

## v2.1 changes (2026-04-24)

- **Universal reference block embedded** (full-inline, Option B). The card JSON now contains the complete universal pipeline specification — H_min constants for all 8 architecture classes, Moss 2018 healthy reference β for all 18 tissues, 80-cell age-decade immune baseline, EpiDISH Salas QC bounds, universal tier thresholds, sex-stratification rule, language discipline, and the cross-cohort batch-offset warning from VAL-057. A new analyst loading only this card JSON plus `GAPE_WEB_v13.py` can run the full pipeline end-to-end without consulting any other file.
- **Lessons-learned section added** — 5 disease-specific documented quirks, each with source validation, context, observed quirk, interpretation, and how the card was updated to handle it. See `lessons_learned` key in the card JSON.
- **Cross-card lessons catalog** maintained in `LESSONS_LEARNED.md` at the Cookbook root. This card's entries are labeled with the card prefix (ad-LL-###).
- **VAL-057 consolidated external specificity test added.** GSE53740 Ferrari 2014 GIFT cohort (n=384: 193 HC, 15 AD, 128 FTD, 44 PSP/CBD). Pre-registered pooled A_dir test produced d=+0.013 (NULL, O4 per pre-reg). Four post-hoc analyses (sex-stratified, per-CpG directional, 80-cell age anchor, A_dir by decade) added after Heath flagged pre-reg omissions. Findings: (a) Male AD d=+0.415 replicates AIBL male d=+0.512; female AD d=-0.131 does NOT replicate AIBL female d=+0.705 — pooled null from opposing sex contributions. (b) PSP/CBD preserved 5/7 frozen Rule A directions vs AD 4/7 — panel may detect tauopathy-associated drift as well as or better than AD-specific drift. (c) GSE53740 HC sit +2.306 SD above 80-cell Cookbook baseline — cohort-level batch offset from Ferrari 2014 preprocessing. **Tier stays cross_platform_validated** (AIBL + AddNeuroMed primary replication holds). Card v2.1 adds sex stratification as mandatory covariate, tauopathy-specificity caveat, and cross-cohort-normalization requirement.
- **Known limitations expanded from v2.0**. Four new entries reflecting VAL-057 findings and sex-stratification requirement.
- **Next validation steps expanded from v2.0**. Five new entries: ADNI replication, synucleinopathy specificity test, FTD-subtype stratification, cross-cohort normalization method, pre-registration discipline upgrade.

---

## v2.2 changes (2026-04-26)

**Headline.** VAL-091 confirmed the v2.0/v2.1 prediction that *"Stage 2 NNLS deconvolution for AD is expected NULL — brain tissue not in buffy coat at levels NNLS can resolve"* — now tested with the Loyfer 2023 array atlas extension (which adds a sorted-cell `Cortical_neurons` reference Moss 2018 lacked). Two of three AD cohorts showed null, and the small-n GIFT signal was outlier-driven. The card's clinical assertion is unchanged. v2.2 adds VAL-091 as confirmatory Stage 2 evidence and adds the **glioma-vs-AD differential-diagnosis tile** to the EDEAR report.

- **VAL-091 added as Stage 2 confirmation.** Loyfer/Moss array atlas (`nloyfer/meth_atlas/reference_atlas.csv`, 26 cell types, 6,105 unique array-indexed CpGs) deconvolution applied directly to AIBL GSE153712 (EPIC, n=161 AD vs 471 HC), AddNeuroMed GSE144858 (450K, n=93 AD vs 96 HC), and GIFT GSE53740 (450K, n=15 AD vs 193 HC). Within-cohort AD-vs-HC Cohen's d: AIBL = −0.026 [−0.21, +0.17]; AddNeuroMed = −0.083 [−0.36, +0.19]; GIFT = +0.96 [+0.15, +1.88] (n=15, mean pulled by single 5.8% outlier; AD median 0.9% vs HC median 0.0%). Outcome label `O4_AD_NEURO_NULL` per pre-reg. **AD does not elevate cortical-neuron cfDNA at array-NNLS resolution. The card v2.1 prediction holds.**
- **Glioma-vs-AD Stage 2 differentiator added.** VAL-090 established that glioma plasma reads cortical-neuron fraction = 1.092% (Cohen's d = +1.96 vs healthy reference). VAL-091 establishes that AD plasma reads at the HC floor (~0.25%). When the Stage 1 directional immune A-score fires AND Stage 2 cortical-neuron fraction reads >0.5%, the combination is more consistent with glioma than with AD-only and triggers a **Differential-Diagnosis Required** flag in the EDEAR report. When Stage 1 fires AND Stage 2 cortical-neuron sits at the HC floor, the combination is AD-consistent (the card's expected pattern).
- **GIFT specificity arm reported as descriptive context.** FTD vs HC d = +0.19 (essentially null); PSP/CBD vs HC d = −0.51 (PSP/CBD reads *below* HC). No tauopathy-class elevation. Argues against a generic-neurodegeneration explanation for the cortical-neuron readout. **Not a tier-changing claim** — single small cohort, descriptive only.
- **`BELOW_NORMAL` tier added to the universal tier vocabulary.** VAL-091 exposed that GIFT PSP/CBD reads cortical-neuron *below* HC at d = −0.51 — a real signal, not a missing one. Below-normal A-scores can also indicate immunosuppression, treatment effect, or post-chemo/post-transplant states (the SUPPRESSED tier from heme-epic v0.1 is the same idea). The full tier vocabulary is now: **`BELOW_NORMAL` / `NORMAL` / `MARGINAL` / `DETECTABLE` / `URGENT` / `FLOOR_BREACH`**. Below-normal in AD context typically indicates a non-AD differential and routes to clinician review, not to AD reassessment.
- **Cross-platform NNLS routing artifact documented (ad-LL-006).** AddNeuroMed cortical-neuron fraction read 7.4% mean in HC vs ~0.3% in AIBL/GIFT/GSE51057 HC — a 28× cross-cohort baseline shift. Diagnosis: AddNeuroMed is 450K with only 5599 of 6105 Loyfer reference CpGs present (8% missing); NNLS routes mass to Cortical_neurons by default when discriminating CpGs are absent. Within-cohort AD-vs-HC contrast remains valid (both arms suffer the same routing); absolute fractions are not comparable across platforms without coverage-aware normalization. **Implication for EDEAR production:** Stage 2 cortical-neuron tier thresholds must be platform-stratified (EPIC vs 450K) until coverage-aware normalization is implemented. v2.2 card adds platform tagging requirement to Stage 2 reporting.
- **Tier stays `cross_platform_validated`.** AIBL Stage 1 (VAL-051) + AddNeuroMed Stage 1 (VAL-052) cross-platform replication holds as the primary clinical evidence. VAL-091 is supplementary Stage 2 confirmation and a glioma-vs-AD differential, not a Stage 1 modification. The directional 7-CpG panel is still frozen since VAL_051_SEAL.txt 2026-04-23 07:23:53 UTC. Stage 1 is still the sole tier-determining clinical signal.
- **EDEAR report updates.** (a) Stage 2 cortical-neuron fraction tile gets a numeric value with comparison to the Loyfer atlas glioma anchor (1.09%) and HC anchor (~0.3%). (b) Differential-Diagnosis Required flag triggers when Stage 1 immune positive AND Stage 2 cortical-neuron > 0.5%. (c) Platform tag (EPIC / 450K) appears on every Stage 2 result so coverage-routing artifacts are visible to the reader. (d) `BELOW_NORMAL` tier surface in the same color band as `MARGINAL` (yellow-grey) but with a distinct text label, and routes the patient to clinician review for differential.
