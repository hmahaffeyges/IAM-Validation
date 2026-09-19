# VAL-12X+3 (provisional ID) — Crohn's-Disease Blood Methylation Baseline Exploratory VAL

**Prereg version:** v1.0-DRAFT
**Date drafted:** 2026-05-02
**Card:** gastric+esophageal-epic v0.1 sprint (per Heath Q3 sign-off: Crohn's pathway documented in BOTH gastric-epic and hcc-epic v0.3.1 amendment)
**Prereg type:** Phase C exploratory disease cohort scoring (run-everything regime applied to non-target disease)
**Depends on:** VAL-12X (Boccellato calibration) sealed first

**Status:** This is a **subordinate exploratory VAL**, not a primary card validation. The findings inform the Crohn's-pathway language to add to gastric-epic and hcc-epic v0.3.1 known_limitations — not a Crohn's-card claim.

---

## 1. VAL identification + scope

- Provisional VAL ID: **VAL-12X+3**
- Cohorts: GSE87650 (Adams 2014 IBD blood, n=240) + GSE99788 (Ventham 2016 IBD blood, n=549) — both publicly accessible Tier 1
- Comparison strategy: **Welch CD vs UC vs HC** in each cohort separately, with cross-cohort triangulation
- Not a Crohn's-with-HCC validation (no such cohort exists at 450K/EPIC platform — documented as gap)
- Document goal: establish **what long-standing CD blood looks like at run-everything atlas resolution**, so future patient-level CD interpretations can reference a baseline

## 2. Hypothesis (pre-locked, EXPLORATORY)

**Stage 1 hypothesis (Xu-538):** Crohn's disease blood shows architectural drift signature distinct from healthy. Direction not pre-locked (BIDIRECTIONAL): chronic systemic inflammation could elevate (per CCL-021 4% cfDNA detection floor + immune-class drift) OR depress (per heme-LL-003 SUPPRESSED tier in chronically ill cohorts) the A_immune signal. |d_unpaired CD vs HC| ≥ 0.3 pre-registered as positive finding.

**Stage 2 hypothesis (cell-of-origin tile pattern in CD blood):** under run-everything regime, run all atlases. CD blood is whole-blood substrate, so tile readouts reflect circulating-cell methylation only. Predicted patterns:
- BoccellatoStomachRef tiles read NULL on CD blood (gastric tissue does not shed at meaningful fraction into peripheral blood except in advanced GI-tract disease)
- Loyfer Upper_GI / Colon_epithelial_cells / Hepatocytes tiles: **POSSIBLE elevation** if CD intestinal involvement causes detectable enteric-cell shedding into circulation
- Caggiano TIM tiles: signature of chronic inflammation expected — TIM cells (T-cell exhaustion, regulatory T-cell expansion) elevation in CD vs HC

**Stage 3 hypothesis (immune sub-composition in CD blood):** CD-vs-HC differences in lymphoid/myeloid balance, neutrophil expansion, naïve/memory T-cell ratios. Magnitude direction not pre-locked.

**Bidirectionality declaration:** Per Heath's reminder, all outcome thresholds use magnitude-based |d| with explicit direction labels.

**The Crohn's-blood-baseline finding is the deliverable:** sealed per-tile A-score distributions for CD vs UC vs HC across run-everything stack. Future patient-level interpretations reference this baseline.

## 3. Pre-locked decision criteria (CHK-2.1)

### Per cohort, per stage outcome
- **CD vs HC primary contrast** (Welch d, magnitude-based |d| with direction label)
- **CD vs UC differential contrast** (does CD differ from UC? is the IBD signature substantially shared or substantially distinct?)

### Multi-disease detection patterns (run-everything mandate)
1. **CD-with-tissue-leakage** — Stage 2 Upper_GI / Colon_epithelial_cells tile elevation in CD subset
2. **CD-systemic-inflammation** — Caggiano TIM signature + Stage 1 Xu-538 directional shift
3. **CD-vs-UC differentiation** — does the run-everything stack distinguish CD from UC at all? If yes, this is a CD-specific pattern; if no, the IBD signature is generic-inflammation
4. **Inflammaging baseline shift** — CD blood mean A_immune compared to age-matched HC; tests whether long-duration CD produces accelerated inflammaging signature (relevant to Crohn's→HCC pathway)

## 4. Pre-locked stratifications

| Stratum | Source | Note |
|---------|--------|------|
| CD vs UC vs HC | Cohort-provided diagnosis | Primary stratification |
| Disease duration (CD subset) | Cohort metadata if available | Long-duration CD is the Marcus-pathway-analog |
| Treatment status (azathioprine, anti-TNF) | Cohort metadata if available | Azathioprine is the implicated treatment in CD-HCC literature |
| Sex | Cohort metadata | Per CCL-002 |
| Age | Cohort metadata | Required for inflammaging analysis |

## 5. CHK-3.1A and CHK-3.1B substrate gates

GSE87650 + GSE99788 are HM450 platform, processing pipeline cohort-specific. CHK-3.1A may apply within-cohort self-cal envelope rather than TCGA-substrate-floor (per VAL-108/109/110 cardio precedent).

**Pre-flight check required:** verify substrate-class for each cohort. If raw-IDAT or sesame-pipeline output, apply VAL-106 substrate floor; if minfi-preprocessFunnorm or GenomeStudio AVG_Beta, apply within-cohort self-cal substrate floor.

## 6. CHK-2.17 cohort-substrate-coverage pre-flight

Same gate as STAD/ESCA — sample 5-10 random samples per cohort, verify Xu-538 + Boccellato + Loyfer + Caggiano + Salas + UniLIFE coverage at ≥90% mean / ≥80% q5. Halt seal if any atlas fails.

## 7. CHK-3.2 cross-cohort baseline check

- Within-cohort: HC mean A-score vs anchor (GSE51057 / Hannum 80-cell baseline)
- Cross-cohort: GSE87650 HC vs GSE99788 HC vs GSE51057 HC (three EU IBD/healthy cohorts at HM450; expected reasonably-aligned baselines based on common Illumina pipeline; flag mismatches in anchor-SD units)

## 8. CCL-025 application (cross-card observation)

CD chronic systemic inflammation is a chronic-driver pattern analogous to HBV/HCV in liver, smoking in lung, H. pylori in stomach. **Therefore:**
- The hcc-epic v0.3.1 amendment adds CD as a chronic-driver risk factor for HCC paired-tissue-arm BLUNTING (predicted analogous to viral hepatitis subgroup) but ccfDNA POSITIVE detection (predicted analogous to VAL-059 GSE298812)
- The gastric+esophageal-epic v0.1 known_limitations add CD-IBD comorbidity as a stratification need for any future gastric-card cohort

This VAL produces the empirical baseline to compare against when (in future) a CD-with-HCC or CD-with-gastric-cancer cohort becomes available through biobank acquisition.

## 9. CHK-7.6 reproducibility triple

- **Source code:** `val12X+3_crohns_blood_baseline.py` (~200 lines, multi-cohort with stratification)
- **Inputs:** GSE87650 + GSE99788 series matrices + clinical metadata (publicly accessible via GEO FTP); 6 atlas matrices SHA-sealed
- **Environment:** Python 3.x standard scientific stack
- **Expected output:** `val12X+3_crohns_blood_baseline_results.json`

---

## Awaiting sequence

1. ⏳ VAL-12X (Boccellato calibration) sealed first
2. ⏳ This prereg sealed alongside VAL-12X+1 STAD + VAL-12X+2 ESCA
3. ⏳ Heath sign-off on (a) outcome thresholds, (b) the "this is exploratory not primary" framing, (c) the CCL-025 prediction language for the v0.3.1 hcc-epic amendment

## Honest scope-check

This VAL does NOT validate that EDEAR can detect CD-driven HCC. It documents what CD blood looks like at run-everything atlas resolution. The CD-HCC pathway prediction (analogous to viral-hepatitis subgroup paired-d blunting + ccfDNA positive detection) is empirically untested at the cohort level — the cohort doesn't exist publicly.

What this VAL DOES contribute:
1. Establishes Crohn's-aware blood baseline as a Type 2 calibration artifact (per atlas calibration typology doc)
2. Provides the third confirming data point for CCL-025 if smoking-stratified ESCC + H. pylori-stratified STAD also confirm
3. Documents the CD methylation signature so future patient-level interpretations have a reference

This VAL does NOT contribute to gastric-epic v0.1 OR hcc-epic v0.3.1's clinical claims. The outcome of this VAL goes into a "Crohn's-aware blood baseline" annex, not a "CD detects HCC" claim.
