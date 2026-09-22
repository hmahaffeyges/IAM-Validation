# VAL-068 Pre-Registration — pancreatic-epic Multi-Substrate Comparison

**Sealed:** 2026-04-25 UTC
**Card:** pancreatic-epic
**Card version target:** v0.1 (third tissue anchor alongside VAL-066 TCGA-PAAD n=5 paired and VAL-067 GSE49149 n=196 case-control)
**Cohort:** GSE74071 (Tjensvoll et al.) — 14 PDAC tumor + 7 adjacent normal + 4 pancreatic juice circulating cancer cells + 3 cancer-associated fibroblasts + 1 primary culture cancer cells = 28 samples on Illumina HM450
**Platform:** GPL13534 (HM450)

## Background

VAL-068 brings two unique elements to pancreatic-epic v0.1: (1) **pancreatic juice as a non-blood, non-tissue specimen** with potential clinical-deployment implications (analogous to how VAL-065 attempted urine for prostate-epic), and (2) **cancer-associated fibroblasts (CAFs) as a stromal-class comparator** to the secretory-class tumor signal. PDAC has the densest fibroblast/stromal compartment of any common cancer (Hosein 2020, Ohlund 2014); separating tumor secretory signal from CAF stromal signal at the per-substrate Xu-538 level is informative for understanding what the Stage 2 deconvolution actually sees.

## Pre-registered hypotheses

**H1 (paired tumor vs adjacent-normal, n=7 patients with both):** Same secretory-class A-score elevation prediction as VAL-066. Paired Cohen's d > 0.

**H2 (pancreatic juice circulating cancer cells vs adjacent-normal, unpaired n=4 vs n=7):** Pancreatic juice cancer cells should show A-score elevation comparable to or greater than tumor tissue, because juice cells are tumor-derived but enriched (less stromal admixture).

**H3 (CAFs vs adjacent-normal, unpaired n=3 vs n=7):** Cancer-associated fibroblasts should be scored against H_min(stromal) = 0.862950, NOT H_min(secretory). Tested as supplementary stratum to verify the class-assignment matters and that CAFs do not confound a secretory-class scoring on bulk tumor tissue.

**H4 (CCL-027 mandatory bidirectional check):** Per-CpG direction split for tumor vs adjacent-normal arm; document whether VAL-068 replicates the VAL-066 mixed-direction pattern (46.9% positive) or the VAL-067 50/50 split or some third pattern.

## Outcome thresholds

- O1: TUMOR_VALIDATED if H1 paired d > +0.3 with lower CI > 0
- O2: WEAKER if H1 paired d in [0, +0.3]
- O3: NULL if H1 CI straddles zero
- O4: INVERTED if H1 paired d < 0 with upper CI < 0
- O5: UNEXPECTED otherwise — convene with Heath
- Bonus deliverable regardless of outcome: pancreatic juice substrate signal (H2) and CAF stromal-class signal (H3) reported as exploratory supplementary.

## Methods

- Same H_min(secretory) = 0.843264 for tumor + adjacent-normal + juice cells, plus H_min(stromal) = 0.862950 for CAF supplementary.
- Same Xu-538 panel SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`.
- Same QC ≥ 400 valid Xu-538 CpGs per sample.
- RNG seed 20260425.

## Patient pairing for H1

The 7 adjacent-normal samples have the following patient mapping based on sample title prefix matching:
- PH64A (tumor) ↔ PH64B (normal) — paired
- PH67A (tumor) ↔ PH67B (normal) — paired
- PH70B (normal) — no matching tumor in series
- 314001-314005 are juice cells (no normal pairing)
- 314006-314008 are CAFs (no normal pairing)
- 314009/314010, 314011/314012 — paired tumor/normal
- GEMM samples — pair by adjacency (16↔15? 18↔17? 22↔21?) — confirmed at runtime by sequential numbering pattern

If pairing cannot be confirmed for some, use unpaired d as the primary statistic for the tumor-vs-normal arm.

## Reproducibility anchors

- Pre-registration SHA-256: (computed at seal)
- Cohort SHA: (computed from sorted GSM list at seal)
- Xu-538 panel SHA: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
- β matrix SHA: (computed at run)

## Deliverables

1. `val068_pancreatic_epic_gse74071.py` — reproducible Python 3 stdlib script
2. `VAL-068_prereg.md` — this document
3. `VAL-068_outcome.md` — outcome doc
4. `VAL-068_results.json` — primary + multi-substrate
5. `GSE74071_manifest.json` — sample-to-substrate map

GitHub destination: `Biological_Physics/validation_runs/`
