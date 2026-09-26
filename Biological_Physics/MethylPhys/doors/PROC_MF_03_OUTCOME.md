# PROC-MF-03 — outcome: NOT COMMISSIONED. B1–B6 met again; B7 failed on the fifth laboratory — its healthy null is 4–25× wider than the four (by cell and laboratory), and its Breast and Prostate lines were set by two controls elevated on both cells at once.

**Sealed 2026-09-26** against [`PROC_MF_03_PREREG.md`](PROC_MF_03_PREREG.md), fixed before the EPIC-Italy markers were
extracted. Four 450K laboratories as before (48 arrays, 768 spikes); GSE51032 re-extracted from the raw series matrix
on the author's disk at **all 1,506 of the detector's markers** (median 3,887 of 4,444 deconvolver markers present per
array), 424 controls, 6,784 spikes, scale-mapped through `GSE51032_450K` exactly as PROC-EPIC-01 did.
Evidence: [`PROC_MF_03.json`](../kit/results/PROC_MF_03.json) · [`PROC_MF_03_null.json`](../kit/results/PROC_MF_03_null.json)
· [`PROC_MF_03_extraction.json`](../kit/results/PROC_MF_03_extraction.json) · [`PROC_MF_03_null_mad_by_lab.json`](../kit/results/PROC_MF_03_null_mad_by_lab.json)
· [`PROC_MF_03.py`](../kit/PROC_MF_03.py) · [`PROC_MF_03_extract.py`](../kit/PROC_MF_03_extract.py) · [`PROC_MF_03.png`](../plates/PROC_MF_03.png)

**A correction first.** MF-02's documents call GSE51032 "EPIC" and "a fifth platform". It is a **450K array**
(GPL13534); "EPIC-Italy" is the cohort's name. It is a fifth *laboratory* on the same platform, with author-processed
betas rather than our stage-1 calibration — which, in the chain's terms, is what makes it a different laboratory.

## Detection limit on each laboratory's own line (≥ 90 % of spikes detected)

| cell | 450K NNLS | 450K inverse-variance | GSE51032 inverse-variance |
|---|---|---|---|
| Breast | 5 % | **0.5 %** | > 5 % |
| Colon epithelial | 2 % | **0.5 %** | 5 % |
| Cortical neurons | 5 % | **1 %** | > 5 % |
| Prostate | 5 % | **1 %** | > 5 % |

## Bars

| bar | result |
|---|---|
| B1 450K limit lower than NNLS, ≥ 3 of 4 | **MET** — 4 of 4, on per-laboratory lines |
| B2 honest σ | **MET** — 3.1 % of null beyond 2σ (MF-02's sealed σ; a per-laboratory-centred σ my first draft used gives 13.5 % and is reported, not used) |
| B3 unbiased, both cohorts | **MET** — 450K +0.0004 / +0.0008; GSE51032 −0.0003 / −0.0008 |
| B4 centred null on all five laboratories | **MET** — worst \|median\| 0.0000 |
| B5 same-laboratory weights ≤ 20 % better | **MET** |
| B6 blood composition unchanged | **MET** — median Δ 0.0001 |
| B7 GSE51032 limit ≤ 2 % on its own line, all four cells | **FAILED** — one cell at 5 %, three beyond 5 % |
| B8 (recorded) the 450K line applied to GSE51032 | false positives 43–46 % on every cell, as MF-02 predicted |

## What the fifth laboratory shows — measured

1. **Its healthy null is 4–25× wider than the four, at full marker resolution.** MAD of f̂ for Breast: the
   four laboratories 0.0010–0.0014, GSE51032 **0.0084** (6–8×); colon 0.0002–0.0006 vs **0.0043** (7–22×); neurons
   0.0004–0.0014 vs **0.0099** (7–25×); prostate 0.0013–0.0030 vs **0.0132** (4–10×). Against the median of the four:
   6.7×, 10.0×, 8.5×, 6.3×. This is the bulk of the distribution, not a tail.
2. **This overturns MF-02's diagnosis.** MF-02 attributed most of its B7 failure to the reduced marker set, on the
   evidence that restricting the *450K* null to 783 markers widened it 7×. That was a measurement on the 450K null,
   and it did not transfer: restoring all 1,506 markers narrowed the GSE51032 null only 1.6× (0.0138 → 0.0084 for
   Breast). The laboratory, not the marker set, is the dominant factor. MF-02's outcome is corrected to say so.
3. **The Breast and Prostate lines were set by two arrays elevated on both cells at once.** GSM1236355 and GSM1236248
   read **0.20 / 0.18 on Breast and 0.29 / 0.27 on Prostate**; on neurons they read only 0.07 / 0.06 and on colon they
   are not elevated. The neuron line was set by a different array, GSM1235872 (0.16 on neurons, 0.06–0.07 on Breast and
   Prostate). Two solid-tissue cells rising together in a control is consistent with the substrate-mismatch pattern of
   PROC-TISSUE-01 and DISC-BLADDER-003, but it is **not** the every-class signature that document describes — the
   honest statement is that these arrays are not blood-like *on the solid-tissue columns*, and that a specificity rule
   (one cell rising more than the others) would have flagged them. The pre-registered line rule, the 1 − 1/n quantile,
   is on 424 arrays the second-highest value, and these arrays are it. Without them the Breast p99 is 0.030; the bulk
   width alone would still put the Breast limit near 5 %, so removing them would not have passed B7 — the bar fails on
   the width, not only on the outliers.
4. **Why the width.** The weights (1/variance per locus) were estimated on the four laboratories' stage-1-calibrated
   residuals. GSE51032's betas are author-processed and reach the chain through a scale map; its residual structure
   is different, and weights that are optimal for the four are not optimal for it. The detector is per-laboratory in
   its *line* here; it is not yet per-laboratory in its *weights*.

## Decision, by the pre-registered rule

*Any bar failing → not adopted.* Nothing enters the chain from MF-03. Three procedures have now established, on
four laboratories, that inverse-variance detection reads 0.5–1 % of a foreign cell where the chain reads 2–5 %, with an
honest σ and no bias — and that this does not yet extend to a laboratory whose data reaches the chain by a different
route.

**What a PROC-MF-04 would have to do, if the author commissions it** — three things, all following from measurements
above rather than from preference:

- **Per-laboratory weights as well as line**, estimated on the laboratory's own commissioning panel (GSE51032 has
  424 controls; 36 sufficed on 450K).
- **The chain's own gates ahead of the detector.** A control that reads 0.2 on every foreign cell should never
  reach a detection line; the composition guard and a "rises more than the other classes" specificity rule
  (PROC-BRAIN-01 B5) exist for exactly this, and the panel that sets the line must be the panel that passes them.
- **A line rule that is not the maximum.** 1 − 1/n on a large panel hands the line to the single worst array; a
  stated quantile (p99) with a minimum panel size is the same false-positive intent without that fragility. This
  is a design error in my pre-registration and is recorded as one; the bar was not moved.

## In the author's terms

The tool is sensitive — half a percent, proven three times on four laboratories. What it does not yet have is the
ability to be *commissioned* at a new laboratory from that laboratory's own panel: the line transfers by measurement
now, the weights do not yet, and the panel must be gated before it sets anything. That is the same shape as the
laboratory zero, one level up.

**Corrected 2026-09-26 after audit:** the first sealed text said the two arrays read 0.18–0.29 on "Breast, Prostate and neurons" and that the null was "4–11×" wider. The per-cell readings in [`PROC_MF_03_null.json`](../kit/results/PROC_MF_03_null.json) show neurons at 0.06–0.07 for those arrays and colon not elevated, and the width ratios run to 25× on colon and neurons. Both statements are corrected above; the verdict does not change.
