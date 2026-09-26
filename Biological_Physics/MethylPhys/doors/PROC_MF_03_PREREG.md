# PROC-MF-03 — pre-registration: the inverse-variance detector with a per-laboratory threshold, tested on a fifth laboratory at full marker resolution

**Written 2026-09-26, after PROC-MF-02 sealed and before the EPIC-Italy markers were extracted or any array
scored under these bars.** MF-02 met B1–B6 and failed B7 because (a) a threshold set on four laboratories was
applied to a fifth, and (b) the EPIC-Italy matrix used carried 783 of the detector's 1,506 markers. This procedure
fixes both in advance. Nothing moves after results are visible.

## In the author's terms

The detector already reads half a percent of a foreign cell in blood on four laboratories. What MF-02 showed is that
the *line* — the value above which the report says "detected" — belongs to each laboratory, the way each laboratory
already owns its zero. This procedure gives every laboratory its own line from its own healthy panel and asks whether
the reading still holds on a laboratory and platform the detector never saw.

## What changes from MF-02 — two things, both fixed now

1. **Per-laboratory threshold.** For laboratory L, the detection threshold for cell c is the 1 − 1/n quantile
   of f̂ on L's own healthy panel (n = panel size), i.e. ≤ 1 false positive per panel. For the four 450K
   laboratories, n = 12 (the same 48 arrays as MF-01/02); for EPIC-Italy, the panel is the **424 controls**
   (no cancer diagnosis in the series), so the line is their 99.76th percentile. The *weights* stay
   leave-one-laboratory-out from the 450K residuals; only the centre and the line are the laboratory's own.
2. **Full marker resolution.** The EPIC-Italy matrix is re-extracted from the raw series matrix on the author's
   disk (`GSE51032_series_matrix.txt.gz`, 845 arrays, 485,577 loci) at the deconvolver's **1,506 markers**. The
   extraction records how many of the 1,506 are present; if fewer than 1,350 (90 %) are, B7 is NOT ASSESSED
   rather than run on a different detector.

## Spikes and null

450K: as MF-01/02 — 48 healthy arrays, 768 real-array spikes (Breast, Colon_epithelial_cells, Cortical_neurons,
Prostate × 0.5, 1, 2, 5 %). EPIC-Italy: the 424 controls are the null; spikes are the same four cells × four
fractions into **each control array** (6,784 spikes), on the EPIC-Italy line.

## Bars

| bar | requirement |
|---|---|
| B1 | 450K detection limit with per-laboratory lines lower than NNLS on ≥ 3 of 4 cells (MF-02's result must survive the change of line) |
| B2 | honest σ on the 450K null: \|f̂/σ\| > 2 in 2–10 % |
| B3 | unbiased at 2 % and 5 %: median (f̂ − f) within ±0.005, both platforms |
| B4 | centred null on every laboratory including EPIC-Italy: \|median f̂\| ≤ 0.003 per cell |
| B5 | 450K same-laboratory weights no more than 20 % better than leave-one-out |
| B6 | blood composition unchanged: median max \|Δ\| < 0.005 |
| B7 | **EPIC-Italy false-positive rate on its own line ≤ 1/424 by construction — so the bar is the detection limit:** ≥ 90 % of spikes detected at **≤ 2 %** for all four cells on the 424-control line |
| B8 | **transfer of the 450K line is NOT assumed:** the 450K threshold applied to EPIC-Italy is reported, and if its false-positive rate exceeds 0.02 the outcome says so — this is recorded, not a pass/fail bar, because MF-02 already established it fails and the procedure is designed around that fact |

**Decision rule.** B1–B7 met → adopted as **Stage 2d, foreign-cell detection**, with the threshold a
laboratory-commissioned runtime quantity stored beside the laboratory zero (`detection_thresholds_v1.json`, one
entry per laboratory per cell, with n and the date), reported per foreign cell as (f̂, σ, detected yes/no) in the
bundle and on the Cells tab, and a red flag `FOREIGN_CELL_DETECTED` when any foreign cell clears its line. A
laboratory without a commissioned line reports "detection not commissioned for this laboratory" — never a
borrowed line. Any bar failing → not adopted; the failing bar and its number recorded.

## Evidence files (named before they exist)

In `kit/`: **PROC_MF_03.py**, **PROC_MF_03_extract.py**. In `kit/results/`: **PROC_MF_03.json**,
**PROC_MF_03_null.json**, **PROC_MF_03_extraction.json**. In `plates/`: **PROC_MF_03.png**. Linked from the outcome
once they exist.
