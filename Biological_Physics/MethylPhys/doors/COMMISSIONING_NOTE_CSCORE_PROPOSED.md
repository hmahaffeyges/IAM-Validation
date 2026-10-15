# The clustering meter (C-score): proposed for commissioning, 2026-10-15

**Status: proposed, waiting for the author's sign-off.** Not a diagnostic test.

**What it reads.** Whether a sample's departures from healthy come in clumps (whole stretches of DNA moved together, the pattern a
tumor leaves) or scattered (the pattern of ordinary aging and wear). It rides on Met-A's chain: Stage 1 calibration, then the C-score,
then the same-run tare against healthy arrays on the same slide.

**What it has passed** (every bar fixed before its data were read; record `CSCORE_COMMISSIONING_PLAN.md`):

| check | bar | result |
|---|---|---|
| healthy band, laboratory 1 (GSE315366) | >= 95 % not above 1.152 | 34 of 35 |
| healthy band, laboratory 2 (GSE184269) | >= 95 % not above 1.152 | 22 of 23 |
| healthy band, laboratory 3 (GSE147740) | >= 95 % not above 1.152 | 172 of 173 |
| same person measured twice (GSE147740, 13 pairs) | within-person SD <= 0.125 | 0.0379 |
| clustered loss planted in real healthy arrays (step 2) | caught on >= 95 % of placements | 100 % |

**Reference sets:** the band was set on 617 healthy arrays from 19 series: GSE110530, GSE112618, GSE118144, GSE122244, GSE123914, GSE141682, GSE142512, GSE155952, GSE161678, GSE166503, GSE200376, GSE222927, GSE225544, GSE235717, GSE247193, GSE247195, GSE250556, GSE276323, GSE286313 (`data/DEV_CSCORE_TARE_01_arrays.csv`); each array is tared against healthy arrays on its own slide (at least 3), else its own series. None of the three test laboratories helped set the band.

**Code and outputs:** `data/CSCORE_COMMISSIONING/read_repeat_test.py` -> `data/CSCORE_COMMISSIONING/repeat_GSE147740.json` and
`repeat_GSE147740_rows.csv`; `data/CSCORE_COMMISSIONING/read_band_test.py` -> `data/CSCORE_COMMISSIONING/band_test_GSE184269.json`;
`data/CSCORE_COMMISSIONING/band_test_GSE315366.json`; `data/CSCORE_COMMISSIONING/plan_numbers.py` -> `plan_numbers_output.txt`.

**Scope.** EPIC v1 arrays; healthy whole blood (the band's own population, 555 of 617 arrays) and purified neutrophils. The band:
upper limit 1.152 (blocks of 10), one-sided: above is a departure; below is "less clustered than healthy" and is not a departure.
Detection limit: C_rel 1.5 at 5 percent of sites clustered.

**Not yet shown.** Real cancers read by the C-score with a sealed bar; 450K and EPIC v2 arrays; plasma. Those are the next tests,
and none of them is a condition of this proposal.
