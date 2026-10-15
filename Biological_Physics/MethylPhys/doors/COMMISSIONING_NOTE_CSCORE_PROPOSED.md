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
| same person measured twice (GSE147740, 13 pairs) | within-person SD <= 0.125 | 0.038 |
| clustered loss planted in real healthy arrays (step 2) | caught on >= 95 % of placements | 100 % |

**Scope.** EPIC v1 arrays; healthy whole blood (the band's own population, 555 of 617 arrays) and purified neutrophils. The band:
upper limit 1.152 (blocks of 10), one-sided: above is a departure; below is "less clustered than healthy" and is not a departure.
Detection limit: C_rel 1.5 at 5 percent of sites clustered.

**Not yet shown.** Real cancers read by the C-score with a sealed bar; 450K and EPIC v2 arrays; plasma. Those are the next tests,
and none of them is a condition of this proposal.
