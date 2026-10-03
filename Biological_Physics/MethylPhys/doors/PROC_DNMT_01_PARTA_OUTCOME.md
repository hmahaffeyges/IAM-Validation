# DNMT-01 Part A — Met-A under a known DNMT1 block (development measurement, 2026-10-01)

**Why.** A known cause with a known direction: block the enzyme that copies methylation and see whether Met-A reads it, from the right channel,
and not when an inactive look-alike is given.

**Data.** GSE135205 (Pappalardi et al. 2021), 51 EPIC arrays, 3 AML cell lines; raw IDATs through our Stage 1. Each line read against its own 4 vehicle
(DMSO) arrays, leave-one-out; canon site rule (≤ 3,000 sites per channel).

**What we measured.**
- Vehicle arrays: 0.968–1.048 (12 arrays). Inactive analog GSK3510477, 10 µM: 1.002–1.032 (6 arrays).
- Active drug GSK3685032 at ≥ 80 nM, day-4 dose series: 1.16–1.85, up to 40 times the width of Normal (all ≥ 80 nM arrays, time series included: 1.16–1.87; data/dnmt_arrays_readings.csv). At 3.2–16 nM: 1.001–1.028, no change.
- Second active compound GSK3484862, 1 µM, day 4: 1.73–1.81 in all 3 lines (days 2 and 4 together: 1.60–1.85).
- Channel: methylated sites carry the change (median A_meth − 1 = +1.56); unmethylated sites +0.017. Blocked copying lets methylated sites lose their mark;
  unmethylated sites have nothing to lose.

**What we learned about the gauge: it saturates at the entropy ceiling.**
- Between 16 and 80 nM the methylated sites fall from β ≈ 0.94 to 0.57–0.89, and by 400 nM to 0.36–0.60. Per-site entropy peaks at β = 0.5 (1 bit),
  so A_meth stops at its ceiling, 1/H(floor) ≈ 2.8 (observed 2.66–2.85). Past β = 0.5, further loss lowers H again (MV4-11 at 2,000 nM: β 0.37, A 1.72).
- Over time at 400 nM, two lines are already at the ceiling by day 2 (MV4-11 1.87, NOMO-1 1.77); THP-1 climbs 1.45 → 1.74 → 1.79 → 1.82 over days 1–6.
- So Met-A is monotone only while the methylated sites stay above β = 0.5. A reading near the ceiling needs the mean β printed beside it, or a change
  well past half-loss can read lower than a smaller one.
- One array is unexplained: NOMO-1 at 10 µM keeps β 0.91 and reads 1.16 (the same line reads 1.81 at 400 nM).

**Chain changes this points to.**
1. Report the methylated-site mean β next to Met-A, and flag "past the entropy ceiling" when it is below 0.5.
2. To see the slope: a dose series in the range where β stays above 0.5 (here 16–80 nM).

Noise index N on these arrays 0.163–0.304; reading each line against its own vehicle arrays removes it. Cancer cell lines, one array per condition, one lab.
