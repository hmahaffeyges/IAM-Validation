# PROC-DNMT-01 Part A — outcome (2026-10-01). Met-A on arrays under a known DNMT1 block. Scored as pre-registered (sha 2256dda819e6da7f)

**Data.** GSE135205 (Pappalardi et al. 2021), 51 EPIC arrays, 3 AML cell lines; raw IDATs through our Stage 1. Each line read against its own 4 vehicle (DMSO) arrays,
leave-one-out; canon site rule (≤ 3,000 sites per channel).

| | prediction | result | |
|---|---|---|---|
| P1 | vehicle arrays in Normal | **12/12** (0.968–1.048) | PASS |
| P2 | inactive analog GSK3510477, 10 µM, in Normal | **6/6** (1.002–1.032) | PASS |
| P3 | day-4 dose 0→2,000 nM, Spearman ρ ≥ 0.9 in each line | THP-1 1.00; MV4-11 0.77; NOMO-1 0.71 | FAIL (1/3) |
| P4 | 400 nM: day 6 > 4 > 2 > 1 in ≥ 2/3 lines | THP-1 only (1.45 → 1.74 → 1.79 → 1.82) | FAIL (1/3) |
| P5 | second active compound GSK3484862, 1 µM, day 4, above Normal | **3/3** (1.73–1.81) | PASS |
| P6 | rise carried by the methylated channel | methylated channel +1.56 (median), unmethylated +0.017 (ratio 0.01) | PASS |

**The size of the reading.** Vehicle and inactive analog sit at 1.00 ± 0.02. Every active-drug array at ≥ 80 nM reads 1.16–1.85, up to 40 times the
width of Normal. Up to 16 nM there is no change (0.997–1.028). The change is carried entirely by the methylated sites, as the physics says it must be:
blocked copying lets methylated sites lose their mark; unmethylated sites have nothing to lose.

**Why P3 and P4 failed: the reading saturates at the entropy ceiling.** The response is a step, then a plateau, not a slope.
- **Dose:** between 16 and 80 nM the methylated sites fall from β ≈ 0.94 to 0.57–0.89, and by 400 nM to 0.36–0.60. Per-site entropy peaks at β = 0.5 (1 bit),
  so A_meth stops at its ceiling, 1/H(floor) ≈ 2.8 (observed 2.66–2.85). Below 80 nM nothing moves, so the rank test is set by noise
  (MV4-11: 1.015 at 0 nM vs 1.001 at 3.2 nM). Past the ceiling, further loss lowers H again (MV4-11 at 2,000 nM, β 0.37, A 1.72 < 1.75).
- **Time:** by day 2 at 400 nM two lines are already at the ceiling (MV4-11 1.87, NOMO-1 1.77), so later days cannot rise.
- One array is unexplained: NOMO-1 at 10 µM keeps β 0.91 and reads 1.16 (the same line reads 1.81 at 400 nM).

**What this shows.** Met-A reads a known loss of copy fidelity, from the right channel only, with no signal from vehicle or the inactive look-alike.
Two compounds agree. The monotone dose and time predictions failed because the chosen doses jump past the entropy ceiling, not because the reading
went the wrong way. Next: a dose series in the range where β is above 0.5 (here, 16–80 nM), where H is monotone in β, to test the slope.
Noise index N on these arrays 0.163–0.304 (above the blood floor range; within-line comparison removes it).

**Limits.** Cancer cell lines, one array per condition, one lab.
