# PROC-NEUT-TEST-01 T2 — outcome (2026-10-01): FAIL as pre-registered. Cause found: array noise in the second lab.

**Test.** Isolated neutrophils from two men, another lab (GSE247193: 30 y; GSE247195: 54 y), 8 times of day × 3 arrays. Read by chain v3
(bc4a651), unchanged, against the frozen EPIC neutrophil floor with no tare, as the chain's rule then stated. 33 of 48 arrays were read;
15 failed to download (GEO throttling). They are re-read in the diagnostic below.

| prediction | result |
|---|---|
| T2a ≥ 80 % in Normal | **2/33 — FAIL** (A 0.86–1.26) |
| T2b within-person SD ≤ 0.010 | **FAIL** — 30 y: mean 1.193, SD 0.041; 54 y: mean 1.057, SD 0.051 |
| T2c replicate median range ≤ 0.010 | **FAIL** — 0.041 and 0.025 |
| time of day (descriptive) | no consistent pattern above the replicate noise |

**Diagnosis (development, after looking; all 48 arrays plus 6 Salas, through the chain's own Stage 1/A/M).**
- **Purity is not the cause.** The NNLS neutrophil fraction is 0.95–1.00, median 1.00. The purity-matched reading equals the own-floor reading.
- **Array noise is the cause.** Median |β − neutrophil profile| at the 6,000 sites: Salas 0.002, the 54-year-old's arrays 0.007, the 30-year-old's arrays 0.018.
  The 30-year-old's arrays also have lower call rates (0.956–0.983), and in that set lower call rate goes with higher A (rho −0.61). The identity sites sit near β = 0 or 1,
  where H is steepest, so array noise raises mean H. The gauge amplifies instrument noise exactly where it reads.
- The two men's difference (1.19 vs 1.06) follows their arrays' noise, not their age.

**Consequence for the chain (to apply after this pre-registered battery is scored; chain unchanged until then).**
1. **Isolated cells must also be tared** against same-run healthy references. The "own floor, no tare" rule holds only for arrays as clean as the floor's.
2. **Instrument-noise gate at intake**: an array-level noise index read from sites invariant across all blood cells. Above the floor's own
   noise, the reading is withheld or must be tared. The threshold is to be set on the floor arrays and tested on independent arrays.
3. A same-person time series in a clean lab is still the right test of within-person stability. It is to be repeated on a lab whose noise index passes.
