# PROC-UNMIX-01 — outcome: NOT ADOPTED. The dilution-line inversion is exact arithmetic the composition solver cannot feed.

**Sealed 2026-09-27** against the bars in [`PROC_UNMIX_01_PREREG.md`](PROC_UNMIX_01_PREREG.md) (clarification to B6 dated
2026-09-27, before any array was scored). Script: kit/PROC_UNMIX_01.py (archived privately); results kit/results/PROC_UNMIX_01.json (archived privately).
Three constructed mixtures (typical, neutrophil-heavy, lymphocyte-heavy) from exact atlas means, plus 48 whole-blood arrays
(12 per commissioning laboratory) through the chain's own pipeline map and composition solver.

| bar | rule | measured | |
|---|---|---|---|
| B1 | every present cell NORMAL after re-zero + inversion | present cells read 0.67–0.93 | **FAILED** |
| B2 | max \|A − 1.00\| ≤ 0.015 on the constructed mixtures | 0.328 | **FAILED** |
| B3 | majority cells (neutrophils, monocytes) move ≤ 0.005 | 0.276 | **FAILED** |
| B4 | a planted CD4 at own A = 1.06 reads ELEVATED, others NORMAL | reads 0.856; others not NORMAL | **FAILED** |
| B5 | p90−p10 spread not wider on ≥ 4 of 5 blood cells, 48 arrays | wider on 5 of 5 (0.05–0.08 → 0.22–0.40) | **FAILED** |
| B6 | inversion is the identity at f = 1 | \|Δ\| = 0 | MET |

## Diagnostic written after the bars (not a bar)

Unmixing the same constructed mixtures with the **true** fractions instead of the solver's returns every cell to
A = 0.9998 (= 1.000 within the re-zero). So the inversion `own_β = (β − Σ_{c≠cell} f_c μ_c) / f_cell` is exact; what fails is
its input. The solver's fractions carry ±0.03 per cell and 3.5–4.5 % of mass outside the true mixture (measured
2026-09-26 on the same constructions); dividing by f_cell turns a 0.03 error on a 0.07 cell into a ~40 % error on its
own β. The unmixed A is therefore a reading of the solver's error, not of the cell.

## Step (1), the re-zero, on its own

Trimming each cell's identity loci until its own atlas profile reads exactly H_min moves the f = 1 readings from
0.9362–1.0204 (median 0.9904) to 0.9896–1.0369, keeping a median 240 of 264 loci. It was scored only as part of the
package and is **not adopted** here. Whether the identity-locus set should be re-zeroed so a cell's own profile reads
1.000 by construction is a separate question for a separate pre-registration; recorded, not started.

## What this decides

The fraction confound on minority cells (FRACTION_AND_A.md; NK at 5–10 % reading 0.946, CD8 0.961, CD4 1.031 on a
perfect specimen) is real and is **not removed by inverting the dilution line** with the present composition solver.
Fraction stays a detection gate and is not applied to A. Routes recorded, not started: (i) a solver whose fractions are
precise to < 0.005 on minority blood cells — the atlas-v2 covariance (PLAN 27) is the first candidate; (ii) scoring a
minority cell only on loci where the other present cells are near-constant, so the inversion's denominator matters less.
