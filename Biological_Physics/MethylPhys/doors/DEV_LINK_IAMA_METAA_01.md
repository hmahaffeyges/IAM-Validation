# DEV-LINK-IAMA-METAA-01 — what IAM-A's copy error predicts for Met-A (written 2026-10-09, before any data are read)

**Status labels.** The relation below is DERIVED (algebra from the two definitions) under one stated ASSUMPTION; the numbers are
CALCULATED from frozen inputs; nothing is fitted. The tests are open.

## Definitions (both frozen in the chain)
- IAM-A = H(ε) ÷ (P · H(ε₀)), ε = isolated copy errors per opportunity on methylated molecules; H(ε₀) = 0.2043 bits (thermal floor);
  healthy neutrophils read 1 at ε_h, where H(ε_h) = P · H(ε₀) (ε_h ≈ 0.0385, P = 1.1492).
- Met-A = mean over 6,000 identity sites of H(β_i) ÷ the healthy mean (floor 0.33026), H(b) = −b log₂ b − (1−b) log₂(1−b);
  3,000 methylated sites (healthy β 0.75–0.95) and 3,000 unmethylated (0.05–0.25), healthy β_i = μ_i (`metA_floors_v1_3.json`).

## Derivation
A cell copying less faithfully than healthy has ε = ε_h + δ(1 − ε_h): each CpG on a methylated molecule that healthy maintenance keeps
methylated is lost with extra probability δ. At a methylated identity site the fraction of methylated molecules becomes β′ = μ(1 − δ).
**Assumption A (loss only):** IAM-A measures only loss on methylated molecules, so unmethylated sites are taken unchanged (β′ = μ).
**Bound B (symmetric):** unmethylated sites gain at the same rate, β′ = μ + (1 − μ)δ — an upper bound on the Met-A shift.
Then **Met-A_pred(δ) = mean H(β′_i) ÷ mean H(μ_i)** and δ follows from the measured IAM-A by inverting H(ε) = IAM-A · H(ε_h).

## Calculated (frozen identity sites and the 6 reference arrays)
| IAM-A | ε | δ | Met-A predicted, A (loss only) | Met-A predicted, B (symmetric) |
|---|---|---|---|---|
| 1.00 | 0.03850 | 0 | 1.000 | 1.000 |
| 1.02 | 0.03952 | 0.0011 | 1.006 | 1.012 |
| 1.05 | 0.04106 | 0.0027 | 1.015 | 1.030 |
| 1.10 | 0.04368 | 0.0054 | 1.030 | 1.060 |
| 1.16 | 0.04689 | 0.0087 | 1.049 | 1.096 |

\calculated Met-A moves 0.30 (A) to 0.60 (B) as much as IAM-A. A cell can leave IAM-A's Normal band (1.05) and stay inside Met-A's.

## Predictions and how each can fail (falsifiable)
1. **Forward:** for cells measured both ways, measured Met-A lies between curve A and curve B at the cell's measured IAM-A (within
   Met-A's stated precision). Below curve A: Met-A is not seeing the copy error IAM-A measures. Above curve B: Met-A sees more than copy
   error (e.g. a fraction of cells switched state) — then the reverse test below applies.
2. **Reverse (upper limit):** from a measured Met-A, curve B gives the largest IAM-A copy error it can come from if all drift is copy
   error; measured IAM-A above that limit falsifies the relation.
3. **Same-file test first:** one WGBS file read for IAM-A (molecules) and for a sequencing Met-A (site means at the same identity sites,
   against a Loyfer healthy sequencing reference) — no array, no laboratory difference. Needs the sequencing Met-A commissioned.
Healthy cells alone do not test this (both read 1 by construction); the test needs cells with different known amounts of change.
