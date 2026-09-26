# Fraction and A — what mixing does to the per-cell A, measured on constructed specimens, 2026-09-26

**Author's rule this document serves:** *"it doesnt matter the fraction that is a detection issue."* Measured, that
rule is exactly right — and it also says what to do above the detection gate.

## The generator can now test the chain

`Synthetic_Patient_Generator.compose_cells()` composes a specimen from the **115 cell-type means over every atlas
address (483,092)** with known fractions. Run through `run_sample.py --betas … --pipeline atlas_scale` (the identity
scale map the chain already carried for specimens that never went through Stage 1), a textbook healthy blood
reads: **every present cell NORMAL against its own centre, class gauge NORMAL in band, no spurious cell above 1 %,
sky assessable.** Before today a synthetic patient read BREACH from coverage alone (the cohort generator emits a
class-mean subset) — and ran through the wrong scale map, which adds +0.10 to every cell. Both are recorded.

Noise does nothing to A: σ = 0 and σ = 0.044 (the residual sd of a real array against its own composition
expectation) agree to four decimals. H of the mean β is immune to zero-mean noise. The scale map is the whole
difference.

## What mixing does — one straight line per cell

Dilute one cell into a blood background at known fractions and score it on its identity loci:

| cell | A at fraction 0 (reading the blood) | A pure | detected from |
|---|---|---|---|
| Colon_epithelial_cells | 0.869 | 1.004 | 2 % |
| Breast | 0.905 | 0.994 | **35 %** |
| Cortical_neurons | 1.065 | 0.986 | 1 % |
| CD56_NK-cells | 0.942 | 0.992 | 12 % (in a blood background that already contains NK) |
| GMP | 0.919 | 0.986 | 0.5 % |
| CD4_T-cells | 1.037 | 0.991 | 2 % |

A moves **linearly** from what the rest of the specimen reads at the cell's loci to the cell's own value. That is
what mixing does to a mean β, and it is why healthy blood could not give a presence floor for any abundant cell:
neutrophils are never below 33 % of blood, so blood cannot show where their A stops depending on fraction.

## Above the gate, the cell's own A is recoverable — by arithmetic the chain already has the inputs for

The mixture is invertible: own β = (specimen mean β − background expectation) / f, where the background is the
other deconvolved cells at their found fractions and the atlas gives what they read at these loci.

| cell | f | A mixed | **A unmixed** | A pure |
|---|---|---|---|---|
| Cortical_neurons | 0.03 | 1.063 | **0.993** | 0.986 |
| GMP | 0.03 | 0.922 | **0.979** | 0.986 |
| CD56_NK-cells | 0.08 | 0.947 | **0.992** | 0.992 |
| CD4_T-cells | 0.08 | 1.033 | **1.004** | 0.991 |
| Colon_epithelial_cells | 0.08 | 0.879 | 1.115 | 1.004 |
| Breast | < 0.20 | — | not found | — |

Four cells recover to within 0.01 of their pure value at 3–8 %. **Colon does not**, and the reason is the one
PROC-COV-01 measured: the deconvolver's composition is slightly wrong about the *background* at colon's loci, and
dividing by a small f amplifies that misfit. **Breast is a detection limit**, not a scoring one — the deconvolver
does not find it below 20 % in blood.

## The architecture this fixes, in the author's terms

1. **Fraction is the detection gate.** A cell the deconvolver does not find is not scored. Period.
2. **Above the gate, report the unmixed A** — the cell's own A, with its uncertainty set by (background
   misfit)/f. Where that uncertainty exceeds the NORMAL half-width, print the A and withhold the tier word, and
   say why in one line: *not enough of this cell to score it confidently.*
3. **No per-cell "healthy band."** The physics defines healthy at A = 1.0. [`percell_reference_identity_v1_0.json`](../chain/Runtime%20Matrices/Percell_Reference/percell_reference_identity_v1_0.json)
   is the calibration record — the instrument's noise floor per cell and its scale check — and stays that.

Evidence: `kit/results/PRESENCE_FLOOR_synthetic_dilution.json`, [`UNMIX_TEST.json`](../kit/results/UNMIX_TEST.json),
[`PRESENCE_FLOOR_blood_observed_range.json`](../kit/results/PRESENCE_FLOOR_blood_observed_range.json); example run `chain/example_runs/MethylPhys_SYNTH_CELLS_HC.html`.
