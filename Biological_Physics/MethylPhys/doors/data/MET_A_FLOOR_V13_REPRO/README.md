# Commissioned Met-A neutrophil matrices rebuilt from public data (2026-10-10)

Inputs: the 6 physical GSE110554 neutrophil arrays (GEO IDATs) through chain Stage 1 (`../DEV_NOISE_02/build_noise_sites_01.py ... calib`).

| chain file | what it sets | rebuilt by | result |
|---|---|---|---|
| `metA_floors_v1_3.json` | the neutrophil floor and its 6,000 identity sites (every Met-A reading) | `reproduce_floor_v13.py` (runs `chain_tests/freeze_v13.py` unchanged, paths only) | floor identical (difference 0); sites and references identical |
| `metA_floors_v1_3_loo.csv` | held-out precision printed with each reading (not used to compute A) | same | frozen-site readings identical to 6 decimals; re-chosen-site readings within 0.0006 (precision SD 0.01969 vs 0.01976) |
| `neutrophil_reference_v1_2.json` | per-site healthy mean and spread, C-score baseline (block 10) | `reproduce_reference_v12.py` (freeze_v13.py's formulas, block 10) | H mean and shrunk SD identical; clustering values identical (median 1.0103) |

The re-chosen-site difference: readings on the frozen sites match exactly, so the betas there are identical. Re-choosing sites ranks every
array probe by its SD, so a few probes elsewhere on the array differ between the box's Stage 1 run (2026-09-30) and this one. That file only feeds the
printed precision. `metA_floors_v1_3_loo.csv` is freeze_v13's output with the columns renamed (session cell 2026-10-01 22:50).
