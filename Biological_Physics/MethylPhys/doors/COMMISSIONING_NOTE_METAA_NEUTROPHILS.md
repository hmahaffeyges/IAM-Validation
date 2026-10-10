# Met-A on neutrophils (EPIC v1) — commissioning note (COMMISSIONED 2026-10-09)

**Status: COMMISSIONED 2026-10-09 by the author** ("I agree with commissioning Met-A with the detection limits printed on every report").
Every row cites the record it comes from; no number here is new. From this date Met-A readings in this scope are results, not development readings.

**Detection limits printed on every report:** a loss of 2 % of the neutrophil pattern (purified neutrophils) or 5 % (whole blood) is the
smallest that reliably reads outside Normal (DEV-METAA-SENS-01). Code: `conductor_v3.METAA_COMMISSIONING`; report: Stage 5 section.

## Scope
Met-A for neutrophils, on EPIC v1 arrays, from isolated neutrophils or whole blood (any neutrophil fraction, each reading with its own detection limit; DEV-LOWFRAC-01), stages 0, 1, 2, 5, 6, 8
(self-tare II then the median tare), 9 (noise gate), 13. Everything else in the chain stays development: sky (11/12), direction (10),
trace/foreign cells (3b/3c), other cell types, EPIC v2, IAM-A.

## Bars and results

| # | Bar (where it was written) | Result (record) | Met |
|---|---|---|---|
| 1 | Same-person replicates: within-person SD ≤ 0.020 (CHAIN_COMMISSIONING stage 5/8) | 0.0164 (Box Run 1 job A) | yes |
| 2 | Same-person replicates ≥ 95 % Normal | 62/63 (job A) | yes |
| 3 | Other laboratories' purified neutrophils Normal on tared A | 68/68 (job A) | yes |
| 4 | Floor (reference) arrays 6/6 Normal | 6/6 (job A) | yes |
| 5 | Healthy arrays across all test sets read Normal (specificity) | 525/541 (97.0 %), 18 series, medians 0.994-1.005 (job B) | yes |
| 6 | The chain refuses what it cannot read, with the reason | 242 healthy arrays of out-of-scope specimens and 72 EPIC v2 arrays refused, each named (job B); intake 1,569/1,569 end to end (DEV-INTAKE-02) | yes |
| 7 | Neutrophil fraction in whole blood recovered within RMSE 0.02 (DEV-NILC-01) | 0.019 chain composition, 0.014 atlas_e, 12 EPIC mixtures from another laboratory (GSE182379); FACS bloods 0.007-0.031 by group (GSE112618) | yes |
| 8 | Met-A responds to a known loss of pattern by the amount the model predicts (DEV-METAA-SENS-01, checks 1-2) | ratio 0.998 (neutrophils), 1.043 (whole blood) | yes |
| 9 | ≥ 95 % leave Normal at a 1 % loss (DEV-METAA-SENS-01, check 3) | 0.875 neutrophils, 0.083 whole blood; 100 % at 2 % (neutrophils) and 5 % (whole blood) | **no** |
| 10 | C-score healthy band (0.751-1.409, tared) holds on laboratories not used to set it (≥ 95 %) | 23 / 26 healthy granulocytes of a new laboratory (GSE226298) inside, 88.5 %; all three outside are low (0.63-0.74) (DEV-NEWLAB-GRAN-01) | **no** |
| 11 | A new laboratory's healthy cells read Normal on tared Met-A (≥ 95 %) | 26 / 26 healthy granulocytes, GSE226298 (DEV-NEWLAB-GRAN-01) | yes |

## Author decisions (2026-10-09)
1. **Bar 9:** commissioned with the measured detection limits printed on every report (option a).
2. **Bar 10:** the C-score is not part of this commissioning; the author asked for a plan to commission it (`doors/CSCORE_COMMISSIONING_PLAN.md`).

## Evidence since commissioning (sealed before reading)

| test | result | record |
|---|---|---|
| Identical DNA at three laboratories (SEQC2 EpiQC, 30 EPIC arrays, 7 cell lines) | technical repeats within 0.0114 (10/10); same DNA across three laboratories within 0.0084 untared, 0.006 tared | DEV_EPIQC_ARRAY_01_OUTCOME.md |
| Specificity: 4–15 % liver, lung, colon or neuron DNA in blood DNA (Moss 2018, 9 arrays) | all Normal, within 0.010 of the unmixed blood (9/9) | DEV_CSCORE_MOSS_01_OUTCOME.md |

## Reproduce from public data
Every matrix the commissioned path reads rebuilds from GEO with committed code (identical; `data/MET_A_FLOOR_V13_REPRO/README.md`):
1. `python3 doors/data/DEV_NOISE_02/build_noise_sites_01.py WORK calib 0 1` (91 Salas purified arrays through chain Stage 1), then
   `build_noise_sites_01.py WORK build` and `build_noise_gate_01.py WORK` (noise sites and gate).
2. `python3 doors/data/MET_A_FLOOR_V13_REPRO/reproduce_floor_v13.py WORK/betas WORK2` (floor, 6,000 identity sites, held-out precision).
3. `python3 doors/data/MET_A_FLOOR_V13_REPRO/reproduce_reference_v12.py WORK/betas` (per-site healthy spread, C-score baseline).
4. `python3 doors/data/MET_A_FLOOR_V13_REPRO/reproduce_blood_composition.py WORK/betas WORK3` (whole-blood composition).
