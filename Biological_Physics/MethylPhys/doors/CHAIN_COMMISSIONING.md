# Chain v3 commissioning — stage by stage (development, 2026-10-03)

The class-era (v2) commissioning table that stood here is kept in the private archive with the retired v2 chain.

Order: SOP v3 section 2b (base chain first, then the toolkit in the commissioning order). Each check was written in a dated `doors/` note before
the data were read; the outcome is under the line in the same note. A stage is wired into `run_sample.py` / `conductor_v3.py` / `report_v3.py`
only if its check passed. Data: 56 GEO series in the project bucket, 5,069 physical arrays (DEV-BASE-CHAIN-01 manifest), box ssh:methylphys-cpu-01.

| stage | check (note) | result | wired |
|---|---|---|---|
| 0 Intake | (a) every array runs end to end or stops with a named reason (DEV-BASE-CHAIN-01) | run 1: 26 crashes + 38 EPIC v2 arrays misread as sex mismatch; fixed (EPIC v2 refused at intake, unreadable IDAT a named stop, manifest failure = ENVIRONMENT_MISSING_MANIFEST); run 2: **5,069/5,069, 0 crashes** | running |
| 1 Calibration | (a) as above | run 2: PASS (one unreadable IDAT is a named stop) | running |
| 2 Composition (8 groups) | exercised by (a), (d) | runs on every whole blood; deconvolver conformance PASS on 4,978 reports. Development finding: fails the held-out truth bars that stages 3/4 were held to (DEV-NILC-01) | running |
| 5 Met-A (neutrophils) | (b) purified healthy neutrophils Normal on tared A_rel | **FAIL**: floor arrays 6/6; other laboratories 42/49 (0.810-1.068) | running |
| 6 C-score | exercised by (a), (d) | computed on every reading; band not set | running |
| 7 IAM-A (Stage Q) | (e) constructed single-molecule data | **PASS** (IAM-A 1.0 Normal; other pipeline refused; 30/30 errors) | running |
| 8 Same-run tare | (c) GSE250556 within-person SD; targets SD <= 0.020, >= 95 % Normal | **not met**: SD 0.037, 48/63 Normal | running |
| 9 Noise gate | (c) withhold rate | 0/63 withheld when tared; N > 0.149 on 59/64 (would be withheld untared) | running |
| 13 Report | (d) every section SOP 2b requires, on every report past Stage 0 | run 1: 5,481/5,553 (the 72 EPIC v2 refusal reports lacked the Stage 1 line; fixed); after re-render **5,553/5,553 PASS** | running |
| 4 NILC | held-out truth (cord-blood DNA mixtures; constructed mixtures) + replicate spread (DEV-NILC-01) | **FAIL** (toolkit module and NILC-e) | no |
| 3 Atlas deconvolution (held patch atlas_e) | as NILC + agreement with 8 groups + variant-e rule (DEV-ATLAS-EPIC-02) | **FAIL** truth bars; rule reproduced, repeatability and agreement PASS | no |
| 5 per cell (B cells, the one qualifying cell) | pure-cell precision >= 95 % Normal; replicate SD <= 0.020 (DEV-PERCELL-01) | **FAIL** (72.7 %; 0.037) | no |
| 10 Directional decomposition | replicates no direction; treated series known direction (DEV-DIRECTION-01) | not assessable as built (cohort-trained disease-direction panel) | no - author decision |
| 11 Sky map | healthy replicate skies consistent with the shuffled null (DEV-SKY-01) | **FAIL** (median power 6.0x null at l 2-8; 100 % of arrays above null in bands 1-5) | no |
| 12 Sky statistics | look-elsewhere by simulation holds its rate (DEV-SKYSTAT-01) | not run: tool not built; stage 11 open | no |
| 3b Trace-cell detection | DEV-TOOLKIT-ADDED-01 | not run: line must be re-set without a population | no - author decision |
| 3c Foreign-cell detection | DEV-TOOLKIT-ADDED-01 | not run: population-quantile line, class-era beta scale | no - author decision |
| 11b Surface brightness | DEV-TOOLKIT-ADDED-01 | not run: class-era inputs | no - author decision |
| 12b Difference map | same-person check; same-person differences below other-person differences >= 95 % (DEV-TOOLKIT-ADDED-01) | **PASS** (348/348; q99 0.060 vs 0.174) | **yes** (`--prior-betas`, release check E5) |

Release check: `kit/release_check.py` on the box copy (git repository, commit 9593d66 = main 058b646 + these changes, `doors/` included): **PASS, 12 of 12** (F1, F1b, S1-S4, E1-E5, M1); `kit/results/release_check.json`.
