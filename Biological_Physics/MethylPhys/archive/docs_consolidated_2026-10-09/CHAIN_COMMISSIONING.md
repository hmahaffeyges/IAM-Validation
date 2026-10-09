> **Archived 2026-10-09.** Merged into [`STATUS.md`](../../STATUS.md) (plan, commissioning record) or the LOG [`development/METHYLPHYS_DEVELOPMENT_LOG.md`](../../../../development/METHYLPHYS_DEVELOPMENT_LOG.md) (chain changes). Kept as a record; not updated.

# Chain v3 commissioning — stage by stage

**Commissioned:** Met-A on neutrophils, EPIC v1 (isolated neutrophils; whole blood with neutrophil fraction ≥ 0.20), stages 0, 1, 2, 5, 6,
8, 9, 13 — **2026-10-09**, by the author (`COMMISSIONING_NOTE_METAA_NEUTROPHILS.md`). Detection limits 2 % (purified) / 5 % (whole blood)
loss of pattern, printed on every report. Everything else below is development.


**DEVELOPMENT - not commissioned.** Development mode (author ruling O): no sealed pre-registrations; each check was written in a dated `doors/DEV_*.md`
note before the data were read and the outcome is under the line in the same note.

## Round 4 (Box Run 1 completion and constructed checks, 2026-10-08/09)

| Stage | Check | Result | Bar met |
|---|---|---|---|
| All test sets re-read with the adopted tare | Box Run 1 job B (2026-10-08, 5,006 s, exit 0) | 4,845 rows: 4,520 read, 129 stopped at intake, 196 without input. Healthy arrays with tared A: 541 in 18 series, **525 Normal (97.0 %)**, median 1.0001, 2.5-97.5 % 0.958-1.045. Every one of the 18 series has median 0.994-1.005 | yes (healthy specificity) |
| Chain refusals on real data | job B | 242 healthy arrays read nothing, each with its named reason: sorted B/T/NK/monocyte/eosinophil/basophil cells, bone marrow, PBMC (no reference yet) and 72 EPIC v2 arrays | as designed |
| Positive controls in the test sets | job B | none in scope: treated and myeloid sets are cell lines, bone marrow or have < 3 same-run references; infection set disease and healthy arrays on different slides; GSE118144 patient neutrophils 0.999 vs healthy 0.999 | not assessable |
| Met-A sensitivity, constructed loss of pattern | DEV-METAA-SENS-01 | response matches the chain's own model (ratio 0.998 neutrophils, 1.043 blood); all arrays leave Normal at 2 % loss (neutrophils) and 5 % (blood) | 2 of 3 checks; check 3 (1 % loss) not met |
| Composition, 12 constructed EPIC mixtures of 12 purified cell types (GSE182379, another laboratory) | Box Run 1 job E, scored against the depositors' fractions; bars DEV-NILC-01 (neutrophils RMSE ≤ 0.02, other groups ≤ 0.03) | **neutrophils: atlas_e 0.014, chain composition 0.019, both within 0.02**. atlas_e meets 6 of 8 groups (eosinophils 0.034, CD8 T 0.056 outside); chain composition 5 of 8 (basophils 0.030, B 0.032, CD8 T 0.030 outside). Correlation with truth 0.95-1.00 for every group | neutrophils yes; all groups no |
| Met-A C-score band on held-out laboratories | job B | not assessable: the healthy series in job B that were not used to set the band are sorted non-neutrophil cells, bone marrow or PBMC, all refused before Met-A | not assessable |
| Stage Q0 IAM-A intake | DEV-Q0-HEALTHY-01 | three healthy hg19 files proceed; the real hg38 copy stops (GENOME_BUILD_MISMATCH) | yes |

## Round 3 (Box Run 1, 2026-10-05)

| stage / item | check (note) | result | wired |
|---|---|---|---|
| 8 Self-tare II then median tare, on the box, wired | Box Run 1 job A (bars from DEV-SELFTARE-02) | replicate SD 0.0164 (<= 0.020); 62/63 Normal; other laboratories 68/68; floor 6/6 - every bar met | **yes** (Stage T step 1, PR #27) |
| 6 Met-A C-score on every healthy array | Box Run 1 job C | median (2.5-97.5 %): GSE250556 0.840 (0.690-1.201), healthy_repeat 1.067 (0.728-1.852), DEV_BASE_CHAIN_01 1.165 (0.836-1.809) | printed; band not set |
| 11 / 12 Sky with the apodised mask | Box Run 1 job D | band 1 1.84 -> 1.18; bands 1-4 still 1.13-1.18 (bar 0.9-1.1); look-elsewhere 0.611 (bar 0.084) | no |
| 3 / 4 Composition against FACS (GSE112618, 6 bloods) | Box Run 1 job E, scored locally | mean abs error atlas_e 0.005-0.044, chain composition 0.007-0.031 by group; donor overlap with the references still to check | no (flags) |
| All test sets re-read with the adopted tare | Box Run 1 job B | not complete: segfault after the first set (902/902 ok) | - |

## Round 2 (2026-10-04)

| stage / item | check (note) | result | wired |
|---|---|---|---|
| 0 Intake: age and sex optional (F), blood specimens only (L), ids hashed (B), EPIC v2 refused (M) | DEV-INTAKE-02 | 1,569/1,569 end to end, 0 crashes; 955 refused naming the specimen, 613 blood specimens read; 0 stops on age or sex; typed id in 0 of 1,837 bundles and ledgers, in 1,837/1,837 report titles | **yes** |
| 9 Noise gate: withhold below 90 % noise-site coverage, with the reason (A) | DEV-INTAKE-02 check 4; release check E8 | no real array below 90 % (lowest 48,127 of 48,528); constructed array withheld with the explanation | **yes** |
| 5 Met-A, purified healthy neutrophils, enlarged set | DEV-INTAKE-02 check 6 | tared A_rel: floor 6/6; other laboratories 56/68 Normal (round 1: 42/49) | running (unchanged) |
| 1 Detection: poobah against the Gaussian negative-control test (E) | DEV-DETECTION-01 | 4,996 arrays, 55 strata: poobah better in 31, Gaussian in 0, neither in 24 | poobah stays (no change) |
| 8 Self-tare on type II fixed sites, then median tare (G) | DEV-SELFTARE-02 | replicate within-person SD 0.0164, 62/63 Normal; other laboratories 49/49; floor 6/6 - every bar met | flag `--dev-selftare-ii`; author decision to make it the tare |
| 3 / 4 Composition against another laboratory's mixtures (H) | DEV-COMPOSITION-TRUTH-02 | no adult EPIC mixture set found; GSE77797 (450K): atlas_e within 0.02 except granulocytes 0.041; NILC-e granulocytes 0.061 | flags `--dev-atlas-e`, `--dev-nilc` |
| 10 Directional decomposition, physics only (I) | DEV-DIRECTION-02 | treated arrays 12/12 toward disorder; replicates 56/63 no direction (bar 95 %); vehicle 4/6 no direction | flag `--dev-direction` |
| 11 / 12 Sky map against the block-shuffle null; sky statistics (J) | DEV-SKY-02 | median power ratio band 1 1.84, bands 2-6 1.03-1.15 (bar 0.9-1.1); look-elsewhere rate 91 % (bar 8.4 %) | flag `--dev-sky` |
| 3b / 3c / 11b on the array's own noise (K) | DEV-TOOLKIT-ADDED-02 | 3b: 82.5 % called at 5 % (bar 95 %), 0/360 unspiked; 3c: 100 % called at f = 0 (bar <= 5 %); 11b: 3.8 % of repeat pairs inside the interval (bar 95 %) | flags `--dev-trace`, `--dev-foreign`, `--dev-brightness` |
| 7 IAM-A C-score (C) | DEV-IAMA-CSCORE-01 | constructed independent sites C 0.823 (limit +-0.80); clustered 447.5; real whole files 604-1,047 | **yes** (printed; band not set) |
| 7 IAM-A on real single-molecule files | DEV-IAMA-REAL-01 | 3/3 end to end; 60 MB heads reproduce the floor file exactly; whole files 1.0394, 1.0632, 1.0344 (2/3 Normal) | running |
| 5 New cell: monocytes, B cells (D) | DEV-NEWCELL-01 | monocytes 11/13, SD 0.032; B cells 16/22, SD 0.037 - neither meets the new-cell rule | no (B behind `--dev-percell-b`) |
| EPIC v2 through SeSAMe (M) | DEV-EPIC-V2-01 | 72/72 v2 calibrated; identity sites median 5,391 of 6,000 (< 5,400); v2 minus v1 at shared sites mean -0.034, SD 0.054; no v2 floor, no v2 replicates | flag `--dev-epic-v2`; EPIC v2 stays refused |
| Development flags (N) | DEV-FLAGS-01 | 63/63 readings identical with every flag on; every flag labelled; development section on every report | **yes** |

Release check: `kit/release_check.py` on a fresh git copy on the box (public clone at `185f609` plus the four round-2 commits applied as patches, box commit `edc6ae6`, `doors/` included): **17 of 17 checks PASS** (F1, F1b, S1-S4, E1-E10, M1; 258 s); `kit/results/release_check.json`. E10 ran with the atlas v2 parquet; the sky block was NOT_RUN there (healpy not in that environment).

## Round 1 (2026-10-03)

The class-era (v2) commissioning table that stood here before round 1 is kept in the private archive with the retired v2 chain.

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


## IAM-A, round 1 (2026-10-09, development)
| Check | Result | Record |
|---|---|---|
| Intake stops bad files (wrong build, cut file, low conversion) | real hg38 file and two low-conversion runs stopped; healthy Loyfer files proceed | DEV-Q0-HEALTHY-01, session 2 |
| Repeatability (halves; donors; sequencers) | within 0.004; 0.009 (largest 0.0086); 0.009 (largest 0.0087) | session 2 |
| Another laboratory's healthy neutrophils read Normal | TruSeq yes (1.047, 1.042); Swift no (1.16) | session 2, DEV-IAMA-KIT-01 |
| Kit independence | not met: 0.12 between kits on the same cells; not a read-end or coverage artefact | DEV-IAMA-KIT-01 |
