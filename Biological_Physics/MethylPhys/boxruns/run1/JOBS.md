# Box Run 1: job list (arrays only)

Prepared 2026-10-04. Development mode: readings are logged in `development/METHYLPHYS_DEVELOPMENT_LOG.md`. The one commissioning
check (job A) uses the bars already written in `doors/CHAIN_COMMISSIONING.md`; no new bars are set here.

All inputs are already in `s3://methylphys-data-945451304272-us-west-2-an/`. All outputs go to `results/BOXRUN1/<job>/`, with a
progress log at `results/BOXRUN1/log.txt`. The driver's last step stops the instance, and it stops the instance on a crash as well.
The driver reaches S3 through boto3 with the default credential chain (credentials configured on the box; the instance has no IAM role).

## Inputs

| S3 prefix | Files | GB | Used by |
|---|---|---|---|
| `results/DEV_BASE_CHAIN_01/betas` | 56 | 26.6 | calibrated betas of 5,069 arrays: saves recalibrating (A, B, C) |
| `downloads/G_chain_tests/GSE250556` | 131 | 1.1 | same-person replicates (A, C, D) |
| `downloads/G_chain_tests/neutrophil_ref` | 28 | 9.1 | purified neutrophils, other laboratories (A) |
| `downloads/G_chain_tests/healthy_repeat` | 66 | 20.6 | healthy repeats (B, C, D) |
| `downloads/G_chain_tests/infection` | 43 | 39.4 | B |
| `downloads/G_chain_tests/myeloid` | 41 | 21.2 | B |
| `downloads/G_chain_tests/autoimmune` | 22 | 14.8 | B |
| `downloads/G_chain_tests/prediagnosis` | 10 | 7.7 | B |
| `downloads/G_chain_tests/longitudinal` | 18 | 7.6 | B (and the difference map) |
| `downloads/G_chain_tests/neutrophil_state` | 8 | 2.5 | B |
| `downloads/G_chain_tests/neutrophil_ref_GSE128733` | 4 | 0.03 | GSE128733 neutrophils, 2 EPIC arrays (A, another laboratory) |
| `atlas_v2/IAMAtlas_v2.parquet` | 1 | – | atlas v2 (D, E) |

About 151 GB is read in total (sum of the table; the atlas size is not recorded here). Every job works from the calibrated betas where they exist and calibrates only the arrays that lack them.

## Jobs, in run order

**A. Commissioning check: self-tare II, then the median tare, on purified neutrophils.**
The author adopted self-tare II on 2026-10-04: each array is tared on its own type II fixed sites, then by the same-run median, with
nothing fitted. The bars are the ones already in `CHAIN_COMMISSIONING.md` for stage 5 and stage 8:
- same-person replicates (GSE250556): within-person SD ≤ 0.020 and ≥ 95 % Normal;
- purified neutrophils from other laboratories: Normal on tared A;
- floor arrays: 6/6 Normal.

Development round 2 met all three (0.0164; 62/63; 49/49; 6/6). This run repeats the check with self-tare II as the chain's own Stage T
rather than as a flag, on the full 68-array set plus the 2 new GSE128733 arrays. Output: one table per bar and one row per array.
If the GSE128733 folder is missing or empty, the driver writes a warning to the log (and to the job's state) and job A continues without
those 2 arrays.

**B. Every chain test set read again with the adopted tare.** These are development readings: Met-A, noise index, C-score, direction
and composition, one row per array, per set.

**C. Met-A C-score on every held-out healthy array.** The output is the spread (median, 2.5–97.5 %), per laboratory and per set. No band
is set here; the author sets it after reading the spread.

**D. Sky statistics with the apodised mask.** The same test that withheld the sky, on healthy replicates against the block-shuffle
null. It is run with the hard mask and the apodised mask side by side. The bars come from `CHAIN_COMMISSIONING.md` (DEV-SKY-02): band
power ratios 0.9–1.1, and a look-elsewhere rate at the stated 8.4 %. The question is whether band 1 (1.84 with the hard mask) comes down.
Needs: the apodised-mask code (GitHub session task 1) and the atlas v2.

**E. Atlas composition (atlas_e) on bloods with known composition.** This runs only if GSE112618 (FACS validation, 6 arrays) or
GSE182379 (constructed mixtures of 12 cell types, 12 arrays) has been downloaded first. Both are small and can be fetched to S3 before
the run. Needs the atlas v2. If their location is not given, or holds no files, job E is skipped with the reason in the log; this is not
a failure.

## Box and time

The 32xlarge is not needed: arrays are light and calibration is already done for most inputs. On the Mac, calibrating one EPIC
array took about 6 s (2026-10-04); the number of arrays per set and the time per reading are measured by the driver's local test.
- Proposed: **m7a.8xlarge** (32 cores, 128 GB, about $1.85/h). The run time and cost are filled in after the local test; the plan is a
  run of a few hours at most.

## Still to do before the run (no box)
1. Driver script `boxruns/run1/run1_driver.py` (GitHub session task 2): runs A–E, resumes after a stop, logs to S3, stops the
   instance at the end and on a crash.
2. Apodised mask (GitHub session task 1), needed by job D.
3. The 2 GSE128733 arrays are at `downloads/G_chain_tests/neutrophil_ref_GSE128733`; optionally fetch GSE112618 and GSE182379 for job E.
4. Local test of the driver on 3 arrays per job, with the shutdown step switched off.

## Job E data in S3 (recorded 2026-10-05)

| Set | What | S3 location | In Run 1 |
|---|---|---|---|
| GSE112618 | 6 whole bloods with FACS-counted cell fractions (EPIC v1) | `downloads/G_chain_tests/jobE_GSE112618/` (12 IDATs); the RAW tar, signal file and series matrix with the FACS fractions are also in `downloads/G_chain_tests/healthy_repeat/GSE112618/` | yes, job E |
| GSE182379 | 12 constructed DNA mixtures of 12 cell types, known fractions (EPIC v1) | `downloads/G_chain_tests/healthy_repeat/GSE182379/` (RAW tar 0.57 GB, signal intensities, series matrix) | **no**: the run's job E prefix points only at GSE112618 |

**Follow-up (to do):**
1. After Run 1: score job E's atlas_e fractions against the FACS fractions in GSE112618's series matrix. This is local work with no box.
2. Next box run: job E on GSE182379, with `--only E --force E --job-e-prefix downloads/G_chain_tests/healthy_repeat/GSE182379/`. Then score it against the
   mixture fractions in its series matrix.

## Run 1 outcome (2026-10-05) and the rerun
- **A done** (all four commissioning bars met), **C done**, **E done** (GSE112618 FACS bloods). Outputs in `s3://…/results/BOXRUN1/`.
- **B failed**: two threads wrote the same array's cached betas at once (an array that sits in two test sets). Fixed: each thread writes
  to its own temporary name.
- **D did not run**: `healpy` is not installed in the box's chain environment. Before the rerun: `/home/ubuntu/env/bin/pip install healpy`.
- Rerun B and D only (`--only B,D`), on a fresh 500 GB scratch disk that is deleted afterwards.

## Rerun of B and D (2026-10-05 07:26–08:57 UTC) and the plan for the next box run

**Status by job:** A done (bars met) · B NOT done · C done · D done (bars not met) · E done (6 FACS bloods scored locally, below).

**D result** (433 healthy whole-blood arrays; band power over the block-shuffle null, bar 0.9–1.1 in every band; look-elsewhere bar ≤ 0.084):

| mask | band 1 | 2 | 3 | 4 | 5 | 6 | look-elsewhere |
|---|---|---|---|---|---|---|---|
| hard | 1.835 | 1.034 | 1.143 | 1.146 | 1.079 | 1.048 | 0.905 |
| apodised 2.0° | 1.182 | 1.148 | 1.160 | 1.134 | 1.064 | 1.050 | 0.611 |

**E result, scored locally against the FACS fractions in the GSE112618 series matrix** (`jobE_vs_FACS_GSE112618.csv`; mean / max
absolute error over the 6 bloods). Check before using it as held-out truth: whether these 6 donors are among the purified-cell donors
behind `blood_composition_EPIC_v1` (GSE110554).

| group | atlas_e mean | atlas_e max | chain composition mean | chain composition max |
|---|---|---|---|---|
| neutrophils | 0.016 | 0.028 | 0.031 | 0.046 |
| granulocytes (NEU+EOS+BASO) | 0.021 | 0.033 | 0.017 | 0.026 |
| monocytes | 0.005 | 0.011 | 0.010 | 0.017 |
| B cells | 0.013 | 0.019 | 0.007 | 0.017 |
| NK | 0.036 | 0.055 | 0.031 | 0.050 |
| CD4 T | 0.013 | 0.020 | 0.023 | 0.043 |
| CD8 T | 0.044 | 0.073 | 0.024 | 0.051 |

**B failure (attempt 4):** the first set read (902 arrays, all ok); the worker then died with a segmentation fault (exit −11), so
`B_all.csv` was never written. Attempt 2 failed on a parallel-write race (fixed in `a0aae763`); attempt 3 on pandas broken by numpy 2.

### Next box run (planned for Thursday or later): do these in order
1. **Done 2026-10-05 (`4136a5e0`):** B restores the sets an earlier attempt finished from S3 (healthy_repeat is already there) and
   parquet/tar reads are serialised. Original note: change worker B to write one CSV per test set as each set finishes, and to resume by skipping
   sets whose CSV already exists; add `--workers` per job so B can run with 8 threads. Reproduce the segfault locally on one set with
   30 threads if possible; if it is pyarrow/parquet under threads, read the parquet betas once in the main thread.
2. **Box:** m7a.8xlarge; create a 500 GB gp3 scratch disk in us-west-2a and attach it as /dev/sdf; fresh 12-hour STS credentials.
3. **Environment check before starting** (stop if it fails, and stop the box): `/home/ubuntu/env/bin/python -c 'import numpy, pandas,
   scipy, healpy, methylprep'` with numpy 1.26.4, pandas 1.5.3, scipy 1.17.1, healpy 1.17.3. Never `pip install` without pinning numpy < 2.
4. **Run:** `run1_driver.py --work /mnt/scratch/boxrun1 --only B,E --force B --force E --workers 8
   --job-e-prefix downloads/G_chain_tests/healthy_repeat/GSE182379/` (B plus E on the 12 GSE182379 mixtures).
5. **After:** add the results to `development/METHYLPHYS_DEVELOPMENT_LOG.md`, update `doors/CHAIN_COMMISSIONING.md`, `doors/PLAN.md`,
   `doors/DATA_REGISTER.csv`, then detach and delete the scratch disk and confirm the box is stopped.
