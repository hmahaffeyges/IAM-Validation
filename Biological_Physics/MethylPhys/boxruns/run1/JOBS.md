# Box Run 1: job list (arrays only)

Prepared 2026-10-04. Development mode: readings are logged in `development/METHYLPHYS_DEVELOPMENT_LOG.md`. The one commissioning
check (job A) uses the bars already written in `doors/CHAIN_COMMISSIONING.md`; no new bars are set here.

All inputs are already in `s3://methylphys-data-945451304272-us-west-2-an/`. All outputs go to `results/BOXRUN1/<job>/`, with a
progress log at `results/BOXRUN1/log.txt`. The driver's last step stops the instance, and it stops the instance on a crash as well.

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
| GSE128733 neutrophils (2 arrays, local, to upload) | 4 | 0.03 | A (another laboratory) |

About 151 GB is read in total (sum of the table). Every job works from the calibrated betas where they exist and calibrates only the arrays that lack them.

## Jobs, in run order

**A. Commissioning check: self-tare II, then the median tare, on purified neutrophils.**
The author adopted self-tare II on 2026-10-04: each array is tared on its own type II fixed sites, then by the same-run median, with
nothing fitted. The bars are the ones already in `CHAIN_COMMISSIONING.md` for stage 5 and stage 8:
- same-person replicates (GSE250556): within-person SD ≤ 0.020 and ≥ 95 % Normal;
- purified neutrophils from other laboratories: Normal on tared A;
- floor arrays: 6/6 Normal.

Development round 2 met all three (0.0164; 62/63; 49/49; 6/6). This run repeats the check with self-tare II as the chain's own Stage T
rather than as a flag, on the full 68-array set plus the 2 new GSE128733 arrays. Output: one table per bar and one row per array.

**B. Every chain test set read again with the adopted tare.** These are development readings: Met-A, noise index, C-score, direction
and composition, one row per array, per set.

**C. Met-A C-score on every held-out healthy array.** The output is the spread (median, 2.5–97.5 %), per laboratory and per set. No band
is set here; the author sets it after reading the spread.

**D. Sky statistics with the apodised mask.** The same test that withheld the sky, on healthy replicates against the block-shuffle
null. It is run with the hard mask and the apodised mask side by side. The bars come from `CHAIN_COMMISSIONING.md` (DEV-SKY-02): band
power ratios 0.9–1.1, and a look-elsewhere rate at the stated 8.4 %. The question is whether band 1 (1.84 with the hard mask) comes down.
Needs: the apodised-mask code (GitHub session task 1).

**E. Atlas composition (atlas_e) on bloods with known composition.** This runs only if GSE112618 (FACS validation, 6 arrays) or
GSE182379 (constructed mixtures of 12 cell types, 12 arrays) has been downloaded first. Both are small and can be fetched to S3 before
the run.

## Box and time

The 32xlarge is not needed: arrays are light and calibration is already done for most inputs. On the Mac, calibrating one EPIC
array took about 6 s (2026-10-04); the number of arrays per set and the time per reading are measured by the driver's local test.
- Proposed: **m7a.8xlarge** (32 cores, 128 GB, about $1.85/h). The run time and cost are filled in after the local test; the plan is a
  run of a few hours at most.

## Still to do before the run (no box)
1. Driver script `boxruns/run1/run1_driver.py` (GitHub session task 2): runs A–E, resumes after a stop, logs to S3, stops the
   instance at the end and on a crash.
2. Apodised mask (GitHub session task 1), needed by job D.
3. Upload the 2 GSE128733 arrays to S3; optionally fetch GSE112618 and GSE182379 for job E.
4. Local test of the driver on 3 arrays per job, with the shutdown step switched off.
