# DEV-REPL-V3-01 — technical replicates on chain v3 with the median tare (development note, 2026-10-03; \measured)

**What it shows.** With the median tare, the 64 technical replicates of GSE250556 read with a within-person SD of 0.037
and 48 of 63 in Normal. The 0.008 / 63-of-63 figures came from the fitted tare that was removed.

Run plan: `doors/DEV_REPL_V3_01_PLAN.md` (commit 87cfa65). Chain: commit 87cfa65, unchanged. Exact commands: `doors/data/DEV_REPL_V3_01_run/COMMANDS.md`. Counting rules written before the run:
`doors/data/DEV_REPL_V3_01_run/ANALYSIS_RULES_set_before_reading.md` (also in S3 at `results/PROC_REPL_V3_01/`, timestamp 2026-10-03 05:41:38 UTC, before the box job started).

## Run
- Data: `GSE250556_RAW.tar` from GEO (1,117,624,320 bytes; sha256 in `RAW_tar_record.json`), 128 IDATs = 64 arrays. Person and replicate labels from
  `GSE250556_series_matrix.txt.gz`.
- Chain v3, `run_sample.py --engine v3 --specimen "whole blood" --array-type EPIC_v1`, `--sex`/`--age` from the series matrix. Two passes.
  Pass 2: `--slide-ref-table` = pass-1 A of the other arrays on the same slide (>= 3 present on every slide, so the batch fallback was never used),
  the array itself excluded. Stage T = median tare, nothing fitted. Noise gate `noise_gate_EPIC_v1.json` (N_max 0.149).
- Box: methylphys-cpu-01, job 9a90e094, 05:41–05:50 UTC. IDATs and every output copied to S3 as produced.

## Measured
| item | value |
|---|---|
| arrays read end to end | **63 / 64** |
| not read | GSM7981500 (subjectC, pooled, replicate 1): Stage 0 QUARANTINE before calibration, hard failure `ctrl_qc` (control-probe QC). Same array failed in RUN3. |
| tared | 63 / 63 read (references: 7 same-slide arrays for 56; 6 for the 7 on slide 205832330169, which holds the quarantined array) |
| withheld by the noise gate | 0 (every reading is tared, so the gate does not withhold). N is above 0.149 on 59 of 63 (N 0.137–0.195): untared, 59 would be withheld. |
| tared A_rel | mean 1.001, range 0.920–1.077 (per array: `doors/data/dev_repl_v3_01_readings.csv`) |
| within-person SD (pooled) | **0.037** — subjectA 0.030 (16), subjectB 0.035 (16), subjectC 0.040 (15), subjectD 0.041 (16) |
| SD over all | 0.040 |
| in Normal (0.95–1.05) | **48 / 63 = 76.2 %** (8 below, 7 above) |

Grouping rule: person = the `subjectA`..`subjectD` label in the series-matrix sample title (the series matrix has no separate subject field;
age 24 / 39 / 54 / 66 confirms four people). All 16 replicates of a person are grouped, pooled and unpooled DNA alike.
Within-person SD = sqrt( sum over persons of sum of squared deviations from the person mean / sum of (n − 1) ).

## Targets for commissioning (not a verdict; development run)
| target | development reading today |
|---|---|
| >= 62 / 64 read end to end | 63 / 64 |
| within-person SD <= 0.020 | 0.037 |
| >= 95 % of tared readings in Normal | 76 % (48 / 63) |

## Side by side: RUN3 (fitted tare, removed) vs this run (median tare)
Source for RUN3: `doors/data/chain_v3_dev3_readings.csv`, column `A_rel_tared`, 64 GSE250556 rows.
| | RUN3 fitted tare | PROC-REPL-V3-01 median tare |
|---|---|---|
| read / tared | 63 / 63 | 63 / 63 |
| within-person SD | 0.008 | 0.037 |
| SD over all | 0.013 | 0.040 |
| in Normal | 63 / 63 | 48 / 63 |
| range | 0.967–1.026 | 0.920–1.077 |
Untared A is identical in both runs (largest difference 0.0000). The RUN3 file gives 63 / 63 in Normal; the RUN3 note's table says 62 / 63.

## Observations
- Untared A within-person SD is 0.036; the median tare leaves it at 0.037. Dividing by a slide median does not remove array-to-array spread.
- Untared A tracks the noise index N on these arrays (r = 0.84; with neutrophil fraction r = 0.35). The removed fitted tare used N; that is where its 0.008 came from.
- For comparison only: dividing by the median of all 62 other arrays instead of the same slide gives within-person SD 0.030
  and 55 / 63 in Normal. Still outside bars 2 and 3.

Nothing was tuned after reading.

## Next (development)
The median tare moves every array by the same amount; it cannot remove noise that differs from array to array (untared A tracks the noise index,
r 0.84). Being developed: the array tares itself from its own 48,528 fixed sites (derived per array, nothing fitted, no cohort), then rerun on the
same 64 arrays. For deployment: a known-pattern control DNA on every slide (physical tare).
