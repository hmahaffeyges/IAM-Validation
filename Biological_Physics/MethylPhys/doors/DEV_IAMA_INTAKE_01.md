# DEV-IAMA-INTAKE-01 — an intake stage for single-molecule files (Stage Q0), specification (development)

**DEVELOPMENT - not commissioned. Specification only; nothing is wired.**

**Why.** An array cannot reach Met-A without Stage 0 (control probes, detection, bead count, call rate, sex check, decision gate) and
Stage 1 (calibration). A single-molecule file reaches IAM-A (Stage Q) with only one check: it must come from the pipeline the floor
file was built with (`loyfer_pat_v1`). A file with poor bisulfite conversion, thin coverage or heavy duplication would be read anyway.

**Stage Q0, before Stage Q.** Each check stops the file with a named reason, as Stage 0 does for arrays.
| step | check | refusal |
|---|---|---|
| Q0.1 | the file is a .pat(.gz) or site table with the expected columns and genome build (hg38), readable to the end | QUARANTINE_UNREADABLE_PAT |
| Q0.2 | pipeline is `loyfer_pat_v1` (already in Stage Q; moved here) | PIPELINE_MISMATCH |
| Q0.3 | bisulfite conversion: non-CpG cytosine methylation (or the spike-in, when the sequencing record gives one) below a set rate | QUARANTINE_CONVERSION |
| Q0.4 | coverage: enough reads cover enough of the cell's identity sites | QUARANTINE_LOW_COVERAGE |
| Q0.5 | read length: enough reads span at least the CpG count Stage Q needs per read | QUARANTINE_SHORT_READS |
| Q0.6 | duplicate fraction below a set rate | QUARANTINE_DUPLICATES |
| Q0.7 | specimen is a purified cell type with a floor file (IAM-A reads purified cells only) | SPECIMEN_REFUSED |
| Q0.8 | decision gate: proceed, or stop at the first refusal, and record every measured value in the report | — |

**How the limits are set (development, then commissioning).** Measured on the healthy single-molecule files already behind the floor
file, plus the 8 paired-neutrophil WGBS runs (GSE128731) when Box Run 2 reads them: each limit is the edge of the healthy files' own
range, written in a dated note before any test file is read. Negative controls, as for Stage 0: a truncated file, a file from another
pipeline, a file with conversion deliberately degraded, a file subsampled below the coverage limit; each must stop with its named reason.

**Where it goes.** `chain/stage_q0_intake.py`, called by `run_sample.py` before `stage_q_iam_a.read`; listed in `doors/CHAIN_SEQUENCE.md`
after the author approves this specification.
