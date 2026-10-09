# Box Run 2: job sheet (single-molecule sequencing; IAM-A on the paired neutrophils)

Prepared 2026-10-05. Development mode: readings go in `development/METHYLPHYS_DEVELOPMENT_LOG.md`; no bars are set here.

## Goal
Read IAM-A (Stage Q) and the IAM-A C-score on the purified neutrophils of GSE128731 (WGBS), whose same specimens were read for
Met-A on EPIC arrays (GSE128733, DEV-PAIRED-01: self-tared A 1.0437 and 1.0426). This gives Met-A and IAM-A on the same cells.

## Pipeline, pinned to the one the floor file and the Loyfer 2023 atlas used (`loyfer_pat_v1`)
From Loyfer et al., Nature 2023, Methods: paired-end FASTQ mapped with **bwa-meth v0.2.0, default parameters**, to the human genome with
lambda, pUC19 and viral genomes added; BAM by **SAMtools v1.9**; reads stripped of non-CpG nucleotides and written as PAT files by
**wgbstools v0.1.0** (`bam2pat`). Genome **hg19** (28,217,448 CpGs, the index the book uses); the atlas also ships hg38.
Install these exact versions on the box; Stage Q refuses any file whose pipeline field is not `loyfer_pat_v1`.

## Inputs
| what | where | size |
|---|---|---|
| GSE128731 neutrophil WGBS runs (2 donors, Sample6 and Sample7, 4 runs each across Swift, QIAseq and TruSeq kits) | SRA, run list `GSE128731_runs.csv` | 414 GB for all 8; ~100 GB for one run per donor |
| hg19 + lambda + pUC19 reference, bwa-meth index | built once in session 1, saved to `s3://…/reference/hg19_bwameth/` | ~20 GB |
| Loyfer 2023 Blood-Granulocytes .pat files (the floor's own pipeline), for the format check | GEO GSE186458 | per sample, a few GB |

## Sessions
1. **Session 1 (1–2 h, m7a.8xlarge, 500 GB scratch):** install bwa-meth 0.2.0, SAMtools 1.9, wgbstools 0.1.0 into their own environment
   (never into the chain environment); build the hg19 index and save it to S3; download one Loyfer Blood-Granulocytes .pat file;
   then take ~1 million read pairs of one GSE128731 run through the whole path and confirm the PAT file has the format and CpG
   indexing of a Loyfer .pat (same columns, same CpG index for the same position). **Stop here if it does not match.**
2. **Session 2 (~8–12 h, about $15–25):** one run per donor (the TruSeq runs, or the kit closest to Loyfer's, to be read from the
   sequencing records first): SRA download → bwa-meth → SAMtools → `wgbstools bam2pat` → Stage Q0 checks once wired (DEV-IAMA-INTAKE-01)
   → Stage Q IAM-A and C-score. Outputs to `s3://…/results/BOXRUN2/`.
3. **Session 3 (later, ~1 day, about $40–50):** the other 6 runs, to measure how IAM-A depends on library kit and sequencer.

## After each session
Log in the development log; update `STATUS.md` and `doors/DATA_REGISTER.csv`; delete the scratch disk; confirm the box is stopped.

## Update 2026-10-08 (before session 2)
- **Pipeline pin, completed from Loyfer 2023 Methods:** bwa-meth 0.2.0 (Python 3.6 environment) → SAMtools 1.9 → **Sambamba 0.6.5 markdup**
  (`-l 1 -t 16 --sort-buffer-size 16000 --overflow-list-size 10000000`) → SAMtools view `-F 1796 -q 10` → wgbstools 0.1.0 bam2pat, hg19
  (28,217,448 CpGs, checked). Sambamba 0.6.5 installs on the box (checked 2026-10-08).
- **Session 2 reads with position v2** (P = 1.1492, whole files) through Stage Q0 then Stage Q, with `--specimen "isolated neutrophils"` and
  `--alignment-qc` (conversion from the lambda spike-in, duplicate fraction from Sambamba).
- **Size:** each neutrophil run is 105-182 Gbases. A first reading per donor uses a random subsample of read pairs (≈ 10× depth); P is a
  whole-genome rate, which a random subsample estimates without bias. The read count and hours are set from session 1's measured alignment speed.
- **Before session 2 reads any GSE128731 file:** set the Q0 limits (conversion, ≥ 6-call share, duplicates) from the three healthy Loyfer
  granulocyte files' own values, in a dated note.
