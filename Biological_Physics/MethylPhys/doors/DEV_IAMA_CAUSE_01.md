# DEV-IAMA-CAUSE-01 — why three EMSeq LAB02 libraries read low (development; design 2026-10-12; nothing read yet)

**Why.** DEV-IAMA-COMMISSION-01 held commissioning under its stopping rule: across laboratories was not met because three EMSeq LAB02 libraries
(SRR13051256, SRR13051128, SRR13051268) read copy error below every other EMSeq library; they are the three most deeply sequenced EMSeq runs, and
the pipeline reads the start of each file (values in `DEV_IAMA_COMMISSION_01_OUTCOME.md` and `data/DEV_IAMA_COMMISSION_01/why_iama_commission_01_output.txt`).
This is a development read to find the cause; it does not re-score anything.

**Question.** Does the low copy error belong to the library, or to the part of the file read?

**Design.** The three low runs and two that read normally in the same laboratory (SRR13051263, SRR13051136), each read from three parts of the
run (start, middle, end, by SRA spot range) through the same pipeline, with read length, base quality, duplicate fraction and conversion recorded
per part. Box script `data/DEV_IAMA_CAUSE_01/cause_01.sh` (each part written with SRA fastq-dump by spot range, then read by the Stage B
script unchanged, one part per call); decision `decide_cause_01.py`, rule in its header, fixed before any part is read: per part, IAM-A of each
low run over the mean of the two normal runs; LIBRARY if below 0.97 at all three parts, FILE-PART if below 0.97 at the start only and within
0.97-1.03 at the other two, otherwise MIXED. Check: the start part reproduces each run's Stage B reading within 2 %.
**Synthetic test:** `decide_cause_01.py --planted` (output `decide_cause_01_planted_output.txt`, four tables with known answers: PASSED); the
orchestration with stand-ins for SRA and the pipeline (output `cause_01_orchestration_test_output.txt`: every part staged before its call: PASSED).
The run is entered in the ledger before launch.
**Not a milestone:** design (development).

**First run stopped itself after part 1 (2026-10-13, fixed).** The job passed `SHUTDOWN=1` to `cause_01.sh`, and the shell exported it to the
pipeline this script calls; `session_epiqc.sh` stops the machine when it sees that variable, so the spot machine shut down after the first of
15 parts. Part 1 is complete and in S3 (`results/BOXRUN10_EPIQC/cause01/SRR13051256p0.pat.gz`, 24.5 M reads, 99.99 % mapped, conversion
0.99777) and is skipped on the rerun. Fix: the inner call now clears the variable, so only this script stops the machine, after the last part.
Both controls are in `data/DEV_IAMA_CAUSE_01/cause_01_shutdown_control_output.txt`: with the fix the pipeline sees it unset; with the old line
it saw 1. Machine time lost: about 35 minutes, roughly $1.

