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
**Synthetic test:** DEV_IAMA_CAUSE_01/decide_cause_01.py, DEV_IAMA_CAUSE_01/decide_cause_01_planted_output.txt, DEV_IAMA_CAUSE_01/cause_01_orchestration_test_output.txt, DEV_IAMA_CAUSE_01/cause_01_shutdown_control_output.txt

The decision rule was exercised on four planted tables with known answers (PASSED) before any part was read; the orchestration and shutdown controls are the other two outputs. `decide_cause_01.py --planted` (output `decide_cause_01_planted_output.txt`, four tables with known answers: PASSED); the
orchestration with stand-ins for SRA and the pipeline (output `cause_01_orchestration_test_output.txt`: every part staged before its call: PASSED).
The run is entered in the ledger before launch.
**Not a milestone:** design (development).

**First run stopped itself after part 1 (2026-10-13, fixed).** The job passed `SHUTDOWN=1` to `cause_01.sh`, and the shell exported it to the
pipeline this script calls; `session_epiqc.sh` stops the machine when it sees that variable, so the spot machine shut down after the first of
15 parts. Part 1 is complete and in S3 (`results/BOXRUN10_EPIQC/cause01/SRR13051256p0.pat.gz`, 24.5 M reads, 99.99 % mapped, conversion
0.99777) and is skipped on the rerun. Fix: the inner call now clears the variable, so only this script stops the machine, after the last part.
Both controls are in `data/DEV_IAMA_CAUSE_01/cause_01_shutdown_control_output.txt`: with the fix the pipeline sees it unset; with the old line
it saw 1. Machine time lost: about 35 minutes, roughly $1.

**In plain words.** Our second instrument reads single DNA molecules from sequencing. In its commissioning four of five
checks passed; the fifth failed, where the same kit was run in two laboratories on identical reference DNA. Rather than
adjust anything, our own rule puts the whole commissioning on hold until we know why. This test is that search: we read the
suspect libraries from several positions in each file, with controls that passed, to find whether the gap came from where in
the file we sampled or is real. **What it would take to finish.** The 15 readings now running answer it. If sampling explains
the gap, the check is rerun properly and commissioning can finish; if not, the failure stands, that kit is recorded as not
trustworthy across laboratories, and the instrument's scope shrinks to what passed.

## Outcome, scored once under the rule committed 2026-10-12: MIXED (2026-10-13)

**Read.** 15 parts (five libraries x three positions in the file: 0, 40, 80 % through), aligned on a rented machine with the three
committed scripts carried with the job and their checksums printed, then read by the chain's Stage Q
(`read_parts_01.py`, committed before any part was read; output `cause_parts_01.csv`). Each part file was read whole first, so a
truncated or doubly written part could not reach the decision. Decision: `decide_cause_01.py` -> `cause_decision_01.json`.

| library | line | r at part 0 | r at 40 | r at 80 | call |
|---|---|---|---|---|---|
| SRR13051128 | HG007 | 0.8578 | 0.9791 | 0.9591 | MIXED |
| SRR13051256 | HG006 | 0.8740 | 0.9775 | 0.9635 | MIXED |
| SRR13051268 | HG004 rep2 | 0.8912 | 0.9870 | 0.9872 | FILE-PART |
| SRR13051263 | HG005 (control) | 1.0012 | 0.9870 | 0.9818 | - |
| SRR13051136 | HG003 (control) | 0.9988 | 1.0130 | 1.0182 | - |

**What it shows.** All three suspect libraries read 11 to 14 % low at the START of the file and recover to within 1 to 4 % deeper in,
while both controls sit flat at about 1.00 throughout. The laboratory gap is therefore largely an artifact of reading only the head of
each file - where a sequencer's first cycles are least settled - and not a property of those libraries. It is called MIXED rather than
FILE-PART because the sealed rule required all three inside +-3 % at parts 40 and 80, and two land just outside at 0.959 and 0.964.
The tolerance was not loosened after seeing the numbers.

**The reproduction check passed.** Each library's part-0 copy error reproduces its sealed Stage B value to 0.0 % relative, all five.
The carried-script run read the same files the same way as the commissioning run, so this diagnosis is about the data and not about a
changed pipeline.

**What follows, under this note's stopping rule.** Across laboratories stays NOT MET for EMSeq and IAM-A commissioning stays HELD. The
cause is now known and the remedy is a pipeline rule rather than a mystery: read spread across the file rather than from its head. That
rule must be written, exercised on planted data and sealed BEFORE the check is rerun, and the rerun then needs the whole stage re-read
on the new rule - not only these five libraries, which have now been spent on this diagnosis.

**In plain words.** Our second instrument failed one of its five checks: the same reference DNA, prepared with the same kit in two
different laboratories, did not read the same. Rather than adjust anything, we asked why. We took the three libraries that disagreed,
plus two that agreed, and read each one from three different places in its data file. The three disagreeing ones all read far too good
at the very beginning of the file and then settled down to nearly normal further in; the two controls read the same everywhere. So the
disagreement came mostly from reading only the first part of each file, which is the least reliable part of a sequencing run. **What it
would take to finish:** write the rule that reads across the whole file, test it on data with a known answer, seal it, and then read the
whole stage again. Until that happens the instrument stays held, and the failed check stays failed.

**Not a milestone:** a diagnosis of a held check, not a result about cells or about the law.

