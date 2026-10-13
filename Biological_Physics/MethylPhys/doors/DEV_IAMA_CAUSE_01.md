# DEV-IAMA-CAUSE-01 — why three EMSeq LAB02 libraries read low (development; design 2026-10-12; nothing read yet)

**Why.** DEV-IAMA-COMMISSION-01 held commissioning under its stopping rule: across laboratories was not met because three EMSeq LAB02 libraries
(SRR13051256, SRR13051128, SRR13051268) read copy error below every other EMSeq library; they are the three most deeply sequenced EMSeq runs, and
the pipeline reads the start of each file (values in `DEV_IAMA_COMMISSION_01_OUTCOME.md` and `data/DEV_IAMA_COMMISSION_01/why_iama_commission_01_output.txt`).
This is a development read to find the cause; it does not re-score anything.

**Question.** Does the low copy error belong to the library, or to the part of the file read?

**Design.** The three low runs and two that read normally in the same laboratory (SRR13051263, SRR13051136), each read from three parts of the
run (start, middle, end, by SRA spot range) through the same pipeline, with read length, base quality, duplicate fraction and conversion recorded
per part. The read script, its planted test and the decision rule are written and committed before the run; the run is entered in the ledger first.
**Not a milestone:** design (development).
