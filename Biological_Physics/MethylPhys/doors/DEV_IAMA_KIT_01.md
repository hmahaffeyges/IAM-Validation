# DEV-IAMA-KIT-01 — is the Swift kit's high IAM-A a read-end artefact? (development, 2026-10-09)

**DEVELOPMENT - not commissioned.** Written before the clipped readings below were made.

**Observation (session 2).** Donor 6's neutrophils: TruSeq / HiSeq X IAM-A 1.047 (Normal); Swift / HiSeq X 1.167 and Swift / NovaSeq 1.159.
Swift Accel-NGS libraries add a low-complexity tail at one read end, and the bases next to it read artificially unmethylated. At a truly
methylated site that is an isolated T: exactly what IAM-A counts as a copy error.

**Test.** The two HiSeq X BAMs of donor 6 (SRR9888332 Swift, SRR9888333 TruSeq) through wgbstools 0.1.0 bam2pat with `--clip` 0, 10 and
15 (ignore the first and last N bases of every read; wgbstools' own option for biased read ends), then Stage Q with P v2. The M-bias table
(`--mbias`) is written at clip 0.

**Reading rules (set now).**
1. Read-end artefact if: clipping lowers Swift IAM-A by more than its two halves differ, toward TruSeq, AND changes TruSeq by less than 0.01.
2. If both move together, clipping removes real molecules' ends in both, and P (measured on Loyfer files with no clip) cannot be compared;
   recorded, no conclusion.
3. Whatever the result, no chain setting changes without the author: a clip in the pipeline changes what P was measured on.
