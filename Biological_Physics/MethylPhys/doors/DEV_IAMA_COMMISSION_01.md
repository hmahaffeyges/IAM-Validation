# DEV-IAMA-COMMISSION-01 — commissioning bars for IAM-A, narrow scope (STATUS section 0, R1; design 2026-10-11; no pass values yet; nothing read)

**Scope to be commissioned.** IAM-A on whole-genome bisulfite sequencing of sorted neutrophils, hg19, pipeline `loyfer_pat_v1`, Stage Q0 intake,
read against same-run healthy references (same-run tare), with the healthy position P from the three Loyfer granulocyte whole files
(DEV-IAMA-P-WHOLE-01). Nothing else: no whole blood, no RRBS, no other cell type, no other pipeline.

**Data (public only).** Identical DNA across laboratories and kits: SEQC2 EpiQC sequencing (PRJNA646948; seven lymphoblastoid reference lines,
TruSeq, Swift, EM-seq and others, two or more technical replicates; CANDIDATE_EPIQC.md). Healthy neutrophils from another laboratory: the
laboratory-G runs already read in development (GSE128731) are not independent; a further public set is searched for.
The EpiQC lines are not neutrophils: they test the instrument (repeatability, laboratory and kit offset), not the healthy neutrophil reading.
Transfer rests on DEV-IAMA-XCELL-01 (the kit offset is the same for CD4 T cells and neutrophils) and is stated as a limit.

**The five tests.** Pass values are not written here: each is set from its synthetic test (below) and committed with it, before any EpiQC
or neutrophil file is read.
| test | what is read |
|---|---|
| C1 intake | Stage Q0 conversion limit decided on identical DNA (copy error against conversion across kits), before C2-C5 (DEV-IAMA-INTAKE-01 open) |
| C2 repeatability | technical replicate libraries of the same DNA, laboratory and kit: IAM-A rep2 / rep1 |
| C3 offset removed | identical DNA across laboratories and kits, each library against the median of the other lines in its own run (same-run tare): spread of each line's tared IAM-A |
| C4 healthy reads Normal | sorted healthy neutrophils from a laboratory not used for P or the tare, same-run tared: share within the Normal band |
| C5 a known change is seen | planted loss on real healthy neutrophil files through Stage Q (measured response, insilico_loss_02.py): movement against C2's repeat spread |

**Stopping rule.** If no public healthy neutrophil set qualifies for C4, C2, C3 and C5 are still read, and the commissioning record states that a
healthy-Normal reading outside the Loyfer laboratory is untested. If any bar fails, commissioning stops there until the cause is found.

**Synthetic tests before the seal and before any EpiQC download** (push-gate rule 7):
1. Sampling floor of C2: split a real Loyfer whole file's molecules at random into two halves; the halves' IAM-A ratio at EpiQC depth is the
   smallest repeat spread any library pair can show. C2's pass value is set above it.
2. C3 resolving power: plant laboratory and kit offsets (the measured Swift/TruSeq offset, DEV-IAMA-KIT-01) on split files; the same-run tare
   must remove them to within 0.03 with seven lines per run.
3. Depth: subsample to the EpiQC run depths (from the SRA run table) and confirm Stage Q0 coverage passes.
4. Conversion: each EpiQC kit's reported conversion against the Q0 limit (C1), from the paper's tables, before download.

**Synthetic test 1 result, 2026-10-11** (`data/DEV_IAMA_COMMISSION_01/split_floor_01.py`, output `split_floor_01_output.csv`; summary `split_floor_01_summary.txt`). One healthy
granulocyte whole file (GSM5652313) split at random into halves, three seeds, thinned to 100 %, 25 %, 5 % and 1 % of its molecules, read by
Stage Q. Largest |IAM-A ratio - 1| between halves: 0.0024 (whole file), 0.0029 (25 %), 0.0024 (5 %),
0.0071 (1 %, 1,595,409 molecules per half). Counting noise is below 1 % even at 1 % depth, so
C2's pass value is set by library preparation and sequencing, not by sampling. This is the floor only; the real repeat spread comes from
technical replicate libraries (EpiQC), and C2's pass value is set after synthetic tests 2-4.

**Synthetic test 2 result, 2026-10-11 (C3 resolving power)** (`data/DEV_IAMA_COMMISSION_01/offset_tare_02.py`; outputs
`offset_tare_02_share05_summary.txt`, `offset_tare_02_share20_summary.txt`). Real healthy molecules split into two runs of seven lines (lines differ
by planted copy error, as the seven EpiQC cell lines do); the second run carries a region-clustered artefact of the kind measured for Swift against
TruSeq. Bar written before running: every line's tared IAM-A in the two runs within 0.03. Passed at both depths. With about 296,000 opportunities
per library the untared run offset was 1.0143-1.0519 (artefact 0.02) and 1.0404-1.0806 (0.04), tared difference at most 0.0277 and 0.0272. With four
times the molecules the untared offset was 1.0321-1.0415 and 1.0574-1.0716 and the tared difference at most 0.0081 and 0.0113. Most of the shallow
residual is counting noise; the same-run tare itself removes a 3-8 % run offset to about 1 %.

**Synthetic tests 3 and 4, 2026-10-11 (metadata only)** (`data/DEV_IAMA_COMMISSION_01/epiqc_design_03_04.py`; run table `epiqc_runs.csv`;
output `epiqc_design_03_04_summary.txt`). The EpiQC reads are runs SRR13050956-SRR13051274 (the paper's data statement); a project search on
PRJNA646948 returns no runs, so the run range is used. Whole-genome libraries Stage Q can read: 78 (TruSeq, MethylSeq, SPLAT, TrueMethyl BS:
one laboratory, 7 lines x 2 replicates each; EMSeq: two laboratories, 7 lines each, 4 lines with 2 replicates in each). Test 3 (depth) passed:
every library sits far above the counting-floor depth of test 1, so the box run needs only a sample of reads per library (about 2x), not the
full depth. Test 4 (conversion, reported from the paper): TruSeq's non-CpG figures put it at Q0's 0.98 limit, so C1 (the conversion rule) is
decided on identical DNA before C2-C5 are read.

**What the EpiQC design can and cannot test.** C2 (repeatability): every kit, same laboratory. C3 across kits: five kits on identical DNA in one
laboratory. C3 across laboratories: EMSeq only (two laboratories). The EpiQC lines are lymphoblastoid cell lines, not neutrophils, as stated above.

**Synthetic test 5, 2026-10-11 (conversion through Stage Q)** (`data/DEV_IAMA_COMMISSION_01/conversion_05.py`; output `conversion_05_output.csv`,
summary `conversion_05_summary.txt`). Incomplete conversion planted on real healthy molecules (every T call read as C with probability 1 - c),
read by Stage Q. Bar written before running: copy error falls by the derived factor c within 0.005. **Failed**: copy error falls about half as much
(ratio 0.9904 at c 0.98, 0.9855 at c 0.97), because Stage Q counts only isolated errors on molecules at least 80 % methylated. Dividing by c
therefore over-corrects (IAM-A 1.0080 at c 0.98); uncorrected IAM-A reads 0.9927 at c 0.98 and 0.9889 at c 0.97.

**C1, the conversion rule (draft, from tests 2 and 5; not sealed).** Conversion c of every library is measured the same way, from human non-CpG
(CHH) calls on autosomes, with the lambda spike-in as a cross-check where a kit carries one (the EpiQC kits differ: the paper states lambda for
EMSeq at one laboratory, MethylSeq and TrueMethyl). Stage Q0 keeps its 0.98 limit (ENCODE WGBS standard); no conversion correction is applied. At
the limit the measured bias, IAM-A 0.9927, is smaller than the tare residual of test 2. Because the EpiQC lines have a different methylation
landscape from neutrophils, the box run repeats test 5 on one library per line before any C2-C5 value is read; C1 holds only if that bias at
c 0.98 is within the same bound. Mapping-quality filter required: the lines carry EBV, absent from the hg19 reference.

**Draft pass values for C2-C5 (2026-10-11; for the author before the seal; nothing read).** One tolerance, 0.03, runs through all of them: it is
the bar the same-run tare met in test 2, and four times the counting floor of test 1 at the depth the box run will read.
| test | pass value (draft) | why this value |
|---|---|---|
| C2 repeatability | in every kit, at least 6 of 7 lines' replicate pairs read IAM-A rep2/rep1 within 0.03 of 1 | a repeat must not move more than the tare is required to hold; the whole-blood TruSeq repeats (DEV-IAMA-CONVERSION-01) differed by more, so TruSeq may fail, and that is recorded if it does |
| C3 across kits | each line's tared IAM-A in each kit within 0.03 of that line's median over the five kits, in at least 6 of 7 lines | the offset the tare removed to about 1 % in test 2 |
| C3 across laboratories | EMSeq only (the one kit made in two laboratories): each line's tared IAM-A at the two laboratories within 0.03, in at least 6 of 7 lines | the only two-laboratory comparison EpiQC holds |
| C4 healthy reads Normal | not testable on neutrophils (no second laboratory with 3 or more donors); recorded untested, GSE126754 a one-donor spot check; met on neurons if DEV-IAMA-NEURON-01 goes ahead | stopping rule |
| C5 a known change is seen | a planted loss of 1 % of methylated calls on every EpiQC library, read through Stage Q, moves IAM-A by more than 0.03 in every library | Stage Q's measured response on vehicle molecules (insilico_vehicle_SRR25322252.csv) |
Pass values are written before any EpiQC read; the seal follows the scorer and its planted test.

**Scorer and its planted test, committed before the seal.** `data/DEV_IAMA_COMMISSION_01/score_iama_commission_01.py` fixes every rule: a library
is kit x laboratory x line x replicate (pooled runs); evaluable means Stage Q0 PROCEED and conversion >= 0.98; C1 holds when the planted-0.98
bias is >= 0.99 in every checked library, else C2-C5 are not scored; the same-run tare needs >= 3 references; the k-of-n rule passes a group of
n >= 4 when at least n-1 are within TOL = 0.03 (n < 4 not assessed); a kit enters the commissioned scope only on C2, C3-kits (EMSeq also
C3-labs), C1 and C5. `planted_scorer_test_01.py` (output `planted_scorer_test_01_output.txt`): nine synthetic library tables, each tripping one
rule or none, all giving the stated verdict -> PASSED. The seal guard was tested both ways (no seal and uncommitted seal refuse; a committed seal
with the unchanged scorer opens; a changed scorer refuses).

**Synthetic test:** `data/DEV_IAMA_COMMISSION_01/planted_scorer_test_01.py`, output `data/DEV_IAMA_COMMISSION_01/planted_scorer_test_01_output.txt`.

**How the libraries are read (fixed before the seal, 2026-10-11).**
1. Runs: `epiqc_libraries_runs.csv`. One sequencer per tare group (kit x laboratory x replicate); NEB covers every group, so every library is
   read from its largest NEB-sequenced run. The first 10,000,000 read pairs of that run are streamed from ENA FASTQ.
2. Pipeline: Box Run 10 (`boxruns/run10_epiqc/session_epiqc.sh`), the pinned Box Run 2 pipeline unchanged. Conversion is
   `conversion_rate_human_chr1` from `alignment_qc.py --both` (human chr1 CHH for every library); lambda is recorded as a cross-check only.
3. q0 is Stage Q0 checks Q0.1-Q0.6. Q0.7 (specimen) is not applied, because no reading here is made against a healthy position P.
4. eps_planted: 1 % of methylated calls lost (insilico_loss_02.py rule). c1_iama: conversion_05.py rule at c 0.98. Both are computed on the laptop
   from each library's .pat file.
5. Stage A: the 7 TruSeq replicate-1 libraries (`epiqc_stageA_runs.csv`), plus a 1,000,000-pair format check of one library of each other kit
   (`epiqc_pilot_runs.csv`). Format checks are not scored.
6. **Stop rule.** Stage B (the other 71 libraries) runs only if C1 is met on Stage A. A kit whose format check maps below 70 % of read pairs
   (samtools flagstat) is not run in Stage B and is recorded as not readable by the pinned pipeline.
7. Stage B depth: 10,000,000 pairs. If Stage A's median opportunities are below 2,400,000 (twice the 1,180,000 at which synthetic test 2 held the
   tare to about 1 %), Stage B uses 10,000,000 x 2,400,000 / median pairs, capped at 25,000,000. The depth is never lowered.

