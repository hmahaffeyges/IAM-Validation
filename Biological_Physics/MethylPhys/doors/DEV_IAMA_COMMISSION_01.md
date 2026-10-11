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

