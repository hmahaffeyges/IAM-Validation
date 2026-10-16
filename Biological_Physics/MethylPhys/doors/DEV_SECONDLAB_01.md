# DEV-SECONDLAB-01: IAM-A's self-reference line in a second laboratory (PHY-047), with the meter's own error (PHY-050)

**Status: rule, bars and samples committed 2026-10-15 before any ENCODE read is counted. Development.** One of the two gates left before
IAM-A's grand sealed test (COM-IAMA-052).

**In plain words.** On 2026-10-15 we found that a healthy cell's tagging mistakes follow one straight line set by how much of its DNA it keeps
loose, across 56 kinds of healthy cells, all from one laboratory (Loyfer). If that line is physics, a different laboratory, with different
machines, software and donors, should land its healthy cells on the same line. This test reads five kinds of healthy blood cells sorted by
the ENCODE project (Bernstein laboratory) and asks whether they do. The same reads also measure the sequencer's own error, the correction
IAM-A needs before its grand test.

**Data.** ENCODE whole-genome bisulfite alignments (gemBS, GRCh38), one file per cell, public (`data/DEV_SECONDLAB_01/samples.tsv`): B cells
(ENCFF092SBB), T cells (ENCFF605NRU), NK cells (ENCFF133QNK), CD14 monocytes (ENCFF704SCQ), CD34 myeloid progenitors (ENCFF969NGH).
Reference: ENCODE's GRCh38_no_alt_analysis_set_GCA_000001405.15.

**Prior reading:** PROC-CHANNEL-01 (2026-10-10) read ENCODE alignments only for the sequencing error (work_errors.csv); no loose share and no
held error of these cells was ever computed. Loyfer's 56 cells set the line; nothing from ENCODE helped set it.

**Counter.** `data/DEV_SECONDLAB_01/secondlab.py`: the own-read counter of `development/loose_share_2026-10-15/` re-implemented for gemBS
alignments (the call comes from the read base against the reference on the read's conversion strand). Molecules are read pairs; alternate
molecules give the loose share (2,000-CpG tiles, loose below 0.70, tiles with at least 200 calls) and the held error (molecules with at least
6 calls, at least 80 % methylated, isolated unmethylated interior calls). Sequencing error: A/T-reference mismatches on 1 read in 20.
Synthetic MET (`secondlab_synthetic.txt`): planted loose share, held error and sequencing error recovered on both strands; the first synthetic
compared against a formula that did not apply to its short molecules and then read past its planted span, both corrected before any data.

**Synthetic test:** DEV_SECONDLAB_01/secondlab.py, DEV_SECONDLAB_01/secondlab_synthetic.txt

**The line (fixed).** Held error = 0.0298 + (0.0555 - 0.0298) x loose share (own reads, every CpG; `opensea_line.json`).

**Bar (development, showcase).** S1: at least 3 of the 5 cells' held error within 0.003 of the line at their own loose share (Loyfer's own
cells: 41 of 56 within 0.003). Reported, not scored: each cell beside the same Loyfer cell type; held error with the measured sequencing error
subtracted (PHY-050).

**Disclosed differences.** GRCh38 tiles instead of hg19 tiles; read pairs merged by the counter instead of wgbstools; read ends trimmed by 3
bases; a different sequencer whose error is measured and reported. Four of the five cells come from one donor.

**What it would take to finish.** The five files counted once on a rented machine, the bar read, and the result recorded here.

## Outcome (2026-10-15): S1 as first scored 3 of 5; CORRECTED below to NOT MET (0 of 5) like for like

Counted once on spot machine i-0ae982785e17a071d (code 4a087820); results `data/DEV_SECONDLAB_01/out/`, scored `secondlab_scored.json`,
side by side `secondlab_side_by_side.csv`.

| ENCODE cell | loose share | held error | the line predicts | residual | within 0.003 | sequencing error |
|---|---|---|---|---|---|---|
| B cells | 0.337 | 0.0329 | 0.0384 | -0.0056 | no | 0.0032 |
| T cells | 0.379 | 0.0375 | 0.0396 | -0.0020 | yes | 0.0031 |
| NK cells | 0.253 | 0.0358 | 0.0363 | -0.0004 | yes | 0.0033 |
| CD14 monocytes | 0.329 | 0.0340 | 0.0383 | -0.0043 | no | 0.0032 |
| CD34 myeloid progenitors | 0.106 | 0.0325 | 0.0325 | +0.0000 | yes | 0.0041 |

**Reported, not scored.** (1) Beside Loyfer's same cell types the ENCODE loose shares are about six times higher (B cells 0.337 against
0.055, monocytes 0.329 against 0.045, NK 0.253 against 0.058, T cells 0.379 against CD4 T 0.048) and the held errors higher too (0.033-0.038
against 0.028-0.029): the two laboratories' readings differ in scale, and the cells moved roughly along the line rather than sitting at
Loyfer's positions. Every residual is at or below the line (median -0.0020). (2) The meter's sequencing error is 0.0031-0.0041 per base on
A/T positions; subtracted, the held errors read 0.028-0.034 (PHY-050).

**What it means.** The line held in a second laboratory by the bar written before the count. What it does not yet show is that one
laboratory's loose share means the same as another's: the next step is to find why ENCODE's loose share reads six times higher on the same
cell types (candidates: GRCh38 against hg19 tiles, the two pipelines' read filtering, a global methylation offset), with Loyfer files
counted by this same extractor as the bridge.

## Correction (2026-10-15, the same evening): the first scoring compared unlike quantities

The line's held errors are scale-matched: `opensea_line.py` multiplies every own-read held error by 0.7412824486 (the committed
`own_reads_genome_cells.csv` constant, gerr_s / gerr, matching PROC-CHANNEL-01's scale). The rule above named the line but not that
factor, and the first scoring compared ENCODE's RAW held errors with it. Found when Loyfer's own B-cell samples (raw 0.036-0.041) were put
beside the line's B-cell value (0.0282). Like for like (`secondlab_like_for_like.csv`, `.json`):

| ENCODE cell | held, raw | held x 0.741 | the line | residual |
|---|---|---|---|---|
| B cells | 0.0329 | 0.0244 | 0.0384 | -0.0141 |
| T cells | 0.0375 | 0.0278 | 0.0396 | -0.0117 |
| NK cells | 0.0358 | 0.0266 | 0.0363 | -0.0097 |
| CD14 monocytes | 0.0340 | 0.0252 | 0.0383 | -0.0131 |
| CD34 myeloid progenitors | 0.0325 | 0.0241 | 0.0325 | -0.0084 |

**Verdict like for like: NOT MET, 0 of 5.** The same holds on the raw axis (line 0.0402 + 0.0347 x L): every ENCODE cell sits well below.
On the same cell types, ENCODE reads a lower held error than Loyfer (B cells raw 0.0329 against 0.0380) with a loose share six times
higher (0.337 against 0.055): the two laboratories' loose shares are not on one scale, so the line cannot yet be carried from one laboratory
to another. This blocks IAM-A's grand test until the loose share is defined so that it reads the same in two laboratories.

**What it would take.** Find why ENCODE's loose share reads six times higher on the same cell types: ENCODE's own published per-CpG calls
for the same run (ENCFF703XLD) are being read with the same tile rule, to separate the extractor from the data.

## Diagnosis (2026-10-15): the fixed 0.70 cut turns a laboratory's overall methylation offset into a six-fold loose share

`data/DEV_SECONDLAB_01/scale_diagnosis.py` -> `scale_diagnosis.json`, the same tile rule on published per-CpG data. ENCODE's own calls for the
counted B-cell run read a loose share of 0.359 (this extractor: 0.337), so the extractor is not the cause. ENCODE's B cells are less
methylated overall than Loyfer's (genome-wide mean 0.7271 against 0.8045; tile median 0.767 against 0.811). The cut at 0.70 sits on the steep
side of the tile distribution: on Loyfer's B cell alone the share is 0.054 at 0.70 and 0.133 at 0.75. A fixed cut therefore carries each
laboratory's overall methylation level into the loose share.

**What it would take.** A loose share that reads the same in two laboratories: the cut defined from the sample itself rather than fixed at
0.70 (for example, relative to the sample's own firm level), designed and synthetic-tested first, then re-read on both laboratories' cells.
Whether ENCODE's lower overall level is biology, donor or protocol is not settled here.

**Follow-up the same evening (author chose: the cut from the sample's own firm level).** Checked first on the two B cells already read
(`scale_diagnosis.json`): the difference is the SHAPE of the tile distribution, not its level. Loyfer B 10th percentile / median / 90th
percentile 0.735 / 0.811 / 0.865; ENCODE B 0.545 / 0.767 / 0.886: the top ends match, ENCODE carries a long tail of much less methylated
tiles. A cut set from the sample's own median still leaves ENCODE about five times Loyfer's share (0.278 against 0.053). ENCODE's earlier,
deeper run of the same experiment on its earlier pipeline (ENCFF774VLD, coverage 15) shows the same tail (loose share 0.326, 10th percentile
0.539): not the extractor, not the newer pipeline, not coverage. The tail belongs to this donor's sample (biology or preparation).
**Next:** healthy B cells from a third source, to learn whose tail is typical, before the definition is chosen.

**Third source (2026-10-15): healthy B cells from another laboratory look like Loyfer's; ENCODE's sample is the outlier.** GSE49627 (Hansen
laboratory), resting B cells of three healthy donors, published per-CpG calls, same tile rule (`scale_diagnosis.json`): genome-wide
0.8222 / 0.838 / 0.8378, 10th percentile 0.762 / 0.779 / 0.782, loose share 0.031 / 0.023 / 0.021 (Loyfer B: 0.8045, 0.735, 0.054; ENCODE B:
0.7271, 0.545, 0.359). The long undermethylated tail is specific to the ENCODE donor's sample; it resembles the large hypomethylated blocks the
Hansen laboratory reported in B cells after culture and transformation. The sealed result stands as NOT MET; this explains it. Only the
loose share of GSE49627 has been read (no molecule, no held error).
