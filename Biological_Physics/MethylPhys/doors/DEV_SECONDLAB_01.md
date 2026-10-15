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

**The line (fixed).** Held error = 0.0298 + (0.0555 - 0.0298) x loose share (own reads, every CpG; `opensea_line.json`).

**Bar (development, showcase).** S1: at least 3 of the 5 cells' held error within 0.003 of the line at their own loose share (Loyfer's own
cells: 41 of 56 within 0.003). Reported, not scored: each cell beside the same Loyfer cell type; held error with the measured sequencing error
subtracted (PHY-050).

**Disclosed differences.** GRCh38 tiles instead of hg19 tiles; read pairs merged by the counter instead of wgbstools; read ends trimmed by 3
bases; a different sequencer whose error is measured and reported. Four of the five cells come from one donor.

**What it would take to finish.** The five files counted once on a rented machine, the bar read, and the result recorded here.
