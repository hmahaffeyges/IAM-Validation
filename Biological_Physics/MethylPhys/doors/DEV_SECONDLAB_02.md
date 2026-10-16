# DEV-SECONDLAB-02: IAM-A's self-reference line in a second laboratory, on healthy resting B cells (GSE49627, Hansen lab)

**Status: alignment running; rule not yet written. Nothing has been counted.** Manifest PHY-047, PHY-074. Replaces the ENCODE sample of
DEV-SECONDLAB-01, whose B cell carries an undermethylated tail that healthy B cells from two other laboratories do not (PHY-074).

**Prior reading:** GSE49627's processed per-CpG files of the three resting B cells were read for their tile shape only (loose share and
tile percentiles), for the DEV-SECONDLAB-01 scale diagnosis on 2026-10-15. No held error has been computed on any GSE49627 sample.

**Step 1 (running): alignment only.** `data/DEV_SECONDLAB_02/box/align_hansen.sh`: full runs SRR948844, SRR948845, SRR948846 from ENA,
Trim Galore, Bismark paired-end on hg38 (the tumor work's tools and index), deduplication, sorted and indexed BAMs to S3. No count.

**Step 2 (before any count): the rule.** Written and committed after step 1 and before step 2's count: the bar against the chain's line
(`chain/Runtime Matrices/IAM_A_SelfRef/selfref_line_v1.json`), the synthetic test through chain Stage Q2 count_bam on Bismark alignments of
this read length, and the sample-shape check (COM-IAMA-053) applied first.

**Step 3: the count**, chain Stage Q2 count_bam only, stamped, then scored once by the committed rule.
