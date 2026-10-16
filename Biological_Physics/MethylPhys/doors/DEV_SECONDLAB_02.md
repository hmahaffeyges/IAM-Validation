# DEV-SECONDLAB-02: IAM-A's self-reference line in a second laboratory, on healthy resting B cells (GSE49627, Hansen lab)

**Status: alignment running; rule S1 sealed; shape rule to be added before the count. Nothing has been counted.** Manifest PHY-047, PHY-074. Replaces the ENCODE sample of
DEV-SECONDLAB-01, whose B cell carries an undermethylated tail that healthy B cells from two other laboratories do not (PHY-074).

**Prior reading:** GSE49627's processed per-CpG files of the three resting B cells were read for their tile shape only (loose share and
tile percentiles), for the DEV-SECONDLAB-01 scale diagnosis on 2026-10-15. No held error has been computed on any GSE49627 sample.

**In plain words.** IAM-A reads a sample against its own loose share, using one line measured on healthy cells in one laboratory. This
test asks whether that line holds for healthy B cells from a different laboratory. The bar was fixed from the line's own scatter before any of
these cells is counted. What it would take to succeed: at least two of the three donors pass the shape check and read within 0.0067 of
the line.

**Step 1 (running): alignment only.** `data/DEV_SECONDLAB_02/box/align_hansen.sh`: full runs SRR948844, SRR948845, SRR948846 from NCBI's open copy in AWS (fasterq-dump; ENA served them at about 0.26 MB/s, so the first machine, i-08f7918ad0a721c4c, was stopped after 35 minutes with nothing downloaded),
Trim Galore, Bismark paired-end on hg38 (the tumor work's tools and index), deduplication, sorted and indexed BAMs to S3. No count. Run 2 (i-0629519b42e52ed71) produced no trimmed reads (tools unpacked outside the path they were built for); run 3 unpacks them where they were built, runs a 100,000-pair preflight first and uploads every log.

**Step 2: the rule (sealed 2026-10-15 by the author, "Seal as drafted"; before any count).**

- **S1:** at least 2 of the 3 donors read within +/- 0.0067 of the chain's self-reference line (version 1,
  `chain/Runtime Matrices/IAM_A_SelfRef/selfref_line_v1.json`). The bar is the 90th percentile of the line's own scatter over Loyfer's 56
  cell types, computed by the chain (`data/DEV_SECONDLAB_02/bar_secondlab02.py` -> `bar_secondlab02.json`, stamped), not chosen.
- **Reported, not scored:** the lab shift, each donor's residual against Loyfer's own B cells' residual.
- **Applied first:** the sample-shape check (COM-IAMA-053), sealed 2026-10-15 in DEV_CHAIN_Q2_01.md: tile 10th percentile no more than the
  chain-computed margin below the lowest healthy Loyfer B cell; a donor it refuses is not scored, and with fewer than
  2 donors left the test reads NOT ASSESSED rather than MET or NOT MET.
- Synthetic test: chain Stage Q2 count_bam on planted Bismark (XM) paired alignments, T2 of the chain synthetic (MET), plus the read length
  of these runs (checked from the first aligned BAM before the count).

**Synthetic test:** DEV_CHAIN_Q2_01/q2_synthetic.py, DEV_CHAIN_Q2_01/q2_synthetic.json

**Step 3: the count**, chain Stage Q2 count_bam only, stamped, then scored once by the committed rule.
