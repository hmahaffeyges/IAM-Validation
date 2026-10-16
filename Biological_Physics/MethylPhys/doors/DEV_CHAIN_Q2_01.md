# DEV-CHAIN-Q2-01: IAM-A's self-reference moved into the chain (Stage Q2), with stamped outputs and a gate

**Status: synthetic MET 2026-10-15. Development.** Manifest OTH-057. Author's rule, 2026-10-15: side scripts are for exploration; any
synthetic test or test that is logged and pushed must run through the chain.

**In plain words.** Until today, IAM-A's newest part (reading a sample's loose share, counting its held error, and comparing it with the
line) ran in separate scripts beside the chain. One of them copied the line's numbers without a conversion factor built into them, and a
test was first scored as a pass when it had failed. Now that part lives in the chain itself, the line is stored in the same units the
counter writes so no conversion exists, every result the chain writes carries a stamp, and the push gate refuses any new test that did not
run through the chain.

**What moved.** `chain/stage_q2_selfref.py`: the .pat reader (exact port of `development/loose_share_2026-10-15/own_reads_loose.py`), the
aligned-read reader for gemBS (XB) and Bismark (XM) alignments (port of `doors/data/DEV_SECONDLAB_01/secondlab.py`, plus XM), the loose share
(2,000-CpG tiles, loose below 0.70), the held error, the line (built by `build_line`, stored as a stamped runtime matrix on the raw axis,
refused if edited), `predict` and `residual`, and the stamp.

**Synthetic test:** DEV_CHAIN_Q2_01/q2_synthetic.py, DEV_CHAIN_Q2_01/q2_synthetic.json

Result (`q2_synthetic.json`, stamped): T1 the chain's .pat reader gives the development counter's loose share and held error to the last
digit on the same molecules and recovers planted loose shares 0.2, 0.5, 0.8 (first version placed the held-error molecules inside two
planted tiles; moved off the tiles, the chain was not changed); T2 the aligned-read reader recovers planted loose share, held error and
sequencing error for gemBS and Bismark, both strands; T3 a planted line is rebuilt, a sample is read against it, and a line file edited after
the chain wrote it is refused; T4 the stamp verifies.

**The gate.** `CANON/chain_gate.py` (rule 23b), run by checked_push only after its seven controls (`CANON/chain_gate_controls.py`) all give
their stated answer: the DEV-SECONDLAB-01 pattern, typed line numbers, an unstamped result, a hand-edited result, a result from an
uncommitted chain and a scratch/ citation are refused; a correct test is accepted. Committed chain versions are listed in
`CANON/chain_versions.json` (checkpoint.sh registers each).

**What it would take to finish.** The line rebuilt by the chain from Loyfer's 153 samples (a box run of `count_pat`), checked against the
development line, then stored as `selfref_line_v1.json`; the ENCODE five recounted through the chain; then the second-laboratory test on
GSE49627 run through the chain from the start.

## The line rebuilt by the chain (2026-10-15)

Loyfer's 153 samples counted by chain Stage Q2 (`out/GSM*.json`, stamped; spot machine i-0f3380b0fa6d2bd0a, code 0f90bfce, chain
e49788e3db38; 13 samples GEO refused with server-busy errors were counted by the same committed chain on the laptop). Every count equals the
development count to machine precision (loose share, held error, opportunities). `build_line_v1.py` verifies every stamp, builds the line with
`Q2.build_line` and stores it as `chain/Runtime Matrices/IAM_A_SelfRef/selfref_line_v1.json` on the counter's raw axis (no scale factor).
It agrees with the development line read from its own file (`line_v1_vs_development.json`: differences within the old file's printed
rounding). This is a reproduction, not a new test; no verdict changes.

The ENCODE five (`out/ENCFF*.json`, chain Stage Q2 count_bam, identical to `secondlab.py`'s counts) read against the chain's line
(`encode_vs_line_v1.json`): all five below the line, residuals -0.011 to -0.019 on the raw axis, the same verdict as the corrected
DEV-SECONDLAB-01 (NOT MET; explained by that sample's undermethylated tail, PHY-074).

## Step 3: the tile shape, for the sample-shape check (2026-10-15)

**Prior reading:** GSE49627 (Hansen resting B cells) was read for its tile shape only, for the DEV-SECONDLAB-01 scale diagnosis (2026-10-15); it is not counted in this record and no held error has been computed on it.

Chain Stage Q2 now also stores each sample's tile percentiles (tile_p10, tile_p50, tile_p90 of the 2,000-CpG tile means), so the
sample-shape check (COM-IAMA-053) can be built from chain output. Synthetic T5 (`q2_synthetic.py`): the percentiles equal those of the
planted tile means and under-covered tiles are left out; T1-T5 MET. A range rule was checked first and does not work: ENCODE's loose shares
(0.11-0.38) lie inside the line's own range (0.03-0.93), so the check must compare a sample with healthy cells of its own type.
Loyfer's 153 samples and the ENCODE five are counted again through this chain version (box job, `results/DEV_CHAIN_Q2_01/shape/`); held
error and loose share must reproduce step 2 exactly. The shape rule is then drafted from the healthy evidence, shown to the author, and
sealed before any Hansen (DEV-SECONDLAB-02) count.

**The sample-shape check (COM-IAMA-053), sealed 2026-10-15** (author: "KISS, don't make our bars too high ... what is best for IAM"; drafted from the chain's recount, sealed as drafted). A sample is read only if its tile 10th percentile is no more than the margin below the lowest healthy sample of its own cell type. The margin is the 90th percentile of the healthy within-type spread of the tile 10th percentile (0.055 on 146 of Loyfer's samples; recomputed by the chain on all 153 before use, never typed). A cell type with no healthy reference is not read. Code: chain Stage Q2 `build_shape_reference`, `shape_check`; synthetic T6 MET. On the chain's ENCODE counts: B cells, monocytes, T cells and NK cells fall 0.075-0.162 below their healthy types' minimum and are not read; CD34 progenitors have no healthy reference and are not read. Disclosed: Hansen's B-cell tile shape (prior reading, processed files) sits above Loyfer's healthy B-cell minimum; the margin was set from Loyfer's spread alone.
