# DEV-BACKBONE-01: does a tumor's backbone read worse than the writer's nameplate? (rule written 2026-10-15; PREPARED, NOT RUN)

**Status: prepared, not run.** The author's step 4 (whiteboard, 2026-10-15): run once, and only after steps 1 to 3 hold up on paper.

**The question, in plain words.** Healthy cells hold their firmly held territory at the writer's own limit (whiteboard part 2). Tumors widen
their loosely held territory, and on its own that only moves a sample along its own line. The cancer signal on this picture is different:
the firm part itself reading worse than nameplate. This test asks whether a tumor's firmly held territory carries more error than the same
patient's normal tissue, and more than healthy cells ever read there.

**Code:** `data/DEV_BACKBONE_01/backbone_01.py` (its constants and their sources: `backbone_01_constants.json`). The real mode refuses to run
until the author's go is committed.

**Synthetic test:** DEV_BACKBONE_01/backbone_01.py, DEV_BACKBONE_01/backbone_01_synthetic.txt

Run once 2026-10-15: planted tumors worse than nameplate read MET; planted pairs with no tumor excess on firm territory read NOT MET;
SYNTHETIC: MET.

**Data (already held, nothing downloaded).** The six true tumor/normal pairs of early-onset colorectal cancer (GSE284325, WGBS, PROC-TUMOUR-01),
per-site tables in our S3 (downloads/tumour_work/out/EOCRC_*_WGBS_*.parquet: m, t, opp_A, err_A, opp_B, err_B). Development use of spent data.

**Definitions, fixed now, taken unchanged from earlier records.**
- Firm and loose territory: DEV-ARCH-01's cut. 100 kb bins; a bin is loose if its methylation is below 0.70. Here the bin methylation is
  sum(m)/sum(t) over the six normals pooled (the tumors never define the territory). Bins with pooled t < 50 are left out.
- Held error: (err_A + err_B)/(opp_A + opp_B) in the bin set, with PROC-TUMOUR-01's corrections unchanged (sequencing-error subtraction,
  per-patient genotype mask).
- Healthy range on firm territory: 0.0136 to 0.0226, what 56 healthy cell types read on the sites every cell keeps fully methylated
  (PROC-CHANNEL-01 section 7). Reason: that is the range the nameplate claim rests on.

**Bars, each with its reason (LESSONS C15).**
- N (healthy side): each normal's firm-territory error inside the healthy range, allowing 2 standard errors of its own count, in at least
  5 of 6 normals. Reason: the claim is that healthy tissue reads at nameplate on firm territory; one of six may miss on purity or field effect.
- T1 (tumor above its own normal on firm territory): at least 5 of 6 pairs. Reason: PROC-TUMOUR-01 found 6 of 6 genome-wide; the question is
  whether it survives restriction to firm territory.
- T2 (tumor above the healthy range on firm territory): at least 4 of 6. Reason: "worse than nameplate" means outside where healthy cells read;
  tumor purity is unknown and dilutes this, so a third may fall short.
- MET: N, T1 and T2. NOT MET: T1 fails (the tumor excess is only a change of territory, which the author's picture says is not the signal).
  INCONCLUSIVE: N fails (the normals are not at nameplate, so nameplate cannot be the reference here), or fewer than 5,000 qualifying
  opportunities on firm territory in more than two samples.

**Reported beside the verdict, not scored.** The loose-territory error and loose share of each sample; each pair with and without CRC5
(conversion difference 0.0051, just over PROC-TUMOUR-01's 0.005 limit).

**Island arm (added 2026-10-15, before any tumor island count exists).** The tumor's island gain error over its own normal's, counted
with PROC-CHANNEL-01's rule, shown beside the healthy island range (0.0094-0.0162). The counts were not saved for these samples; they need
the raw reads re-aligned (approved by the author). Synthetic (`data/DEV_BACKBONE_01/backbone_01_island.py`, `backbone_01_island_synthetic.txt`):
planted tumors with islands written over 2x read 1.91-1.97x their normals in all six pairs; planted unchanged islands read 0.98-1.02x.

**Limits.** Six pairs from one laboratory; purity unknown; the data were read genome-wide before (PROC-TUMOUR-01), so this is development,
not a sealed test.

**In plain words.** Healthy cells seem to hold their firmly held DNA exactly as well as the copying enzyme allows. If that is right, the
earliest sign of cancer is not more loosely held DNA (healthy designs vary in that too) but the firmly held part slipping below what the
enzyme guarantees. This test will look for that slip in six colorectal cancer patients, each against their own normal tissue, by a rule
written before the look. What it would take to succeed: the normals reading inside the healthy range and the tumors above both their own normal
and that range, by the bars above; the test is finished when it has run once and its verdict is recorded.
