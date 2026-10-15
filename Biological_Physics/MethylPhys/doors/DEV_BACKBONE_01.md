# DEV-BACKBONE-01: does a tumor's backbone read worse than the writer's nameplate? (rule written 2026-10-15; run once 2026-10-15)

**Status: run once 2026-10-15 with the author's go; verdict INCONCLUSIVE by the written rule (tumor side MET, healthy side not at nameplate).** The author's step 4 (whiteboard, 2026-10-15): run once, and only after steps 1 to 3 hold up on paper.

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


## Island arm, the counts (2026-10-15; reported, not scored)

The six pairs re-aligned (box job 4110eb64 and one rerun after AWS reclaimed a spot machine; first 10 million reads per sample, unchanged
committed script) and counted with PROC-CHANNEL-01's rule (`data/DEV_BACKBONE_01/island_counts.csv`, summary `island_counts_summary.json`).
Healthy island range across 56 cell types: 0.0094-0.0162.

| pair | normal | tumor | tumor over own normal | sequencing substitution error, normal / tumor |
|---|---|---|---|---|
| CRC1 | 0.0141 | 0.0102 | 0.72 | 0.0018 / 0.0064 |
| CRC2 | 0.0120 | 0.0168 (above range) | 1.40 | 0.0019 / 0.0020 |
| CRC3 | 0.0198 (above range) | 0.0223 (above range) | 1.13 | 0.0020 / 0.0021 |
| CRC4 | 0.0155 | 0.0210 (above range) | 1.35 | 0.0021 / 0.0022 |
| CRC5 | 0.0135 | 0.0175 (above range) | 1.30 | 0.0037 / 0.0022 |
| CRC6 | 0.0098 | 0.0103 | 1.05 | 0.0065 / 0.0068 |

**What it shows.** 4 of 6 tumors read above the healthy island range (1 of 6 normals does too, CRC3); 3 of 6
pairs read more than 1.2 times their own normal (the synthetic's planted 2x read 1.91-1.97; unchanged islands 0.98-1.02); median ratio
1.211. Islands are written over in some of these tumors, not all, and by less than 2x. CRC1 runs the other way; its
tumor's sequencing substitution error is 3.5 times its normal's, so read quality, not the islands, is the first thing to check there.
CRC5 is the pair flagged for a conversion difference. Six pairs, one laboratory, purity unknown: development. The firm-territory test
(bars N, T1, T2 above) has still not been run; it waits for the author's go, as written.


## Outcome (2026-10-15): INCONCLUSIVE by the rule; the tumor's excess survives on firm territory in 6 of 6

Run once with the author's go (code `data/DEV_BACKBONE_01/backbone_01.py` at commit 69b8a77f; rows `backbone_01_rows.csv`, verdict
`backbone_01_verdict.json`, summary `backbone_01_summary.json`). PROC-TUMOUR-01's genotype mask and sequencing-error subtraction, unchanged.

| pair | normal, firm | tumor, firm | tumor over own normal | separation (standard errors) |
|---|---|---|---|---|
| CRC1 | 0.03427 | 0.04643 | 1.355 | 21.5 |
| CRC2 | 0.03487 | 0.04196 | 1.203 | 10.2 |
| CRC3 | 0.03194 | 0.03791 | 1.187 | 16.4 |
| CRC4 | 0.03176 | 0.03561 | 1.121 | 12.7 |
| CRC5 | 0.03606 | 0.03863 | 1.071 | 3.3 |
| CRC6 | 0.03742 | 0.04126 | 1.103 | 11.7 |

- **T1, MET (6 of 6, bar 5):** every tumor carries more held error than its own normal on firm territory alone, median 1.154 times,
  every pair separated by at least 3.3 standard errors. The tumor excess is not only a change of territory: it is on the backbone itself.
- **T2 (6 of 6 above 0.0226):** met as counted, but read with N.
- **N, not met (0 of 6):** the normals read 0.0318-0.0374 on firm territory, above the healthy range 0.0136-0.0226. By the rule this
  makes the verdict INCONCLUSIVE: the nameplate range cannot serve as the reference for these samples.

**Why N likely failed (a hypothesis, not tested here).** The healthy range comes from PROC-CHANNEL-01, read on Loyfer's sorted cells with
its own counter and scale; these samples are counted with PROC-TUMOUR-01's isolated-error counter on a different laboratory's tissue.
The own-read work of 2026-10-15 measured that the two scales differ by a constant (0.741, own_reads_genome_cells.csv). The scale is not
applied here, because the rule did not name it. A fair nameplate comparison needs a healthy range counted with this counter.

**What it would take to finish.** The healthy firm-territory range counted with PROC-TUMOUR-01's counter on healthy tissue (the 56 cell
types' own reads), written before it is compared; then N and T2 are read against it.

**In plain words.** In six young patients with colorectal cancer, the tumor's DNA holds its firmest marks less well than the same
patient's healthy colon, every time, and clearly. That is the tumor signal the physics points to, seen on the part of the DNA that is
supposed to be held hardest. Whether the healthy colon itself sits exactly at the writer's limit could not be judged yet, because the
healthy reference was counted on a different scale.
