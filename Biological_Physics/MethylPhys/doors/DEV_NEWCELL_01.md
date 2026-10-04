# DEV-NEWCELL-01 - the new-cell rule (three tests) applied to the next cell after neutrophils (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Rule (author decision D), written into the SOP.** A cell type is read only after it passes three tests on chain v3:
1. **Purified-cell Normal.** Purified healthy arrays of the cell from laboratories other than the floor's, median tare against >= 3 same-series
   arrays of the cell (self excluded): >= 95 % of tared readings in Normal (0.95-1.05).
2. **Replicate spread.** Repeated arrays of the same DNA or person: within-person SD of tared A <= 0.020.
3. **Identifiability.** (a) Against the cell's floor, every purified healthy array of every other blood group reads outside Normal (>= 99 % of them);
   (b) the composition stage recovers the cell's fraction within RMSE 0.03 on a held-out mixture truth set.

**Next cell: monocytes** (the next myeloid cell; floor `metA_floors_v1_2_ALLCELLS_development.json`, EPIC `monocytes`). Also run on B cells (round-1
candidate) for the record.
- Test 1 sets: sorted monocytes / B cells labelled healthy, outside GSE110554 / GSE167998 / GSE181034 (GSE122244, GSE180130, GSE118144 and others in the bucket).
- Test 2: no repeated purified monocyte arrays of one person exist in the bucket. The GSE250556 whole-blood replicates are read for the cell against
  the composition-matched expectation (as DEV-PERCELL-01), stating that Stage M's read line (fraction >= 0.20) would withhold the reading.
- Test 3a: purified arrays of the other groups in the bucket (GSE110554 and other laboratories). Test 3b: DEV-NILC-01 H1 NNLS8 RMSE for the cell
  (round-1 record); no new held-out set exists (DEV-COMPOSITION-TRUTH-02).

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box job 5bd710b8. Records: `data/DEV_NEWCELL_01/`.

| cell | 1 purified-cell Normal (bar >= 95 %) | 2 replicate spread (bar <= 0.020) | 3a other groups outside Normal (bar >= 99 %) | 3b H1 RMSE (bar <= 0.03) |
|---|---|---|---|---|
| monocytes | 11/13 (84.6 %) | 0.0320 (fraction median 0.055) | 196/202 (97.0 %) | 0.0112 |
| B | 16/22 (72.7 %) | 0.0368 (fraction median 0.050) | 175/190 (92.1 %) | 0.0301 |

- \measured Monocytes: test 1 11 of 13 (GSE122244 4/5, GSE180130 7/8); test 2 0.032; test 3a 196 of 202; test 3b 0.011. Tests 1, 2 and 3a are outside their bars.
- \measured B cells: test 1 16 of 22 (GSE118144 11/13, GSE122244 1/5, GSE179801 4/4); test 2 0.037; test 3a 175 of 190; test 3b 0.0301 (at the bar).
- \observed Test 2 was read on whole-blood replicates where the cell is ~5 % of the specimen; Stage M's read line (0.20) would withhold that reading.
  No repeated purified arrays of one person exist in the bucket for either cell.
- \observed Neither cell meets the rule. Neutrophils remain the only cell read.

The rule (three tests) is written into SOP section 2b.
