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
