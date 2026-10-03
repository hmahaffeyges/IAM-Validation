# DEV-PERCELL-01 — stage 5 Met-A for each newly separated cell type (development; check written 2026-10-03 before the data were read)

**Commissioning step 3.** One cell type at a time, only for a cell that stage 3 or stage 4 separates within the DEV-ATLAS-EPIC-02 / DEV-NILC-01
bars on both held-out sets and on repeatability ("I'd rather not have a cell than only have part of one"). If no cell other than neutrophils
qualifies, this step is not run and says so.

**Per qualifying cell c.** Floor and identity sites: `metA_floors_v1_2_ALLCELLS_development.json` (EPIC, 6 Salas purified arrays, development;
not changed). (i) Pure-cell precision: purified healthy-labelled arrays of c from other laboratories in DEV-BASE-CHAIN-01 (e.g. GSE122244,
GSE180130, GSE179801), A = mean H(beta) over c's identity sites / floor, median tare against >= 3 same-series references of the same cell, self
excluded. Bar: >= 95 % of tared readings in Normal. (ii) Own replicate test: GSE250556 pooled-DNA replicates, whole-blood A for c against the
composition-matched expectation at c's sites (atlas v2 means of the 8 parent blood entries x stage 2 fractions), median tare per slide. Bar:
within-person SD <= 0.020. Both bars -> the cell is wired as a stage 5 reading; otherwise not.

---
## Outcome (recorded 2026-10-03 after the run; nothing above the line was changed)
Box job 91d7b32f (chain commit 63d55fa; beta vectors from DEV-BASE-CHAIN-01). Records: `data/DEV_TOOLKIT_01/` (`scores.csv`, `repeatability.csv`,
`agreement.csv`, `fractions_long.csv`, `summary.json`, script `toolkit_b.py`).

Qualifying cells (stage 3 atlas_e): **B cells** only (no cell qualified through stage 4). Floor: development EPIC `b cells`.
- (i) Pure-cell precision: 22 purified healthy B-cell arrays of other laboratories, tared per series:
  **16 of 22 Normal = 72.7 %** (bar 95 %): GSE118144 controls 11/13, GSE122244 1/5, GSE179801 4/4. **FAIL.**
- (ii) Own replicate test, GSE250556 pooled replicates (31 arrays): within-person SD **0.037** (bar 0.020). **FAIL.**

**B-cell Met-A is not wired.** Neutrophils remain the only cell read. Per-cell records: `data/DEV_TOOLKIT_01/percell_B_pure.csv`, `percell_B_repl.csv`.
