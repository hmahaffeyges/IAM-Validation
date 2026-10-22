# DEV-ONE-DRAW-01: every instrument on one blood draw (exploration, not sealed)

Author 2026-10-22: "the full chain running end to end, EVERY instrument scoring on one whole blood sample that is separated between Buffy coat
and cfDNA, every cell with enough to score, every dna strand long enough to score, and each one placed on their appropriate globe ...
reproducible with the push of a button." To-do step 40. Every reading is the one command (`chain/MethylPhys_Interface/run_sample.py`); spent
public data only; nothing sealed until the instruments show what is possible.

**The instrument's job:** one draw's buffy coat and plasma, read by the one command, give a reading from every instrument that has enough to read,
and say for each one how much it needed.

## Built 2026-10-22 (development, every reading flagged)
- `--dev-seq-values MIN_CALLS`: a value at each IAM-Atlas v2.1 site from the same sequencing molecules IAM-A reads (`chain/stage_seq_values.py`),
  so the IAM-Deconvolver, Met-A for each cell, Met-A against the atlas and the globe read a sequencing run. The commissioned array readings (Met-A
  on neutrophils, the C-Score) are not applied to sequencing values.
- `--dev-thin FRACTION` (tests only): a seeded thinned copy of the molecules, so a deep sample stands in for a shallow one.
- Already in the one command: `--cell-panel atlas` (every atlas cell type) and Met-A for each cell found.

## Exploration 1, written before it runs: depth
`data/DEV_ONE_DRAW_01/seq_depth_01.py`: the healthy CD8 T-cell WGBS sample GSM5652282 (spent), fetched by the chain, read at full depth and
thinned to 0.3, 0.1, 0.03, 0.01 and 0.003 of its molecules: IAM-A, sites with a value, Met-A against the atlas's CD8 T cells, the cell shares.
