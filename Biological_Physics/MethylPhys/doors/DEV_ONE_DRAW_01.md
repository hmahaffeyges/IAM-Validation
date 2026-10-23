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

## Exploration 2, written before it runs: plasma (2026-10-22)
Stage 0 now accepts plasma cell-free DNA (`stage_0_intake.is_plasma`; development): read on the full atlas panel, Met-A for every source
found, the commissioned whole-blood readings not applied. `data/DEV_ONE_DRAW_01/plasma_02_score.py` reads the bundles of GSE122126's 58
cell-free DNA arrays (spent), run through the one command by the series runner (`plasma_gse122126_samples.csv`): 12 healthy, 22 sepsis,
15 cancers, and 9 from pancreatic islet transplant recipients before and 1 and 2 hours after the transplant, where donor pancreatic DNA is
expected in the blood (a positive control). Reported by group: the sources found, the non-blood share, pancreatic sources, Met-A of each.

## Exploration 3, written before it runs: known mixtures (2026-10-22)
`data/DEV_ONE_DRAW_01/mixes_03_score.py` reads GSE122126's 9 laboratory mixtures (spent): hepatocyte, lung, colon and neuron DNA mixed into
blood DNA at 3.5 to 10 %, each read by the one command on the full atlas panel (series.yml, `--cell-panel atlas`). Reported: each tissue's known
amount against the share found and whether its 95 % interval holds it; the largest share found for a tissue that was not put in.

## Exploration 1, run once (2026-10-22; explore run 39868983687): how few molecules each instrument needs
Output: `data/DEV_ONE_DRAW_01/seq_depth_01_summary.txt`. GSM5652282 (healthy CD8 T cells, WGBS, spent), read by `run_sample.py` at seeded
thinned fractions of its molecules (`--dev-thin`, `--dev-seq-values 1`).
- **IAM-A** reads Normal from all molecules down to 1 % of them (about 3.7 million), and falls below Normal at 0.3 % (about 1.1 million).
- **The IAM-Deconvolver** on the sequencing values reads the sample as CD8 T cells (share 0.99 to 1.0) down to 3 % of the molecules.
- **Met-A against the atlas** on sequencing values reads 0.673 at full depth and falls as the depth falls (0.542 at 30 %, 0.343 at 10 %),
  and is withheld below 10 % (fewer than 90 % of its sites have a call). Two causes, both expected from the physics of a count: (1) a value
  made from k of n calls is 0 or 1 far more often than the cell's true level when n is small, and the entropy of 0 or 1 is 0, so the
  measured disorder is biased low by an amount set only by n; (2) at full depth, sequencing values of a sorted cell are more extreme than an
  array's, so the array-built atlas expectation is not this chemistry's healthy reference. Next: the expected entropy of a k-of-n value is
  computed exactly from n (a binomial sum, derived, nothing fitted) and used as the expectation at each site's own depth; then a same-chemistry
  healthy tare. Not sealed.
