# DEV-EACH-CELL-01: Met-A for every cell of a mixed sample (exploration, not sealed)

Author 2026-10-22: whole blood and plasma are split by the IAM-Deconvolver and every cell it finds is scored; "dont worry about sealing and
preregistration yet...this is exploratory so we learn before we waste our arrays and samples"; "let the instrument tell us...not our
assumptions". Every reading is the one command (`chain/MethylPhys_Interface/run_sample.py`, record `met_a_each_cell`), on spent data only.

**The instrument's job:** each cell's reading moves for the cell whose pattern was lost, by the shift it predicts at that person's share.

## Synthetic test 2, replacing test 1 (written 2026-10-22 before it runs)

**Test 1 below does not count** (author: "it MUST use the chain or it didnt happen"): its readings were the one command, but the loss was
planted by the test script. The one command now plants a constructed loss itself (`run_sample.py --dev-plant CELL:LOSS`, after Stage 1,
scaled by the cell's share, recorded in the bundle and bannered PLANTED). `data/DEV_EACH_CELL_01/synthetic_each_cell_02.py` only downloads
the raw IDAT pair from GEO and runs the one command: as it is, and with a 5 % loss planted in monocytes, CD4 T cells, B cells and NK cells in
turn. Run on GitHub's machines from the repository as committed.

## Synthetic test 1, superseded: the plant was made outside the one command (2026-10-22; `data/DEV_EACH_CELL_01/synthetic_each_cell_01.py`, output `synthetic_each_cell_01_summary.txt`)

The healthy example blood (GSM7508429) and the same blood with a constructed 5 % loss of one cell's pattern at its identity sites, at its share.
Monocytes (share 0.0627) moved 92 % of the predicted shift, every other cell less than a twentieth of it. CD4 T cells (share 0.233) moved
99 % of the predicted shift; CD8 T cells moved about a quarter as much, B and NK cells less, because lymphocytes share identity sites: the
readings of close relatives are not independent, and the report must say so. The planted cell's share did not move (a loss at the identity
sites is not taken up by the Deconvolver). Healthy readings untared sit 1.10 to 1.19 (the atlas's offset; the same-run tare cancels it).

## Next
1. The one command on the full IAM-Atlas v2.1 panel (`--cell-panel`), so every cell the specimen holds is found and scored.
2. Group-A's 278 healthy bloods (spent): which cells appear and at what shares; each cell's healthy spread; the lowest share each can be scored at.
3. Plasma, on plasma array series already read (unread ones stay unread).
4. IAM-A depth: a spent deep CD8 sample thinned toward cell-free DNA depth and fragment length, read at each step (finding F38).

## Exploration 3, written before it runs (2026-10-22): the whole atlas panel

The one command now takes `--cell-panel`: `blood` (the six cells the IAM-Deconvolver is commissioned on, the default), `atlas` (all of
IAM-Atlas v2.1's cell types) or a named list; any panel other than the six is development and every share is flagged. Met-A for each cell reads
every cell the panel finds. `data/DEV_EACH_CELL_01/full_panel_03.py` runs the healthy example blood from its raw IDATs three times (six cells;
the full atlas; the full atlas with a 5 % monocyte loss planted by the chain) and reports every cell found above a share of 0.001.

## Exploration 3, run once (2026-10-22; explore run 39868137014): the whole atlas panel
Output: the run's summary (`full_panel_03_summary.txt`, the run's artifact). Two problems found, both fixed in the chain the same day:
(1) Met-A for each cell read only the cells whose identity sites had an atlas value in all 79 cells; most cells were refused with about half
their sites (fixed: the expectation is mixed over the cells with a value, kept at 99 % of the shares); (2) on 79 cells close relatives trade a
share, so several best-fit shares fell outside their own 95 % range (now flagged not resolved; which estimate is closer to the truth is what
the known mixtures, exploration 3 of DEV-ONE-DRAW-01, measure). The planted monocyte loss was not read, because monocytes were refused (1).

## Synthetic test 2, run once (2026-10-22; explore run 39865611924): every reading and every plant by the one command
Output: `data/DEV_EACH_CELL_01/synthetic_each_cell_02_summary.txt`. From the raw IDATs of GSM7508429, a 5 % loss of one cell's pattern planted
by `run_sample.py --dev-plant` at that cell's share: the planted cell moved 92 % (monocytes), 99 % (CD4 T), 100 % (B) and 100 % (NK) of the
shift predicted for it; the largest move of another cell was 6 % (monocyte plant) to 26 % (CD4 T plant, seen in CD8 T) of the planted cell's
move, the lymphocytes sharing part of their pattern; the Deconvolver's share of the planted cell did not move. Read on the six-cell panel,
untared. The globe in each report shows the change against healthy neutrophils, so a loss in another cell barely moves it (the changed
pixels are in the report's own count): a globe for each cell is not built yet.

## Exploration 4, written before it runs (2026-10-22): Met-A on the 6,000 identity sites against every atlas site
Author: "Met-A should be able to read 800K sites so long as they correspond to the cells in the atlas." `data/DEV_EACH_CELL_01/metaa_allsites_04.py`
runs only the one command on the raw IDATs of GSM7508429: healthy, then a 5 % loss planted by the chain in monocytes, CD4 T, B and NK cells, once
at their identity sites and once at every atlas site they hold (`--dev-plant CELL:0.05:all`). Each bundle carries both readings: identity sites
(I) and every site weighted by its shift (W, `--dev-metaa-allsites`). Reported for each: the planted cell's move in the reading's own error over
sites, and the largest move of another cell as a share of it. Nothing scored. Known before it runs: the error over sites is within one sample,
so it is smaller than the spread between healthy people; that spread comes from many healthy bloods read the same way (Group-A's 278).

## Exploration 5, written before it runs (2026-10-22; to-do 38.1): does a clustered loss light its cell's globe layer?
`data/DEV_EACH_CELL_01/globe_cell_05.py`, only the one command on the raw IDATs of GSM7508429: a loss planted by the chain in 10 runs of 30
neighboring identity sites (`--dev-plant CELL:LOSS:runs`) of monocytes, CD4 T, B or NK cells, at 5, 20 and 50 %, scaled by the cell's share; each
cell's changed blocks on the globe against healthy. An even loss was already shown not to light the layer (2 of 597 blocks before and after).

## Exploration 4, run once (2026-10-22; explore run 39885755066)
Output: `data/DEV_EACH_CELL_01/metaa_allsites_04_summary.txt`. Each move is in the reading's own error over sites (within one sample).
- A loss at the cell's identity sites: the identity-site reading moves 6.0 (monocytes) to 30.9 (CD4 T) errors, other cells 15 to 26 % as
  much; the every-site reading moves 0.5 to 2.1 errors, every cell alike (100 to 118 %).
- A loss at every site the cell holds: the identity-site reading moves 5.6 to 29.9 errors, other cells 92 to 121 % as much; the every-site
  reading moves 44.6 to 186.7 errors, every cell alike (101 to 106 %).
Reading: the every-site version is far more sensitive to a loss across a whole pattern but cannot say which cell (it is a whole-sample
reading); the identity sites say which cell only when the loss sits at that cell's own sites; a loss across a cell's whole pattern cannot be
put on one cell by either, because most held sites are held by its relatives too. Kept: the identity sites for the reading of each cell.
Next: identity sites chosen to be specific to each cell (to-do 39.9); the every-site version as a candidate whole-sample reading.

## Exploration 7, written before it runs (2026-10-22): sites specific to each cell
`stage_metaa_atlas.specific_sites` keeps Met-A's site rule and adds one condition: the cell's atlas value differs by at least a stated amount from
every other cell of the panel (`run_sample.py --dev-metaa-specific`). On the six-cell blood panel the counts at 0.1 and 0.2 are: neutrophils
5,001 and 2,658; monocytes 1,791 and 356; B cells 3,952 and 2,505; CD4 T 672 and 66; CD8 T 273 and 25; NK 2,616 and 553 (of 6,000). CD4 and
CD8 T cells hold nearly the same sites. `data/DEV_EACH_CELL_01/specific_07.py` plants exploration 4's hard case, a 5 % loss at every site a cell
holds, and reads it on identity sites and on both specific sets.

## Exploration 5, run once (explore run 39893275546; output `data/DEV_EACH_CELL_01/run_05/globe_cell_05_summary.txt`)
Clustered losses (10 runs of 30 identity sites) planted by run_sample.py, read on each cell's globe layer. Healthy: 0 to 5 changed blocks per cell.
A clustered loss lights its own cell's layer once loss times share is a few percent of the DNA: CD4 T cells (share 0.233) 2 to 29 blocks at 20 %;
NK cells (0.0807) 2 to 18 at 50 %; B cells (0.0565) 4 to 11 at 50 %; monocytes (0.0627) 2 to 9 at 50 %. Other cells' layers rise by at most 7.
At 5 % no cell's layer lights. The per-cell layer places a clustered loss on the right cell; its reach is set by the cell's share.

## Exploration 7, run once (explore run 39905743359; output `data/DEV_EACH_CELL_01/run_07/specific_07_summary.txt`)
A 5 % loss at every site a cell holds. On the identity sites the planted cell moves 5.6 to 29.9 of its own error, but its relatives move 92 to 121 %
as much: the loss cannot be put on one cell. On sites held by one cell alone the planted cell barely moves (0.1 to 1.5 of its error) and the others
still move as much or more; at a difference of 0.2 too few sites remain for CD4 T. Sites held by one cell alone do NOT solve attribution of a
loss across a whole pattern; the per-cell globe (exploration 5) does for a clustered loss.

## Exploration 8, written and committed before it runs
`data/DEV_EACH_CELL_01/spread_08_score.py` on every fifth of Group-A's 278 healthy bloods (`spread_08_samples.csv`, 56 arrays, one laboratory series),
read by run_sample.py with `--cell-panel atlas --dev-metaa-allsites`: for each cell and each reading, the spread between healthy people and what a
1 % loss moves in units of it. The scorer only summarizes the bundles.

## Synthetic C-Score test 1, written and committed before it runs (plan 41.2, 41.3)
`data/DEV_EACH_CELL_01/cscore_cell_01.py`: the C-Score for every cell the IAM-Deconvolver finds (flagged below 1 % of the sample) and for the whole genome (`chain/stage_metaa_atlas.cscore_in_mix`, the commissioned statistic on each cell's departure map), by run_sample.py --cell-panel atlas, on four spent arrays, one per run: GSE235717 GSM7508429 (healthy whole blood) and three of GSE122126's known mixtures (hepatocytes and lung; colon and neurons; lung and neurons in blood). Each array read clean, then for every cell the IAM-Deconvolver found a loss planted by the chain, clustered (50 % in 10 runs of 30 identity sites) and spread (20 % at every site the cell holds). Expected before it runs: a clustered loss raises the planted cell's C above its clean C, and more than a spread loss does. A first version planted in four blood cells only and was canceled before it ran (author: why not all of them?). Which cells are there is the IAM-Deconvolver's answer, never assumed.

**Exploration 8, outcome (series run 39927627528; `data/DEV_EACH_CELL_01/run_08/spread_08_summary.txt`).** 56 of Group-A's healthy whole bloods,
pass 1 (untared), every reading by run_sample.py. For neutrophils, the cell the planted loss reaches most: on the identity sites the spread
between people is 3.69 % and a 1 % loss moves the reading 2.41 % (0.65 spreads); on every atlas site the spread is 1.98 % and the same loss moves
it 1.83 % (0.92 spreads). So the every-site reading is about 1.4 times as able to see a loss in one person. Its spread is nearly the same for every
cell (1.4 to 2.3 %) and its medians sit near 0.94, which points to one sample-wide factor (the array's scale against the atlas) rather than to
each cell: the factor the atlas ruler is meant to remove. Small shares (the rarer cells) move well under one spread for a 1 % loss of their pattern.

