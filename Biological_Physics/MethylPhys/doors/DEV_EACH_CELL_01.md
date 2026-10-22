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
