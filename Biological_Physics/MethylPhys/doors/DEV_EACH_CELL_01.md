# DEV-EACH-CELL-01: Met-A for every cell of a mixed sample (exploration, not sealed)

Author 2026-10-22: whole blood and plasma are split by the IAM-Deconvolver and every cell it finds is scored; "dont worry about sealing and
preregistration yet...this is exploratory so we learn before we waste our arrays and samples"; "let the instrument tell us...not our
assumptions". Every reading is the one command (`chain/MethylPhys_Interface/run_sample.py`, record `met_a_each_cell`), on spent data only.

**The instrument's job:** each cell's reading moves for the cell whose pattern was lost, by the shift it predicts at that person's share.

## Synthetic test 1 (2026-10-22; `data/DEV_EACH_CELL_01/synthetic_each_cell_01.py`, output `synthetic_each_cell_01_summary.txt`)

The healthy example blood (GSM7508429) and the same blood with a constructed 5 % loss of one cell's pattern at its identity sites, at its share.
Monocytes (6.27 % of the blood) moved 92 % of the predicted shift, every other cell less than a twentieth of it. CD4 T cells (23.3 %) moved
99 % of the predicted shift; CD8 T cells moved about a quarter as much, B and NK cells less, because lymphocytes share identity sites: the
readings of close relatives are not independent, and the report must say so. The planted cell's share did not move (a loss at the identity
sites is not taken up by the Deconvolver). Healthy readings untared sit 1.10 to 1.19 (the atlas's offset; the same-run tare cancels it).

## Next
1. The one command on the full IAM-Atlas v2.1 panel (`--cell-panel`), so every cell the specimen holds is found and scored.
2. Group-A's 278 healthy bloods (spent): which cells appear and at what shares; each cell's healthy spread; the lowest share each can be scored at.
3. Plasma, on plasma array series already read (unread ones stay unread).
4. IAM-A depth: a spent deep CD8 sample thinned toward cell-free DNA depth and fragment length, read at each step (finding F38).
