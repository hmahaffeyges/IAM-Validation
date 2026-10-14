# DEV-ERASE-01 — the half and half: one half writing, one half erasing (prediction sealed 2026-10-14, before any data set is searched)

**The idea (author, 2026-10-14).** "What is the half and half in DNA methylation and now I know, one is erase and one is write." It is the
virial pair of IAM's Law as Part I states it: when a system binds, the well splits into two equal halves, the kinetic it keeps and the heat
it writes (K = Q = |V|/2). In the cell's copy channel the loss has two routes, the writer's own misses and active erasure (TET enzymes); the
author's statement is that they are the two equal halves.

**Why it is worth testing.** The writer alone, measured outside the cell (DNMT1, DEV-WRITER-01), would hold a copy error of 0.0155;
healthy cells hold 0.032. Two equal halves of loss lower the holding by exactly one bit, kT ln 2, which from the writer's measured range
gives 3.17-3.69 kT around the measured 3.41. That match was found after the fact on a known number, so it does not count. What counts is the
prediction below, which no data here has been read for.

**Prediction.** Remove the eraser and the copy error halves. Written as the ratio of the copy error in eraser-removed cells to that in the
same laboratory's control cells, the ratio cancels laboratory, species and instrument offsets. Computed in
`data/DEV_ERASE_01/prediction_erase_01.py` (output `prediction_erase_01.txt`): equal halves give about 0.51; the two virial alternatives
(eraser twice or half the writer) give about 0.34 and 0.67; no eraser share gives 1.

**Rule, fixed now.** MET if the ratio is within 0.40-0.60. NOT MET if it is 0.85 or above (the extra loss is not erasure; missed restores
or something else). Anything else is INCONCLUSIVE and reported as such.

**Before any data is opened.** (1) Metadata-only search for single-molecule sequencing of eraser-removed cells with same-laboratory
controls: TET1/2/3 triple knockout, and IDH1/2-mutant cells or tumors (2-hydroxyglutarate blocks TET); every set found is declared in
`CANON/data_spent.csv` before reading. (2) The copy error is read by Stage Q, the same reading behind 0.032; the IAM-A instrument is still
on hold at commissioning, so any result is development, not commissioned, until it passes. (3) Caveats written now: knockout cells are often
stem cells and adapt over time; IDH mutation also changes other enzymes; mouse cells are another species (the ratio form is chosen for this).

**Synthetic test:** DEV_ERASE_01/synthetic_erase_01.py, DEV_ERASE_01/synthetic_erase_01.txt
Simulated molecules with copy errors planted at 0.032 and at half that, read by Stage Q's own counting: ratio 0.5232 against the 0.517
expected from counting isolated errors only, bar 0.45-0.55: BAR MET. Disclosed: the bar was in the file before it ran, but the file was
committed after (same session, 2026-10-14). A planted check on real healthy molecules is still to do before any real set is read.

**Planted check on real healthy molecules (committed 2026-10-14 before it runs):** `data/DEV_ERASE_01/planted_real_erase_01.py` and
`.sh`. Two healthy Loyfer files already spent for development (granulocytes GSM5652313, CD4 T cells GSM5652281). Arms: unchanged; half of the
real isolated errors turned back (the eraser-removed picture); about as many new ones added (the boosted-eraser picture). Bars: halved/base
0.45-0.55 and the base file's two halves within 0.95-1.05; doubled/base reported, no bar (amended before any real file was read: a
code smoke test on a small made-up file read a doubling as 1.77, because a new error beside an old one leaves neither isolated).

**In plain words.** We think methylation's half and half is writing and erasing. If that is right, cells whose eraser is switched off should
make half as many copy errors as normal cells from the same laboratory. We wrote down the number and the pass and fail lines before looking
at any such data. What it would take to finish: the planted check on real healthy molecules, at least one eraser-removed set with same-laboratory controls has been found and declared, and its ratio has been read once against the lines above, whatever it says.

Related: PHY-040 (the author's half-law), PHY-006 / HYP-002 / SCO-033 (this test's earlier rows), DEV-WRITER-01/-02, LEV-019/-044, LEV-046.
