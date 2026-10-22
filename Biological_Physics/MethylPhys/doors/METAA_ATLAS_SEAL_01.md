# METAA-ATLAS-SEAL-01: Met-A against IAM-Atlas v2.1 on purified monocytes, sealed (to-do step 33)

**The instrument's job:** a healthy purified monocyte, read against the atlas's monocytes and tared against other healthy monocytes of the
same laboratory, reads Normal; a 1 % loss of its pattern stands out.

**Why now.** Exploration 6 (DEV_METAA_ATLAS_01.md; GSE168779, 67 healthy children's monocytes, an independent laboratory) read 100 % Normal
tared on the same slide and 96 % tared across slides, with a planted 1 % loss at 3.6 and 2.6 healthy spreads. The search for an unspent series
(2026-10-22) found no purified healthy series holding four or more healthy arrays of one cell type on most slides, so the scored tare is the
laboratory tare (every other healthy array of the series), the one exploration 6 met at 96 %; the same-slide tare is reported where a slide
holds at least three others.

**Synthetic test:** DEV_METAA_ATLAS_01/synthetic_01.py, DEV_METAA_ATLAS_01/synthetic_01_summary.txt

**Prior reading:** GSE168779 was read in exploration 6 (2026-10-21) and is used here only for the rehearsal; GSE267660, the sealed data, has not been read.

**Sources:** GSE267660 (GEO, public): the raw IDATs of 31 healthy donors' CD14 monocytes, the sealed data; GSE168779 (GEO, public): the
spent healthy monocytes of the rehearsal; IAM-Atlas v2.1 (Zenodo 10.5281/zenodo.23441999, public): the monocyte reference the arrays are read against.

**Data, declared before opening (2026-10-22).** GSE267660: CD14-positive monocytes of 31 healthy donors (the series' HD_mono arrays, EPIC v1,
raw IDATs on GEO; an independent laboratory, not among IAM-Atlas v2.1's sources, never read). The multiple sclerosis arrays and the dendritic
cells are not opened. List: `data/METAA_ATLAS_SEAL_01/seal_samples.csv`.

**The reading.** `data/METAA_ATLAS_SEAL_01/seal_read_01.py`: each array through chain Stage 1 and `chain/stage_metaa_atlas.read`
(monocytes), unchanged; the plant as in exploration 3; the tare by `stage_metaa_atlas.tare`, never using the array itself.

**Bars, to be sealed after the rehearsal** (at most three, from the instrument's job): at least 90 % of the listed arrays receive a tared
reading; at least 90 % of the tared healthy readings are Normal (0.95 to 1.05); the planted 1 % loss moves the tared reading by at least
2 healthy spreads on average.

**Rehearsal first.** The same script, through the chain, on 24 spent healthy arrays (GSE168779's unvaccinated controls,
`rehearsal_samples.csv`). It must run cleanly end to end; its numbers do not set the bars.

## Rehearsal, run once (2026-10-22; explore run 39793753455, commit 6a751add)

The script through the chain on the 24 spent GSE168779 controls ran cleanly end to end: 24 of 24 tared, all Normal, a planted 1 % loss at
3.41 healthy spreads (the same-slide tare, reported: 8 arrays, all Normal, 4.6 spreads). Its numbers set nothing.

**Sealed 2026-10-22** in `data/METAA_ATLAS_SEAL_01/sealed_01.json` (the read script, the sample list, the workflow and the atlas
Met-A stage frozen by SHA-256): B1 at least 90 % of the 31 listed arrays receive a tared reading; B2 at least 90 % of the tared readings are
Normal (0.95 to 1.05); B3 the planted 1 % loss moves the tared reading by at least 2 healthy spreads on average. Read once on GitHub.

## Outcome, scored once (2026-10-22; explore run 39801005038, commit f0cec636): NOT COMMISSIONED

Output: `data/METAA_ATLAS_SEAL_01/sealed_01/sealed_summary.txt` and `sealed_rows.csv`. **Two of three sealed bars met.** B1 met: 31 of 31 arrays tared. B2 met:
90.3 % Normal (bar 90 %), median 0.99936. **B3 not met:** a planted 1 % loss moves the tared reading by +0.05773, which is 1.4 healthy
spreads (spread 0.04119; bar 2). Met-A against the atlas is not commissioned on this test, and the result stands as scored.

**What the numbers show, recorded, not rescored.** The planted move (+0.058) matches the rehearsal (+0.054); what grew is the spread across
this laboratory's healthy arrays: 0.041 here against 0.016 in the rehearsal. One array (GSM8272378) reads 0.81018, far below the rest; it
alone widens the spread. The same-slide tare, reported and not scored, read 9 arrays, all Normal, with the plant at 4.0 spreads, as it did
in exploration 6 (3.6). Both tests point the same way: against the atlas, Met-A needs references from the same slide, and a laboratory-wide
tare is too loose. Next, written before any reading: a sealed test with the same-slide tare scored, on a series with enough healthy arrays
of one cell type per slide.

**Found after scoring (2026-10-22).** The read script took chain Stage 1 and the atlas Met-A stage but not Stage T step 1, the self-tare the
commissioned chain runs before every Met-A reading. The result stands as scored; whether that step changes the spread is exploration 7
(DEV_METAA_ATLAS_01.md), and any next sealed test runs it.
