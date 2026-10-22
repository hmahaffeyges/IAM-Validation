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
