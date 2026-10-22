# DEV-IAMA-SEX-01: do the X and Y move IAM-A's held error on the .pat route? (exploratory, 2026-10-21)

**The instrument's job:** IAM-A reads a healthy sample's held error against what its own loose share predicts, on the same genome for both.

**Why.** Reading the code for the Chain Decoder found (finding F37) that on the `.pat` route chain Stage Q2's `count_pat` counts the held
error on every chromosome, the X and Y included, while the loose share is counted on autosome tiles only (`hg19_tiles` holds chr1 to chr22).
The aligned-read route counts both on the autosomes only. The commissioned line and band were built on the `.pat` route, so they are
self-consistent; the question is how large the difference is, before deciding whether the two routes should be made to match.

**What runs.** `data/DEV_IAMA_SEX_01/held_err_autosomes_01.py`, on GitHub (`explore.yml`): the committed line's own healthy samples of
CD8 T cells (all subsets), monocytes, B cells and NK cells, each `.pat` from GEO counted twice by `count_pat` unchanged: once as committed,
once with every non-autosome line blanked in place (so every other line is split between the two counts exactly as before; checked on a
synthetic file: identical when there are no X lines, and the loose share unchanged when there are). IAM-A by `Q2.reading` on each.

**Not sealed, no bar.** A result here changes nothing in the commissioned instrument by itself. If the X and Y move IAM-A by more than the
band's own healthy width allows to be ignored, the change (autosome-only held error on both routes) would be a new line and band, built and
commissioned in the usual way.

## Outcome (2026-10-21; explore run 39717648496, commit 5b693693; 17 of 17 samples read)

Output: `data/DEV_IAMA_SEX_01/run_01/held_err_autosomes_01_summary.txt` (the per-sample rows are in the run's GitHub output). The loose share is identical on
both counts, as it must be. Counting the held error on the autosomes only moves it, and so IAM-A, by a mean of **−0.32 %** (range −0.71 %
to +0.09 %); the X and Y hold 2.74 % of the held-error opportunities. Band v1 (`chain/Runtime Matrices/IAM_A_SelfRef/band_v1.json`) is about 28 % wide, so the
largest shift is about 2.5 % of the band's width. **Decision:** the commissioned line and band stay as they are; the two reading routes differ here by under 1 %,
recorded in the Chain Decoder (finding F37). No new line or band is needed.
