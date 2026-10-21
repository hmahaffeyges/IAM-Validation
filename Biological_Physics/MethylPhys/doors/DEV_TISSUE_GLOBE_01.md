# DEV-TISSUE-GLOBE-01: the tissue results on the globe (development, 2026-10-21)

**What it is.** A picture, not a reading. The chain's globe in pair mode (`chain/MethylPhys_Interface/report_globe_v4.py`, `build_pair`)
lays one person's tumor against the same person's own adjacent normal, by the globe's fixed rule: each site's difference in the pair's own
spread, read in blocks of 10 neighboring sites, a block changed beyond the 5 % family-wise line. Arrays calibrated by chain Stage 1. Each
page was opened in Chromium on GitHub and screenshotted with the slider at left, middle and right (`data/DEV_TISSUE_GLOBE_01/tissue_pairs_globe.py`,
explore run 39681312575, commit 3db7f1a0; counts in `data/DEV_TISSUE_GLOBE_01/run_01/tissue_pairs_globe.json`). The pairs are the first three
sealed pairs of DEV-TISSUE-COLON-01 and DEV-TISSUE-BREAST-01, already scored and spent; nothing here changes those results.

| Pair (450K arrays) | Blocks changed | Of blocks |
|---|---|---|
| colon A2004 | 1,109 | 38,857 |
| colon A2027 | 1,289 | 38,097 |
| colon A2050 | 298 | 39,209 |
| breast TCGA-A7-A13E | 1,134 | 38,757 |
| breast TCGA-A7-A13F | 912 | 39,669 |
| breast TCGA-AC-A23H | 1,570 | 37,819 |
| **control: two different people's normal colon** | **741** | 38,575 |

**What it shows.** The globe draws each pair's change in clumps, and the browser shows it as drawn (black background, red and blue change).
**What it does not show:** by the count of changed blocks, a tumor against its own normal is not separated from two different people's normal
colon; the control falls inside the tumors' range. Differences between people (including sex, on the X) are the same size as a tumor's
change at this resolution. This is why the instruments read a tumor by what is lost (Met-A, the tissue scorer's sealed rule), not by how much
differs. What would separate them is a question for a test, not this picture: the direction of the change at a cell's identity sites, the
X removed, or a person's own two normals as the control.
