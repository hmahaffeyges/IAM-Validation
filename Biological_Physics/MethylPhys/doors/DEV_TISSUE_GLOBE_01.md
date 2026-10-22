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

## Run 2, planned (written 2026-10-21 before it runs; exploratory, nothing sealed or scored)

The author asked to run every pair we have. `data/DEV_TISSUE_GLOBE_01/all_pairs_02.py` on GitHub (`.github/workflows/tissue_globe.yml`, 10
machines and a combine job): all 191 sealed pairs (colon 88, breast 53, liver 50), each tumor against its own adjacent normal; as many
controls, one person's adjacent normal against another's of the same organ; and colon's 48 healthy donors' mucosa, one against the next.
Every comparison counted with all sites and with the X and Y left out. Reported per tissue: changed blocks (above and below) and the share of
the globe lit, the median and range of each kind, and the share of tumors above the largest control. The question it answers: does a tumor
against its own normal light more than the differences between two people, by which measure, and in which tissue.

## Run 2, run once (2026-10-21; tissue_globe run 39686195039, commit 94bbc421; 430 comparisons, no errors)

Output: `data/DEV_TISSUE_GLOBE_01/run_02/all_pairs_02_summary.txt` and `all_pairs_02_rows.csv`. Changed blocks, X and Y left out (median, range):

| Tissue | Tumor vs own normal | Two people's normal | Two healthy people | Tumors above the largest control |
|---|---|---|---|---|
| Breast (53) | 859 (239-1,948) | 298 (27-602) | | 74 % |
| Liver (50) | 898 (3-2,485) | 316 (41-651) | | 60 % (42 % with the X and Y in) |
| Colon (88) | 917 (49-2,245) | 542 (94-1,358) | 518 (234-1,879) | 15 % (6 % with the X and Y in) |

**What it shows.**
- In breast and liver a tumor against its own normal lights about three times what the difference between two people lights.
- Leaving out the X and Y lowers the controls where women and men are mixed (colon, liver) and not where all are women (breast), as the X's biology predicts; in liver it raised the share of tumors above every control from 42 % to 60 %.
- Direction: colon tumors gain methylation in three blocks for every one they lose (58,658 against 20,363); liver tumors lose more than they gain (26,921 against 16,881); breast is balanced.
- Colon's controls light nearly as much as its tumors. The seven-pair run's two-person control was not a fair yardstick for colon; run 2's colon controls compare people on different slides of the series, while each tumor sits beside its own normal, so a slide difference is likely counted in the controls and not the pairs. Not yet checked: colon controls matched on slide.

Still a picture, not a reading: nothing here is sealed or scored, and no instrument uses it.

## Run 3, planned (written 2026-10-21 before it runs; exploratory)

`data/DEV_TISSUE_GLOBE_01/colon_slide_03.py` on GitHub (`tissue_globe.yml`, script input): colon's controls matched on the slide. On each of
GSE131013's 22 slides, each person's adjacent normal against the next person's on the same slide, and each healthy donor's mucosa against the
next on the same slide; the same rule, with and without the X and Y; compared with run 2's 88 tumors, each already on its own normal's slide.

## Run 3, run once (2026-10-21; tissue_globe run 39695940610, commit d60c37cd; 117 same-slide comparisons, no errors)

Output: `data/DEV_TISSUE_GLOBE_01/run_03/colon_slide_03_summary.txt` and `colon_slide_03_rows.csv`. Changed blocks, X and Y left out (median,
range): colon tumor against its own normal 917 (49-2,245); two people's adjacent normal **on the same slide** 575 (200-1,478); two healthy
donors' mucosa on the same slide 722 (338-2,107). Tumors above the largest same-slide control: 8 % (5 % with the X and Y in).

**What it shows.** The slide is not why colon's controls light so much: matched on the slide, they light as much as in run 2 (542 across
slides). In colon the difference between two people's normal mucosa is nearly as large as a tumor's difference from its own normal, even
between two healthy donors. So in colon a two-person yardstick cannot show a tumor by count alone, unlike breast and liver. A colon tumor
read against its own normal, the person's own reference, remains the comparison that means something. Still a picture, not a reading.

## Step 4: the pairs on the Globe Explorer (2026-10-22; to-do step 28)

`data/DEV_TISSUE_GLOBE_01/explorer_pairs_04.py` draws run 1's seven pictures (3 colon pairs, 3 breast pairs, the two-person control) again
through the chain's pair mode and writes them as one Globe Explorer globe, `tumor-pairs`, labeled development. Run on GitHub (explore.yml);
its pictures are committed under `website/art/out/explorer/` when it lands.
