# DEV-REPLICATE-01: the earlier work, replayed through the chain (tuning)

**The instrument's job:** the chain, one command, reads every array the earlier instruments read, and we see whether the signals they found
are found again. Tuning, not commissioning: only data the earlier work already opened; every data set no instrument has opened stays unread.

## R1: the earlier CPG engine's 11 test arrays

The arrays its test-data manifest lists, fetched from GEO by accession (the IDAT fingerprints are checked against the author's local copies):
rheumatoid arthritis cases and controls (GSE42861, whole blood, 450K), three healthy bloods (GSE87571, 450K), a high-grade colon adenoma
(GSE288652, EPIC) and colorectal carcinoma at stages 1 and 4 (GSE166212, EPIC). Blood: specimen whole blood, the controls and healthy bloods
as references (`r1_blood_samples.csv`); tissue: specimen colon tissue (`r1_tissue_samples.csv`). Scorer `data/DEV_REPLICATE_01/r1_score.py`.

**What the earlier engine reported, written before the chain reads them:** epithelial cells read near zero in healthy blood and appear in
the colon tissues, rising from stage 1 (about 13 % of the sample) to stage 4 (about 25 %), with its disorder reading rising with stage;
one healthy blood (GSM2333905) carried a stem-cell-class elevation; RA cases departed from controls in their immune cells.

**What the chain should show if it agrees:** epithelial (colon) cells found only in the tissues and more at stage 4 than stage 1; the
whole-sample Met-A and the territories higher in the carcinomas than the adenoma, and higher at stage 4; the bloods near one another.

**R1, tissue (series run 40027190291, chain commit 7dc86ca4; `data/DEV_REPLICATE_01/r1_tissue/r1_tissue_summary.txt`).** The IDAT fingerprints
match the author's copies. Stage 0 stopped the stage 1 carcinoma GSM5065990 at intake: bisulfite conversion 0.749, below 0.80
(`r1_tissue/GSM5065990_intake.txt`); the earlier engine had read it. On the other three, untared, development:

| sample | cells found (3 % or more) | whole-sample Met-A | islands | domains |
|---|---|---|---|---|
| adenoma GSM8772491 | colon epithelium 29 %, CD8 TEMRA T 18 %, embryonic stem 10 %, vascular endothelium 9 %, pancreatic acinar 8 %, oligodendrocytes 6 %, regulatory T 5 %, prostate epithelium 4 %, memory B 3 % | 1.14022 | 1.5826 | 1.07795 |
| adenoma GSM8772492 | colon epithelium 42 %, enteroendocrine 13 %, vascular endothelium 11 %, lung bronchus 9 %, CD8 TEMRA T 8 %, memory B 6 %, oligodendrocytes 4 % | 1.21579 | 1.44186 | 1.19155 |
| stage 4 carcinoma GSM5065985 | enteroendocrine 27 %, CD8 TEMRA T 20 %, prostate epithelium 15 %, memory B 11 %, colon epithelium 8 %, pancreatic acinar 8 %, lung bronchus 6 %, smooth muscle 3 % | 1.08983 | 1.14323 | 1.14104 |

Epithelium is found in every tissue, as the earlier engine found; the stage comparison waits on the refused stage 1 array. Cells no healthy colon
holds (stem, pancreas, prostate, lung, brain support cells) are how the IAM-Deconvolver explains a tumor that matches no healthy cell. The islands
read highest in the two adenomas; the domains do not follow stage across these three. Met-A for each cell was withheld on both adenomas (50 to
82 % of identity sites measured, under the 90 % rule); from here each cell is read when at least half its sites are measured, flagged below 90 %.

**Three faults R1 found in the chain, fixed before the rerun.** (1) The globe drew a healthy reference only for blood and isolated cells; for
a tissue it looked for a cell named after the specimen. It now uses the person's own mix as the IAM-Deconvolver found it for every mixed
specimen (tissue, plasma, unnamed), mixed over the cells with an atlas value at each site (kept at 99 % of the shares, read_in_mix's rule).
(2) The information sphere skipped without a word when the globe was not drawn; it now records why. (3) Each cell's Met-A was withheld below
90 % of its identity sites measured; it is now read at half or more and flagged below 90 %. Also: the report's new sections first carried a
word its claim check refuses, so the one command stopped at the report; reworded. Checked on the healthy blood GSM7508429 before the rerun: globe
drawn, 45 of 77,355 blocks changed as before; the sphere read 27 territories.

