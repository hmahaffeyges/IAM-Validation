# DEV-FIRES-01: every tool fires, on blood drawn years before a diagnosis (exploration, not sealed)

**The instrument's job:** read one person's blood with the whole chain, find every cell that is there with nothing assumed, and score every
cell found with every tool that can read it, each reading flagged where it is not commissioned. Nothing here is a test with bars; the readings
are gathered per person so that what departs, in which cell, and how long before a diagnosis, can be seen and then tested on fresh data.

**Data.** GSE51032 (GEO, public): whole blood of 845 participants of the EPIC-Italy cohort, Illumina 450K, raw IDATs. At the last follow-up 424
had stayed cancer-free, 235 had developed breast cancer and 166 colorectal cancer, 20 other cancers; each case carries the years from the blood
draw to the diagnosis (0.02 to 14.4, median 5.4). The series is spent (CANON/data_spent.csv): an earlier, different instrument read its cases
against its controls. It is used here to explore, never to commission. List: `data/DEV_FIRES_01/gse51032_samples.csv` (from the sample labels).

**The reading.** Every array by the one command (`chain/MethylPhys_Interface/run_series.py`, workflow `series.yml`) with `--cell-panel atlas`
(the IAM-Deconvolver on all 79 atlas cells; Met-A for each cell found) and `--dev-metaa-allsites` (the every-site reading). On a 450K array the
commissioned Met-A and C-Score are refused by the chain (no 450K neutrophil reference) and the atlas readers run, flagged.

**Smoke run first:** four arrays from one slide (two controls, one breast and one colorectal case; `smoke_samples.csv`): does a 450K IDAT pass
Stage 0 and Stage 1, and does every atlas reader fire. This is also plan branch 27.1 (a real 450K array with raw IDATs through Stage 1).

**Then:** the whole series in batches across the GitHub machines; per person, each cell's share, Met-A, and the C-Scores as they are built
(plan step 41), set against the years to diagnosis.

---
