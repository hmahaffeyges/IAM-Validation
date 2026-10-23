# DEV-FIRES-01: every tool fires, on blood drawn years before a diagnosis (exploration, not sealed)

**The instrument's job:** read one person's blood with the whole chain, find every cell that is there with nothing assumed, and score every
cell found with every tool that can read it, each reading flagged where it is not commissioned. Nothing here is a test with bars; the readings
are gathered per person so that what departs, in which cell, and how long before a diagnosis, can be seen and then tested on fresh data.

**Data.** GSE51032 (GEO, public): whole blood of 845 participants of the EPIC-Italy cohort, Illumina 450K, raw IDATs. At the last follow-up 424
had stayed cancer-free, 235 had developed breast cancer and 166 colorectal cancer, 20 other cancers; each case carries the years from the blood
draw to the diagnosis (0.02 to 14.4, median 5.4). The series is spent (CANON/data_spent.csv): an earlier, different instrument read its cases
against its controls. It is used here to explore, never to commission. List: `data/DEV_FIRES_01/gse51032_samples.csv` (from the sample labels).

**The reading.** References for every tare: only the 424 who stayed cancer-free (the `ref` column; run_series.py uses only those). Every array by the one command (`chain/MethylPhys_Interface/run_series.py`, workflow `series.yml`) with `--cell-panel atlas`
(the IAM-Deconvolver on all 79 atlas cells; Met-A for each cell found) and `--dev-metaa-allsites` (the every-site reading), and the C-Score for each cell and the whole genome (`stage_metaa_atlas.cscore_in_mix`, flagged; plan 41.2, 41.3). On a 450K array the
commissioned Met-A and C-Score are refused by the chain (no 450K neutrophil reference) and the atlas readers run, flagged.

**Smoke run first:** four arrays from one slide (two controls, one breast and one colorectal case; `smoke_samples.csv`): does a 450K IDAT pass
Stage 0 and Stage 1, and does every atlas reader fire. This is also plan branch 27.1 (a real 450K array with raw IDATs through Stage 1).

**Then:** the whole series in batches across the GitHub machines; per person, each cell's share, Met-A, and the C-Scores as they are built
(plan step 41), set against the years to diagnosis.

## What the earlier instrument saw, written before the chain reads the whole series

The earlier instrument (VAL-047 phase 12: the entropy of each beta value over a small published panel of immune sites, against one fixed
floor, on the series' processed values, with no cell mix, no tare and no Stage 0) read the breast cases as more disordered than the 424
controls in every window before diagnosis: effect size 0.28 at 0-2 years (66 cases), 0.47 at 2-5 (50), 0.44 at 5-10 (75), 0.37 beyond 10
(33), 0.35 over all 224 (permutation p 0.0001). On a 19-site secretory panel the same direction, a difference in A of 0.006 to 0.010. The
colorectal cases showed no clear signal (0.18 over all 76, p 0.15). Source: the earlier results file VAL047_phase12_gse51032_combined_results.json
(the author's evidence folder, not this repository), printed by `data/DEV_FIRES_01/earlier_pattern.py` into `earlier_pattern_summary.txt`.

**The question the chain answers.** With each person's own cell mix found by the IAM-Deconvolver, and each cell read against its own healthy
profile in IAM-Atlas v2.1, do the breast cases still read more disorder than the controls, in which cell, and in which years before diagnosis;
and is there now anything in the colorectal cases? If the earlier signal was the cell mix or the laboratory, the chain will not find it;
that is an answer too, and it decides where IAM is used next (blood, plasma, tissue, or single molecules). Not sealed: the series is spent.

---
