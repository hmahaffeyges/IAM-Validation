# DEV-FIRES-01: every tool fires, on blood drawn years before a diagnosis (exploration, not sealed)

**The instrument's job:** read one person's blood with the whole chain, find every cell that is there with nothing assumed, and score every
cell found with every tool that can read it, each reading flagged where it is not commissioned. Nothing here is a test with bars; the readings
are gathered per person so that what departs, in which cell, and how long before a diagnosis, can be seen and then tested on fresh data.

**Data.** GSE51032 (GEO, public): whole blood of 845 participants of the EPIC-Italy cohort, Illumina 450K, raw IDATs. At the last follow-up 424
had stayed cancer-free, 235 had developed breast cancer and 166 colorectal cancer, 20 other cancers; each case carries the years from the blood
draw to the diagnosis (0.02 to 14.4, median 5.4). The series is spent (CANON/data_spent.csv): an earlier, different instrument read its cases
against its controls. It is used here to explore, never to commission. List: `data/DEV_FIRES_01/gse51032_samples.csv` (from the sample labels).

**The reading.** References for every tare: only the 424 who stayed cancer-free (the `ref` column; run_series.py uses only those). Every array by the one command (`chain/MethylPhys_Interface/run_series.py`, workflow `series.yml`) with `--cell-panel atlas`
(the IAM-Deconvolver on all 79 atlas cells; Met-A for each cell found) and `--dev-metaa-allsites` (the every-site reading), `--dev-territories` (cancer's own territories, DEV-TERRITORIES-01), and the C-Score for each cell and the whole genome (`stage_metaa_atlas.cscore_in_mix`, flagged; plan 41.2, 41.3). On a 450K array the
commissioned Met-A and C-Score are refused by the chain (no 450K neutrophil reference) and the atlas readers run, flagged.

**Smoke run first:** four arrays from one slide (two controls, one breast and one colorectal case; `smoke_samples.csv`): does a 450K IDAT pass
Stage 0 and Stage 1, and does every atlas reader fire. This is also plan branch 27.1 (a real 450K array with raw IDATs through Stage 1).

**Then:** the whole series in four batches (`gse51032_batch1.csv` to `batch4.csv`: every row listed, so the references and slides are the whole series; the `read` column marks the quarter each batch reads, every fourth array in the list's order). Pass 1 of each batch on 20 machines (`series.yml` with `pass1from only`); then pass 2 of each batch reads its quarter tared against the 424 cancer-free references, from all four pass-1 runs (`pass1from` the four run ids); per person, each cell's share, Met-A, and the C-Scores as they are built
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

**Smoke run 1 (series run 39934797643).** Every 450K array stopped at Stage 0 (the one command's logs: `data/DEV_FIRES_01/smoke_01/`): the 450K coverage gate measured 0.525 and 0.534 of the commissioned Met-A references' sites (limit 0.8), so nothing was read (lesson C40). The gate protects the commissioned readers, which refuse 450K themselves; run_sample.py now lets a 450K array whose only intake failure is that coverage go on to the atlas readers, flagged (ADVANCE_FLAGGED_450K). Pass 2 also stopped on the empty first pass; run_series.py now says so plainly. Smoke run 2 follows on the same four arrays, with the information sphere.

**Where the earlier signal sits, and what the chain compares (written 2026-10-23, before the full series is read).** The earlier instrument
read six sites (`earlier_pattern.py`'s source). Four of them (cg03172595, cg05928286, cg17301223, cg11667314) are on the 450K chip but not in
IAM-Atlas v2.1, so the chain cannot read them against the atlas. Of the two that are, cg26203572 is held near 0.97 by every blood cell type,
and cg27091787 separates neutrophils (0.52) from lymphocytes (about 0.92), so it moves with the share of neutrophils to lymphocytes. None is
among any blood cell's Met-A identity sites, and the earlier instrument removed no cell mix. So part of its signal may be a change in the
blood's make-up, which the chain reads as the IAM-Deconvolver's cell shares and Met-A deliberately removes. The comparison is therefore read
in three places, cases against the cancer-free references, window by window, by Group-A's rule (difference in standard errors): (1) the cell
shares, the neutrophil share and the lymphocyte share first; (2) each cell's Met-A and the every-site reading; (3) the territories. A change
in (1) alone says the blood's make-up moved; in (2) or (3) too, the cells themselves moved. Neither outcome is a failure of the chain.

**Where the earlier panel came from (added after reading the author's earlier work folder, before the series is read).** The six sites and the
earlier 538-site immune-class panel are a published set: the blood methylation sites associated with later breast cancer in the Sister Study,
listed in that paper's supplementary tables (sites at p < 1e-7, replicated in EPIC-Italy, plus sites in known breast-cancer susceptibility
genes). So the signal at these sites in EPIC-Italy is already reported, and a reading of them here is a replication, not a discovery. Of the
538, 486 are in IAM-Atlas v2.1: 6 are among any blood cell's Met-A identity sites, 58 are in the islands set and 128 in the domains set
(DEV-TERRITORIES-01), 115 vary between cell types and 218 read alike in all 79. Most of the published signal therefore lies outside Met-A's sites,
which is why the every-site reading and the territories are read here beside Met-A. Several earlier results (top-ranked sites, cross-validated
effect sizes, variance by class, five windows) chose sites or tests on the same arrays; the chain's readings are fixed before any array is read,
so smaller effects than the earlier ones are expected if part of those came from the selection.

**The scorer (committed before any reading is opened).** `data/DEV_FIRES_01/fires_score.py` reads the pass-1 bundles and compares each reading
by Group-A's own rule, the chain's `stage_groupa.compare` unchanged: each reading tared against the cancer-free references on its chip (else the
series), then cases against the cancer-free, |t| >= 2 "groups differ", every result preliminary. Because Group-A tares against the cancer-free
references itself, no separate pass 2 is run. Checked on stand-in bundles before any reading: a planted 3 % shift in 20 of 40 is found, and a
reading with no spread is reported, not divided by.

## First reading of the whole series (exploration; the scorer as committed, then checks added after it, each disclosed)

**As committed** (`data/DEV_FIRES_01/pass1/fires_summary.txt`; 837 of 845 arrays read, pass 1, series runs 40033292750, 40033341848,
40041618829, 40041658839): breast cases read more disorder than the cancer-free on every Met-A reading, in every cell and in both
territories, at 0-2, 2-5 and 5-10 years before diagnosis; over 10 years the same direction without reaching the line. The cell shares do not
differ. Colorectal: small differences, mostly at the islands.

**Added after the first reading, because the breast cases are 233 women and 2 men and the cancer-free 340 women and 84 men:** women only
(`--sex F`, `pass1/fires_summary_sexF.txt`). The breast shift stays (whole-sample Met-A +2.40 %, t +5.51); the colorectal shift mostly
goes. Every case sits on a chip with 3 to 10 cancer-free arrays, so each case is tared against the cancer-free run on its own chip.

**Effect sizes beside the earlier engine** (`pass1/fires_effect_sexF.txt`, `fires_effect.py`; `earlier_pattern_summary.txt`): breast,
whole-sample Met-A, d +0.44, +0.82, +0.67, +0.46 at 0-2, 2-5, 5-10 and over 10 years (earlier 0.28, 0.47, 0.44, 0.37); colorectal, all,
d +0.06 (earlier 0.18). The shape repeats, peaking at 2-5 years.

**But the case arrays are of lower quality** (Stage 0's own measures, women): lower bisulfite conversion control, lower signal and fewer
probes detected than the cancer-free on the same chips. Lower signal and conversion pull values toward the middle, which every Met-A reading
reads as more disorder, in every cell at once, the pattern seen. **Within thirds of each quality measure** (`pass1/fires_quality_sexF.txt`,
`fires_quality.py`; cut points from the cancer-free alone): within the middle and best thirds of conversion and signal the shift stays,
smaller (about 1 %, d 0.5 to 0.7); within the best third of the call rate it fades (d 0.24, p 0.49).

**Matched on every quality measure at once** (`pass1/fires_matched_sexF.txt`, `fires_matched.py`): each breast case paired with the
cancer-free woman on its own chip nearest in all four quality measures, kept only if each is within a quarter of the cancer-free spread.
Only 45 of 230 cases find such a partner (the cases' arrays are mostly weaker than any cancer-free array on their chip). In the 45 pairs,
quality is equal (differences under 0.03 SD) and the case reads higher by a median 0.52 % (Wilcoxon p 0.018; higher in 27 of 45, sign p
0.23); at 0-2 years no difference (19 pairs), at 2-5 and 5-10 years about 1 % (11 and 10 pairs, not significant at this size).

**In plain words:** part of the breast signal is the arrays' quality, which the earlier engine also read; with quality held equal a small
shift remains, about a quarter of the unmatched one, on too few pairs to place it by year. Not settled. Open: why the case samples carry
weaker signal (storage, handling, or the blood itself).
