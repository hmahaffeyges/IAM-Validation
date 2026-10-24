# DEV-ONE-DRAW-01: every instrument on one blood draw (exploration, not sealed)

Author 2026-10-22: "the full chain running end to end, EVERY instrument scoring on one whole blood sample that is separated between Buffy coat
and cfDNA, every cell with enough to score, every dna strand long enough to score, and each one placed on their appropriate globe ...
reproducible with the push of a button." To-do step 40. Every reading is the one command (`chain/MethylPhys_Interface/run_sample.py`); spent
public data only; nothing sealed until the instruments show what is possible.

**The instrument's job:** one draw's buffy coat and plasma, read by the one command, give a reading from every instrument that has enough to read,
and say for each one how much it needed.

## Built 2026-10-22 (development, every reading flagged)
- `--dev-seq-values MIN_CALLS`: a value at each IAM-Atlas v2.1 site from the same sequencing molecules IAM-A reads (`chain/stage_seq_values.py`),
  so the IAM-Deconvolver, Met-A for each cell, Met-A against the atlas and the globe read a sequencing run. The commissioned array readings (Met-A
  on neutrophils, the C-Score) are not applied to sequencing values.
- `--dev-thin FRACTION` (tests only): a seeded thinned copy of the molecules, so a deep sample stands in for a shallow one.
- Already in the one command: `--cell-panel atlas` (every atlas cell type) and Met-A for each cell found.

## Exploration 1, written before it runs: depth
`data/DEV_ONE_DRAW_01/seq_depth_01.py`: the healthy CD8 T-cell WGBS sample GSM5652282 (spent), fetched by the chain, read at full depth and
thinned to 0.3, 0.1, 0.03, 0.01 and 0.003 of its molecules: IAM-A, sites with a value, Met-A against the atlas's CD8 T cells, the cell shares.

## Exploration 2, written before it runs: plasma (2026-10-22)
Stage 0 now accepts plasma cell-free DNA (`stage_0_intake.is_plasma`; development): read on the full atlas panel, Met-A for every source
found, the commissioned whole-blood readings not applied. `data/DEV_ONE_DRAW_01/plasma_02_score.py` reads the bundles of GSE122126's 58
cell-free DNA arrays (spent), run through the one command by the series runner (`plasma_gse122126_samples.csv`): 12 healthy, 22 sepsis,
15 cancers, and 9 from pancreatic islet transplant recipients before and 1 and 2 hours after the transplant, where donor pancreatic DNA is
expected in the blood (a positive control). Reported by group: the sources found, the non-blood share, pancreatic sources, Met-A of each.

## Exploration 3, written before it runs: known mixtures (2026-10-22)
`data/DEV_ONE_DRAW_01/mixes_03_score.py` reads GSE122126's 9 laboratory mixtures (spent): hepatocyte, lung, colon and neuron DNA mixed into
blood DNA at 3.5 to 10 %, each read by the one command on the full atlas panel (series.yml, `--cell-panel atlas`). Reported: each tissue's known
amount against the share found and whether its 95 % interval holds it; the largest share found for a tissue that was not put in.

## Exploration 1, run once (2026-10-22; explore run 39868983687): how few molecules each instrument needs
Output: `data/DEV_ONE_DRAW_01/seq_depth_01_summary.txt`. GSM5652282 (healthy CD8 T cells, WGBS, spent), read by `run_sample.py` at seeded
thinned fractions of its molecules (`--dev-thin`, `--dev-seq-values 1`).
- **IAM-A** reads Normal from all molecules down to 1 % of them (about 3.7 million), and falls below Normal at 0.3 % (about 1.1 million).
- **The IAM-Deconvolver** on the sequencing values reads the sample as CD8 T cells (share 0.99 to 1.0) down to 3 % of the molecules.
- **Met-A against the atlas** on sequencing values reads 0.673 at full depth and falls as the depth falls (0.542 at 30 %, 0.343 at 10 %),
  and is withheld below 10 % (fewer than 90 % of its sites have a call). Two causes, both expected from the physics of a count: (1) a value
  made from k of n calls is 0 or 1 far more often than the cell's true level when n is small, and the entropy of 0 or 1 is 0, so the
  measured disorder is biased low by an amount set only by n; (2) at full depth, sequencing values of a sorted cell are more extreme than an
  array's, so the array-built atlas expectation is not this chemistry's healthy reference. Next: the expected entropy of a k-of-n value is
  computed exactly from n (a binomial sum, derived, nothing fitted) and used as the expectation at each site's own depth; then a same-chemistry
  healthy tare. Not sealed.

## Exploration 4, written before it runs (2026-10-22; plan step 40): one draw, both halves, real healthy donors
GSE186888 holds whole-genome bisulfite sequencing (85x) of white blood cells and the matched plasma cell-free DNA of 23 healthy donors, as
per-CpG count files (.beta). The chain now reads those (`run_sample.py --dev-seq-beta`). Under the scarce-data rule, only the first three donors
by number are read here, declared spent in CANON/data_spent.csv before reading (`data/DEV_ONE_DRAW_01/gse186888_explore_samples.csv`); the other
20 stay unread for the one-draw test that counts. `data/DEV_ONE_DRAW_01/one_draw_04.py` runs the one-draw runner on each donor's two files.
Reported, nothing scored: each half's cells and shares, Met-A for each cell (untared), and what the plasma holds beyond the person's own
white cells. IAM-A is not read: count files carry no molecules.

**Exploration 4 withdrawn before any reading (2026-10-22).** The scarce-data rule reserves matched white cells and plasma from one person for
the test that counts and never explores on them; declaring three donors spent did not satisfy it. The run was cancelled while still queued, so no
GSE186888 file was downloaded or read, and the series is again wholly unread (removed from CANON/data_spent.csv). The `.beta` reader stays,
checked on a constructed file.

## Exploration 5, written before it runs (2026-10-22): constructed draws with known answers
Because the matched set is reserved, the one-draw path is explored on draws the chain builds itself (`run_sample.py --build constructed-draw`)
from the spent example blood: hepatocyte, vascular endothelium and colon epithelium each added to the plasma at 0 (control), 0.5, 1, 2 and 5 %,
plasma read at 10, 30 and 60 reads per site (one GitHub run each), buffy coat at 30. `data/DEV_ONE_DRAW_01/constructed_draw_05.py` reports each
added tissue's share found against the truth and the largest share found for a tissue not added (the noise floor at that depth). A first local
check at 15 reads (5 % hepatocyte and 5 % vascular endothelium) found 4.8 % and 5.2 %, with tissues not added up to 1.9 %.

## Exploration 1 again, with the read-count correction (2026-10-22; explore run 39892463332, commit 82c9105f)
Output: `data/DEV_ONE_DRAW_01/run_02/seq_depth_01_summary.txt`. Met-A against the atlas's CD8 T cells, at all, 30 % and 10 % of the
molecules (median 31, 9 and 3 reads per site): 0.757, 0.798, 0.897 (before the correction 0.673, 0.542, 0.343). Below that, too few identity
sites have a read, as before. IAM-A and the Deconvolver are unchanged (IAM-A Normal down to 1 % of the molecules).
Reading: the correction removed the slide downward as reads fall. Two things remain. (1) At full depth the reading is 0.757, not 1: that is
the gap between sequencing and the array-built atlas, a chemistry offset, which only a tare against healthy samples read the same way can
cancel (same chemistry, same depth). (2) As reads fall, the reading moves toward 1, because with few reads both the measured disorder and its
expectation are made mostly of counting; the reading is not wrong there, but it carries less of the sample. How much a loss still moves it at
each depth is for a planted loss at each depth to show.

## Exploration 6, written before it runs (2026-10-22; plan step 40): what a loss still moves at each depth
`run_sample.py --build constructed-draw` now takes `--dev-plant CELL:LOSS`, planting the loss in the blood's values before the reads are counted,
scaled by the cell's share as the one command scales a plant. `data/DEV_ONE_DRAW_01/seq_plant_06.py` builds a healthy draw and draws with CD4 T
cells at 5 and 20 % and NK cells at 20 %, counts them at 3, 10, 30 and 60 reads per site (one GitHub run each), and reads the buffy-coat half by
the one command. Reported: each planted cell's Met-A move, in the reading's own error over sites, and the largest move in another cell.

Explorations 5 and 6, first dispatch: stopped by a path bug in the test files (a relative output folder, while the chain's builds run from
their own folder); no draw was read. Fixed (absolute folder); nothing else changed; run again.

## Exploration 3, run once through the one command (2026-10-22; series run 39879101605)
Output: `data/DEV_ONE_DRAW_01/run_03/mixes_03_summary.txt`. GSE122126's 9 known mixtures (real arrays: DNA of hepatocytes, lung, colon and
neurons put into blood DNA at 3.5 to 10 %, spent), each read by run_sample.py on IAM-Atlas v2.1's full panel. Over the 15 tissues put in, the
best fit's mean absolute error is 0.0057 (resampled middle 0.0079); the largest share found for a tissue NOT put in is 0.0030; the resampled
95 % range holds the known amount in 10 of 15. Reading: the full-atlas split finds tissue DNA at 3.5 % in blood and puts it at the right size,
with almost nothing found that was not there. The best fit is the closer estimate; the resampled range is too narrow (it holds the truth 67 %
of the time, not 95 %), so it is reported as the spread of the fit, not as a 95 % range, until widened by a rule tested on these mixtures.

**Exploration 6 at 10 reads per site (2026-10-22; explore run 39904730217):** `data/DEV_ONE_DRAW_01/run_06/seq_plant_06_depth10_summary.txt`.
A loss planted before counting moves the planted cell's Met-A clear of its own error at 10 reads: CD4 T cells 5 % by 6.2 of its error, 20 % by
23.4; NK cells 20 % by 7.4. The closest relative moves about a quarter as much (CD8 T for CD4 T; CD4 T for NK). The other depths are running.

**Exploration 6 at 30 reads (explore run 39904765477):** CD4 T cells 5 % moves 8.7 of its own error, 20 % 32.1; NK cells 20 % 10.8; the closest
relative moves 16 to 26 % as much (`run_06/seq_plant_06_depth30_summary.txt`).

**Exploration 5 at 10 reads per site in the plasma (explore run 39904598005):** `run_05/constructed_draw_05_depth10_summary.txt`. Hepatocyte DNA
added at shares 0.005, 0.01, 0.02 and 0.05 is found at 0.0049, 0.0082, 0.0197 and 0.0525; vascular endothelium at 0.0045, 0.0152, 0.0316 and
0.0616; colon epithelium is not found below 0.05 (0.0294 at 0.05). With nothing added, each tissue reads at most 0.0001. The summary's "largest not added" (basophils, about
3 %) is a cell of the blood itself, which the constructed plasma carries; the line should list only tissues outside the blood, and is
corrected in the next runs' summaries. Reading: at plasma-like depth the full-atlas split finds liver DNA from 0.5 % and puts it at the right
size; vessel lining is found but read high; colon is missed at small shares, which is what the known mixtures showed on arrays as well.

**Exploration 6 at 3 reads (explore run 39904694167):** CD4 T cells 5 % moves 4.6 of its own error, 20 % 16.8; NK cells 20 % 6.1. Even at 3 reads
per site a 5 % loss in one cell stands clear (`run_06/seq_plant_06_depth3_summary.txt`).

**Exploration 5 at 60 reads (explore run 39904662144):** `run_05/constructed_draw_05_depth60_summary.txt`. More depth did NOT make the small
shares better: hepatocyte added at 0.005 is found at 0.0008 (at 10 reads: 0.0049), and vascular endothelium reads 0.0047 with none added. Each
depth draws its reads with its own seed, so the small shares move with the draw. Reading: below about 0.02 the limit is the split itself, not
the counting; the Deconvolver's resampled range (too narrow; a plan branch of step 39) must say so before any small share is reported.

**Exploration 6 at 60 reads (explore run 39904799192):** CD4 T cells 5 % moves 11.0 of its own error, 20 % 43.0; NK cells 20 % 14.2; the closest relative 17 to 26 % as much (`run_06/seq_plant_06_depth60_summary.txt`). Over 3, 10, 30 and 60 reads, a 5 % CD4 T loss moves 4.6, 6.2, 8.7 and 11.0 of its error.

**Exploration 5 at 30 reads (explore run 39904630856):** `run_05/constructed_draw_05_depth30_summary.txt`. Hepatocyte added at 0.005, 0.01, 0.02 and 0.05 found at 0.0013, 0.0067, 0.0192 and 0.0507; vascular endothelium found at 0.0028 with none added; colon epithelium found from 0.02 (0.0047) and at 0.05 (0.0326). As at 10 and 60 reads: below about 0.02 the small shares move with the draw. The largest share not added is a blood cell the constructed plasma carries (naive CD8 T cells, basophils, eosinophils), not a tissue.

## Real plasma, sequenced (plasma_seq_01)

Written and committed before any read of these plasmas is opened (tuning, not sealed; author: "Spend 30 healthy only").

**The instrument's job:** the one command reads a real healthy plasma's molecules end to end, and every instrument either reads it, flagged,
or says why it cannot.

**Data.** PRJNA929650 (public SRA): plasma cell-free DNA, shallow genome-wide bisulfite sequencing (about 0.58 times coverage, `plasma_seq_01/source_publication.txt`; about 16
million read pairs per sample, DNBSEQ-G400) of 278 healthy women and 239 with nonmetastatic breast cancer, Ho Chi Minh City (the
publication 10.3389/fonc.2023.1127086; its Table S1 gives each sample's group, matched to all 517 runs by sample code,
`data/DEV_ONE_DRAW_01/plasma_seq_01/prjna929650_labels.csv`). Spent for tuning: 30 healthy, chosen by a fixed rule from the labels alone (the
healthy runs in run-accession order, every ninth, the first 30), listed in `plasma_seq_01_samples.csv`. The other 248 healthy and all 239
cases stay unread, reserved (CANON/data_spent.csv).

**The run.** For each sample, `erase_test.yml` with `full=1`: the committed reads-to-molecules pipeline of DEV-ERASE-01 (bwa-meth 0.2.0 on hg19
with lambda and pUC19, Sambamba markdup, wgbstools bam2pat) on every read pair, then `run_sample.py --pat ... --specimen plasma
--dev-seq-values 1`. For shallow sequencing three readings were set to read and flag, written before any plasma is read: the
IAM-Deconvolver reads with at least 500 of its marker sites measured (`stage_v4.SEQ_READ_MARKERS`, flagged below half); Met-A for each cell
with a tenth of its identity sites (`READ_COVER_MIN_SEQ`); and IAM-A on a mixed specimen is read on its self-reference line with the
sample-shape check not applied (a mix has no single cell's healthy shape). Whole-sample Met-A and the territories now take each site's
expected disorder at its own read count (`expected_H_counts`), as each cell's Met-A already did. Checked on one saved human file before any
plasma: on GSM2387333 (cultured embryonic stem cells, spent), read as plasma, the Deconvolver found embryonic stem cells at 81 % on 22 % of
its markers, IAM-A read on the line, and every reading fired; a whole blood array reads exactly as before (`plasma_seq_01/precheck_summary.txt`).

**What will be reported, nothing scored:** each plasma's cell shares, IAM-A on the line, Met-A for the whole sample and each cell, the
territories, the distance to the ceiling, the globe and the sphere; and how they spread across the 30 healthy plasmas, the first healthy
band for plasma on sequencing.

**First plasma read end to end** (`plasma_seq_01/plasma_01_SRR23279758_summary.txt`): every instrument read it, flagged. Two things the
healthy band must settle before any of it means anything: IAM-A's self-reference line was built on purified cells, and cell-free DNA's
short fragments give a loose share (0.405) far above any purified cell's, so its "below Normal" may only be plasma lying off the cells' line;
and the IAM-Deconvolver's answer (lung interstitial macrophages, erythrocyte progenitors and bone-marrow progenitors largest) is recorded
as the instrument gave it. The other 28 were dispatched on this result.

**The band's scorer, committed before 28 of the 30 are read** (`plasma_seq_01/plasma_band.py`): for each reading, the median and the 25-75 %
and 5-95 % ranges over the healthy plasmas; IAM-A's held error and loose share apart, and whether IAM-A follows the loose share (the second
plasma, K0619, `plasma_seq_01/plasma_02_SRR23279768_summary.txt`, read held error 0.0383 against the first's 0.0392 (rounded), loose share 0.185 against 0.405, IAM-A 0.82 Normal against 0.72); the
cell shares' medians. Nothing fitted; whatever band the 30 give is the first one for plasma on sequencing.
