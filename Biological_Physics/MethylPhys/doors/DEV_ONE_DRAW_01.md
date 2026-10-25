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
plasma, K0619, `plasma_seq_01/plasma_02_SRR23279768_summary.txt`, read held error 0.038256 against the first's 0.039202, loose share 0.1852 against 0.4052, IAM-A 0.8196 Normal against 0.722); the
cell shares' medians. Nothing fitted; whatever band the 30 give is the first one for plasma on sequencing.

## The healthy plasma band, 30 plasmas (`data/DEV_ONE_DRAW_01/plasma_seq_01/plasma_band_summary.txt`; scorer `plasma_band.py`, committed before 28 of them were read)

All 30 healthy plasmas of PRJNA929650 read end to end by the one command (shallow bisulfite sequencing). Medians, with the 5 % to 95 % range:

| Reading | Median | 5 % to 95 % |
|---|---|---|
| held error | 0.03854 | 0.03716 to 0.04019 |
| loose share | 0.2864 | 0.1305 to 0.4019 |
| IAM-A | 0.7733 | 0.7145 to 0.8398 |
| Met-A, whole sample | 0.8302 | 0.7583 to 0.9406 |
| territory islands | 0.8391 | 0.6967 to 0.9684 |
| territory domains | 0.86 | 0.7499 to 1.059 |
| globe changed blocks | 21.5 | 12 to 41.85 |
| sphere territories changed | 0 | 0 to 1 |

- **The held error barely moves between people,** a few percent across all 30, inside the range seen in purified healthy cells. On
  cell-free DNA too it behaves like a constant.
- **IAM-A tracks the loose share almost exactly** (Spearman rho -0.94, p 2.7e-14), and half the plasmas sit just below the Normal band
  built on purified cells. The plasmas' loose shares lie far above any purified cell's, so the self-reference line is read beyond the range
  it was built on. **Lesson:** cell-free DNA needs its own healthy band for IAM-A, or a loose share measured in a way short fragments do not
  inflate; the purified-cell line does not carry over as it stands.
- **Met-A and the territories read below 1** (whole sample 0.8302), less disorder than the atlas mix predicts. Open: the read-count
  expectation at one or two reads per site, or the mix itself (next point), each of which would bias it this way.
- **The IAM-Deconvolver's largest shares:** lung interstitial macrophages 0.385, neutrophils 0.178, erythrocyte progenitors 0.174.
  IAM-Atlas v2.1 holds no megakaryocytes, which the published work on plasma names as a main source of its DNA; that DNA has to land in
  some held cell type, and the macrophage share is a candidate for where it went. This is the atlas's known limit (a cell type it does not
  hold) appearing on real data; the shares are recorded as the Deconvolver returned them, flagged.

## Why IAM-A reads low on plasma: synthetic test cfDNA 1 (written and committed before it runs; plan 40.11)

**The instrument's job:** a healthy plasma reads Normal on IAM-A.

Two explanations of the 30 plasmas (IAM-A tracking the loose share, rho -0.94), each with its own fix:
- **Mixing.** The loose share is the share of 2,000-site tiles whose mean over every molecule falls below 2/3. Plasma mixes many cell types;
  where they disagree, a tile's mean falls although no molecule is loose, while the held error, counted molecule by molecule, is untouched.
  Then the fix is a loose share counted molecule by molecule, and IAM-A on plasma stays free of the atlas.
- **Fragment length.** Cell-free DNA comes in short pieces, so fewer calls per molecule. Then the fix is a correction for length.

`data/DEV_ONE_DRAW_01/synthetic_cfdna_01.py`: constructed molecules on every autosome, two cell types with the same held error, each holding
a different 8 % of tiles unmarked; read alone, mixed 70/30 and 50/50, and cut from 12 calls to 5. Every file through `run_sample.py
--specimen plasma`. **Expected if mixing is the cause:** the loose share rises with mixing (about twice as high at 50/50) while the held error
stays within 3 %, and IAM-A falls. **Expected if length is the cause:** cutting the molecules moves the loose share; mixing does not. Nothing
scored; the test says which fix to build.

**Synthetic test cfDNA 1, read** (`data/DEV_ONE_DRAW_01/synthetic_cfdna_01_summary.txt`; run 1 stopped at intake because the constructed
molecules began at CpG index 0, and the tiles of chr15 to chr22 had to be chosen inside the genome dictionary's ranges, plan 14.9; run 2 added
the 8- and 6-call cuts after 5 calls proved too short to count). **Both causes are real, and both pull IAM-A down.**

| Molecules | Cell A alone | Mix 70/30 | Mix 50/50 |
|---|---|---|---|
| 12 calls: loose share | 0.0795 | 0.1159 | 0.1534 |
| 12 calls: held error | 0.03186 | 0.03134 | 0.03195 |
| 12 calls: IAM-A | 0.7407 | 0.7078 | 0.701 |
| 8 calls: held error | 0.02786 | 0.02812 | 0.02820 |
| 6 calls: held error | 0.03000 | 0.03072 | 0.02942 |
| 5 calls | not read: no molecule qualifies | | |

- **Mixing nearly doubles the loose share** (0.0795 to 0.1534 at 50/50) while the held error does not move, so IAM-A falls: as predicted
  for mixing. Real plasma mixes many cell types; its loose share (median 0.2864) sits far above any purified cell's.
- **Short molecules lower the held error** (0.03186 at 12 calls, 0.02786 at 8), and below 6 calls nothing is counted. The qualifying rule
  (at least 6 calls, at least 80 % methylated) allows 2 errors in 12 calls but only 1 in 8, so shorter molecules carrying a second error are
  dropped. In the 30 real plasmas only 0.1125 to 0.1395 of molecules reach 6 calls (`plasma_read_length_summary.txt`).
- The absolute IAM-A of a constructed cell (0.7407) is set by the loss rate chosen for it; only the movements are the result.
- **Neither involves the atlas:** IAM-A reads the molecules directly. **The fixes, each for its own synthetic test (plan 40.11):** a loose
  share counted molecule by molecule (the share of long molecules that are loose), which a mix leaves unchanged because it is an average
  over molecules; and the qualifying rule's length bias computed exactly for each molecule length, as the read-count correction was.

## Where megakaryocyte DNA lands: absorption test 1 (written and committed before it runs; plan 40.13)

**The question:** IAM-Atlas v2.1 holds no megakaryocytes, which the published plasma work puts at about a quarter of healthy cell-free DNA. Until
the cell enters the atlas through its own sealed test, which atlas cells does the IAM-Deconvolver give their DNA to? Those cells' share is then
a stand-in for the megakaryocyte share, read with an abnormal IAM-A on another cell's molecules as a second, independent indicator (plan branch 40, the paired indicators).

**Data spent (author: "Spend donor 1 only"):** GSE87197 donor 1, 33 megakaryocyte samples from bone marrow (23 single cells, 10 pools of 5);
donors 2 and 3 stay unread for the sealed test. The files are on GRCh38; each atlas site is taken at its hg38 CpG from the Zhou lab's public
EPIC annotation (one single cell matched 84,563 atlas sites). `data/DEV_ONE_DRAW_01/mk_absorb_01/mk_absorb_01.py`: the pool read as plasma,
then planted into one spent healthy plasma at 0, 10, 25 and 40 % of its reads; every file through `run_sample.py --dev-seq-beta`.
**Expected:** the pool lands mostly on one or a few atlas cells, and those cells' shares rise as more is planted.

**Absorption test 1, read** (`data/DEV_ONE_DRAW_01/mk_absorb_01/mk_absorb_01_summary.txt`; the stand-in check `mk_standin.py`, added after
the reading, output `mk_standin_summary.txt`). Donor 1's 33 megakaryocyte samples pooled: 708961 of 786902 atlas sites with reads, a median of 3.

- **Pure megakaryocyte DNA lands on four atlas cells:** lung interstitial macrophages 0.429, erythrocyte progenitors 0.293, MEP (bone
  marrow) 0.146, eosinophils 0.111 (together 0.979).
- **The same four already make up most of a healthy plasma** (0.764 in SRR23279758), and they are its largest shares in all 30 plasmas. The
  published plasma work first deconvolved its plasma to about 30 % erythrocyte progenitors, then found megakaryocytes once it had their
  profile; the same absorption appears here.
- **Planted megakaryocyte DNA raises their sum as mixing predicts:** 0.796, 0.831 and 0.878 at 10, 25 and 40 % against 0.786, 0.818 and
  0.850 predicted.
- **So the stand-in is the sum of the four,** read against the person's own level. It is a noisy stand-in: it also holds real erythroid
  and macrophage DNA, and one donor at about 3 reads per site built it. The sealed atlas entry on donors 2 and 3 replaces it.

**Synthetic test cfDNA 2** (`data/DEV_ONE_DRAW_01/synthetic_cfdna_02.py`, output `synthetic_cfdna_02_summary.txt`; written with its expectation
before it ran, committed together with its result): Stage Q2 now also records the loose share molecule by molecule (`mol_low_share`, below the
0.70 cut the tile share uses; `mol_mid_share`, held neither marked nor unmarked), beside the tile loose share, used by no reading yet. Test 1's two cells, each
with 5 % of tiles held undecided in its own set:

| Sample | Tile loose share | Molecules below the cut | Undecided molecules |
|---|---|---|---|
| cell A | 0.1295 | 0.1271 | 0.0554 |
| cell B | 0.1295 | 0.1258 | 0.0554 |
| mix 70/30 | 0.1727 | 0.1286 | 0.0569 |
| mix 50/50 | 0.1750 | 0.1268 | 0.0551 |

**Met as written:** mixing raises the tile share by a third; both molecule-level shares stay within 3 %. Next: a self-reference line on the
molecule-level share, built on the purified healthy cells (which needs their molecule files counted again with the new fields), then the 30
plasmas reread against it; the length correction separately.

## A provisional megakaryocyte reference: test 1 (written and committed before any donor 2 file is read; plan 40.18)

**Prior reading:** GSE87197 donor 1 (33 single megakaryocytes) was spent for absorption test 1; donor 2 (19 cells) is spent for this test
(author); donor 3 (19 cells) stays unread. The spent healthy plasma (SRR23279758) is plasma_seq_01's; earlier sections of this record read GSE122126's
plasma arrays and mixtures (spent); none of them is read again here.

**Synthetic test:** chain/stage_provisional.py, DEV_ONE_DRAW_01/mk_absorb_01/provisional_synthetic_01.py, DEV_ONE_DRAW_01/mk_absorb_01/provisional_01/provisional_synthetic_01.txt

The test is itself constructed: real megakaryocyte reads planted into a spent plasma at known shares; no patient sample is read. The column (built, donor 1): 708,961 atlas sites with reads, a median of 3 per site.

**The idea (author):** until purified data good enough for a full atlas fit exist, a cell the atlas lacks gets a provisional column, kept
apart from IAM-Atlas v2.1, so its DNA is named instead of absorbed into look-alike cells. **Built:** `chain/stage_provisional.py` (the
column's level is the pooled reads through the atlas's own inverse transfer; its spread is the counting spread, the median donor spread of
the 79 atlas cells, and the atlas reference's laboratory and floor terms), built by the one command (`run_sample.py --build provisional`)
from donor 1 only, and read beside the atlas (`run_sample.py --dev-provisional megakaryocytes`: a second deconvolution with the column
added, recorded as `cell_shares_provisional`, flagged; the atlas's own shares unchanged).

**The test** (`data/DEV_ONE_DRAW_01/mk_absorb_01/mk_provisional_01.py`, absorption test 1's code with the donor and the reading flag
changed): donor 2's pool read pure, then planted into the same spent healthy plasma at 0, 10, 25 and 40 %. **The instrument's job:** name
the megakaryocyte DNA instead of absorbing it. **Bars:** (1) the pure donor 2 pool reads megakaryocytes (provisional) as its largest share,
at least 0.50; (2) the megakaryocyte share rises at each step, 0 < 10 < 25 < 40 % (revised before donor 2 was read; see the synthetic check); (3) the four stand-ins summed rise by at
most 0.02 from 0 to 40 %.

**Synthetic check 1, read** (donor 1, the column's own donor, so it should pass easily; `provisional_01/provisional_synthetic_01.txt`):
the pure pool reads megakaryocytes 1.0 with the column (the four stand-ins 0.979 without it); with 40 % planted the stand-ins fall (0.691 to
0.494) instead of rising. The share is low: 0.366 at 40 % against simple mixing's 0.472, so bar 2's first wording (within 0.05 of simple
mixing) was replaced, before donor 2 was read, by a rise at each step; the distance to simple mixing is recorded, not scored. **Noted for
later:** in the healthy plasma the stand-ins still hold 0.691 with the column (lung interstitial macrophages the largest), so megakaryocytes
explain only part of what those cells take in plasma.

**Provisional reference test 1, read** (`data/DEV_ONE_DRAW_01/mk_absorb_01/provisional_02/mk_provisional_01_summary.txt`): **not met**
(bar 1 not met, bar 2 met, bar 3 not met). Donor 2's pure pool reads megakaryocytes (provisional) 0.2082, the largest share but below 0.50;
planted into the plasma the share rises at each step but only from 0.1205 to 0.1378 at 40 % (simple mixing 0.1556), and the stand-ins rise
by 0.0373. **What it shows:** the column worked on donor 1 (1.000 pure, 0.366 at 40 %) because it read donor 1's own noise: at about 3 reads
a site, the sites chosen as megakaryocyte markers were partly sites where donor 1 looked extreme by chance. **Disclosed:** the test's own
scoring stopped on an error; the chain's sample-id hashing had replaced the cell name inside the pure pool's bundle, whose id was the same
word ("megakaryocytes"); the same bars were read from the saved bundles by `mk_provisional_01_score.py`. The hashing fault goes on the
chain's fix list. **Next:** a version 2 column, written before it runs, using only sites with enough reads and a spread honest about
single-cell noise; development on donor 2 (spent); confirmation on donor 3 (unread).
