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

## Provisional megakaryocyte column, version 2: test 2 (written and committed before version 2 is read on donor 2; plan 40.18)

**Prior reading:** GSE87197 donor 1 built both columns; donor 2 was read in test 1 (spent), so this test is **development**; donor 3 stays
unread for the confirmation.

**What was wrong with version 1** (found after test 1): the counting spread was put on the array scale with the transfer's local slope, which
at the ends understated it. Sites with 1 or 2 reads (all read as exactly 0 or 1) came out narrower (median spread 0.111 and 0.075) than sites
with 3 (0.208), so the Deconvolver trusted the thinnest sites most and chose donor 1's chance extremes as markers. **Version 2**
(`chain/stage_provisional.py::column_v2`, stated before it was read): the level from the shrunk proportion (m + 1/2)/(n + 1); the counting
spread carried through the atlas's own inverse transfer by its interval's two ends, no slope; only sites with at least 5 reads (197,580 of
donor 1's). Built by the one command (`--build-cell megakaryocytes@2`); read with `--dev-provisional megakaryocytes@2`. Its spread now falls
with depth (median 0.154 at 5 reads, 0.111 at 10; `data/DEV_ONE_DRAW_01/mk_absorb_01/provisional_03/column_spread.txt`).

**The test** (`data/DEV_ONE_DRAW_01/mk_absorb_01/mk_provisional_02.py`): test 1's saved donor 2 readings (the pure pool; the spent healthy
plasma with 0, 10, 25 and 40 % planted), read again with version 2. **The instrument's job:** name the megakaryocyte DNA instead of
absorbing it. **Bars:** test 1's three.

**Synthetic test:** chain/stage_provisional.py, DEV_ONE_DRAW_01/mk_absorb_01/provisional_synthetic_01.py, DEV_ONE_DRAW_01/mk_absorb_01/provisional_03/provisional_synthetic_v2.txt

The synthetic is donor 1 in-sample: version 2 reads its own donor's pure pool 0.880 (version 1: 1.000, its own noise), and with 40 %
planted the megakaryocyte share rises from 0.085 to 0.358 while the stand-ins fall (0.679 to 0.538).

**Provisional reference test 2, read** (`data/DEV_ONE_DRAW_01/mk_absorb_01/provisional_03/mk_provisional_02_summary.txt`): **not met**, all
three bars. Donor 2's pure pool reads megakaryocytes 0.1954 with version 2 (not its largest share; the stand-ins take 0.7770); planted into
the plasma the share does not rise at each step (0.0847, 0.1065, 0.0992, 0.1062) and the stand-ins rise by 0.0994. Version 2 fixed the
spread, but donor 2's DNA is still read as marrow progenitors, so the column's spread was not the main problem.

**What the sample titles show:** each cell's ploidy (megakaryocytes copy their DNA as they mature: 2N young, 8N to 32N mature, the
platelet-making cells that shed into plasma). Donor 1: 2N 8, 4N 6, 8N 6, 16N 6, 32N 7. Donor 2: 2N 8, 4N 2, 8N 2, 16N 0, 32N 7. Donor 3: 2N 9,
4N 2, 8N 1, 16N 2, 32N 5. Donor 2's set leans young, and a young megakaryocyte is close to its progenitor (MEP), one of the stand-ins. A
hypothesis, not a finding. **Ploidy check 1** (`data/DEV_ONE_DRAW_01/mk_absorb_01/mk_ploidy_01.py`, written before it runs; development on
donor 1, spent): donor 1's 2N cells (8) and its 8N-and-up cells (19) pooled separately and each read once through the one command, the atlas
only. If the young pool lands on the marrow progenitors and the mature pool does not, the next column is built from mature cells only, the
cells that shed into plasma.

**Ploidy check 1, read** (`data/DEV_ONE_DRAW_01/mk_absorb_01/ploidy_01/`): **ploidy does not separate them.** Donor 1's mature cells (8N and
up, 19 pooled, median 2 reads a site) read lung interstitial macrophages 0.419, erythrocyte progenitors 0.319, MEP (bone marrow) 0.144,
eosinophils 0.091; its young cells (2N, 8 pooled, median 1 read) read erythrocyte progenitors 0.256, lung interstitial macrophages 0.229, lung
alveolar macrophages 0.189. Both land on the same stand-ins, so the young-cell explanation of test 2 is not supported. **What it leaves:**
single cells at 1 to 3 reads a site are too thin to build a reference column from, however the spread is handled. **Next:** look for public
megakaryocyte methylation from bulk cells at real depth (purified or cultured megakaryocytes, sequencing or arrays) before a version 3
column; GSE87197 donor 3 stays unread.

**The tile table, checked (plan 14.9, development):** Stage Q2's chromosome counts (`stage_q2_selfref.HG19_CPGS`) are exact for chr1 to chr13
and short from chr14 to chr22 against the genome dictionary's ranges (`Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json`), so tiles
from chr15 on sit at shifted addresses and many chr21 and chr22 molecules fall in no tile. `run_sample.py --dev-tiles dictionary` builds the
tiles from the dictionary instead. On one spent healthy plasma (`data/DEV_ONE_DRAW_01/tiles_01/tiles_compare.txt`): more tiles are read,
the loose share moves in the fourth decimal, the held error is unchanged (it uses no tiles). The commissioned tiles stay the default until
the self-reference line's healthy samples are recounted with the dictionary's tiles.

**The megakaryocyte stand-in, printed by the one command** (development). Every plasma sample's bundle now carries
`megakaryocyte_stand_in`: the summed shares of the four cells pure megakaryocyte DNA lands on in absorption test 1 (lung interstitial
macrophages, erythrocyte progenitors, MEP (bone marrow), eosinophils), each cell's share beside it, read against the person's own level.
It is read where it was measured, with IAM-Atlas v2.1 as the Deconvolver's ruler. Against the IAM Sequencing Atlas it is not read, and the
bundle names the stand-in cells that atlas lacks (MEP and eosinophils). Absorption test 1 is its check (planted megakaryocyte DNA raised the
sum as mixing predicts); nothing new is scored.

## The 30 tuning plasmas read again against the 57-cell IAM Sequencing Atlas (plasma_seq_02)

Written and committed before any of them is read (development; the 30 were spent for tuning in plasma_seq_01). **The instrument's job:** a
healthy plasma, read as a mix of the atlas's cells in its own shares, reads Normal. **Why again:** plasma_seq_01 read them against
IAM-Atlas v2.1 with each site's level treated as exact, before Met-A, the territories and the Deconvolver were made to work on sequencing
against the sequencing atlas (healthy passes 5 and 7), and before the atlas held granulocytes. **The run:** the same 30
(`plasma_seq_02/plasma_seq_02_samples.csv`), each aligned again from SRA by the committed pipeline (`erase_test.yml`, `full=1`; their
earlier molecule files are no longer kept) and read by the one command with the healthy passes' settings (`--full-chain --dev-seq-ref auto
--dev-whole-sites --dev-seq-level donors --dev-terr-counted`; no donor half, since no plasma is an atlas input). **Reported, nothing scored:**
the band of every reading over the 30 by the committed band scorer (`plasma_seq_01/plasma_band.py`), beside plasma_seq_01's. The
megakaryocyte stand-in is not read (it was measured on v2.1, and the sequencing atlas lacks MEP and eosinophils). The commissioning test
for plasma waits for the plasma line to read Normal here, and is then written before any untouched plasma is read.

**The first plasma, and a check written before the other 29 are opened.** SRR23279758 read end to end by the one command (median 1 call per
site, 407263 sites). Its islands read 2.59257 (the plug-in expectation 2.58541, so not the counted method;
`plasma_seq_02/plasma_floor_check_first.txt`), with 1.6 points more methylation than
its mix predicts (level change 0.01623); its library floor (unmarked side) is 0.022906, the atlas inputs' median 0.0062855: a difference of 0.0166, the size
of the islands' excess. The atlas's own inputs carried only 0.488 of their floor differences at the islands, so one plasma does not decide
it. **Written now, before the other 29:** `plasma_seq_02/plasma_floor_check.py` (each plasma's floor against its islands; the slope through
the origin, 1 if every false call at the floor sites also falls at the islands). Development; nothing scored.

**An interim look at 12 of the 30** (`plasma_seq_02/plasma_floor_check_12.txt`, the committed check; nothing scored): floors 0.022102 to
0.029562 (median 0.023822), the islands' level change on the floor difference through the origin 0.974, the islands reading against the
floor rho +0.608. In plasma, nearly every false call at the floor sites also falls at the islands (the atlas's own inputs carried 0.488,
where the floor differences are a few thousandths).

## The same 30 plasmas with the common floor (plasma_seq_03)

Written and committed before any of it runs (development; the 30 are spent for tuning). **The instrument's job:** a healthy plasma, read as
a mix of the atlas's cells in its own shares, reads Normal. **The run:** each of the 30 counted again from the molecules plasma_seq_02 saved
(`erase_test.yml`, `full=1`, `patfrom` the plasma_seq_02 run; no new alignment), read by the one command with plasma_seq_02's settings and
`--dev-floor-equal` (`plasma_seq_03/plasma_seq_03_samples.csv`). **Reported, nothing scored:** the committed band scorer and floor check,
beside plasma_seq_02's. The flag stays off by default (healthy pass 6) until a plasma test written before reading says otherwise.

**plasma_seq_03's first reads: the flag was not applied (my error, disclosed).** 25 of the 30 were counted again with `--dev-floor-equal` on
the command line, but the one command set the sample's floor (and the flag) only where it builds values from a `.beta` file or bigWig tracks,
not from molecules (`--pat`), which is how every plasma comes in; the floor synthetic read `.beta` files, so it did not see this. The first
bundle opened carried no floor record and read exactly as in plasma_seq_02; none of those 25 is used. **Fix:** the floor is set in one place
after every sequencing input is turned into values. One plasma is read first and its bundle checked for the floor record before the other
29 are sent (new run names, `pl3b_`).

**plasma_seq_02: the 30 against the 57-cell atlas** (`plasma_seq_02/plasma_band_02_summary.txt`, `plasma_floor_check.txt`; the committed
scorers on a GitHub machine; nothing scored). All 30 read. Against plasma_seq_01 (IAM-Atlas v2.1): Met-A at the identity sites of the cells
found 0.7594 to 0.9918; whole-sample Met-A 0.8302 to 0.9205; the domains 0.86 to 0.8666; the globe's changed blocks 21.5 to 0. The largest
share is now granulocytes (0.319), which v2.1's plasma reading gave to lung interstitial macrophages. IAM-A reads each plasma's own
molecules, and the same committed pipeline reproduced them exactly (IAM-A 0.7733, held error 0.03854, both as in plasma_seq_01). **The
islands read 2.681**, against 0.8391 on v2.1: every plasma's floor (0.022102 to 0.037124, median 0.024661) is several times the atlas's,
and the islands' level change follows it with slope 0.939 through the origin (rho +0.719). Against the array-based v2.1 the same plasmas'
islands read low; against a sequencing atlas whose own floor is 0.0063 they read high, so the ruler's floor matters as much as the
sample's.

**plasma_seq_03: the common floor brings plasma's islands to 0.9272** (`plasma_seq_03/plasma_band_03_summary.txt`,
`plasma_floor_check_03.txt`; the same committed scorers on a GitHub machine; nothing scored). All 30 counted again from plasma_seq_02's
molecules with `--dev-floor-equal` (after the fix). The islands read 0.9272 (25 to 75 % 0.8803 to 0.956; 5 to 95 % 0.8194 to 0.9637),
against 2.681 without the floor; against the floor the islands now run slightly the other way (rho -0.428), as the slope below 1 (0.939)
predicts: a little over-corrected. Everything else is unchanged, as it should be (the floor enters only the islands and sorted-cell Met-A):
whole-sample Met-A 0.9205 (its islands part 2.711, where the floor is not applied), the domains 0.8666, IAM-A 0.7733. **Open on plasma:**
the domains and whole-sample Met-A below 1, and IAM-A (which falls as the loose share rises, rho -0.94, as in plasma_seq_01).

**Plasma at plasma depth: synthetic (written and committed before it runs)** (`plasma_seq_03/plasma_depth_synthetic.py`). Six constructed
plasmas, healthy by construction, with a spent plasma's own read count at every atlas site and its own cell shares, every read drawn from
the atlas's pooled level of a cell of the mix; read by the one command with plasma_seq_02's settings. Bars: the median domains reading, and
the median whole-sample Met-A, each within 0.02 of 1. If met, the instrument reads Normal at plasma depth and the plasmas' shortfall is in
the plasmas (cells the atlas lacks, or the plasmas themselves); if not, it is the reading at that depth. Run on a GitHub machine from
SRR23279758's saved molecules and plasma_seq_02 bundle.

**Plasma at plasma depth: synthetic not met** (`plasma_seq_03/plasma_depth_synthetic.txt`; the bars unchanged). At SRR23279758's own depth
(407263 sites, median 1 read per site) six plasmas healthy by construction read the domains 0.9307 to 1.1279 (median 0.9766) and
whole-sample Met-A 0.9507 to 1.0816 (median 0.9759): both just outside 0.02 of 1. What the run shows beyond the bars: at 1 read per site
the reading of one plasma moves by several hundredths between draws of the same truth, and the Deconvolver does not hold the planted mix
(the sum of |returned - planted| over the planted cells 0.4359 to 0.7826, so about a fifth to two fifths of the mix placed elsewhere);
Met-A at the identity sites of the cells found reads 0.9594 to 1.0724, the islands 0.8832 to 1.0314. The real plasmas' domains (0.8666)
sit below even this. **Next, written before it runs:** the same construction at stated depths, to find the depth at which plasma reads
Normal; it spends no data. (`plasma_seq_03/plasma_depth_sweep.py`: D in 2, 5, 10, 20 reads, three each; bar, at 20 both medians within 0.02 of 1.)

**The depth sweep: met at 20 reads** (`plasma_seq_03/plasma_depth_sweep.txt`). Constructed healthy plasmas (the same truth and shares) at
Poisson(D) reads per site, medians of three: at 2, domains 0.9574 and whole-sample Met-A 0.9531; at 5, 0.9854 and 0.9874; at 10, 0.9685 and
0.9801; at 20, 1.0137 and 1.0078 (bar met), with the islands 1.0036 and Met-A at the identity sites 1.0034. The Deconvolver's share gap
falls from 0.523 at 2 reads to 0.1925 at 20, never below 0.1 on this fourteen-cell mix (recorded). **What it says about the 30 plasmas:**
at their 1 read per site, depth alone moves the domains a few hundredths (the synthetic at that depth read 0.9766); the real plasmas read
0.8666, so most of their gap is in the plasmas, not the reading. **Next, written before it runs:** a constructed plasma holding a cell the
sequencing atlas lacks. Megakaryocytes are the first candidate on our own evidence: pure megakaryocyte DNA lands on erythrocyte
progenitors and other cells in IAM-Atlas v2.1 (absorption test 1), and the plasmas' largest share after granulocytes is erythrocyte
progenitors.

**A plasma holding a cell the atlas lacks: synthetic (written and committed before it runs)** (`plasma_seq_03/plasma_mk_synthetic.py`).
The depth sweep's construction at 20 reads, with a megakaryocyte fraction m of 0, 0.1 and 0.25 at the provisional megakaryocyte column's
pooled level, read against the 57-cell atlas (which has no megakaryocytes); only atlas sites where that column has at least 10 reads
(or 8, or 6, whichever first leaves the Deconvolver 500 markers), since a level counted from a few reads looks ordered. Bar: the control
(m = 0) reads Normal. Recorded: how the domains and whole-sample Met-A move with m, against the real plasmas' 0.8666 and 0.9205, and where
the Deconvolver puts the megakaryocyte DNA.

**A plasma holding megakaryocyte DNA: the control not met; megakaryocytes a small part** (`plasma_seq_03/plasma_mk_synthetic.txt`; bar
unchanged). 146979 atlas sites with at least 8 megakaryocyte reads (19581 domain sites, 1033 Deconvolver markers; 10 reads left fewer
than 500 markers). The control (m = 0) read the domains 1.0210 and whole-sample Met-A 1.0176: 0.001 outside the bar on the domains, not met.
With megakaryocyte DNA: m 0.1, domains 1.0089 and whole-sample Met-A 1.0039; m 0.25, 0.9866 and 0.9719 (the islands 1.0028 to 0.9280);
the Deconvolver spreads it, mostly onto erythrocyte progenitors (0.248 at m 0.25) and NK cells. So a quarter of the DNA from a cell the
atlas lacks lowers the domains by about 0.034, a small part of the plasmas' gap (0.8666). In the 30 real plasmas the domains follow
whole-sample Met-A (rho +0.973, `plasma_seq_02/plasma_floor_check.csv`), not the floor (rho -0.055), and spread 0.6857 to 1.0699, wider
than depth alone gives. **Next (diagnostic, nothing scored):** `plasma_seq_02/plasma_shares_diag.py`, each plasma's domains and
whole-sample Met-A against every cell's share and the sites read.

**The plasmas' domains follow the erythrocyte progenitor share** (`plasma_seq_02/plasma_shares_diag.txt`, `domains_by_cell.txt`;
diagnostics, nothing scored). Over the 30, the domains fall as the Deconvolver's erythrocyte progenitor share rises (rho -0.928; median
share 0.093); no other cell comes close (granulocytes +0.384), nor the sites read (-0.119). The erythrocyte progenitor column carries the
most expected disorder at the domains of all 57 cells (pooled level 0.652, disorder 0.8534, against a median of 0.4573), and that is in each
person: its donor spread there is moderate (0.0815) and its three inputs read their own domains 0.9879, 1.0287 and 0.9712 in the 57-cell
healthy pass. **So the plasma DNA the Deconvolver gives to erythrocyte progenitors is more ordered at the domains than that cell.** The
megakaryocyte synthetic did not reproduce it (its megakaryocyte DNA went mostly elsewhere). What that DNA is stays open.

## The 30 plasmas with erythrocyte progenitors held out of the Deconvolver (plasma_seq_04)

Written and committed before any of it runs (diagnosis only; the 30 are spent for tuning). **The question:** what is the plasma DNA the
Deconvolver gives to erythrocyte progenitors? Held out of its reference (`run_sample.py --dev-deconv-drop`, new, flagged in the bundle as
`cell_shares.held_out`), that share goes to the cells that remain. If it is erythrocyte progenitor DNA, the domains should read no closer
to 1 and the share should scatter; if it is a cell whose domains are ordered, the domains should come toward 1 and the share land on a
few cells. **The run:** the 30 counted again from plasma_seq_02's saved molecules with plasma_seq_03's settings (the common floor) and
`--dev-deconv-drop erythrocyte_progenitors` (`plasma_seq_04/plasma_seq_04_samples.csv`). **Reported, nothing scored:** the committed band
scorer and share diagnostic, beside plasma_seq_03's. Not a reading the chain uses: the Deconvolver's reference keeps every cell.

**plasma_seq_04: holding erythrocyte progenitors out trades one gap for another** (`plasma_seq_04/plasma_band_04_summary.txt`,
`plasma_shares_diag_04.txt`; the committed scorers; nothing scored). All 30 read. The domains come to 1.023 (5 to 95 % 0.9239 to 1.104;
from 0.8666) and whole-sample Met-A to 1.044 (from 0.9205), but Met-A at the identity sites of the cells found rises to 1.109 (from 0.9918),
and the islands read 0.9149 (from 0.9272). The share goes mostly to blood cells (granulocytes 0.347, monocytes 0.087, NK cells 0.029), and
the domains no longer follow any one cell (largest rho -0.243). **What it says:** the plasma DNA the Deconvolver gives to erythrocyte
progenitors matches that cell at the identity sites (taking it away raises the disorder there) but not at the domains (keeping it lowers
the domains). It reads as erythroid at its identity sites and more ordered than the atlas's bone-marrow erythrocyte progenitors at the
domains. Neither panel reads every reading Normal; the Deconvolver's reference keeps every cell. What that DNA is (an erythroid cell at a
stage the atlas does not hold, or a cell it lacks) stays open, and needs data of that cell, not a change to the reading.

**Public data for the erythroid question (a search; nothing read).** The published plasma study of erythroblast and megakaryocyte DNA used
bone-marrow erythroblasts from the same series as the atlas's erythrocyte progenitors (GSE186458), so it adds no new erythroid stage. The
same study deposited platelet whole-genome bisulfite sequencing (GSE206818), a bulk source from the megakaryocyte lineage, untouched and
kept for commissioning unless the author chooses to spend it on a megakaryocyte column.

## IAM-A on plasma, on the molecule-level loose share (diagnostic, written and committed before it runs on the plasmas)

**The instrument's job:** a healthy plasma reads Normal on IAM-A. On the 30 tuning plasmas it reads 0.7733, following their loose share
(rho -0.94), and synthetic test cfDNA 2 showed why: mixing raises the tile loose share, while the share of loose molecules stays put. The
atlas's 153 inputs already carry the molecule-level share in their count records (`iama_mol_01/healthy_count_records.csv`, copied from
them). `iama_mol_01/iama_mol_lines.py` builds a line on each share the way the chain builds its own (per cell, a Theil-Sen line across 56
cells) and reads every plasma bundle on both, on a GitHub machine. On the healthy inputs alone, before any plasma: 0.9498 on the tile
line, 0.9678 on the molecule line. **Expected if the mix is the cause:** the plasmas read near the healthy inputs on the molecule line and
low on the tile line. Nothing scored; it says whether the molecule-level line is worth building into the chain, synthetic first.

**Read: the molecule-level line does not lift plasma** (`iama_mol_01/iama_mol_lines.txt`; GitHub run on plasma_seq_02's 30 bundles). The 30
plasmas read 0.7618 on the molecule line, against 0.7733 on the tile line; the healthy inputs read 0.9678 and 0.9498. The expectation for the
mix as the cause is not met: plasma's molecule-level share (0.3310) sits above the healthy inputs' (0.2730) as its tile share does, so the
mix is not what holds plasma's IAM-A down. That leaves the other cause synthetic test cfDNA 1 found: plasma's short molecules, where the
qualifying rule (at least 6 calls, at least 80 % methylated) drops molecules carrying a second error and lowers the held error counted.
**Next:** that length bias computed exactly for each molecule length, as the read-count correction was, synthetic first.

## Plasma's short molecules: the length bias removed, synthetic test cfDNA 3 (written and committed before it runs; plan 40.11)

**The instrument's job:** a sample's held error does not depend on how long its molecules are. The held error counts isolated losses on
molecules that qualify (at least 6 calls, at least 80 % methylated); a short molecule carrying a second loss fails to qualify, so short
molecules count low. Derived exactly (`stage_q2_selfref.qual_iso_expect`, a walk over every pattern of a molecule's calls; nothing fitted;
`length_bias_expect.txt`): for each length, the expected isolated losses on a qualifying molecule at a per-call loss e. Stage Q2 now records the qualifying molecules by
length, and `held_err_length`: the e at which the expected losses of the molecules actually counted equal the losses counted, written as the
held error a molecule would show with no qualifying rule, e (1 - e)^2. It is used by no reading yet. The derivation reproduces synthetic test
cfDNA 1's measured held errors (per opportunity, at a loss of 0.035: 0.02979, 0.02811 and 0.03161 expected at 6, 8 and 12 calls, against
0.03000, 0.02786 and 0.03186 counted). **Synthetic test cfDNA 3** (`synthetic_cfdna_03.py`): one constructed cell, molecules cut to 6, 8,
12 and 20 calls, through the one command. Bars: the corrected held error within 5 % of the known answer at every length, and its spread
across lengths under half the raw spread. If met, the healthy line is rebuilt on the corrected held error (the 153 inputs counted again for
their tallies by length) and the 30 plasmas read on it.

**Synthetic test cfDNA 3: met** (`synthetic_cfdna_03_summary.txt`). Truth 0.03259. Raw held error 0.02956, 0.02864, 0.03153 and 0.03260 at
6, 8, 12 and 20 calls; corrected 0.03232, 0.0333, 0.03251 and 0.03267, every length within 5 % (bar 1 met); spread across lengths 0.00098
against 0.00396 raw (bar 2 met).

## IAM-A on plasma on the length-corrected line (written and committed before any reading)

**The instrument's job:** a healthy plasma reads Normal on IAM-A. With synthetic test cfDNA 3 met, the healthy line is rebuilt on the
corrected held error. **Readings:** healthy pass 8, the atlas's 140 inputs counted again by the one command with pass 7's list and settings
(`DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/healthy_fullchain8_samples.csv`), and plasma_seq_05, the 30 tuning plasmas counted again from
plasma_seq_02's saved molecules with its settings (`plasma_seq_05/plasma_seq_05_samples.csv`), both on GitHub. **Scorer**
(`iama_len_01/iama_length_line.py`): a raw line and a corrected line across the healthy cells, built as the chain builds its line; every
sample read on both. **Expected if length is the cause:** the plasmas read near the healthy inputs on the corrected line, and about 0.77 on
the raw line. Nothing scored. Both passes' readings go into the A-score database when they land.

**Read: the length-corrected line does not lift plasma either** (`iama_len_01/iama_length_line.txt`; healthy pass 8, 140 of 140 jobs, and
plasma_seq_05, 30 of 30, both on GitHub; scored on a GitHub machine). The lines were built from 48 cells (137 healthy inputs with both
values). Healthy inputs: 0.9491 on the raw line, 0.9420 on the corrected line. Plasmas: 0.7768 raw, 0.7485 corrected. The correction raises
the healthy inputs' held error (0.04461 to 0.05139) as well as the plasmas' (0.03854 to 0.04331), and plasma stays below what its loose
share (0.2864 against 0.1339) predicts on either line. **The expectation for length as the cause is not met.** With the mix ruled out
before, neither cause synthetic test cfDNA 1 found explains plasma's low IAM-A. Plasma's held error is lower, and its loose share higher,
than a healthy cell's; why is open, and is not assumed. `held_err_length` stays recorded and used by no reading.

**Disclosed (mine):** the dispatcher for plasma_seq_05 matched the registry's scoring runs as samples and sent two meaningless runs, which
failed and were removed; the same error as before, now fixed by matching sample identifiers.

**The platelet set cannot build a megakaryocyte column (its files checked; no methylation value read).** GSE206818's public files are one
merged level track of its three platelet donors (a bedGraph of methylated fractions, and its bigWig); the reads are protected, and there is
no per-donor file and no read count. The IAM Sequencing Atlas counts every cell from donors' reads and seals its donors into two halves, one
to build and one to test, so a merged level track can be neither counted nor split. The set stays unspent. The erythroid question still
needs bulk data, with reads, of a megakaryocyte or erythroid cell the atlas lacks.

**PRJNA929650's stage and subtype, from its publication (a table, no methylation value read)** (`plasma_seq_01/prjna929650_clinical.py`,
`plasma_seq_01/prjna929650_clinical.csv`). The publication's Table S1, taken from Europe PMC's open-access supplementary files, gives every
one of the 517 plasmas its age, tumor stage, tumor type, tumor count and diameter, and molecular subtype; all 517 match the run list. Of the
239 nonmetastatic breast cancers: stage I 44 (I, IA, IB), stage II 100 (II, IIA, IIB), stage III 34 (III, IIIA), not recorded 61. The
A-score database now carries each plasma's stage and subtype (`CANON/ascore_labels.py`). The 239 stay unread until plasma reads Normal on
healthy plasma on every reading; they are then a cancer set with recorded stage, as the A-score database needs.

**IAM-A in the plasma's own mix: one open question, not two** (`plasma_seq_02/iama_mix_diag.py`, `iama_mix_diag.txt`; added after both
were read, disclosed; nothing scored). Read against a line across cells, IAM-A puts 13 of the 30 tuning plasmas inside the healthy inputs'
band (median 0.7733). Read against the plasma's own mix of cells, which the chain already prints, 25 of 30 sit inside the healthy inputs'
band (median 0.9189; healthy inputs 0.9977, 80 read). What remains follows the domains (rho +0.756) and the erythrocyte progenitor share
(rho -0.634): the plasmas that read lowest on IAM-A in the mix are the ones whose domains read lowest and that put most DNA on erythrocyte
progenitors. So plasma's low IAM-A and its low domains point at the same DNA, which the atlas reads as erythroid at the identity sites
and as more ordered than its bone-marrow erythrocyte progenitors at the domains. What that DNA is stays open, and needs data of that cell,
with reads.

## Is plasma's "erythroid" DNA megakaryocyte DNA? (written and committed before it runs; development, diagnosis only)

Absorption test 1 found that pure megakaryocyte DNA, read on IAM-Atlas v2.1, lands partly on erythrocyte progenitors, and the published
plasma study reports megakaryocyte DNA as a large share of healthy plasma. The IAM Sequencing Atlas holds no megakaryocyte column; its
provisional one (donors 1 and 2 of the single-cell series, spent) failed Met-A's test on donor 3 for too few reads per site, but the
Deconvolver needs only its marker sites. **The chain:** `stage_seq_atlas.deconv_reference(..., with_mk=True)` (`run_sample.py --build
seq-deconv-ref --build-with-mk`) builds a development reference that keeps that column, and `run_sample.py --dev-deconv-ref PARQUET` reads a
sequenced sample's cell shares against it, flagged in the bundle. **Synthetic first** (`seq_atlas/deconv_mk_synthetic.py`): six plasma-like
mixes planted from that reference's levels, with megakaryocyte and erythrocyte progenitor shares from 0 to 0.3, read against it and against
the committed reference; bars: both shares within 0.05 of planted in every mix against the megakaryocyte reference. The build and the
synthetic run on one Google spot machine. **If met:** the 30 tuning plasmas read again from their saved molecules against it. **Expected
if the DNA is megakaryocyte DNA:** the erythrocyte progenitor share moves to megakaryocytes and the domains move toward 1.

**The megakaryocyte reference and its synthetic: met** (`chain/Runtime Matrices/Seq_Atlas/deconv_reference_mk_dev.parquet`, built by the
one command on a Google spot machine from the 57-cell pieces and the provisional megakaryocyte pieces: 58 cells, 5799 marker sites;
`seq_atlas/deconv_mk_synthetic.txt`). In all six mixes the megakaryocyte and erythrocyte progenitor shares read within 0.05 of planted
(largest error 0.015 and 0.013). Recorded, not scored: against the committed reference, planted megakaryocyte DNA goes mostly to
granulocytes and monocytes and little to erythrocyte progenitors (0.3 planted, 0.041 to them), unlike absorption test 1 on IAM-Atlas v2.1.
**Next:** plasma_seq_06, the 30 tuning plasmas read again from plasma_seq_02's saved molecules with `--dev-deconv-ref
deconv_reference_mk_dev.parquet` (`plasma_seq_06/plasma_seq_06_samples.csv`), scored with the committed scorers.

**plasma_seq_06: megakaryocytes take a small share; the domains could not be read** (`plasma_seq_06/mk_shares_compare.py`,
`mk_shares_compare.txt`, `plasma_shares_diag_06.txt`, `plasma_band_06_summary.txt`; 30 of 30 read on GitHub; nothing scored). With the
megakaryocyte reference, the erythrocyte progenitor share falls from 0.0926 to 0.0358, while megakaryocytes take 0.0235 (5 to 95 % 0.0038 to
0.0675); the fall does not follow the megakaryocyte share (rho -0.186), and the megakaryocyte share does not follow plasma_seq_02's domains
(rho +0.230). Adding a cell also changes the marker sites, so other shares move (monocytes 0.0459 to 0.0132). **What it says:** megakaryocyte
DNA is at most a small part of the DNA read as erythroid. **The domains and whole-sample Met-A were not read,** for two reasons in the chain:
1. a reading for each cell stopped at the megakaryocytes, which are not a cell of IAM-Atlas v2.1 where Met-A's identity sites are chosen,
and the readings after it in the same step were not reached (fixed: `stage_metaa_atlas.read_in_mix`, `departure_in_mix` and `read_whole`
now name such a cell and read the rest); 2. even read, a site is kept only where
cells holding 99 % of the shares have a value, and the sequencing atlas's site tables hold no megakaryocyte column, so a share of 0.024
would blank every site. **Next:** the provisional megakaryocyte column at v2.1's sites, development only, so the mix has its levels.

**Plasma from a laboratory the atlas has never seen, healthy and cancer (GSE186007; written and committed before any of its plasmas is read).**

**The instrument's job:** a healthy plasma from a new laboratory, tared on that laboratory's other healthy plasmas, reads Normal on the gauge.

One laboratory's 100 human plasmas (34 healthy, 50 colorectal cancer, 16 pancreatic cancer; GEO labels only were read), posted as
Bismark coverage files on hg19, the atlas's own numbering, so the chain reads them with no alignment (`--build seq-table`, then the one command
with the plasma passes' flags). Job `data/DEV_ONE_DRAW_01/plasma_lab3/plasma_lab3_read.sh`: each plasma is uploaded the moment it is read and
one already read is skipped, so a restarted machine resumes. Sample list `plasma_lab3_samples.csv`; scorer `plasma_lab3_score.py`.
**Bars** (healthy plasmas only, at least 20 read): 1. the tared whole-sample Met-A median within 0.95 to 1.05; 2. at least 80 % of the healthy
plasmas' tared whole-sample Met-A within 0.95 to 1.05. Tare: each healthy plasma against the median of the laboratory's other healthy plasmas
(leave-one-out); each cancer plasma against the median of all its healthy plasmas. **Recorded, not scored:** the untared readings (the
laboratory's offset), the islands and domains tared the same way, the C-Score, the Deconvolver's shares, the library floor, and the cancer
plasmas by type, for the A-score database (where healthy ends and cancer begins is for experts to set).

**Sources:** GSE186007 (public, GEO: the 100 human plasmas' Bismark coverage files on hg19 and their labels, group, age and sex); the IAM Sequencing Atlas (public, Zenodo: the healthy reference every plasma is read against); the hg19 CpG dictionary (public, the repository's hg19-bwameth-index-v1 release).

**Prior reading:** none of these 100.

**Plasma from a new laboratory, second reading (written before it runs; the first reading's flags could not read whole-sample Met-A).** While the first reading ran, 9 of its first 10 healthy plasmas returned no whole-sample Met-A ("only 0 sites with an expectation";
bundles in the run's artifacts). Cause, from the chain's code (`stage_metaa_atlas.py`, `MIX_COVER = 0.99`): whole-sample Met-A counts a site only
when cells with an atlas value there hold at least 99 % of the sample's shares, and the first reading's flags read the shares against the
Deconvolver's development reference with a megakaryocyte column (`--dev-deconv-ref deconv_reference_mk_dev.parquet`) that the IAM Sequencing
Atlas does not hold; every plasma with a megakaryocyte share above 1 % (1.9 % to 5.5 % in these) then has no site to read. The one plasma read had
0.14 %. The first reading runs to its end as written and is scored as written. This second reading changes only the flags, to the earlier plasma
pass's committed reference (`plasma_lab3_samples_r2.csv`, the rsflags column; the list is otherwise the same), with the same job, the same bars and
the same scorer. No whole-sample Met-A value of these plasmas decided this change; only the refusal reason and the shares were read.
**Change for later (chain code, after the running readings finish):** a cell the Deconvolver finds but the atlas does not hold should not empty
the whole-sample reading.

**Plasma from a new laboratory, first reading: not met** (`plasma_lab3/reading_1/plasma_lab3_r1_summary.txt`, `plasma_lab3_r1_readings.csv`; the 34 healthy plasmas, each read by the one command,
one per GitHub job.) 9 of 34 returned a whole-sample Met-A, under the 20 the bars need, so neither bar is met. The other
25 returned "only 0 sites with an expectation" (counted from the bundles), each with a megakaryocyte share of 1.3 % to 8.6 %; the 9 read had 0.98 % or less: the cause found during the run (no megakaryocyte column in the atlas, above). The
values of the 9 read are in the summary as recorded, not scored. The second reading (above) runs next.

The scorer's not-read list now gives the whole-sample reason when the bundle has no top-level one (it listed none of the 25, and the record first said 0; corrected from the bundles). Bars and tare unchanged.

**Plasma from a new laboratory, second reading: not met** (`plasma_lab3/reading_2/plasma_lab3_r2_summary.txt`, `plasma_lab3_r2_readings.csv`; scored on GitHub by the committed scorer, run
42313573989.) All 34 healthy plasmas read. Bar 1 met: the tared whole-sample Met-A median is 1.0002. Bar 2 not met: 0.147
of them within 0.95 to 1.05 (at least 0.80 needed); the tared readings run from 0.228 to 1.515 (standard deviation 0.257).
The islands and domains, tared the same way, scatter as widely (standard deviations 0.281 and 0.271).
**Recorded, not scored:** the C-Score of every one of the 34 reads within 0.95 to 1.05 (median 0.9985, standard deviation 0.0053),
on plasma from a laboratory the atlas has never seen. The cancer plasmas stay unread: the healthy plasmas have not read Normal. **Next
(development):** why the whole-sample reading scatters on these plasmas when it held on the earlier plasma passes; these plasmas are read at 1 to 3
calls a site, and on sequencing Met-A falls with depth (DEV_METAA_ATLAS_CURVE_01), so depth is checked first, from the bundles.

**Depth does not explain the second reading's scatter** (`plasma_lab3/reading_2/plasma_lab3_r2_depth.txt`, `plasma_lab3_depth.py` on GitHub from the bundles; development). The healthy
plasmas' median calls a site run from 1 to 20; Spearman with the tared whole-sample Met-A -0.191 (p 0.28), with its distance from 1 +0.211 (p
0.23); the ten deepest (15 to 20 calls a site) still read 0.79 to 1.51. Next: the Deconvolver's shares, from which the reading's expectation is
built (`plasma_lab3_depth.py` version 2).

**The Deconvolver's shares do not explain it either** (`plasma_lab3/reading_2/plasma_lab3_r2_shares.txt`; development). Of the cells holding at least 1 % of the healthy plasmas, the strongest
link to the tared whole-sample Met-A is colon macrophages, Spearman +0.331 (p 0.056); granulocytes +0.301 (p 0.084); the unexplained remainder
-0.014. Next (`plasma_lab3_depth.py` version 3): the library floor (false calls at sites every cell holds unmarked or marked), the ceiling
statistics, the foreign fraction and the coverage by chromosome, each from the bundles.

**The library method, and a measure of copies (development; synthetic met, written before any plasma is measured).** Version 3 of the check (`reading_2/plasma_lab3_r2_library.txt`) put the scatter in the genome's bulk ("the rest" of the sites, Spearman
+0.997 with the whole-sample reading) and tied it most strongly to the turnover share, the share of sites read past 0.5 to the far side of
their healthy state (Spearman -0.475, p 0.0045; this laboratory's healthy plasmas median 0.110, up to 0.19, against 0.068 on the earlier plasma
passes' 60 healthy plasmas, whose whole-sample Met-A spread was 0.067 against 0.26 here). Neither a per-batch tare (the two depth groups) nor
depth removes it. GSE186007's method is linear amplification-based bisulfite sequencing (its series title). If its calls are copies of fewer
molecules, every site reads nearer 0 or 1 than its molecules allow, and by an amount set by each library's amplification. **The measure**
(`plasma_lab3/plasma_lab3_copies.py`): at the library-floor sites every healthy atlas cell holds unmarked, a methylated call is a false call;
the dispersion index D of the false calls is near 1 for independent molecules and near the copies per molecule for copies. **Synthetic (met,
`plasma_lab3_copies_synthetic.txt`):** D recovers 1, 2, 4 and 8 copies at about 2 and 15 calls a site. **On the 34 healthy plasmas (job
`plasma_lab3_copies.sh`, one per GitHub job, cancer plasmas left unread):** D for each, and Spearman of D with the tared whole-sample Met-A
and with its distance from 1. If D is well above 1 and tracks the scatter, the chain must count molecules, not calls, for such libraries.

**The copies measure on the 34 healthy plasmas: the calls are copies, and the copies drive the scatter** (`plasma_lab3/reading_2/plasma_lab3_r2_copies.txt`; one plasma per GitHub job, run 42326420243; development). Every healthy plasma's false
calls at the floor sites are overdispersed: D median 2.358, range 1.192 to 270.461 (independent molecules give 1). The ten deep plasmas (15 to
20 calls a site in the depth file: 153, 154, 158, 160, 161, 163, 165, 166, 167, 168, each GSM7426 followed by these digits) are the most copied, D 4.6 to 270 in
the rows: their depth is mostly copies.
D tracks the reading: Spearman with the tared whole-sample Met-A -0.480 (p 0.004); the three lowest readings (0.228, 0.455, 0.568) have D 22.8, 11.1, 5.3.
**What it means:** the chain counts every call as a molecule; on a linear-amplification library the molecules are fewer, by each library's
own D, so the chain's expectation of counting noise is wrong for that library, and wrong by a different amount for each. The false-call rate
itself is ordinary (median 0.00586). **Next (chain code, after the running readings finish; synthetic first):** each sequenced sample's own D,
measured at the floor sites, carried into the chain so that a sample's calls are counted as molecules (calls divided by D) wherever counting
noise enters a reading; the synthetic plants copies in a constructed plasma and must show the readings move with copies before and not after.
The cancer plasmas stay unread.

**The copies correction in the chain (development flag; written before its synthetic runs).** `run_sample.py --dev-seq-copies` (off by default; without it every reading is unchanged). Before any reading, the chain measures the
sample's own copy distribution at the library-floor sites (`chain/stage_seq_copies.py`, `copy_distribution`): a false molecule is rare there,
so the calls at a site with a false call are one molecule's copies, and the sites holding two false molecules are removed by deconvolution
(their share follows from the false-call rate and each site's molecules; Poisson). Then every expectation built on the sample's own calls
(`stage_metaa_atlas.expected_H_counts`: whole-sample Met-A, each cell's Met-A, the ceiling) is the expected disorder under that copy process
(`expected_H_copies`: each site's molecules drawn from its calls and the measured mean copies, each copied by the measured distribution, as a
ratio to the same simulation without copies times the exact binomial expectation). Measured from the sample; nothing fitted. Two forms were
tried first on the expectation formula alone and not used (working notes, not committed: they call chain pieces directly): counting calls / D
over-corrected (the dispersion index is weighted toward the most-copied molecules, so it runs above the mean copies), and calls / mean copies,
rounded, left a staircase error at the shallowest depth. **Synthetic through the one command** (`plasma_lab3/copies_synthetic.py`): a healthy
plasma by construction at the array-layout sites, in one GSE186007 healthy plasma's own shares (`copies_synthetic_shares.csv`), molecules
Poisson(6), the same molecules copied 1, 4 and 12 times, each read with and without the flag. Bars: 1. copies 1: the flag moves whole-sample
Met-A by at most 0.01; 2. copies 4 and 12 with the flag within 0.03 of copies 1 without it. **Then** one spent healthy plasma re-read without
the flag must give its second reading's value exactly (the chain unchanged without the flag), before the 34 are read with it.

**The copies synthetic: met; the third reading, written before it runs.** (`plasma_lab3/copies_synthetic_summary.txt`, GitHub run 42333729784.) Without the flag the constructed plasma read whole-sample Met-A
0.98611 uncopied, 0.73372 at 4 copies and 0.70158 at 12; with it 0.99543, 1.00398 and 0.99919, the chain measuring 1.0267, 4.0988 and 12.2511
copies. Both bars met. **Third reading:** the same 34 healthy plasmas, the same job, bars and scorer as the second, the flags of the second plus
`--dev-seq-copies` (`plasma_lab3_samples_r3.csv`). **Unchanged-chain check, alongside:** GSM7426135, read without the flag on the code that now holds
it, must give the second reading's whole-sample Met-A exactly (1.18639). If it does not, the third reading is not used. The cancer plasmas stay unread.

**Unchanged-chain check: met.** GSM7426135 read without the flag on chain 2c28120 (which holds it) gave whole-sample Met-A 1.18639, its second-reading value exactly (GitHub run 42337323172); the third reading is used.
