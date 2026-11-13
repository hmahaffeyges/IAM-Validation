# PHY-084: the referee test of the healthy baseline (SEALED 2026-11-13, before any library is read)

**In plain words.** From one number of physics (how many thermal kicks one ATP is worth) we worked out how often a healthy cell should
lose a DNA mark, then added the measured share caused by the cell's own eraser enzymes. Before looking, we wrote down what a new laboratory's
healthy blood cells should read on our meter, how wide the allowed range is and why, and what counts as a miss. Then we read them once.

**What it would take** to pass: all four healthy cell types land inside their ranges. To make the test sharper next time: the eraser's share
measured in mature human blood cells, and each cell type's own energy delivery, which together set most of today's width.

**Where IAM's Law is.** The prediction is the candidate (CANON `eps0_candidate`, 0.04454 per site): the loss of a mark held against
thermal kicks by a constant push of two coins (two ATP, M = 20.94 kT each) per restore, from M alone (PHY-083). One measured input the law
does not set is added: the eraser's share of the loss, taken as the band between the two measured ratios, 0.8642 (all erasers, stem cells,
DEV-ERASE-01 recount) and 0.905 (TET2 alone, blood progenitors, PHY-085). Nothing is fitted to the libraries read here.

**Data (reserved, never read).** GSE173787 (another laboratory, Shenzhen): the 58 healthy-control libraries, 24 CD19 B, 12 CD4, 12 CD8,
10 CD14 monocytes; whole-genome bisulfite, 24-252 million pairs each (`data/DEV_CHAIN_Q2_01/referee_samples.csv`, `referee_runs.csv`).
The 75 multiple-sclerosis libraries are not read.

**How each library is read.** The one command, on GitHub (`erase_test.yml` through `erase_gh.sh`): aligned to hg19 as the DEV-ERASE-01
recount, then `run_sample.py --dev-count-only` (Stage Q2), 10 million read pairs in blocks across the whole run, the genome-coverage check
kept (human). Canary first (one library, short); then all 58. A library the pipeline cannot read is reported, not replaced.

**Amendment 1 (2026-11-13, before any held error was computed from a referee library).** The canary (GSM5278426, 2 million pairs, GitHub run 43403531488)
was stopped at intake by the genome-coverage check: chr13, 16, 17, 19, 21 and 22 received no reads and chr4 13 %. GSE173787's runs are
stored in genome order, so the plan's 20 blocks each fall on a few chromosomes. The read method is changed, nothing else: 200 equal blocks
(50,000 pairs each, the same 10 million in total) evenly spaced through each run (`blocks` column of referee_runs.csv, read by
erase_blocks.py; the default of 20 is unchanged and gives byte-identical plans for all 32 earlier samples). The prediction, the bar and
pass/fail are untouched. The coverage check still decides; a library it refuses is reported, not read by another route.

**The prediction, through the meter.** For each cell type, the counter's lone-loss reading (`held_err_length`) that the forward model gives
at that cell type's own measured clustering (median `coupling` of its libraries, an instrument property read from the same libraries),
interpolated on the committed grid (`data/DEV_CHAIN_Q2_01/forward_out/pg_<ratio>_<x>.csv`: forward_candidate.py, seq mode, 3 samples of
120,000 molecules each, ratios 0.8642 and 0.905, clustering 1.19-2.14, read by the one command on GitHub). Middle of the two ratios. A
cell type whose clustering falls outside the grid is not extrapolated: the grid is extended by the same command first, and that is reported.

**The observation.** Each cell type's zero-loose intercept of `held_err_length` against `loose_share` (Theil-Sen; 2,000 resamples of
libraries, seed 20261113), as PHY-050.

**The bar (GUM, k = 2),** per cell type, relative to the prediction's middle: the spread of the healthy hold between cell types (8.8 %,
55 types, PHY-050 celltype_net.csv); the eraser band as uniform (about 1.4 %); the forward model's sampling (the grid's replicate spread);
copier lag in resting blood (0.4 %, lag_bound_blood.py); the intercept's own resampled error. About +/-18 % before the intercept's error.

**Pass and fail, fixed now.** MET if all four cell types' intercepts lie inside their bars. NOT MET if any lies outside; then the result
is reported as it is, with each cell type's pull, and nothing is re-tuned.

**Known before reading (stated so they are not hidden).** (1) In the Loyfer atlas data (spent), these four types read 0.88-0.98 of the
55-type median held error, so the fresh libraries are expected to sit at or below the middle of the bar. (2) The bar is wide because the
cell type's own delivery and the eraser's share in mature human blood are not measured; a pass shows the law plus the measured eraser is
in the right place, not that it is right to a few percent. (3) Another laboratory's preparation can carry its own molecule-end error;
the lone-loss counter is the one PHY-050 used, and each library's end damage is reported beside its count.

**Code.** `data/DEV_CHAIN_Q2_01/referee_score.py` (the scorer; usage in its header), committed with this record.

**Synthetic test:** DEV_CHAIN_Q2_01/forward_candidate.py, DEV_CHAIN_Q2_01/forward_candidate_summary.txt

The scorer's own synthetic test: `data/DEV_CHAIN_Q2_01/referee_score_synthetic.py` (`referee_score_synthetic.txt`): planted on the
prediction, MET with pulls -0.17 to -0.03; planted 30 % high, NOT MET with pulls 3.07-3.33.

**Decision.** The seal was made by the agent under the author's standing authority ("you know my answers 99 percent of the time so I give
you the authority to make them, just tell me what you did and why"). Why now: every input is measured and committed; waiting would add
nothing the referee data could not then be accused of being tuned to.
