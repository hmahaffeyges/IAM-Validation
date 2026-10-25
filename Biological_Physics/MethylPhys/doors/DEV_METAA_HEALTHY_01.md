# DEV-METAA-HEALTHY-01: Met-A's healthy middle line, predicted from the array itself (development)

**The instrument's job:** a healthy sample reads 1 on whole-sample Met-A, from any laboratory, with no healthy cohort and no reference arrays.

## Why healthy does not read 1 today

Untared, healthy whole blood reads differently from laboratory to laboratory (about 1.11 on GSE235717's example blood, about 1.03 on
GSE42861's controls, about 0.98 on AIBL, about 0.96 on GSE87571). The atlas expectation e is the same physics for each of them; what
differs is the array. An array does not report a site's true level t. It reports a compressed, noisy copy: a design's lowest readings sit
above 0 and its highest below 1, and every reading carries noise. Disorder H is curved, so both move H(x) away from H(t) even when nothing
in the blood has changed. EPIC-Italy showed the same thing from the other side: weaker arrays read more disorder.

## The prediction (written and committed before any test)

For each probe design d (I, II), from the array's own anchor sites (the atlas ruler's invariant sites: held below about 0.05 or above about
0.95 in every one of the 79 cell types, `Runtime Matrices/Stage_T/atlas_ruler_v1.json`):

- the array's own straight line, x = a_d + b_d e, through the medians of its low and high anchors against the atlas's medians there;
- the array's own noise, s_d = the median absolute residual of x from that line at the anchors, scaled to a standard deviation.

Then, at every site the whole-sample reading uses, the healthy expectation of disorder is what that array would show if the blood were
exactly the atlas in this sample's mix:

  E[H] = E over noise N(0, s_d^2) of H(clip(a_d + b_d e + noise, 0, 1)), by 20-point Gauss-Hermite quadrature,

and the predicted healthy reading is **A_pred = sum w E[H] / sum w H(e)**, with read_whole's own weights w = |0.5 - e|. The reading that
should be 1 in healthy blood is **A / A_pred**. Nothing is fitted: a, b and s are measurements of the array on sites where every healthy cell
agrees; the atlas and the shares are the chain's own.

**What is not derived yet, stated:** the atlas expectation e itself is measured (built from healthy cells), and the held error and each
cell's gap sit inside it; replacing e with a value predicted from the held error and the gap is the next layer (plan 14.4).

**Sources:** IAM-Atlas v2.1 (public, committed in this repository: each cell type's healthy level at every site) and its ruler file (public, committed: each site's probe design and the anchor sites); GEO GSE235717, GSE42861, GSE87571, GSE153712 and GSE51032 (public: raw IDATs of healthy whole bloods, all spent).

**Synthetic test:** DEV_METAA_HEALTHY_01/synthetic_02.py, DEV_METAA_HEALTHY_01/synthetic_02_summary.txt, DEV_METAA_HEALTHY_01/gain_counter_01.py, DEV_METAA_HEALTHY_01/gain_counter_01_summary.txt

**Prior reading:** every series here was read before by this project: GSE110554 (atlas sources and Met-A's six commissioning arrays; step 3 version 4), GSE235717 (the example healthy blood), GSE42861, GSE87571 and GSE153712 (replays R1 to R4), GSE51032 (EPIC-Italy) and GSE186458 (the atlas's own molecules); this is development, chosen from spent data on purpose, and no commissioning claim rests on it. GSE116229 (confirmation 1): spent for the leukemia replication; no count of its three cord-blood samples had landed or been read when confirmation 1 was written.

## Bars (at most three)

1. **Synthetic:** arrays built from the atlas mix with a known line and noise. A / A_pred reads 1.000 within 0.005 across the line and noise
   range seen on real arrays, and a planted 5 % loss of the held pattern still raises it by at least half as much as it raises A.
2. **Real healthy blood, five laboratories** (spent, healthy only: GSE235717, GSE42861 controls, GSE87571, AIBL controls, EPIC-Italy
   cancer-free): each laboratory's median A / A_pred lies within 0.03 of 1.
3. **The laboratories close up:** the spread of the five medians of A / A_pred is smaller than the spread of the five medians of A.

Synthetic first; the real check runs through `run_sample.py` on GitHub's machines.

## Synthetic test 1, run 1 (`data/DEV_METAA_HEALTHY_01/synthetic_01_run1_summary.txt`): bar 1 not met as written

- **Healthy:** A / A_pred read 0.98843, 0.99194 and 0.99714 on the mild, typical and weak arrays (1.0 on the ideal one), so not within 0.005
  of 1 on the first two. The measured line's low point came out about 0.003 above the true one (I: a 0.02276 against 0.02): a median of noisy
  values does not follow the straight line when the anchors' own levels are unevenly spread; a mean does.
- **The planted loss vanished** (A / A_pred +0.0000 on the ideal array while A rose by 0.1784). The plant moved every site, the anchors
  included, so the array's own line took the loss in as if it were the array's blur. **This is physics, not a fault to tune away:** a loss
  spread evenly over every site, the anchors included, is indistinguishable from a weaker array by any reading made on the array alone.
  Cancer's losses are not even; they fall in the loosely held domains, not at sites every one of the 79 cell types holds firmly.

**Changes for run 2, written before it runs:** the line through the anchors' MEANS (x = a + b e holds for means exactly, apart from clipping
at 0 and 1); and the planted 5 % loss placed off the anchors, at every site except the atlas ruler's anchor sites. Bar 1 is otherwise
unchanged. What the array alone cannot see, an even loss at every site, is stated as the reading's blind spot.

## Synthetic test 1, run 2 (`data/DEV_METAA_HEALTHY_01/synthetic_01_run2_summary.txt`): bar 1 met

| Synthetic array | Healthy A / A_pred | A 5 % loss raises A by | and A / A_pred by |
|---|---|---|---|
| ideal | 1.0 | 0.1720 | 0.1720 |
| mild | 0.99603 | 0.1310 | 0.1043 |
| typical | 0.99581 | 0.1158 | 0.0876 |
| weak | 0.9984 | 0.0943 | 0.0641 |

Healthy reads within 0.005 of 1 on every array, and A / A_pred keeps 68 % to 100 % of a loss's rise, above the half required. The array's
line and noise were measured within 0.003 of what was built in. **In plain words:** the prediction removes the array's own blur, so a healthy
sample reads 1 on any array, and a loss off the firmly held sites still shows. Next, bar 2: real healthy bloods from five laboratories,
through `run_sample.py` on GitHub's machines.

## Test 1 does not count (the chain gate, rule 23b)

Test 1 called the Met-A stage directly and took its shares from a file copied out of a chain result by hand. Under the rule that a result
the chain did not produce did not happen, it is kept above as the record of what was learned (the mean-based line; the even-loss blind
spot) and is replaced by **synthetic test 2** (`data/DEV_METAA_HEALTHY_01/synthetic_02.py`, written before it ran): a real healthy blood
read from its raw IDATs, then the synthetic arrays written as betas files, every one read by `run_sample.py`, so the self-tare and the
IAM-Deconvolver act on them as on any array. Same bar 1.

## Bar 2 and bar 3 (lists and scorer committed before any reading)

Twenty healthy whole bloods from each of five laboratories, the first twenty healthy arrays of each series in accession order
(`data/DEV_METAA_HEALTHY_01/bar2_labels.csv`): GSE235717, GSE42861 (its normal controls), GSE87571, GSE153712 (AIBL's healthy controls),
GSE51032 (cancer-free); all spent. Each through `run_sample.py` on GitHub's machines; scorer `bar2_score.py`.

## Synthetic test 2 (explore run 40101305939, commit f22216a4; `data/DEV_METAA_HEALTHY_01/synthetic_02_summary.txt`): bar 1 not fully met

Every array through `run_sample.py`.

| Array | A | A_pred | A / A_pred | a 5 % loss off the anchors: A rises | A / A_pred rises |
|---|---|---|---|---|---|
| real healthy blood GSM7508429 (raw IDATs) | 1.11291 | 1.09895 | 1.0127 | | |
| mild | 1.29338 | 1.29741 | 0.99689 | 0.1207 | 0.0948 |
| typical | 1.41335 | 1.41524 | 0.99866 | 0.1034 | 0.0736 |
| weak | 1.5511 | 1.56238 | 0.99278 | 0.0878 | 0.0583 |

- **Healthy within 0.005 of 1 on the mild and typical arrays, not on the weak one** (0.0072 below). Bar 1 is not fully met.
- **A loss still shows:** 66 % to 79 % of its rise kept, above the half required.
- **The real healthy blood:** its own measured line is nearly ideal (design I offset 0.0084, noise 0.00492), yet that small offset raises the
  disorder at nearly unmarked sites steeply; predicting it takes the reading from 1.11291 to 1.0127.
- **Open, not settled:** through the whole chain the IAM-Deconvolver re-reads each synthetic array's mix, and the mix drifts as the blur
  grows (the shares move by a total of 0.0499, 0.0904 and 0.1033 on the mild, typical and weak arrays; `share_drift.py`, added after the run, output `share_drift_summary.txt`), so the atlas expectation the
  prediction uses is not the one the array was built from. The typical array drifts almost as much and reads within 0.002, so the drift
  alone does not explain the miss.

## Layer 1: the atlas's healthy levels from the held error (written and committed before any file is counted)

**The instrument's job:** a healthy cell's level at a site it holds is set by how faithfully it holds the mark, so the atlas's healthy values
there come from physics and only which sites each cell holds is measured.

**The prediction.** Each healthy cell type c has two held errors, counted on its own molecules by Stage Q2: on the loss side (a lone
unmethylated call between two methylated ones, on molecules held marked: `held_err`) and, new, on the gain side (a lone methylated call
between two unmethylated ones, on molecules held unmarked, at least 6 calls and at most 20 % methylated: `gain_err`). Each counted lone rate
r is a per-call rate eps through r = eps (1 - eps)^2. Then, where c holds the mark, its healthy level is **1 - eps_loss(c)**; where c holds it
absent, **eps_gain(c)**. What is left, the atlas's level minus the prediction, is **the gap**: the part only biology sets (sites held partly,
or differently between cells of one type), measured once.

**Sources:** the published sequencing atlas of human cell types, GEO GSE186458 (public: 153 healthy .pat files of 56 cell types, spent;
`data/DEV_METAA_HEALTHY_01/layer1_samples.csv`), and IAM-Atlas v2.1 (public, committed: each cell type's healthy level at every site).

**Bars:**
1. **The counter (synthetic; `gain_counter_01.py`, output `gain_counter_01_summary.txt`; first run by calling the counter directly, then rewritten so each constructed file is read by `run_sample.py --dev-count-only`, with identical results):** constructed
   molecules with known loss and gain rates, recovered within 5 %, and the two sides within 5 % of each other at equal rates. **Met:** held
   side 0.956 to 0.965 of the expected rate, gain side 0.969 to 0.992; at equal rates gain over held 1.011. The slight shortfall is the same
   on both sides (the qualifying filter drops molecules carrying many errors); it is stated, not corrected.
2. **Marked side:** at the sites marked at 0.75 or above in all 79 cell types, |atlas median - (1 - eps_loss)| <= 0.02 in at least 45 of
   the 56 cell types.
3. **Unmarked side:** at the islands unmarked in all 79 (`stage_metaa_atlas.territory_sets`), |atlas median - eps_gain| <= 0.02 in at
   least 45 of the 56.

Every file through `run_sample.py --dev-count-only` (`count_atlas_01.py`, on GitHub's machines); scorer `layer1_score.py`. If the unmarked
side misses with the atlas above the prediction, that excess is the healthy islands' accumulated drift, a reading in its own right
(plan 48.3).

**Layer 1, scored** (`data/DEV_METAA_HEALTHY_01/layer1_summary.txt`; 148 of 153 samples counted, 56 cell types, each file through
`run_sample.py --dev-count-only` on GitHub's machines): bar 2 **not met** (22 of 56 within 0.02 on the marked side; 45 needed); bar 3
**not met** (37 of 56 on the islands). The misses split by platform, not by biology:

| Cells in the atlas | Marked side, median gap | Islands, median gap |
|---|---|---|
| measured by sequencing and arrays (17) | -0.004 | +0.028 |
| measured by sequencing only (39) | +0.027 | +0.004 |

IAM-Atlas v2.1 stores its levels on the array scale; the predictions come from sequencing molecules; the scorer compared them without
conversion. **Version 2** (`layer1_score_v2.py`, written and committed after version 1's gaps were seen, before it runs): one change, each
atlas median put on the sequencing scale through the atlas's own measured transfer (`DEV_ATLAS_V21_01/transfer_v21_final.json`); counts,
sites, predictions and bars unchanged; it rereads the same counts. Version 1's result stands as scored. If version 2 also misses, a second
cause is stated now: the predictions are each cell's genome-wide held error, while the bars' sites are those held most firmly in all 79
cell types.

**Layer 1 version 2, scored** (`data/DEV_METAA_HEALTHY_01/layer1_v2_summary.txt`, all 153 samples; `layer1_v2_summary_same148.txt`, the 148
version 1 read): **bar 2 met (54 of 56), bar 3 met (56 of 56), on both sets.** Version 1's scorer rerun on the same 148 gives its 22 and 37
again, so the scale is the only change. Outside 0.02: erythrocyte progenitors on the marked side (-0.067; the atlas holds them below the
prediction) and smooth muscle (+0.021).

**Disclosed:** (1) version 2 was written after version 1's gaps were seen, on the same counts, so this is a development result, not a sealed
one; a confirmation on samples that played no part (another laboratory's sequencing of healthy cell types) is the next step. (2) The first
run of version 2 printed 0 of 56 for both bars: a comment added in the edit had cut off the line that keeps the tally; the per-cell gaps
were right; the tally was restored and the scorer rerun (commit after `a75d1610`). (3) Five samples (GSM5652176, GSM5652220, GSM5652267,
GSM5652304, GSM5652348) landed after version 1 was scored; the result is the same with and without them.

**What it means:** each healthy cell type's level at the sites it holds is set by its own held error, counted on its own molecules (1 - eps
where the mark is held, eps where it is held absent), to within 0.02 in 54 of 56 cell types, with nothing fitted. The levels come from
physics; only which sites each cell holds is measured.

## Step 3: the derived neutrophil mark (written and committed before any granulocyte molecule is counted)

**The instrument's job:** a healthy neutrophil array reads Normal on Met-A against a mark derived from the cell's own held errors, with no
reference arrays.

**Derivation** (`chain/stage_m_met_a.derived_baseline`): at each of Met-A's 6,000 commissioned neutrophil identity sites
(`metA_floors_v1_3.json`), the healthy level is 1 - eps_loss where the mark is held and eps_gain where it is held absent (layer 1), put on
the array scale through IAM-Atlas v2.1's measured transfer (its inverse, PCHIP through the knots of `transfer_v21_final.json`); the mark is
the mean entropy over the sites. Held or held absent is IAM-Atlas v2.1's neutrophil level at the site (>= 0.5). Nothing is fitted. IAM-Atlas v2.1 has a neutrophil level at 5,553 of the 6,000 sites (2,931
held, 2,622 held absent); the other 447 take the same proportion, so the mark is computed on the 5,553.

**Data:** the three healthy blood granulocyte samples of the published sequencing atlas (GSE186458: GSM5652313, GSM5652314, GSM5652315;
`data/DEV_METAA_HEALTHY_01/step3_samples.csv`), molecules never counted before, each read by `run_sample.py --dev-count-only` through
`count_atlas_01.py` (now taking a sample list; default unchanged) on GitHub's machines. Granulocytes are mostly neutrophils; the atlas has no
purified-neutrophil sequencing. The six arrays: the commissioned neutrophil references, read by the chain against the measured mark on the
frozen sites (`metA_floors_v1_3_loo.csv`); against the derived mark each reading is multiplied by measured / derived.

**Bar 1:** all six read Normal (0.95 to 1.05) against the derived mark. Scorer `step3_score.py`.

**Stated before:** the derivation leaves out the array's own measurement blur, which can only raise the entropy an array reads. If the
derived mark falls below the measured one so that the arrays read above Normal, the gap is that blur, and the chain already measures blur
on the atlas's anchor sites; adding it would be version 2, written before it runs.

## Confirmation 1: layer 1 on samples that played no part (written and committed before any of them is counted)

**Prior reading:** GSE116229 is spent for the leukemia replication (DEV_ALL_INFO_01.md); no count of its three cord-blood samples had landed or been read when this was written.

Version 2 was written after version 1 was seen, so it is confirmed on another laboratory's samples: the three cord-blood CD19+ B cells of
GSE116229 (GSM4914800, GSM4914801, GSM4914802), counted by the leukemia replication's pass 1 (10 million read pairs, `run_sample.py
--dev-count-only`). Version 2's rule unchanged, against IAM-Atlas v2.1's B-cell level. **Bars:** marked side within 0.02; unmarked side
within 0.02 (the three pooled). Scorer `confirm_01_score.py`. **Stated before:** cord blood is newborn and the atlas's B cells adult; the count
is at fixed depth, while layer 1 read whole files. A second confirmation on another cell type follows, its series chosen by a fixed rule.

**Step 3, scored** (`data/DEV_METAA_HEALTHY_01/step3_01/step3_summary.txt`): **bar 1 not met (0 of 6).** Granulocyte held error: loss
0.04182, gain 0.01385; on the array scale, held 0.9591 and held absent 0.0617; derived mark 0.28792 bits against the measured 0.33026
(derived / measured 0.8718); the six arrays read 1.139 to 1.156 against it. **Disclosed:** the first scoring stopped with an error: the
derivation inverted the 42-knot transfer table directly, which fails where the curve is flat; it now inverts the smooth curve the atlas
was built with, exactly as the atlas converted its sequencing readings (`atlas_block_v21.py`, mode v21inv). That is the run scored here.

**Diagnostic, after the result (disclosed):** the stated candidate was the arrays' measurement blur. IAM-Atlas v2.1's own neutrophil levels at
the same 5,553 sites give a mean entropy of 0.35084 bits, above the arrays' 0.33026, so blur does not explain the gap. The held-absent side
agrees (atlas median 0.0654, derived 0.0617); the held side does not (atlas median 0.9387, derived 0.9591). The neutrophil's identity marks
are held less firmly than the cell's backbone, the constitutively marked sites layer 1 counted. **Version 2 (to be written and committed
before it runs):** the held error counted on the granulocyte molecules at Met-A's own identity sites, the mark derived from it; still the
cell's own molecules, nothing fitted.

## Step 3 version 2: the held error at the identity sites (written and committed before its counts exist)

Written after version 1 was scored and its diagnostic read (disclosed). One change: the granulocytes' held error is counted only at Met-A's
neutrophil identity sites (`run_sample.py --dev-count-sites neutrophils`, `stage_q2_selfref.count_pat_sites`: Stage Q2's own rules, the
same held and held-absent molecules classed from all their calls; only the calls counted are restricted). A lone error at a site reads
eps_site (1 - eps_nb)^2, its neighbors being mostly backbone CpGs, so eps_site = r / (1 - eps_nb)^2 with eps_nb version 1's genome-wide
per-call rate on the same side. **Synthetic check (laptop, `sites_synthetic_01.py`, output `step3_01/sites_synthetic_01.txt`):** planted
rates 0.06 (loss) and 0.05 (gain) at a site set, 0.03 and 0.02 elsewhere, are recovered once the neighbors' rate is divided out. The derivation, the sites, the six arrays and bar 1 are version 1's.
Counts: the three granulocyte samples again, on GitHub (`count_atlas_01.py ... step3_samples.csv --dev-count-sites neutrophils`).
Scorer `step3_score_v2.py`.

**Step 3 version 2, scored** (`data/DEV_METAA_HEALTHY_01/step3_02/step3_v2_summary.txt`): **bar 1 not met (0 of 6), further off than
version 1.** At the identity sites the lone error on held molecules is lower than genome-wide (loss 0.03160 against 0.04182), so the
derived level moves away from the atlas's (held 0.9909 on the array scale against the atlas's median 0.9387) and the mark falls to 0.07700
bits against the measured 0.33026. **What it shows:** isolated errors on held molecules do not set an identity site's level; the share of
molecules there that do not hold the mark (the loose share at those sites) does. Layer 1's backbone sites carry almost none, which is why
layer 1 met. **Next (version 3, to be written before it runs):** the molecule classes at the identity sites (held, held absent, loose) and
each class's level, from the same counts. **Disclosed:** a version 3 would be the third on the same three granulocyte samples and six
arrays; whatever it reads stays development until confirmed on samples that played no part.

**Which cells stand on each side of step 3** (checked after version 2, before any version 3; `atlas/sources/roster_v2/atlas_v2_roster.csv`,
`atlas/v2/inputs/roster_samples.csv`): the three granulocyte samples (GSE186458) were candidates for IAM-Atlas v2.1 and were left out as a
mixture of neutrophils, eosinophils and basophils; the atlas's neutrophil column comes from purified neutrophil arrays only (two published
array series, 12 samples). So the comparison is not circular, but its two sides are not the same cells: the derived side is granulocytes, the
measured mark purified neutrophils. Part of any step 3 gap may be that difference. A version 3 states it before it runs; a confirmation
needs sequenced purified neutrophils, if a public set exists.

**Why versions 1 and 2 missed: the site rule** (author: "what are we missing?"; read in `chain/stage_m_met_a.py`, the commissioned rule):
Met-A's identity sites are chosen on purified neutrophil arrays with a mean of 0.75 to 0.95 (or 0.05 to 0.25) and a spread across arrays of
at most 0.05, up to 3,000 per side. Fully held sites (0.97 to 0.99) are excluded by design, because a mark pinned at the end shows no wear.
So the atlas's 0.939 at those sites is partly the selection itself. Layer 1 matched because its backbone sites are not filtered this way;
versions 1 and 2 predicted a fully held level (0.959 and 0.991) for sites selected for not being fully held. Version 2 also showed that
isolated errors are rarer at these sites than genome-wide, so their lower level comes from copies that do not hold the mark (a loose share at
the sites). Smaller contributors: the transfer's flat top near 0.959 on the sequencing scale (any level above it inverts to the end), and
granulocytes standing in for purified neutrophils.

**The question, restated:** at sites in the 0.75 to 0.95 window, how much of the drop below full holding is a loose share and how much is
error? That is IAM-A's question (the loose share against the held error), asked at Met-A's sites. **Version 3 (to be written before it
runs):** at each identity site, every call on the granulocyte molecules, split into held molecules, held-absent molecules and loose
molecules; the site's level from all of them; the mark from those levels on the array scale. That is a baseline measured from the cell's own
molecules, not yet a derivation: the derivation is the physics of the loose share at those sites, an open question beside the firm rate's.

## Step 3 version 3: the molecule classes at the identity sites (written and committed before its counts exist)

Written after versions 1 and 2 were scored (not met) and the site rule read (disclosed). **What it rests on, stated before the bars (lesson
C41):** Met-A's site rule (mean 0.75 to 0.95 or 0.05 to 0.25, spread at most 0.05, chosen on purified neutrophil arrays); and IAM-A's
physics, already stated as the source of Met-A's baseline: a site's level = held share x held level + loose share x loose level + absent
share x absent level. Each term is counted on the granulocytes' molecules at the identity sites (`run_sample.py --dev-count-sites
neutrophils`, now also `stage_q2_selfref.count_pat_site_classes`: every call, the molecule classed from all its calls by Stage Q2's rules),
the sites split by the atlas's neutrophil level (held at 0.5 or above). The site level goes through the atlas's own inverse transfer to the
array scale and the mark follows; scorer `step3_score_v3.py`; the sites, the six arrays and bar 1 are version 1's. **What it is:** a baseline
measured from the cell's own molecules, not yet a derivation (the physics of the loose share at these sites is open). **Known differences:**
granulocytes stand in for purified neutrophils; third version on the same samples and arrays, so a pass stays development until confirmed on
samples that played no part. **Checks:** a constructed molecule file over the identity sites, read by the one command, is recovered: planted levels 0.8944 and
0.0964, read 0.8939 and 0.0973 (`step3_03/sites_synthetic_02.txt`); the one command also ran the count on a spent plasma molecule file.

**Synthetic test:** chain/stage_q2_selfref.py, DEV_METAA_HEALTHY_01/sites_synthetic_02.py, DEV_METAA_HEALTHY_01/step3_03/sites_synthetic_02.txt

## Bars 2 and 3, read (`data/DEV_METAA_HEALTHY_01/bar2_01/bar2_summary.txt`): both met

100 of 100 healthy bloods read through `run_sample.py` on GitHub's machines, 20 from each of five laboratories.

| Series | median A | A_pred | A / A_pred |
|---|---|---|---|
| GSE153712 | 0.9834 | 0.9940 | 0.9869 |
| GSE235717 | 1.0536 | 1.0541 | 0.9986 |
| GSE42861 | 1.0257 | 1.0236 | 1.0022 |
| GSE51032 | 0.9932 | 1.0192 | 0.9719 |
| GSE87571 | 0.9614 | 0.9834 | 0.9779 |

- **Bar 2 met:** every laboratory's median A / A_pred lies within 0.03 of 1.
- **Bar 3 met:** the laboratories close up; the spread of the five medians falls from 0.0921 (A) to 0.0303 (A / A_pred).
- **Bar 1 (synthetic) is not fully met** (synthetic test 2: the weak array 0.0072 below the 0.005 line), so by the test's own rule the
  healthy line is not commissioned. **What it shows:** each healthy blood's whole-sample Met-A is predicted from the array's own measured
  line, with no healthy cohort and no reference arrays, in five laboratories. All five series are spent, so this is development.

**Step 3 version 3, scored** (`data/DEV_METAA_HEALTHY_01/step3_03/step3_v3_summary.txt`): **bar 1 not met (0 of 6)**, but the gap closes
from version 2's 0.2332 of the measured mark to **0.8087** (0.26707 bits against 0.33026). The molecule classes at the identity sites: at held
sites 0.9053 of calls on held molecules, 0.0929 on loose molecules, 0.0018 on held-absent ones; at held-absent sites 0.9754 held absent, 0.0239
loose. **Where the gap is** (side diagnostic, added after the scoring, disclosed: `step3_side_v3.py`, output `step3_03/step3_side_v3.txt`):

| Side | Sites | Atlas level (arrays) | Its entropy | Version 3 level | Its entropy |
|---|---|---|---|---|---|
| held | 2931 | 0.9362 | 0.3404 | 0.9344 | 0.3493 |
| held absent | 2622 | 0.0699 | 0.3625 | 0.0263 | 0.1755 |

**The held side is predicted from the molecules** (0.9344 against 0.9362): the loose share beside the held error is what sets it. **The whole
gap is on the unmarked side:** the molecules say these sites are nearly unmarked (0.0114 on the sequencing scale), the arrays read them at
0.0699. Probe design does not explain it (type I unmarked sites 0.0646, type II 0.0701; 98 % of the identity sites are type II). The
unmarked sites were chosen because the arrays read them at 0.05 to 0.25, so this half of the measured mark carries the array's own reading
at nearly unmarked sites, a property of the array, not of the cell. **Next (version 4, to be written before it runs):** put the molecule
levels on each array through that array's own measured line (the healthy line's measurements, which met bars 2 and 3 in five laboratories),
not through the atlas's population transfer. Disclosed: a fourth version on the same samples; development until confirmed on untouched ones.

## Step 3 version 4: each site's array-against-sequencing difference, measured on other cells (written and committed before any file is read)

**Prior reading:** GSE110554 (its B, CD4, CD8, NK and monocyte arrays: atlas sources; its six neutrophil arrays are Met-A's commissioning
references and are not read here) and GSE186458 (the same five cell types by sequencing: atlas sources) are spent; this is development, the
fourth version on the same three granulocyte samples and six arrays.

**What it rests on (lesson C41):** version 3 predicted the marked side from the cell's own molecules (0.9344 against the arrays' 0.9362);
the whole gap was the unmarked side, where the arrays read nearly unmarked sites at about 0.07, and Met-A's site rule selected those sites
on the arrays (0.05 to 0.25). A whole-array line cannot move particular sites, so version 4 measures each identity site's own
array-against-sequencing difference on OTHER cells: B, CD4, CD8 and NK cells and monocytes, 29 arrays of the same laboratory's purified blood
(GSE110554) and 15 sequencing samples (GSE186458), every one read by the one command (`v4_read.py`; `v4_samples.csv`). At each site the
difference is array mean minus the sequencing mean on the array scale (the atlas's own inverse transfer), per cell, using only cells nearly
unmarked there for the unmarked side (at most 0.25) and nearly marked for the marked side (at least 0.75), the median over them. Version 4's
level = version 3's level for the side + the site's difference; the mark is the mean entropy (`v4_score.py`). No neutrophil or granulocyte
array is used. **Known:** the arrays are the same laboratory's as the six commissioned ones, so a laboratory term at those probes is shared.
**Bar 1 (version 1's):** all six commissioned arrays read Normal (0.95-1.05) against the derived mark. **What it would mean:** the
neutrophil's healthy mark from the cell's own molecules and a measured property of each probe, with no neutrophil array; development.

**Synthetic test:** DEV_METAA_HEALTHY_01/v4_synthetic.py, DEV_METAA_HEALTHY_01/step3_04/v4_synthetic.txt

The synthetic (one made-up cell type on both platforms through the one command; per-site differences planted at the identity sites)
recovers them exactly (median error 0.0000 on both sides). A first run at 40 calls a site rounded 0.01 to 0 reads; rebuilt at 100.
