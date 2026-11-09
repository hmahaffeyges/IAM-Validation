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

**Step 3 version 4, scored** (`data/DEV_METAA_HEALTHY_01/step3_04/step3_v4_summary.txt`; 44 of 44 readings; the scorer's count line was
fixed for the installed pandas before it ran, nothing else changed): **bar 1 not met (0 of 6)**; the derived mark is 0.29092 bits against
0.33026, **0.8809** of it (version 3: 0.8087). Each identity site's array-against-sequencing difference was measured on five other blood
cell types at 2925 of 2931 marked and all 2622 unmarked sites: median 0.0032 on the marked side, **0.0300 on the unmarked side**. That moves
the unmarked side's mean level from 0.0263 to 0.0479; the neutrophil arrays read 0.0699. **What is left:** about 0.022 on the unmarked side,
the part of the neutrophil arrays' reading that is neither the granulocytes' molecules nor the probe offset other blood cells show there. Two
causes cannot be told apart on these data: the granulocyte samples stand in for purified neutrophils, and the offset may be larger in
neutrophils than in lymphocytes and monocytes. **Four versions on the same samples:** tuning stops here; the next step is data that splits
the two causes (purified neutrophils measured on both platforms), not a fifth version on these.

## Step 3 version 5: purified neutrophils by sequencing (written and committed before any BLUEPRINT file is read)

**Prior reading:** none of the seven BLUEPRINT donors' purified mature neutrophil sequencing (public GRCh38 per-CpG level and coverage
tracks) has been read by this project. Three are spent here for development (C000S5, C0011I, PB100713: every other donor in name order);
four stay unread for the confirmation (C0010K, C001UY, PB130513, PB270313), read only after this result is recorded, with the same scorer and
bar. Version 4's per-site differences (`step3_04/v4_offsets.csv`, from GSE110554 and GSE186458, spent) are reused unchanged.

**What it rests on (lesson C41):** version 4 left 0.022 on the unmarked side, from either the granulocyte stand-in or a probe offset larger
in neutrophils. Purified neutrophils measured by sequencing split the two. **The cause split, stated before reading:** their mean sequencing
level at the unmarked identity sites against the granulocytes' 0.0114: at least 0.020, the stand-in accounts for part of the gap; below 0.015,
it does not; between, undecided. **Version 5's mark:** at each site the donors' mean sequencing level on the array scale (the atlas's inverse
transfer) plus version 4's per-site difference; per site, where version 4 used one level per side (disclosed). Read by the one command's new
bigWig route (`run_sample.py --dev-seq-beta CALL.bw,COV.bw`, `stage_seq_values.values_from_bigwig`, the sites' GRCh38 positions in
`atlas/v2.1/sites_key_hg38.parquet`); `v5_read.py`, `v5_score.py`. **Bar 1 (version 1's):** all six commissioned arrays read Normal
(0.95-1.05) against the derived mark. **What it would mean:** the neutrophil's healthy mark from purified neutrophils' molecules and a
measured property of each probe, with no neutrophil array; development until the four reserved donors confirm it.

**Synthetic test:** DEV_METAA_HEALTHY_01/v5_synthetic.py, DEV_METAA_HEALTHY_01/step3_05/v5_synthetic.txt

The synthetic (a constructed bigWig pair at the atlas sites, one site in ten at the next base, read by the one command) runs on GitHub, where
the bigWig reader installs, before any donor is read. **Result:** every identity site found (2931 of 2931 marked, 2622 of 2622 unmarked), levels read exactly (max error
0.0000), 78,914 sites read at the next base as planted.

**Step 3 version 5, scored** (`data/DEV_METAA_HEALTHY_01/step3_05/step3_v5_summary.txt`; the three development donors, 5407 of 5553 identity
sites with at least 10 reads; the watcher fetched only .json files, so the readings were downloaded by hand, nothing changed): **bar 1 not
met (0 of 6); the derived mark is 0.7854 of the measured one.** **The cause split is decided:** purified neutrophils read **0.0053** at the
unmarked identity sites (per donor 0.0056, 0.0043, 0.0062), lower than the granulocytes' 0.0114, so **the granulocyte stand-in does not
account for the gap.** At the marked sites they read 0.9355 (granulocytes 0.9344). **What it shows:** at Met-A's unmarked identity sites the
neutrophil's DNA is almost entirely unmarked (about half a percent), and the neutrophil arrays read about 0.07 there, more than other blood
cells' arrays read at the same probes (0.030 above their sequencing). The unmarked half of Met-A's healthy mark is set by how the array reads
nearly unmarked DNA, not by the cell. Those sites were chosen because the neutrophil arrays read them between 0.05 and 0.25, which selects
probes where the array reads above a nearly unmarked truth. **What follows:** the term version 1 named as not included, each array's own
measurement blur, is the remaining part; the healthy line measures it on each array's own anchor sites (bars 2 and 3 met in five
laboratories). The four reserved BLUEPRINT donors stay unread.

## Step 3 version 6: each array's own reading of bare and of fully marked DNA (written and committed before its readings exist)

**Prior reading:** the six commissioned neutrophil arrays (GSE110554) and the three development BLUEPRINT donors (version 5) are spent;
the four reserved BLUEPRINT donors stay unread. This is the sixth version on these arrays; development.

**What it rests on (lesson C41):** version 5 found the neutrophil's DNA nearly bare (about 0.005) at Met-A's unmarked identity sites, where
the arrays read about 0.07; the array's blur (0.004 to 0.008) and reference purity (the other blood cells are also nearly unmarked there)
cannot account for it. An array reads bare DNA above zero and fully marked DNA below one. **Version 6 measures that on each array itself:**
F0, its mean reading at sites the neutrophil's molecules hold bare (at most 0.01), and F1, at sites they hold fully marked (at least 0.99),
over atlas sites that are not identity sites. At each identity site the array should read the straight line through (m0, F0) and (m1, F1),
m0 and m1 the molecules' mean levels at those anchors; the array's derived mark is the mean entropy over the identity sites
(`v6_read.py`, `v6_score.py`). Two means per side on the array's own sites; nothing fitted; the identity sites take no part in the line.
**Bar 1 (version 1's):** all six commissioned arrays read Normal (0.95-1.05), each against its own derived mark. **What it would mean:** Met-A's
healthy mark comes from the cell's own molecules and the array's own reading of bare and fully marked DNA, with no reference arrays of the
cell; development until the four reserved donors confirm it.

**Synthetic test:** DEV_METAA_HEALTHY_01/v6_synthetic.py, DEV_METAA_HEALTHY_01/step3_06/v6_synthetic.txt

The synthetic (constructed molecules and two constructed arrays with their own lines, read by the one command) recovers each array's line
and reads both arrays 1.0000. Its first run read 0.977 and 0.969: the rule took the bare anchors as exactly 0; corrected to the line through
the anchors' own molecule levels before any real reading (disclosed).

**Step 3 version 6, scored** (`data/DEV_METAA_HEALTHY_01/step3_06/step3_v6_summary.txt`): **bar 1 not met (0 of 6); the arrays read 0.77 to
0.81 against their derived marks, now above them.** Each array reads DNA the molecules hold bare at about 0.05 (F0 0.048 to 0.060) and
fully marked DNA at about 0.935. **Side diagnostic** (added after the scoring, disclosed: `v6_side.py`, `step3_06/v6_side.txt`): at the
unmarked identity sites the line predicts 0.0557 where the arrays read 0.0636 (molecules 0.0049), **so the unmarked half of the gap is the
array reading bare DNA above zero.** At the marked sites the arrays read 0.9390 where the molecules read 0.9391 and the straight line predicts
0.8871: **the array's response is a curve, not a line;** binned by molecule level it is raised at the bottom (0.0527 at 0 to 0.01) and flat
at the top (0.9112 at 0.90 to 0.95, 0.9305 at 0.95 to 0.99, 0.9365 above 0.99). Only 2540 identity sites were read by all three donors (one donor
is shallower).

## Step 3 version 7: each array's own response curve (written after version 6's diagnostic, committed before it is scored)

**Disclosed:** written after version 6's side diagnostic was read, on the same arrays and donors; the seventh version. **The rule:** each
array's own response curve, binned means of its readings against the molecules' level at atlas sites that are not identity sites, with fixed
bin edges stated in `v7_score.py`; each identity site's expected reading is the curve at its molecule level; nothing fitted; the identity
sites take no part in the curve. **Bar 1 (version 1's):** all six arrays Normal. **The confirmation, decided now:** whatever version 7 reads on
the development donors, it is then read once on the four reserved BLUEPRINT donors (`v6_read.py OUT confirmation`), with the same scorer and
bar, and that reading decides it.

**Synthetic test:** DEV_METAA_HEALTHY_01/v7_synthetic.py, DEV_METAA_HEALTHY_01/step3_07/v7_synthetic.txt

The synthetic (constructed curved arrays, read by the one command) recovers both arrays at 0.9998 and 0.9997.

**Step 3 version 7, scored on the development donors** (`data/DEV_METAA_HEALTHY_01/step3_07/step3_v7_summary.txt`): **bar 1 not met (0 of
6; 0.8423 to 0.9026).** **Selection diagnostic** (added after the scoring, disclosed: `v7_select.py`, `step3_07/v7_select.txt`): against each
array's own response curve, the marked identity sites read **+0.0257** above it; other sites passing Met-A's array window read +0.0025, and
all other sites at the same molecule levels +0.0011. At the unmarked side the identity sites read +0.0053, other sites passing the window
+0.0136. **What it shows:** the window alone does not lift a site; what lifts the marked identity sites is the rest of their selection, the
ranking on these same arrays of where neutrophils differ most from other cells. Sites ranked on array readings are probes that read high, so
**Met-A's healthy mark at array-chosen sites carries a selection effect that no prediction from the molecules can give.** **Conclusion of
step 3 (seven versions):** the mark at the commissioned sites is a measured quantity and stays measured. A derived Met-A baseline needs its
identity sites chosen on the molecules (the sequencing atlas, or purified sequencing such as BLUEPRINT), not on arrays, so that the sites
are not selected on the readings they are later compared with; that is a new Met-A version, designed and tested in its own record. The
confirmation reading on the four reserved donors, decided before version 7 was scored, is read and recorded when it lands.

**Step 3 version 7, the confirmation** (`data/DEV_METAA_HEALTHY_01/step3_07/step3_v7_confirm_summary.txt`; the four reserved BLUEPRINT
donors, read once as decided before version 7 was scored; 4312 identity sites with molecules): **not met (0 of 6; 0.8449 to 0.8972)**, the
development reading reproduced on donors that took no part (0.8423 to 0.9026). The gap is a property of the commissioned sites, not of the
donors; the conclusion of step 3 stands. The BLUEPRINT neutrophil donors are now all spent.

**The ceiling's A, captured** (`data/DEV_METAA_HEALTHY_01/bar2_01/bar2_rows.csv`, written by `CANON/bundle_rows.py` from the one command's
bundles; values copied, nothing computed): every reading of the 100 healthy bloods with its A and the A its ceiling sits at (every site at 1 bit); the ceiling's A is listed in the
A-score database, on the A-score scale (1 healthy; the ceiling to the right, the floor to the left).

## Synthetic test 3: the real arrays' range (written and committed before it runs)

**Why.** Test 2's weak array missed bar 1 by 0.0072. Checked offline (working notes, chain functions called directly, not evidence): the healthy
line's formula is exact with the planted line and noise (all three arrays 1.0000 within 0.0003) and with the line the chain measures from each array
(the weak array 0.9995); the miss follows the IAM-Deconvolver's mix drifting on a compressed array (adding the drift the chain reported on the mild
array reproduces its 0.99689). But test 2's grid was set without measuring real arrays, and every real array the chain has measured is far less
compressed than even the mild one.

**The instrument's job:** a healthy blood array reads Normal against IAM-Atlas v2.1 on Met-A's gauge.

**Sources:** GEO GSE235717 GSM7508429 (public: one healthy whole blood's raw IDATs, the base and the cell shares); IAM-Atlas v2.1 (public, in this
repository and on Zenodo: each cell's healthy level at every site, and the ruler's anchor sites).

**Reference:** IAM-Atlas v2.1 only, in the mix of cells the IAM-Deconvolver measures; no same-run tare, no controls.

**Synthetic test:** `DEV_METAA_HEALTHY_01/synthetic_03.py` (every array through the one command, on a Google machine that saves as it goes).

**Range of the data:** real arrays (the line and noise the chain measured on 331 arrays: R14 lung on EPIC and the 450K blood rehearsal) have a
design I slope 0.993 to 1.009, offset -0.006 to 0.002, noise 0.003 to 0.015, and a design II slope 0.999 to 1.030, offset -0.018 to -0.004, noise
0.006 to 0.024. The synthetic arrays: the real medians (center), and both tails at the real extremes with the largest real noise (design I slope
0.9928 and 1.0085, noise 0.0148; design II slope 0.9991 and 1.0301, noise 0.0240); test 2's weak array (slope 0.88 and 0.78, noise 0.05 and 0.06),
beyond the real range, recorded, not scored.

Bars (scored on the three arrays in the real range):
- **Bar 1:** each of the 3 healthy arrays reads A / A_pred within 0.005 of 1.
- **Bar 2:** on each of the 3, a 5 % loss of the held pattern off the anchors raises A / A_pred by at least half as much as it raises A.

**Prior reading:** GSM7508429 is spent (tests 1 and 2); nothing else is read.

## The per-person rehearsal on spent data, atlas only (written and committed before any of its readings exist)

The step between the synthetic and a fresh test (plan 60.29.1): the same 100 spent healthy bloods as bars 2 and 3, read again through the one
command on every atlas cell, now scored for each person, not each laboratory's median. Bars 2 and 3 above were medians; a median can be met while
many people read outside Normal.

**The instrument's job:** each healthy whole blood reads Normal on Met-A's gauge against IAM-Atlas v2.1.

**Sources:** GEO GSE153712, GSE235717, GSE42861, GSE51032 and GSE87571 (public: each blood's raw IDATs; the 100 listed in
`data/DEV_METAA_HEALTHY_01/bar2_labels.csv`); IAM-Atlas v2.1 (public, in this repository and on Zenodo).

**Reference:** IAM-Atlas v2.1 only, in each blood's own Deconvolver mix; no same-run tare, no controls, no laboratory median.

**Scorer:** `data/DEV_METAA_HEALTHY_01/wb100_score.py` (it only summarizes the one command's bundles; series run wb100_atlas).

Bars (scorer): 1. at least 90 of the 100 healthy bloods each read A / A_pred inside Normal, 0.95 to 1.05; 2. in each of the five laboratories,
at least 16 of its 20 each read inside Normal. Reported, not scored: untared A against the atlas (bar 2's earlier run: 80 of 100 inside 0.95 to
1.05, per laboratory 9 to 20 of 20); each blood's measured line; the cell shares; the C-Score's whole-genome reading against the atlas and its
derived limit (scored only after its own synthetic, plan 60.29.2).

Scored only after synthetic test 3 is met; if it is not, these readings are recorded and not scored.

**Prior reading:** all 100 are spent: bars 2 and 3 read them and looked only at each laboratory's median of A / A_pred; the earlier
untared A per person is the line above.

## Synthetic test 3, read: not met (`data/DEV_METAA_HEALTHY_01/synthetic_03_summary.txt`; Google, the one command)

| Array | A / A_pred | bar 1 (within 0.005 of 1) |
|---|---|---|
| center (the real medians) | 0.99987 | met |
| low tail | 0.99913 | met |
| high tail (design II slope 1.0301, offset -0.0183) | 0.98813 | not met |
| test 2's weak array (beyond the real range; recorded) | 0.99791 | |

Bar 2 met (a 5 % loss shows on all three). **Cause, checked offline (working notes, chain functions called directly, not evidence):** on a
high-tail array about 1.3 % of readings fall at 0 and 1.6 % at 1, and the line is measured from plain means of the anchors, which clipping pulls
inward. In this test the chain measured design II slope 1.0268 and offset -0.0167 (`synthetic_03_summary.txt`; planted 1.0301 and -0.0183).
In the offline check, the same kind of array read about 1 with the planted line and about 0.989 with the line measured from it (working notes).
The fix measures the line with readings at 0 or 1 counted as censored (`stage_metaa_atlas._line_censored`, run_sample.py `--dev-line-censored`,
development, off by default, so no running reading changes). Offline it brought the high tail close to 1 and showed that on the noisiest real-range
arrays one array's A / A_pred moves by up to about 0.005 from the noise draw alone, so one draw per condition cannot tell a bias from
noise. Test 3 stands as not met under its bar.

## Synthetic test 4's design (restored; disclosed)

Test 4 (`DEV_METAA_HEALTHY_01/synthetic_04.py`, scorer `synthetic_04_score.py`, both committed before it ran) was test 3's three real-range
arrays with ten noise draws each, through the one command with `--dev-line-censored`, against IAM-Atlas v2.1 only. Bars: 1. each of the 30 healthy
arrays inside Normal, 0.95 to 1.05; 2. each condition's mean of its 10 draws within 0.005 of 1; 3. a 5 % loss shows on each. Its design text and
test 3's result were written into this record before test 4 ran, but a commit then in flight refreshed the working copy when it finished and
erased them, so the commit that sent test 4 carried its scripts and not this text; it was restored after test 4 had run. The bars above are the
ones in the committed scorer. From now on the record is written only when no commit is in flight, and checked on main before a test is launched.

## Synthetic test 4, read: not met, and it ran without its flag (disclosed; my error) (`data/DEV_METAA_HEALTHY_01/synthetic_04_summary.txt`; ten draws on Google)

Bar 1 met (all 30 arrays inside Normal); bar 3 met; bar 2 not met: the means over ten draws are center 1.00065, low tail 0.99853, high tail
0.99192 (spread 0.0010 to 0.0017). **But the censored line never ran:** no draw's recorded line carries line = censored. The flag was set in
run_sample.py only inside the two sequencing branches, which an array never enters. So test 4 measured the plain line; what it does show is that
the plain line's bias on the high tail is real and about 0.008 (ten draws, spread 0.0017), not noise. The flag is now set right after the command's
arguments are read, on every route. Test 5 is test 4 again on the fixed command; its scorer refuses any draw whose recorded line is not censored.

## Synthetic test 5: the censored line, ten noise draws, the flag checked (written and committed before it runs)

**The instrument's job:** a healthy blood array reads Normal on Met-A's gauge against IAM-Atlas v2.1.

**Sources:** GEO GSE235717 GSM7508429 (public: one healthy whole blood's raw IDATs, the base and the cell shares); IAM-Atlas v2.1 (public, in this
repository and on Zenodo).

**Reference:** IAM-Atlas v2.1 only, in the mix of cells the IAM-Deconvolver measures; no same-run tare, no controls.

**Synthetic test:** `DEV_METAA_HEALTHY_01/synthetic_05.py` (the one command with `--dev-line-censored`; one noise draw per Google machine, saved as
it finishes; scorer `DEV_METAA_HEALTHY_01/synthetic_05_score.py`, which refuses a draw whose line is not censored).

**Range of the data:** real arrays (331 measured by the chain): design I slope 0.993 to 1.009, offset -0.006 to 0.002, noise 0.003 to 0.015; design
II slope 0.999 to 1.030, offset -0.018 to -0.004, noise 0.006 to 0.024. The synthetic arrays: test 3's center and both tails at the real extremes
with the largest real noise, ten new noise draws each; test 2's weak array once, recorded, not scored.

Bars (scorer), test 4's unchanged: 1. each of the 30 healthy arrays reads A / A_pred inside Normal, 0.95 to 1.05; 2. in each of the 3 conditions the
mean of its 10 draws is within 0.005 of 1; 3. on each of the 3, a 5 % loss raises A / A_pred by at least half as much as it raises A.

**Prior reading:** GSM7508429 is spent (tests 1 to 4); nothing else is read.

## Synthetic test 5, read: met (`data/DEV_METAA_HEALTHY_01/synthetic_05_summary.txt`; ten draws on Google, every draw's line censored)

| Array | inside Normal | mean of 10 draws | spread (sd) |
|---|---|---|---|
| center | 10 of 10 | 1.00093 | 0.00100 |
| low tail | 10 of 10 | 0.99965 | 0.00231 |
| high tail | 10 of 10 | 1.00069 | 0.00256 |

All three bars met; test 3's weak array (as the scorer names it: test 2's weak condition carried into test 3), beyond the real range, read 0.99685 (recorded). With the line measured from censored anchors, the high
tail's bias under the plain line (test 4: mean 0.99192) is gone. **Next:** the per-person rehearsal on the 100 spent healthy bloods, read with
`--dev-line-censored` (its design above); then a fresh test on held healthy arrays.

## Lock 1, the reference door (written and committed before its synthetic runs)

The first of the five anti-drift locks (the author: healthy comparisons come only from the atlases; "we need to make sure this doesnt happen
again"). `chain/reference_door.py`: before any reading the one command checks every IAM-Atlas v2.1 file against its compact_manifest.json and every
IAM Sequencing Atlas file the chain reads against `chain/reference_manifest.json`; a changed or unlisted file stops the run (REFERENCE_REFUSED,
exit 7). Every bundle carries a reference block naming the atlases read and every tare applied. No reading changes.

**The instrument's job:** every reading the chain makes is against the published atlases, and its bundle says so.

**Sources:** IAM-Atlas v2.1 and the IAM Sequencing Atlas's files in the chain (public, in this repository and on Zenodo); no outside sample.

**Reference:** IAM-Atlas v2.1 and the IAM Sequencing Atlas only, checked by checksum.

**Synthetic test:** `DEV_METAA_HEALTHY_01/reference_door_synthetic.py` (the one command on a constructed healthy blood array; GitHub, on a disposable
checkout, since it alters and restores one atlas file there).

**Range of the data:** one constructed whole blood at the real arrays' median noise (design I 0.0066, design II 0.0098); the check itself does not
depend on the sample.

Bars (scorer, in the synthetic): 1. the clean run's bundle says the atlases were checked, names IAM-Atlas v2.1, and records no tare; 2. with one
byte added to one Sequencing Atlas file each run stops before any reading, the file named; 3. a file added to Seq_Atlas that the manifest does not
list stops each run the same way.

**Prior reading:** none; nothing outside the atlases is read.

**Lock 1, the reference door: synthetic met** (`data/DEV_METAA_HEALTHY_01/reference_door_synthetic.txt`; GitHub run 42390866709, the one command on the wired chain).
The clean run's bundle names IAM-Atlas v2.1 and the IAM Sequencing Atlas, checked (60 files), with no tare; one byte added to one Sequencing Atlas
file stopped the run before any reading (exit 7, the file named); an unlisted file in Seq_Atlas stopped it the same way. Lock 1 is live in the one
command; the daily reference line in the day summary follows.

## The per-person rehearsal, read: met (`data/DEV_METAA_HEALTHY_01/wb100/wb100_summary.txt`, `wb100_readings.csv`; GitHub series run wb100_atlas, the one command on
every atlas cell with `--dev-line-censored`; IAM-Atlas v2.1 only, no tare)

98 of the 100 healthy bloods read A / A_pred inside Normal (bar 1 met); per laboratory 20, 20, 20, 18 and 20 of 20 (bar 2 met). The two outside
are both GSE51032 (450K): 0.94451 and 0.94879.

**Recorded, and open before any fresh test:** the ratio centers at 0.9834 (sd 0.0172), where synthetic test 5 centered within 0.001 of 1; the
untared whole-sample A centers at 0.9005, against 0.9977 in bar 2's earlier reading of the same bloods. The difference between the two readings is
the cell panel: this one reads every atlas cell, so the Deconvolver gives these bloods small shares of bone-marrow progenitor and other columns,
while the synthetic arrays were built from six blood cells. **Next (its design written first):** the synthetic built from the full mix the
Deconvolver measured on real bloods, to find whether the center of 0.983 comes from that mix; then the fresh test on held arrays. The
C-Score's whole-genome reading sat within its derived limit for 96 of 100 (recorded; its own synthetic comes first, plan 60.29.2). **Correction
(disclosed):** on 450K arrays that limit carries a platform term (factor 1.00789, `Runtime Matrices/CScore_Limits/platform_term_450k.json`)
calibrated on the spent healthy 450K bloods of this same five-laboratory set, so for the 450K bloods the 96 of 100 is not independent, and the
term itself is taken from labeled-healthy samples, not from the atlases. It is replaced by a term from the atlases (plan 60.29.2), and from now
on each bundle names any calibrated term it used (`reference.calibrated_terms`).

## Synthetic test 6: arrays from the real full mix (written and committed before it runs)

Why: the rehearsal on every atlas cell centered at 0.983; test 5, built from six blood cells and read on the default panel, centered at 1.000.
Test 6 asks whether the center comes from reading on every atlas cell.

**The instrument's job:** a healthy blood array reads Normal on Met-A's gauge against IAM-Atlas v2.1.

**Sources:** GEO GSE235717 GSM7508429 (public: one healthy whole blood's raw IDATs, its full measured mix); IAM-Atlas v2.1 (public, in this
repository and on Zenodo).

**Reference:** IAM-Atlas v2.1 only, in the mix of cells the IAM-Deconvolver measures; no same-run tare, no controls.

**Synthetic test:** `DEV_METAA_HEALTHY_01/synthetic_06.py` (every array through the one command with the rehearsal's flags, `--full-chain
--cell-panel atlas --dev-line-censored`; built from every cell in the real blood's measured mix; ten draws, one per Google machine, saved as each
finishes; scorer `DEV_METAA_HEALTHY_01/synthetic_06_score.py`).

**Range of the data:** real arrays (331 measured by the chain): design I slope 0.993 to 1.009, noise 0.003 to 0.015; design II slope 0.999 to
1.030, noise 0.006 to 0.024; the 100 real bloods' ratio 0.944 to 1.05 around 0.983 (sd 0.017). The synthetic arrays: test 5's center and both
tails (design II slope 1.0175 and 1.0301 and 0.9991, noise up to 0.024), ten new draws each, in the real blood's own mix on every atlas cell.

Bars (scorer), test 5's unchanged: 1. each of the 30 arrays reads A / A_pred inside Normal, 0.95 to 1.05; 2. each condition's mean of its 10 draws
within 0.005 of 1; 3. a 5 % loss shows on each. If bar 2 is met, the method reads its own construction at 1 on every atlas cell and the rehearsal's
0.983 is a property of the real bloods, examined next; if not, the center comes from the method on every cell and is fixed before any fresh test.

**Prior reading:** GSM7508429 is spent (tests 1 to 5); nothing else is read.

## Synthetic test 6, read: not met (`data/DEV_METAA_HEALTHY_01/synthetic_06_summary.txt`; ten draws on Google, the one command with the rehearsal's flags)

| Array | Mean A / A_pred, 10 draws | Range | Test 5 (six cells), mean |
|---|---|---|---|
| center | 1.00555 | 1.00402 to 1.00754 | within 0.0010 of 1 |
| low tail | 0.98735 | 0.98081 to 0.99367 | within 0.0010 of 1 |
| high tail | 1.00011 | 0.99512 to 1.00604 | within 0.0010 of 1 |

Bar 1 met (all 30 inside Normal); bar 2 not met (center 0.0006 over, low tail 0.0077 under); bar 3 met (a 5 % loss shows on each). The real
blood itself reads 1.00222 on every atlas cell (14 cells in its measured mix). **What it shows:** read on the full mix the Deconvolver measures,
the prediction drifts with the array's line (flatter, noisier arrays read lower), where on six cells it did not; the rehearsal's center of 0.983
lies in that direction. The cause is found offline before anything changes in the chain.

**Test 6's drift, diagnosed offline and on three draws (development; nothing scored).** Offline, the line measure reads each planted line correctly on the 14-cell mix (low tail design II slope planted 0.9991,
measured 0.9998) and A / A_pred is within 0.002 of 1; in the chain every condition's line came out near the same values (design II slope 1.027 to
1.030, whatever was planted). So the line was measured against a mix other than the one built. `DEV_METAA_HEALTHY_01/synthetic_06_diag.py` reads
test 6's arrays for three draws again, keeping each array's measured shares and prediction, on Google.

**Test 6's diagnostic, three draws read again with their shares kept (development; nothing scored).** (`synthetic_06_diag_draw1.json` to `draw3`, in the build bucket; summarized here.) On every constructed array the
Deconvolver moved 7 to 24 % of the mix between neighboring subsets (CD56-bright NK to NK, CD8 central memory to CD8, naive CD4 to CD4 and
memory CD4). Read offline with those measured shares, the center reproduces the chain (1.0045 against 1.0053), the low tail does not (0.9954
against 0.9808), and offline the low tail's design II line measures 0.998 where the chain measured 1.030. So inside the chain the line is
measured on values other than the ones rebuilt offline. `DEV_METAA_HEALTHY_01/synthetic_06_capture.py` runs the one command in its own process
on a low-tail and a center array and keeps the exact values the line is measured on (no chain code changed).

## The capture: where test 6's drift comes from, and synthetic test 7 (written and committed before it runs)

**Found** (`synthetic_06_capture.json`, `cap_low.parquet`, `cap_center.parquet` in the build bucket; `synthetic_06_capture.py`; read by
`DEV_METAA_HEALTHY_01/synthetic_06_capture_read.py` into `synthetic_06_capture_read.txt`): `--full-chain`
turns on the development atlas ruler (`--dev-ruler`), so the whole-sample reading is made on the ruler-mapped values, not the array's own. The
values the healthy line was measured on are a stretched copy of the input (low tail, design II: 1.0344 x input - 0.0154; center 1.0156 x input
- 0.0065) and the low tail kept 383,131 of its 770,920 sites. The healthy prediction then measures the array's line again on those values, so
the array's blur is handled twice: on the captured values the line measures design II slope 1.0295 (planted 0.9991). Read on the array's own
values with the chain's own mix, the low tail gives A / A_pred 0.99822 and the center 1.00498 (the remaining center offset is the Deconvolver's
share drift between neighboring subsets). **This also applies to the per-person rehearsal:** it ran with `--full-chain`, so its whole-sample
A and prediction were on the ruler, where test 5 (without it) was on the array's own values; its 98 of 100 is a reading of that combination.

**The fix (development, off by default):** `--dev-pred-raw` (`stage_metaa_atlas.PRED_RAW`): the healthy prediction and the A it is compared
with are read on the array's own values; the ruler's whole-sample A is kept beside it.

**The instrument's job:** a healthy blood array reads Normal on Met-A's gauge against IAM-Atlas v2.1.

**Sources:** GEO GSE235717 GSM7508429 (public, spent); IAM-Atlas v2.1 (public, in this repository and on Zenodo).

**Reference:** IAM-Atlas v2.1 only, in the mix the IAM-Deconvolver measures; no tare.

**Synthetic test:** `DEV_METAA_HEALTHY_01/synthetic_07.py` (test 6 with `--dev-pred-raw`, seed 20261137 + draw; ten draws on Google; scorer
`DEV_METAA_HEALTHY_01/synthetic_07_score.py`, which refuses any draw whose prediction was not read on the array's own values).

**Range of the data:** the real arrays' lines as the chain measured them on 331 arrays, design I slope 0.993 to 1.009 and noise 0.003 to 0.015,
design II slope 0.999 to 1.030 and noise 0.006 to 0.024. The synthetic: test 6's center and both tails (design II slope 1.0175, 0.9991 and 1.0301, noise up to 0.024).

Bars (scorer), test 6's unchanged: 1. each of the 30 arrays reads A / A_pred inside Normal, 0.95 to 1.05; 2. each condition's mean of its 10 draws
within 0.005 of 1; 3. a 5 % loss shows on each.

**Prior reading:** GSM7508429 is spent; nothing else is read.

## The per-person rehearsal read again with the prediction on each array's own values (written and committed before any of its readings exists)

The first rehearsal (98 of 100 inside Normal) ran with `--full-chain`, so its healthy prediction was measured on atlas-ruler values (the
capture, above). The same 100 spent bloods are read again with `--dev-pred-raw` (series run `wb100_raw`, GitHub).

**The instrument's job:** a healthy blood array reads Normal on Met-A's gauge against IAM-Atlas v2.1.

**Sources:** the five spent series of bar 2 (public, GEO; `bar2_labels.csv`); IAM-Atlas v2.1 (public, Zenodo).

**Reference:** IAM-Atlas v2.1 only, in each blood's own Deconvolver mix; no tare.

Bars (scorer `DEV_METAA_HEALTHY_01/wb100_raw_score.py`, the first rehearsal's, unchanged): 1. at least 90 of the 100 each read A / A_pred inside
Normal, 0.95 to 1.05; 2. in each of the five laboratories at least 16 of its 20 do. A bundle whose prediction was not read on the array's own
values is refused.

**Prior reading:** all 100 are spent (bar 2 and the first rehearsal).

**Synthetic test 7, read: not met** (`DEV_METAA_HEALTHY_01/synthetic_07_summary.txt`; all 10 draws.) Bar 1 met: all 30 arrays inside Normal. Bar 3 met: a 5 % loss
shows on each. Bar 2 not met: the center's mean 1.00575 and the high tail's 1.01165 (the low tail 0.99769). With the prediction on each array's
own values the low tail is corrected (test 6: 0.98081); what remains follows the IAM-Deconvolver's share drift between neighboring subsets
(the diagnostic above), which the healthy prediction does not yet carry.

**The per-person rehearsal read again with the prediction on each array's own values: met** (`DEV_METAA_HEALTHY_01/wb100_raw/wb100_raw_summary.txt`, `wb100_raw_readings.csv`; series run wb100_raw, the committed
scorer; none refused.) All 100 healthy bloods read A / A_pred inside Normal, 20 of 20 in each of the five laboratories (the first rehearsal, on
the atlas ruler's values: 98 of 100). Recorded: the center is 0.9869 (laboratory medians 0.9720 to 1.0022), below 1 as synthetic test 7's
centering bar, not met, also shows; the C-Score within its derived limit 96 of 100 (that limit's term was calibrated on labeled-healthy bloods,
disclosed above, so it is not independent). Met-A on whole blood arrays against the atlas alone is not commissioned from this: these bloods
are spent, and test 7's centering is open.

**One default chain (the plan's first step to a working chain): the two fixes that were met are part of the full chain.** The author: get the chain working, then commission. `--full-chain` now turns on the healthy line
with censored anchors (test 5, met) and the prediction on each array's own values (the capture after test 6, test 7's low tail corrected),
so no reader has to remember their flags; the class floor (version 3 not met) stays off. Identity check: test 7's draw 1 again with both flags
removed from its commands (`synthetic_07_default.py`), same seed; its output must equal `synthetic_07_draw1.json` exactly.

**The result is each cell found (the second step to a working chain).** The author: "we dont score bundles of cells we score individual cells that are found by the
deconvolver in a sample". Every bundle now carries `cells_found`: one row per cell the IAM-Deconvolver found (the cells the chain did not mark
not found; no threshold chosen), each with Met-A, the C-Score and its limit, and IAM-A where IAM-A read that cell, each against that cell's own
atlas reference in the sample's own shares (`stage_metaa_atlas.cells_found`). The report's first table and first glance row are this list; the
whole-sample reading follows as a side line. Seen while checking it on a spent 450K breast array: a cell can be found and still have no Met-A
on that platform (breast luminal epithelium, 2,998 of its 6,000 identity sites on the 450K array, under the half the reading needs); the row
says so.

## When a cell's own reading reads that cell (the third step to a working chain; written and committed before it runs)

A cell's Met-A is read at its identity sites in the whole sample, so when the cell is a small share of the sample, those sites carry mostly
other cells' DNA and the reading is no longer about that cell. The chain will state, for each cell, the smallest share at which its reading
reads it; this synthetic measures it, with no number chosen. A cell's reading reads that cell at a share where a loss of held pattern planted
in that cell moves the cell's reading more than the same loss planted in every other cell of the sample at the same sites.

**The instrument's job:** each cell's own reading moves with that cell, not with the cells around it.

**Design** (`share_synthetic.py`, one share and cell per Google machine by `share_synthetic.sh`): IAM-Atlas v2.1 mixed in the shares the
chain found for the spent healthy blood GSM7508429 (`gsm7508429_shares.csv`), through the real arrays' median line and noise per probe design;
the target cell set to 0.2, 0.5, 1, 2, 5, 10, 20 or 50 % and the others scaled to the rest; three arrays each: healthy; a 10 % loss (each level
moved 10 % toward 0.5) in the target at its own identity sites; the same loss in every other cell at those sites. Nine cells: neutrophils,
monocytes, naive CD4 T cells, CD8 central memory T cells, naive B cells, CD56bright NK cells, and hepatocytes, colon epithelium and vascular
endothelium. Every array read by `run_sample.py --betas --full-chain --cell-panel atlas`.

**Sources:** GSE235717 (public, GEO: GSM7508429's raw arrays, the base mix's shares, read earlier); IAM-Atlas v2.1 (public, Zenodo: the cell
levels mixed).

**Prior reading:** GSE235717's GSM7508429 is spent (synthetic tests 3 to 7); here only its measured cell shares are used.

**Reference:** IAM-Atlas v2.1 only; no tare.

**Range of the data:** real cell shares the IAM-Deconvolver reports run from 0.4 % (monocytes in GSM7508429) to 80 % (the main cell of the 89
outside sorted samples, `crosslab_chain_table.csv`), and tissue cells in tissue from 1 to 47 % (the spent breast arrays); real array noise
0.003 to 0.024 per design (331 arrays). Synthetic: shares 0.2 to 50 %, the real median noise (0.0066 and 0.0098).

**Bars:** 1. for each of the nine cells, at 50 % its own loss moves its reading more than the others' loss does (the reading reads the cell at
all). Recorded, not scored: for each cell, the smallest share at which this holds, which becomes the chain's stated share for that cell.

## Each cell's healthy width on arrays, from the atlas's own donors (the fourth step to a working chain; written and committed before it runs)

Normal is measured, not chosen, and 1 is the middle (the author). On arrays the width of healthy for a cell is the spread of that cell's Met-A
across IAM-Atlas v2.1's own build donors (`atlas_donors_samples.csv`: the 456 GEO arrays that entered the build, firm or conditional; the 16
ArrayExpress arrays follow), each read by the one command against the atlas and grouped by the cell the IAM-Deconvolver found most of in it
(never its label; `width_score.py`). Each donor helped make the reference it is read against, so the spread is widened by sqrt(n / (n - 1)),
exact for a reference that is the donors' mean and stated as such for the atlas's modeled levels; the held-out sets check it when instruments
are commissioned. The median of each cell's readings is recorded beside its width, so whether 1 sits in the middle is tested, not assumed.

**The instrument's job:** each cell's healthy width is measured from healthy donors read by the chain itself.

**Synthetic first** (`width_synthetic.py`, written before it runs; parts by `width_synthetic.sh`; scored by `width_synthetic_score.py`): 20
constructed people for each of neutrophils, CD8 T cells and B cells, each person the cell's atlas level at every site plus a planted person
spread at the atlas's own measured donor spread, site by site or half shared by the whole person; through the real arrays' median line and
noise; read by the one command. Known answer: the readings' spread gives back the truth's spread.

**Sources:** the IAM-Atlas v2.1 build arrays (public, GEO: their raw arrays, listed in `atlas_donors_sources.csv`); IAM-Atlas v2.1 (public,
Zenodo: the levels and donor spread planted from).

**Prior reading:** the donor arrays are the atlas's own build inputs; none is a test sample, and none was read before as a test of an instrument.

**Reference:** IAM-Atlas v2.1 only; no tare.

**Range of the data:** real donor spread at the identity sites, measured from the atlas: neutrophils median 0.0043 (5 to 95 % 0.0039 to
0.0061), CD8 T cells 0.0060, B cells 0.0053, monocytes 0.0049, hepatocytes 0.0203; real array noise 0.003 to 0.024 per design. Synthetic: the
atlas's own spread at every site, site by site and half shared; the real median noise.

**Bars** (the synthetic): 1. for each of the three cells and both forms of spread, the readings' spread within 25 % of the truth's; 2. every
person's main cell found is the planted cell. The donors are read only after the synthetic is met.

**The one default chain, identity check: met** (`DEV_METAA_HEALTHY_01/synthetic_07_default_check.txt`). Test 7's draw 1 read with neither flag gives the same readings as with both, to every recorded digit: `--full-chain` now carries them.

**The width synthetic's first run: no reading made (a chain gap, found and fixed).** All 120 constructed people ran through the one command, but none got a per-cell
reading: the chain sent a sample to the atlas readings only when its specimen was whole blood, buffy coat, plasma, tissue or a named atlas cell
('sorted CD8 T cells'), and the synthetic's arrays were labeled 'sorted cells'. The route depended on the label naming a cell. Fixed in the
chain (`stage_0_intake.is_isolated`): any isolated, sorted or purified sample is routed to the atlas readings, and the Deconvolver decides what
it holds; isolated neutrophils keep the commissioned route. The atlas's own donor arrays would have met the same gap. The first run's parts are
set aside in the build bucket (`metaa_healthy/width_first_run/`) and the synthetic runs again unchanged on the fixed chain.

**When a cell's own reading reads that cell, read: not met, and what it shows** (`DEV_METAA_HEALTHY_01/share_synthetic_summary.txt`; 72 parts, every array by the one command).
Bar 1 not met: at 50 % the cell's own loss moved its reading more than the same loss in all the other cells for 8 of the 9 cells, and the two
moves were within about 10 % of each other for all nine (ratio 0.981 to 1.112). **What it shows, recorded:** at every share, the reading moved
with a loss in the cell and with a loss in the others in proportion to the DNA each holds at the cell's identity sites (the ratio of the two
moves follows share / (1 - share): about 0.11 at 10 %, 0.25 at 20 %, 1 at 50 %). So a cell's Met-A read inside a mixed sample is the
sample's disorder at that cell's identity sites. It cannot tell, from one array, whether a change is in the cell or around it. It reads the
cell itself where the cell is nearly all the sample (an isolated sample), and it reads the sample at the cell's sites everywhere else.
**Consequence for the chain:** each row of `cells_found` says which it is; a change is given to one cell only where the physics can place it:
in an isolated sample, molecule by molecule on sequencing (each molecule comes from one cell), or in one person read against their own earlier
draws with the cell shares measured each time.

**The width synthetic, read on the fixed chain: not met** (`DEV_METAA_HEALTHY_01/width_synthetic_summary.txt`; 120 constructed people, every array by the one command).
Bar 2 met: every person's main cell found is the planted cell. Bar 1 not met: the readings' spread is 0.81 to 9.6 times the truth's, and their
median sits near 0.89 where the same people read about 0.999 on their noise-free levels. **What it shows, recorded:** a cell's Met-A read on an
array takes the array's own line and noise straight into the reading; the whole-sample reading already divides by its healthy prediction (the
array's measured line and noise applied to the atlas, A over A_pred), and the per-cell readings do not. The next chain step gives each cell's
reading the same healthy prediction at that cell's identity sites, synthetic first, before any width is measured from the atlas's donors.

**Each cell's reading gets the array's healthy prediction (chain change; the width synthetic runs again unchanged).** `stage_metaa_atlas.cell_prediction`: at each cell's identity sites, the array's own line and noise
(measured once on the whole array, as for the whole-sample reading) applied to the atlas in the sample's mix; the cell's reading in `cells_found`
is A_raw / A_pred, with both kept beside it. The width synthetic's previous parts are set aside (`metaa_healthy/width_second_run/`) and it runs
again with nothing changed in it.

**The width synthetic's third run: the per-cell prediction never ran (my error, disclosed; no new reading).** It came back identical to the second run, digit for digit. Cause: the healthy prediction refuses fewer
than 10,000 sites (written for the whole sample), and a cell has at most 6,000 identity sites, so every cell's prediction was refused and
its reading stayed uncorrected, silently. Fixed: `predicted_healthy` takes a site floor, the per-cell call uses 500 (the floor the chain's
by-place readings already use), and a refused prediction is now written on the row. The third run's parts are set aside
(`metaa_healthy/width_third_run/`) and the synthetic runs again unchanged.

**The width synthetic, read with each cell's healthy prediction: center met, spread not met.** (`DEV_METAA_HEALTHY_01/width_synthetic_summary.txt`; 120 constructed people, every array by the
one command). Bar 2 met (main cell found 120 of 120). The readings now center where the truth does: median 0.9999 to 1.0078 against the
truth's 0.998 to 1.000 (0.89 before the prediction). Bar 1 not met: the spread between people reads 2.6 to 9.3 times the truth's (0.0036 to
0.0162 against 0.0009 to 0.0062). Each person carries a fresh draw of the array's own noise, so a reading's spread is the person spread and
the array's noise together; the bar asked for the person spread alone. Whether the excess is the array's noise, which every real array has,
or something the chain adds, is tested next by a diagnostic through the one command (`width_diag.py`, development, nothing scored): the same
people with the array's line but no random noise (if the chain is right their spread equals the truth's), and one person read 20 times with
fresh noise draws (the array's noise alone). The bar is not changed.

**The width diagnostic, read on 34 of its 48 parts: the excess spread comes from the chain, not from the array's noise (development; nothing scored).** With the array's random noise taken out (its line kept), the readings of the same people still spread 2.3 to 8.9
times the planted truth (B cells 4.1 and 8.9, CD8 T cells 3.0 and 6.7, neutrophils 2.5 and 2.3, for site-by-site and half-shared person spread).
One person read 20 times with fresh noise spreads 0.003 to 0.005: the array's own noise alone. So most of the excess is added by a step of
the chain. Two signs of which step: in neutrophils with half the spread shared by the whole person, the reading over the truth falls as the
truth rises (rank correlation -0.98), so a person-wide shift is partly taken out as if it were the array's own line; in B and CD8 T cells
the excess follows neither the truth nor the Deconvolver's share, and a sorted CD8 sample was read as low as 58 % CD8. A second diagnostic
(`width_diag2.py`) reads the no-noise arm again keeping each person's raw and predicted Met-A, the line the chain measured, the planted
line and the five largest shares. The width synthetic's bar is not changed.

**The second width diagnostic: the extra spread is the array's line, measured on too few sites (development; nothing scored).** The no-noise arm read again (all 24 parts),
keeping each person's raw and predicted Met-A and the line the chain measured. Each cell's prediction follows the measured line's slope
(rank correlation 0.89 to 0.98 at site-by-site spread), and the slope moves from person to person by about 0.0004 with no array noise at all: each
person's own small differences at the ruler's anchors (about 8,000 sites, held below 0.05 or above 0.95 in all 79 cells). Near a level of 0.9 a
slope error of 0.0004 moves the entropy by about 0.25 %, three times the healthy spread between people (0.1 %). Offline, on the same 120 people
built the same way: with the planted line the readings spread 1.05 to 1.6 times the truth; with the line as the chain measures it, 2.6 to 3.7
times (the chain's own 2.5 to 4.1); with the line measured on every site the sample's own mix holds firmly (about 110,000 to 131,000 sites, the
identity sites of the cells found set aside), 1.0 to 1.5 times at site-by-site spread, the slope moving by 0.00006. Where half the spread is shared
by the whole person, the chain also reads the cell's share lower (a sorted CD8 sample as low as 58 % CD8), which doubles the spread again; and a
shift shared by every site cannot be told apart from the array's own offset by any line, so the wide line takes part of it out (neutrophils 0.0025
against a truth of 0.0062). The chain change: `--dev-line-wide` (stage_metaa_atlas.LINE_WIDE, development). The width synthetic runs again,
unchanged but for that flag (`width_synthetic_wide.py`).

**The width synthetic with the wide line: not met, closer** (`DEV_METAA_HEALTHY_01/width_synthetic_wide_summary.txt`; 120 people, the one command with `--dev-line-wide`). The readings now spread
1.6 to 4.1 times the truth (2.2 to 2.7 at site-by-site spread, was 2.5 to 4.1 with the anchors' line); centers 1.0018 to 1.0061; main cell found
120 of 120. What remains is the array's own noise: at site-by-site spread the healthy people differ by only 0.09 % to 0.11 % in a cell's Met-A,
and one array reads that Met-A with about 0.2 % uncertainty of its own (the repeat arm, one person read 20 times, gave 0.29 % to 0.54 % on the
anchors' line), so no single array can give back a spread that small. That bar asked the chain to recover the true spread between people; the
gauge's job is smaller: a healthy person reads Normal, where Normal is the width of healthy people read the same way, the array's own noise
included, as a new person's reading includes it.

**The new-people test of the measured width (written and committed before it runs).** The instrument's job: a healthy person reads Normal on
the gauge. Normal for each cell and spread: the 20 people just read (the donors), median plus or minus 1.96 times their spread. Twenty new
healthy people per cell and spread, built the same way from a new seed (`width_new.py`), read by the one command with `--dev-line-wide`.
Bars: 1. at least 18 of each 20 new people inside Normal, every cell and spread; 2. every main cell found is the planted cell. Reported beside
it, not scored: the donors' spread against sqrt(truth squared plus the instrument's own repeat spread squared), the repeat arm read again on
the wide line (`width_diag3.py`). Scored by `width_new_score.py`. If met, the same rule is applied to the atlas's own donors: each cell's Normal
is the width of its healthy donors read by the one command.

**Normal:** the measured width of the atlas's own donors read by the one command (in this synthetic, its 20 constructed donors stand in for them); no fixed band.

**The new-people test of the measured width: not met, 5 of 6** (`DEV_METAA_HEALTHY_01/width_new_summary.txt`). New healthy people inside the Normal
measured on the donors: neutrophils 19 and 19 of 20, CD8 T cells 18 and 19, B cells 18 where half the spread is shared and 14 of 20 where it is
site by site; main cell found 120 of 120. The B cells' 20 donors read narrower (spread 0.00196) than one person read 20 times (0.00251, the
array's noise alone), a narrow draw for 20 donors; the new people spread 0.00357. A width measured on a finite set of donors carries that
sampling error, and median plus or minus 1.96 spreads does not allow for it.

**The second new-people test (written and committed before it runs).** The instrument's job: a healthy person reads Normal on the gauge.
Fresh donors and fresh new people (new seeds for both, `width_new2.py`), built and read exactly as before.

**Normal:** the measured width of the atlas's own donors read by the one command (here 20 constructed donors stand in), as the prediction interval for one new reading: median plus or minus t(0.975, n - 1) times the donors' spread times sqrt(1 + 1/n); no fixed band.

**Bars:** 1. at least 18 of each 20 new people inside Normal, every cell and spread; 2. every main cell found is the planted cell. Scored by
`width_new2_score.py`. If met, Normal on real samples is each cell's prediction interval from the atlas's own donors (read on Google now).

**Each cell's healthy width on arrays, from IAM-Atlas v2.1's own donors: measured** (`atlas_donor_widths_summary.txt`, `atlas_donor_widths.csv`; 455 of the 456 build arrays read by the one command with `--dev-line-wide` on Google, `donors_wide.sh`; one array had no file). Grouped by the cell the Deconvolver found most
of (never the label); 26 cells with 3 or more arrays. Widths 0.70 % (central memory Th1 cells, 0.0070; NK cells 0.0076) to 5.2 % (neutrophils, 0.0515); blood cells' medians 0.98 to 1.01, so 1
sits in the middle for them, within one width. Five cells' medians sit 2 to 5 widths below 1: cortical neurons 0.922, oligodendrocytes 0.925,
astrocytes 0.906, effector CD8 T cells 0.919, vascular endothelium 0.841 (its arrays only 37 % endothelium by the Deconvolver); each comes from
one series, so the cell and its laboratory cannot be told apart.

What makes the width (`atlas_donor_widths_by_series.csv`, diagnostic, nothing scored): the laboratories. Neutrophils read 0.960 to 1.080 by
series (7 series), and within a series their spread is 2.5 %, half the whole; monocytes 0.965 to 1.016 (within 1.1 % of 2.3 %); CD4 T cells
0.952 to 1.044 (within 1.1 % of 2.2 %). The width across laboratories is the honest Normal for a sample from a laboratory the atlas never
held, and it is wide; the laboratory term on arrays, after each array's own line and noise, is the next thing to measure (what in an
array's chemistry it follows), before any instrument is commissioned on it. No healthy reference from other samples is used for it.

**The second new-people test: not met, 5 of 6** (`width_new2_summary.txt`). New healthy people inside the prediction interval: B cells 20 and 17 of 20,
CD8 T cells 19 and 20, neutrophils 19 and 19; main cell found 120 of 120. The three B cells outside read 1.018 to 1.029 where their truth is
0.995 to 0.998; they are the people with the largest shift shared by every site (-3.0, -2.0 and -1.6 standard deviations), the limit already
recorded (a shift shared by every site partly looks like the array's own line), and the Deconvolver read 2 to 4 % of other cells in them.

My error in the bars: with a Normal that holds 95 % of healthy people, 18 of 20 in each of 6 conditions fails a perfect instrument 37.6 % of
the time by chance; the first and second tests used it. The false-fail rate of a bar is computed before it is sealed from now on.

**The third new-people test (written and committed before it runs).** The instrument's job: a healthy person reads Normal on the gauge.
Fresh donors and fresh new people again (`width_new3.py`), built and read exactly as before.

**Normal:** the measured width of the atlas's own donors read by the one command (here 20 constructed donors stand in), as the prediction interval for one new reading; no fixed band.

**Bars** (false-fail rates for a 95 % Normal computed first): 1. at least 108 of all 120 new people inside (0.7 %); 2. at least 16 of 20 in
every cell and spread (1.5 %); 3. every main cell found is the planted cell. Scored by `width_new3_score.py`. Read by these bars, the second
test's people come out 114 of 120 and every condition 17 or more; that is after the fact and does not count.

**The third new-people test: met** (`width_new3_summary.txt`). 112 of 120 new healthy people inside the Normal measured on their donors (bar 108);
every cell and spread 17 or more of 20 (bar 16): neutrophils 20 and 17, CD8 T cells 18 and 20, B cells 20 and 17; main cell found 120 of 120.
Normal as the measured width of healthy donors read by the one command does its job: a healthy person reads Normal.

**Array step 5: the printed state against the measured widths (chain change; its synthetic, the third new-people test, met).** Each cell
found now carries its Met-A state against that cell's healthy width from IAM-Atlas v2.1's own donors
(`stage_metaa_atlas.width_state`, the table `chain/Runtime Matrices/Met_A_Widths/donor_widths_v1.json`, frozen with its checksum): Normal
within 1 plus or minus k widths (k from the prediction interval for one new reading from that cell's n donors), else above or below Normal,
with its distance from 1 in widths; the report's table 'Each cell found' prints it. 1 is the middle; a cell whose donors read their middle
more than 2 widths from 1 is printed with that note and is not commissioned (7 cells: astrocytes, cortical neurons, cytotoxic CD4 T cells,
lung bronchus epithelium, oligodendrocytes, effector CD8 T cells, vascular endothelium). A cell gets a state only at a share it held among
its donors (share_min; neutrophils 69 %, monocytes 73 %, CD4 T cells 21 %, CD8 T cells 42 %, B cells 50 %, NK cells 66 %); below it, its width
is not measured and the row says so.

**The gap this shows, recorded for the next step:** the atlas's donors are sorted cells, so the widths cover a cell that is most of its sample.
In a regular blood draw the neutrophils are often 50 % to 65 % and the other cells far less, so most cells of a whole blood get no state from
these widths. A width for cells inside a whole blood, from the atlas alone, needs its own design (synthetic first), and the laboratory term
(above) comes with it.

**What the laboratory term follows on arrays: the array's own noise (development, diagnostic; nothing scored).** On the 455 donors (`atlas_donor_noise_relation.csv`),
within each blood cell a donor's Met-A rises with its array's noise as the chain measures it (design II; rank correlation 0.83 neutrophils,
0.69 monocytes, 0.68 CD4 T cells), not with the array type (450K or EPIC) or the line's slope. Pooled over 310 blood-cell donors, an array with
noise 0.010 instead of 0.005 reads about 3.4 % higher, and the noise alone carries about a fifth of the spread (2.95 % to 2.29 % once it is
taken out). The prediction already applies the measured noise, so it is under-applied. The lead, not yet tested: the chain measures the noise
on sites held below 0.05 or above 0.95, where a probe's noise is squeezed against its bound, while a cell's identity sites sit between 0.05 and
0.25 or 0.75 and 0.95, where it is larger. The width synthetic planted one noise at every level, which is why it could not show this. Next:
the synthetic with level-dependent noise (the noise of a probe growing toward the middle, at the real arrays' measured sizes), then the
noise measured by level on the array itself.

**The noise lead: not supported** (`noise_level_diag_rows.csv`; GitHub runs 42499303814, 42499319952, 42499335955; the run at 0.004 still running). With a noise growing toward the middle levels the readings do not rise
with the noise the chain measures (slope -0.46 per unit; the donors' 6.89): error 0.0040, 0.0027 and 0.0000 at a planted noise of 0.007,
0.010 and 0.014; with one flat noise, 0.006 at every size. So the squeezed noise at the firmly held sites does not explain why the donors'
Met-A follows their arrays' measured noise. What the measured noise carries on real arrays is open: the same DNA read in three laboratories
(GSE230132, 30 EPIC arrays of the GIAB lines, DEV_EPIQC_ARRAY_01) differed by only 0.4 to 0.8 % in Met-A on the earlier chain, far less than
the donors' 2 to 5 % between series, which points at the samples (how each laboratory sorted and handled its cells) more than at the arrays.
Next: the same 30 arrays read by today's one command (`epiqc_same_dna_samples.csv`, development; spent), Met-A against the measured noise on
identical DNA.

**Array step 6, the rehearsal, first half: the 100 healthy bloods read with step 5** (`rehearsal_bloods_states.txt`; spent; development). As the step-5
record foresaw, the widths from sorted donors do not reach a whole blood: the neutrophils are a median 55.4 % of these bloods, and only
5 of 100 reach the 69 % their width was measured at; those 5 read Normal (Met-A 0.966 to 0.995); every other cell found at 5 % or more
carries no state (no width at its share, or none for the cell). On a regular blood draw the array gauge has no Normal yet. The breast set's
reading is under way on Google.

**The same DNA in three laboratories, on today's chain** (`epiqc_same_dna_now.txt`; GSE230132, 30 arrays of lymphoblastoid lines, cells the
atlas does not hold, so a diagnostic only). The per-cell Met-A of one line differs between laboratories by 0.9 % to 5.5 % (the earlier
whole-sample reading, under 1 %); it does not follow the array's noise (-0.23) but does follow the Deconvolver's share (-0.63: where a
laboratory's array reads less of the main cell, its Met-A reads higher), which carries about a quarter of it. The same thread as the B cells
outside Normal in the second new-people test: small false shares in the expectation move a cell's reading.

**What this means for the five steps (my reading, for the author):** steps 4 and 5 are met for a cell that is most of its sample. Before the
arrays can be commissioned on a regular blood draw, the chain needs (1) a Normal for each cell inside a whole blood, from the atlas alone, and
(2) the per-cell reading made steady against small false shares; both synthetic first.

## A Normal for each cell inside a whole blood, from the atlas alone (written before it runs)

The instrument's job: a healthy whole blood reads Normal for each cell found in it.

Healthy whole bloods are built from IAM-Atlas v2.1's own donors, the only healthy reference: the calibrated values of the 146 build arrays
that are at least 85 % one blood cell (`blood_donors_values_samples.csv`: neutrophils 57 arrays from 7 series, monocytes 40 from 6, B cells
17 from 3, CD4 T cells 12 from 2, NK cells 12 from 2, CD8 T cells 8 from 3), read again by the one command with their values kept (Google,
`series_google.sh` with KEEP=1). One constructed blood: one donor array of each of the six cells drawn at random, mixed site by site in the
shares the IAM-Deconvolver found in one of the spent 100 healthy bloods (its shares of these six, scaled to 1; a mix of values is the value of
the mixed DNA when every cell gives the array the same amount of signal for its share, stated as the build's assumption). Each constructed blood is read by the
one command as whole blood. 100 constructed bloods (the donors) give each cell's Normal at the shares it held there; 100 more from new draws
(the new people) are scored. Synthetic first: the same construction from the width synthetic's planted people, known answer a Normal reading.

**Normal:** the measured width of the atlas's own donors read by the one command, here mixed into whole bloods, as the prediction interval for one new reading; no fixed band.

**Bars** (false-fail rates for a 95 % Normal computed first): 1. of the new constructed bloods, at least 90 % of all their cell readings
with a width inside Normal (for 600 readings, 6 cells of 100 bloods: fails by chance under 0.1 %); 2. each of the six cells at least 85 of 100
(under 0.1 %); 3. every constructed blood's six cells found, each within 5 points of its built share.

**Array step 6, the rehearsal, second half: the spent breast set (GSE69914, 407 arrays) read with step 5** (`rehearsal_breast_states.txt`; development).
**Prior reading:** GSE69914 was read before (R17, the breast replay, and the earlier VAL tests listed in `CANON/data_spent.csv`); this reading scores nothing. Of 2282 cells found at
5 % or more, 31 get a state (all Normal; vascular endothelium 30, cd4 t cells 1); the rest have no width at their share or none for the cell,
and 518 have no Met-A reading at all. The 30 endothelium states carry the note that the atlas's own endothelium donors read their middle at 0.8411, so they are not commissioned. Breast luminal and basal epithelium and adipocytes have no donor width on arrays. As for the
bloods, the gauge has no Normal yet for the cells of a mixed specimen; the rehearsal shows the chain runs end to end and says so on every row,
and it shows the two things array commissioning needs next (above): a Normal for cells inside a mix, from the atlas alone, and the per-cell
reading made steady against small false shares.

**The constructed bloods from the atlas's donors (written and committed before they run).** `blood_constructed.py`, as the design above, with one tightening:
each cell's donor arrays are split at random into two halves (`blood_donors_halves.csv`; neutrophils 29 and 28, monocytes 20 and 20, B cells
9 and 8, CD4 T cells 6 and 6, NK cells 6 and 6, CD8 T cells 4 and 4), one building the bloods that give Normal and the other the new bloods
scored, so no array is in both. The shares are the 100 bloods' with subtypes counted under their parent cell (`blood_shares_parent_map.csv`). Run
on Google from the values kept by `run_series.py read --keep-values` (series bv2_donors). Scored by `blood_synthetic_score.py`, the bars above.

**The constructed-blood synthetic, read again unchanged.** Its first reading (GitHub runs in `bs_runs`, 16 parts) wrote each report's side files into the
uploaded folder (393 MB a part), too large to bring back; its results were not opened. The same seeds are read again with each blood's work
in a temporary folder, so the readings are the same; only where the files are written changed (the builder for the real bloods likewise).

**The constructed-blood synthetic: not met, on bar 3 only** (`blood_synthetic_summary.txt`, `blood_synthetic_diag.txt`). Bar 1 met: 472 of 514 new healthy cell readings inside the
Normal built from the donor bloods (bar 90 %); bar 2 met: every cell 85 % or more (neutrophils 90 of 100, monocytes 61 of 70, CD4 T 88 of
100, CD8 T 74 of 81, B 76 of 78, NK 83 of 85). Bar 3 not met: 86 of 600 cells were not found, nearly all small (built share median
1.4 %), and the found shares missed by up to 11.5 points (CD8 T cells); the Deconvolver gave about 5 % of each blood
to cells that are not in it (bone-marrow precursors, eosinophils, macrophages).

My error in bar 3: it asked for every cell at any share, below the Deconvolver's reporting floor; a bar on the found shares should have
named that floor. What the synthetic shows: where a cell is found in a whole blood, its Normal from the atlas's donors does its job; the
Deconvolver's false shares are the open problem on blood arrays, the same thread as the same-DNA arrays.

**The constructed bloods from the atlas's donors: not met** (`blood_constructed_summary.txt`, `blood_constructed_donors.txt`). Bar 1 met: 456 of 483 new cell readings inside Normal (94 %).
Bar 2 not met, by NK cells (61 of 72, 84.7 %; neutrophils 94 of 100, monocytes 49 of 52, CD4 T 99 of 99, CD8 T 87 of 88, B 66 of 72). Bar 3 not
met (the Deconvolver's shares, as in the synthetic; largest miss 11.2 points).

Why NK: each cell's Normal rests on few people (distinct donor arrays per half: neutrophils 28, monocytes 19 and 20, B 9 and 8, CD4 T 6, NK
6, CD8 T 4). The 100 donor bloods reuse them, and the interval counted bloods (65 to 100) instead of donors; the NK Normal half's donors read a
median 0.985 to 0.997, the new half's 0.983 to 0.990. My design error: a Normal's interval counts the distinct donors behind it.
What this shows: on arrays from the atlas alone, a per-cell Normal inside a whole blood holds 94 % of new healthy readings; what it needs is
the interval counted by donors, more donors for the small-pool cells, and the Deconvolver's false shares solved.

**The rotation test (written and committed before its readings exist).** The instrument's job: a healthy whole blood reads Normal for each cell found in it.
Diagnostic first, on the 200 bloods already read (`blood_constructed_rotation_diag_summary.csv`, after the fact, does not count): each donor
left out in turn and the interval counted by donors, 923 of 958 readings inside (every cell 90 % or more; neutrophils' Normal 0.93 to
1.07, the laboratory term carried in; monocytes' middle below 1).

**Normal:** the measured width of the atlas's own donors read by the one command, mixed into whole bloods, as the prediction interval for one new reading counted by donors; no fixed band.

The test: 200 new constructed bloods (`blood_constructed.py` MODE all, new seeds, every donor in the pool), read on Google, scored by
`blood_rotation_score.py`. Bars: 1. at least 90 % of all scored readings inside Normal; 2. each cell at least 85 %. Spent data (the atlas's
donors); a pass gives the gauge its Normal on a regular blood draw for development; commissioning needs a sealed test on unseen arrays.

**Correction:** the message of commit 9afcf3d6 gives the rotation diagnostic as 963 of 1000 inside; it was 923 of 958 (96.3 %), as written above and in `blood_constructed_rotation_diag_summary.csv`.

**The Deconvolver's false shares on known mixes (diagnostic, development; nothing scored).** On the 400 constructed bloods already read, whose
shares are known (`deconv_false_shares.txt`): the share given to cells that are not in the blood is a median 6.39 % of each real
constructed blood (10.60 % at the 90th percentile) and 3.80 % of each synthetic one, taken mostly from neutrophils (median
-2.42 points), monocytes (-2.08) and CD8 T cells (-1.54); the cells it goes to are bone-marrow precursors,
eosinophils and tissue macrophages. Three ways forward, each to be tested on these known mixes first: a reporting floor measured on them (a
share below it is not reported as a cell found); identity sites that separate the precursors from the blood cells they come from; or a panel
chosen by the route. The last needs care: the label picks only the route, never the cell (LABEL_RULE), so it is for the author and me to weigh.

**One signal behind two problems: GMP share and neutrophil Met-A (diagnostic, development; nothing scored).** A reporting floor does not answer the
false shares (`gmp_neutrophil.txt`): single false cells reach 14.79 % of a blood, and at a 5 % floor 116 of 631 are still reported.
The largest is GMP, the bone-marrow precursor of neutrophils and monocytes, in 170 of the 200 real constructed bloods, and how much of it
the Deconvolver reports follows which laboratory's neutrophil array is in the blood (2.31 % to 15.09 % of the neutrophil share). It follows
the neutrophils' Met-A too: across the 7 laboratories (rank correlation 0.75) and across the 64 sorted neutrophil donors themselves (0.80).
So the neutrophils' laboratory term and the Deconvolver's largest false share look like one signal: some neutrophil samples carry
precursor-like DNA. Not yet known which: younger neutrophils in some samples (donor or handling), or the atlas's GMP column overlapping
ordinary neutrophil states. Next, on spent data: the GMP and neutrophil columns at their identity sites, and whether the neutrophils' Met-A
moves to 1 when the GMP-like part is read as neutrophil.

**The rotation test: met** (`blood_rotation_summary.txt`). 913 of 939 new healthy cell readings inside the Normal measured on the
atlas's own donors with each donor left out in turn (bar 90 %); every cell 85 % or more (neutrophils 190 of 200, monocytes 109 of 118, CD4 T
181 of 186, CD8 T 168 of 168, B 145 of 146, NK 120 of 121). On arrays, a whole blood's cells found now have a Normal from the atlas alone,
for development. Spent data: commissioning needs the sealed test on unseen healthy blood arrays. The Deconvolver's false shares (GMP above)
stay open; this Normal holds with them in place.

**The rotation test's Normal was centered on the donors' median, not on 1 (my design; disclosed).** The chain's rule is that 1 is the middle. With each Normal
centered on 1, the same 200 bloods give 834 of 939 inside, and monocytes, NK and B cells fall under 85 %
(`blood_rotation_centered_on_1.txt`): inside a whole blood every cell but the neutrophils reads a little below 1 (medians 0.9874 to
0.9945). The offset follows the Deconvolver's false shares (neutrophils rise with them, +0.66; B and NK fall, -0.57 and -0.46), not each
cell's own share. So the rotation test shows a stable Normal around a middle that is not yet 1; the false shares (the missing neutrophil
precursors, above) come first. The commissioning test on unseen healthy whole blood (GSE234461 chosen, metadata only) waits for them.

**The same misplacement in plasma, from records already committed (diagnostic; nothing new read).** The 30 tuning plasmas read with the
megakaryocyte reference (`DEV_ONE_DRAW_01/plasma_seq_06/plasma_shares_diag_06.txt`, `mk_shares_compare.txt`) give monocytes a median 0.013 of
each plasma and colon macrophages 0.059 (lung macrophages 0.014): blood-cell DNA landing on tissue macrophage columns, as the constructed
bloods land on bone-marrow precursors and macrophages here. One GSE186007 healthy plasma (GSM7426135) read again on today's chain with the
committed reference (GitHub run 42538745757; no megakaryocyte column by default) gives granulocytes 0.6591, NK cells 0.1598, monocytes 0.0835.
On both routes the Deconvolver's misplaced shares are the open problem; on blood they follow the missing neutrophil precursors (GMP, above).
The plasma bars were not reached for depth (DEV_ONE_DRAW_01), so the shares were recorded and not read as a misplacement problem until now.

**The true-shares diagnostic: the false shares do not explain the offset** (`blood_true_shares_diag.txt`). On 30 rotation bloods read
again on Google, each cell's reading remade with the true built shares instead of the found ones reads the same, about 1 % below 1 (monocytes
0.9855, B cells 0.9868, NK cells 0.9890, with true shares). The correlation with the false shares above did not hold up as a cause. The 100 spent
healthy bloods, one array each, read below 1 too (neutrophils 0.9812, monocytes 0.9738, B cells 0.9792, NK cells 0.9829, CD8 T 0.9899, CD4 T
0.9987), while the atlas's sorted donors read close to 1 (`blood_true_shares_diag.txt`). So inside a real blood a healthy cell's middle is
not 1. This is the share synthetic's finding seen in healthy blood: a cell's reading in a mix is mostly its neighbors' DNA at its identity
sites. Two ways to read it: the middle inside a blood is measured on the atlas's donors mixed as real bloods are (the rotation test's
Normal), or the in-mix reading is changed so a healthy cell in a mix reads 1. For the author and me to decide before the commissioning test.

## The measured middle on real healthy blood from four laboratories, spent data (written before it is scored)

The instrument's job: a healthy whole blood reads Normal for each main cell found in it, on a gauge where the measured healthy middle is 1.

**Normal:** each cell's middle and width measured on IAM-Atlas v2.1's own donors only, read as the sample is read (the sorted donors where the
cell holds a share its sorted donors held; else the 200 healthy bloods built from those donors, the rotation test): the gauge reads each cell's
Met-A divided by that measured middle, so measured healthy is 1; Normal is 1 plus or minus k widths (k: the prediction interval for one new
reading from the number of donors). Nothing is set from the bloods scored here; the middles and widths are frozen before they are read
(`donor_widths_v1.json`, and the in-blood table built from `blood_rotation_rows.csv`, held out of the chain until this test is scored).

**Data (all spent; development, not commissioning):** 100 healthy whole bloods from each of four laboratories none of which gave the atlas
donors: GSE235717 (the rehearsal's 100, already read on today's chain), and 100 drawn at random (seed 20261109) from the healthy controls of
GSE318173 (Oslo, EPIC), the adults of GSE87571 (Uppsala, 450K) and GSE67393 (Tokushima, 450K), read by the one command on GitHub
(`realblood_*_samples.csv`; --full-chain --cell-panel atlas --dev-line-wide).

**Bars:** 1. in each laboratory at least 90 % of the six cells' readings that have a Normal read Normal; 2. across the four, each cell at least
85 %; 3. in each laboratory each cell's median gauge reading within 2 widths of 1 (no laboratory term left). **Known circularities, stated:**
the atlas columns were built from the same donors whose middles are measured (so the donors read near the prediction partly by construction),
and the built bloods hold six cell types; both are why the test is on laboratories outside the atlas. If monocytes fail across laboratories,
the next step is built bloods that also hold the atlas's own eosinophil and basophil donors (atlas only), not a change from these bloods.
**Prior reading:** all four were read before (GSE318173's healthy controls for Group-A's commissioning, GSE87571 and GSE67393 in earlier development, GSE235717 in the rehearsal; `CANON/data_spent.csv`); this test is development on spent data and commissions nothing.

**The four-laboratory test's scorer and frozen Normal, committed before any of its readings is scored** (`realblood_score.py`;
the in-blood Normal frozen as `realblood_inblood_normal.json`, from the rotation bloods; the sorted Normal is the chain's `donor_widths_v1.json`).
**Disclosed:** GSE235717 (the 100 bloods) was previewed against this same Normal before the test was written, 310 of 369 readings Normal and
monocytes 39 Normal, 46 below; its score in this test is therefore not blind, and the three other laboratories decide the test.

**The four-laboratory spent test of the measured middle: not met** (`realblood_summary.txt`). 1,672 readings of the six cells with a
Normal (1,645 by the in-blood table). Bar 1: Uppsala 487 of 529 (92 %) met; Oslo 227 of 360, Tokushima 301 of 414 and GSE235717 (previewed)
310 of 369 not. Bar 2: neutrophils 345 of 360 met; monocytes 161 of 329, NK cells 148 of 193, B cells 208 of 260 not. Bar 3: monocytes in Oslo
and GSE235717 and NK cells in Tokushima more than 2 widths from 1. Four Tokushima arrays refused at intake (`realblood_diag.txt`).
**What it shows (diagnostic, after scoring):** real healthy people spread 1.2 to 1.8 times wider within a laboratory than the constructed
bloods the in-blood Normal was measured on, and the laboratories' medians differ by 1.5 to 3.7 %. Both follow the Deconvolver's GMP share,
as in the constructed bloods: each blood's distance from the middle rises with it for neutrophils (+0.70) and falls for NK cells (-0.51);
Uppsala, the one laboratory that met bar 1, has the least GMP (median 0.5 %, Oslo 3.7 %, Tokushima 4.9 %). The measured middle is kept out
of the chain. The lead is DNA of maturing neutrophils, which have no column, landing on GMP; next, the in-between neutrophil column from the
public healthy bone-marrow maturation data (two donors, four stages), synthetic first, then these same spent bloods read again.

## Plan: an in-between neutrophil column for IAM-Atlas v2.1 (design only; nothing downloaded or run)

The instrument's job: the IAM-Deconvolver gives DNA of maturing neutrophils its own column instead of GMP's, so a healthy blood's
cells read Normal whatever share of maturing neutrophils it carries.

**Source (public, healthy, metadata checked only):** the bone-marrow neutrophil maturation series of the European blood epigenome project,
GRCh38: donor BM030613 (myelocyte S00JE0A1, metamyelocyte S00JFZA1, band S00JGXA1, segmented S00JHVA1) and donor BM060814 (S00VCUA1,
S00VDSA1, S00VEQA1, S00VFOA1), whole-genome bisulfite, per-CpG level and coverage tracks (a third donor, BM160614, holds RNA only). The
chain reads such track pairs at the atlas's sites already (`stage_seq_values.values_from_bigwig`) and carries provisional array columns
beside the atlas's own shares (`--dev-provisional`, Stage D21), as the megakaryocyte column was carried.

**Steps, each written and committed before it runs:**
1. **Platform bridge:** the two donors' segmented bone-marrow neutrophils, read at IAM-Atlas v2.1's sites, against the atlas's neutrophil
   column (arrays). If they do not read as neutrophils within the neutrophil donors' width, sequencing-made columns cannot join the array
   atlas this way, and the plan stops there.
   Known before it runs: IAM-Atlas v2.1 is on the array scale (its unmarked sites near 0.055, sequencing near 0), so sequencing levels are
   first put on the atlas's scale by the atlas build's own measured term (`Runtime Matrices/Atlas_V21/sequencing_term_v1.json`, the
   inverse of `stage_metaa_atlas.seq_units`); read raw, healthy whole-genome samples read 0.6 to 0.8 against it.
2. **Which stages make the column:** the stages' identity sites against GMP and mature neutrophils; a stage joins the column only if its
   sites separate it from both.
3. **Synthetic:** healthy bloods built from the atlas's columns plus the new column at known shares (0 to 15 %); bars: the new column's
   share within 0.05 of planted, and the GMP share falling to under 1 % where no GMP was planted.
4. **Spent bloods read again** with the column as a provisional column: the four laboratories' 400 healthy bloods. What it should show if
   the lead is right: the GMP share falls, and the laboratories' medians and the within-laboratory spread both shrink.

Two donors are few: the column is provisional and flagged, as the megakaryocyte column was, until more healthy donors are found.

**Step 1 of the column plan, first as a diagnostic: where the Deconvolver puts each maturation stage (written before it runs).** The eight tracks
(`bm_neutrophil_stages_samples.csv`) are read by the one command as sequenced samples whose cells are found against IAM-Atlas v2.1 (no
sequencing-atlas flag; specimen unknown), one GitHub machine each (`bm_neutrophil_stages_read.py`). Nothing is scored. What it shows if the
lead is right: segmented neutrophils found mostly as neutrophils, and the younger stages (myelocyte, metamyelocyte) put largely on GMP. If
the segmented cells are not found as neutrophils, the scale between the two platforms comes first (`sequencing_term_v1.json`).

**The maturation stages on the sequencing route: not readable there for this question (diagnostic, after reading).** All eight
bone-marrow tracks (`bm_neutrophil_stages_read.json`) are found as neutrophils 0.493 to 0.612 and lung interstitial macrophages 0.337 to
0.422, every stage alike, GMP at no share. Sorted blood monocytes on sequencing already put 0.315 on that macrophage column
(`blueprint_share_destinations.txt`), and the healthy plasmas put blood-cell DNA there too, so on the sequencing route myeloid DNA is split
onto it whatever its stage. Step 1 of the plan is not met as written: the segmented cells do not read as neutrophils alone, and the stages are
not told apart, so a column from these tracks cannot join the array atlas this way. The question moves to the array route: GSE42118's one
healthy promyelocyte array and one healthy CD34 array (450K, processed values only), read by the one command (`promyelocyte_read.py`,
written before it runs). If the promyelocyte is found largely as GMP, the lead holds on arrays.
