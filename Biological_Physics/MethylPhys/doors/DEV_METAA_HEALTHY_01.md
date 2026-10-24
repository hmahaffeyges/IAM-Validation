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
