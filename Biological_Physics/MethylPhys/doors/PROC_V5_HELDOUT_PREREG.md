# PROC-V5-HELDOUT — pre-registration: can the atlas v2 posterior predict data it never saw?

**Written 2026-09-29, before any held-out observation is scored.** PLAN V5 / ATLAS_V2_SPEC A4: "mask 5 % of observations on a random
20 blocks, refit, predict from the posterior: 90 % interval covers 85–95 %." This file fixes the details the spec left open. It does
not move after the run.

## What is done
1. **Blocks:** 20 of the 700, drawn with `numpy.random.default_rng(2026).choice(700, 20, replace=False)`.
2. **Mask:** within each block, 5 % of observations (one sample's value at one locus for one cell) are held out, drawn with
   `default_rng(5000 + block)`. Only observations whose (cell, locus) pair has at least two observations are eligible, so every
   held-out value is predicted for a pair the refit still measures. Predicting a pair from the prior alone is a different test
   (and v2 writes such pairs NOT MEASURED).
3. **Refit:** the same model, priors, source terms (source_terms_v1.json), warm-up (600), draws (400), chains (4) and seed as the
   build (`PRNGKey(1000 + block)`), on the remaining 95 %.
4. **Predict:** for each held-out value, the posterior predictive is drawn from the model's own likelihood: for every posterior
   draw, Normal(loc, sqrt(var)) with loc and var exactly as in the fit (array: mu + d, var = donor_sd² + t² + floor; sequencing:
   a + b·mu, var = b²·donor_sd² (÷3 if pooled) + mu(1−mu)/coverage + t² + floor). The 5 %–95 % quantiles of those draws are the
   90 % interval.

## Bars (fixed now)
- **B1 (the spec's bar):** the 90 % interval contains the held-out value for **85–95 %** of held-out values, pooled over 20 blocks.
- **B2:** B1 holds within each data kind (array, WGBS, pooled) with at least 1,000 held-out values; a kind outside 85–95 % is
  reported by name even if B1 passes.
- **B3:** no single block below 80 % or above 98 %.

Reported, not bars: the 50 % interval coverage, the coverage by cell, and the mean absolute prediction error.

## What a result means
- Inside 85–95 %: the atlas's stated uncertainty is honest for new data of the same kinds.
- Below 85 %: the atlas is **over-confident** (intervals too narrow); readings carrying the atlas interval (V8) would overstate
  certainty.
- Above 95 %: **under-confident** (intervals too wide); detection is weaker than it could be.
Either failure is reported with the kind and cells that drive it before any change to the model.
