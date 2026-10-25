# DEV-CSCORE-LIMIT-01: the C-Score's healthy limit, derived (plan 14.12)

Written and committed before any healthy C-Score is scored against the derived limit. Development; nothing here is commissioned.

**The instrument's job:** a healthy sample's C-Score reads at or below its limit.

**Why:** the C-Score's value for scattered departures (1) is derived by construction, but its healthy upper limit has been measured (the
commissioned neutrophil band, the measured band after the same-run tare, from six arrays). Here the limit comes from the statistic itself.

**The derivation** (`data/DEV_CSCORE_LIMIT_01/limit.py`): the C-Score is the variance of block means of 10 neighboring sites (times the
square root of 10) over the variance of the sites. Splitting the sites' sum of squares into between and within blocks, C times the number of
blocks over the number of sites is Beta((blocks - 1)/2, (sites - blocks)/2) exactly when departures are independent from site to site. The
limit is its one-sided 95 % point at the reading's own site count. Nothing is measured on healthy samples. **Stated before scoring:** the
derivation assumes departures independent from site to site. Met-A's identity sites lie far apart, so that should hold for each cell's C.
The whole genome's sites lie close together, where healthy cells co-vary and the atlas's own errors run along the genome, so the
whole-genome C may sit above its limit; if it does, that reading needs a term for neighboring sites, not a measured band.

**Synthetic test:** DEV_CSCORE_LIMIT_01/limit_synthetic_chain.py, DEV_CSCORE_LIMIT_01/limit_01/limit_synthetic_chain.txt

**Bars (at most three):**
1. **Synthetic, through the one command:** 12 constructed whole bloods (the atlas's cells in fixed shares, each site plus an independent
   scatter) read by `run_sample.py`: at least 90 % of their C-Scores (each cell at a share of at least 1 %, and the whole genome) at or
   below the derived limit; one more blood with a planted run of 200 neighboring neutrophil identity sites reads its neutrophil C above it.
2. **The job:** on the five laboratories' 100 healthy bloods (DEV-METAA-HEALTHY-01, spent), at least 90 % of each reading's values at or
   below the derived limit: the neutrophil C-Score's raw statistic, each cell's C (cells at a share of at least 1 %), the whole genome's C.
   Scorer: `data/DEV_CSCORE_LIMIT_01/limit_score.py`.

**Derivation check (before the synthetic, disclosed):** `data/DEV_CSCORE_LIMIT_01/limit_synthetic.py` on the statistic alone. Run 1 took C
as a chi-square over the blocks and ignored that the site variance contains the block variance (output kept as
`limit_01/limit_synthetic_run1.txt`); run 2 uses the exact Beta form (`limit_01/limit_synthetic.txt`). Its 2,000-map window was set too
narrow (about two standard errors) and the same seed drew the same maps in both runs. On many maps
(`data/DEV_CSCORE_LIMIT_01/limit_01/limit_check.txt`) the share above the limit is within a thousandth of 5 % at both 6,000 and 60,000 sites (values in
the file), and the simulated 95th percentile matches the limit to the fourth decimal.

**Bar 1, read: met** (`data/DEV_CSCORE_LIMIT_01/limit_01/limit_synthetic_chain.txt`): 84 values from the 12 independent bloods, share at or
below the limit 0.9643; the planted run's neutrophil C 1.9623 against its limit.

**Bar 2, read: not met overall; met for each cell** (`data/DEV_CSCORE_LIMIT_01/limit_01/limit_score_bar2.txt`). **Each cell's C**, at
identity sites chosen on IAM-Atlas v2.1, sits at or below its derived limit in 0.9495 of 1,009 values: the derived limit holds where its
assumption, sites far apart and departures independent, holds. **The whole genome's C** does so in 0.8000 of 100: its median sits under
the limit, but the genome's neighboring sites co-vary in healthy cells, as stated before scoring, so the whole genome needs a term for
neighboring sites, not a measured band. **The commissioned neutrophil C-Score's raw statistic** does so in 0.8000 of 40, its median above
the others'; its sites were chosen on the same arrays they are scored on, the selection effect Met-A step 3 found for the same sites.
**What follows:** the derived limit can replace the measured band for each cell's C-Score; the whole genome's limit needs the
neighbor term (next); the commissioned neutrophil C-Score keeps its measured band.
