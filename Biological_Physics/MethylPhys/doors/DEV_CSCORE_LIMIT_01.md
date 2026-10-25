# DEV-CSCORE-LIMIT-01: the C-Score's healthy limit, derived (plan 14.12)

Written and committed before any healthy C-Score is scored against the derived limit. Development; nothing here is commissioned.

**The instrument's job:** a healthy sample's C-Score reads at or below its limit.

**Why:** the C-Score's value for scattered departures (1) is derived by construction, but its healthy upper limit has been measured (the
commissioned neutrophil band, 0.877 to 1.152 after the same-run tare, from six arrays). Here the limit comes from the statistic itself.

**The derivation** (`data/DEV_CSCORE_LIMIT_01/limit.py`): the C-Score is the variance of block means of 10 neighboring sites (times the
square root of 10) over the variance of the sites. Splitting the sites' sum of squares into between and within blocks, C times the number of
blocks over the number of sites is Beta((blocks - 1)/2, (sites - blocks)/2) exactly when departures are independent from site to site. The
limit is its one-sided 95 % point at the reading's own site count. Nothing is measured on healthy samples. **Stated before scoring:** the
derivation assumes departures independent from site to site. Met-A's identity sites lie far apart, so that should hold for each cell's C.
The whole genome's sites lie close together, where healthy cells co-vary and the atlas's own errors run along the genome, so the
whole-genome C may sit above its limit; if it does, that reading needs a term for neighboring sites, not a measured band.

**Synthetic test:** DEV_CSCORE_LIMIT_01/limit_synthetic.py, DEV_CSCORE_LIMIT_01/limit_01/limit_synthetic.txt

**Bars (at most three):**
1. **Synthetic:** independent departure maps exceed the limit in 4 to 6 % of 2,000 maps at 6,000 and 60,000 sites (2 to 8 % of 300 at
   770,000); maps with a planted run exceed it in at least 80 %.
2. **The job:** on the five laboratories' 100 healthy bloods (DEV-METAA-HEALTHY-01, spent), at least 90 % of each reading's values at or
   below the derived limit: the neutrophil C-Score's raw statistic, each cell's C (cells at a share of at least 1 %), the whole genome's C.
   Scorer: `data/DEV_CSCORE_LIMIT_01/limit_score.py`.

**Bar 1, read: not met as written** (`data/DEV_CSCORE_LIMIT_01/limit_01/limit_synthetic.txt`; run 1 beside it). Run 1 took C as a chi-square
over the blocks and ignored that the site variance contains the block variance; its limit sat high. Run 2 uses the exact Beta form: 6,000
sites 5.50 %, 60,000 sites 3.65 %, the planted runs 99.8 %. The 60,000-site share falls outside its window. Both runs used the same seed, so
the same maps, and the window is about two standard errors wide. **Disclosed check** (`data/DEV_CSCORE_LIMIT_01/limit_01/limit_check.txt`,
seed 1): on 20,000 and 6,000 maps the share above the limit is 4.90 % and 4.97 %, and the simulated 95th percentile matches the limit to the
fourth decimal. The limit is exact; bar 1's window was set too narrow for its map count.
