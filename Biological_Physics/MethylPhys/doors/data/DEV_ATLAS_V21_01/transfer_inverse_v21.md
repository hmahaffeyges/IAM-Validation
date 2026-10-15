# Atlas v2.1: converted readings (mode v21inv), rule committed 2026-10-15 BEFORE any test runs

**Why.** With the transfer inside the likelihood (mode v21) the model is correct (79 cells distinct; agrees with v2, r 0.9991) but slow:
4.6-7.5 s per locus, every draw at the maximum tree depth (diagnostics 2-5). v2's line in the same model runs 1.10 s per locus once the
read-count noise is taken from m (diagnostic 5). So the slowdown is the curve inside the likelihood.

**Rule (author chose this route; nothing fitted).** Each sequencing reading y is converted once, before the fit, to the array scale:
x = g^-1(y), g the smooth transfer (transfer_v21_smooth.json). Its noise is divided by the local slope k = g'(x) (first-order error
propagation): var = donor spread^2 (/3 pooled) + m(1 - m) / (coverage k^2) + n s(x)^2 / k^2 + floor. k is floored at 0.151, the smallest
non-flat segment slope of the adopted curve, so a reading on a plateau is down-weighted (at most 44 times) instead of given infinite noise.
That floor was chosen before any run, for that reason. Arrays are unchanged. Code: atlas_block_v21.py, mode "v21inv".

**Planted test (planted_inv_v21.py, laptop, run once).** The data generator of planted_strict_v21.py (truth at both extremes, cells seen
only by sequencing, reads drawn through the smooth curve). MET only if all hold: I1 coverage of the 95 percent intervals >= 0.90 overall and
>= 0.85 on the sequencing-only cells' extreme pairs; median |mu - truth| <= 0.03; I2 at most 1 percent of draws at the maximum tree depth.

**Real check (box, block 0's first 60 loci, run once).** MET only if all hold: R1 speed <= 1.6 s per locus (the cost sheet's upper value)
and at most 1 percent of draws at the maximum depth; R2 Flatness Lesson: no cell pair below mean |difference| 0.005; R3 agreement with the
curve-in-the-likelihood fit on the same loci (diagnostic 2 run 3, arm S): r >= 0.999 and median |difference| <= 0.005.
If both are MET, v21inv is the model the atlas is fitted with. If not, the author's choice stands open again (curve as built, about 66 h).
