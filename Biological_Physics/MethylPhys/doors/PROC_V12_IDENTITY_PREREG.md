# PROC-V12-IDENTITY — pre-registration: identity loci built on atlas v2 alone, and the held-out self-read

**Written 2026-09-29, before any v2 identity set is built or any sample is read on one.** Author, 2026-09-29: "I would really prefer
that the new atlas not be dependent on anything from version 1 if its at all possible to have its own better way … all the lessons
we learned from version 1 applied here in version 2." So nothing below reads a v1 file: not the v1.1 loci, not v1 cell names, not the
v1 atlas. What carries over is physics (H_min per class, G-002, frozen) and the selection RULE, because it is physics: an identity
locus is one where the cell sits at its own floor.

## The rule (fixed now)
For each v2 cell c with class k (roster `class_by_draft_rule`; the class is used only to pick H_min):
1. **Target.** b*(k) is the upper-branch beta with H(b*) = H_min(k) (binary entropy, bits). Upper branch only: A uses H(mean β),
   and pooling the two branches puts the mean near 0.5 and inflates A (recorded in v1.1; a physics fact, not a v1 dependency).
2. **Candidates.** Loci where c was MEASURED (n_obs ≥ 2), converged (R̂ < 1.01), and — the v2 tightening — the **whole 95 %
   posterior interval** of μ_c lies inside b*(k) ± 0.05. v1 used the point mean alone; a locus that sits in the window only by noise
   is now excluded.
3. **Size.** At least 100 loci (at n = 100 the sampling error of the mean β moves A by ~0.005). A cell with fewer has **no identity
   set** and is reported NOT READABLE, never scored on a borrowed set.
4. A = H(mean β over the cell's loci) / H_min(k).

## The test (the V12 bar: a cell reading 1.00 on the samples it was fitted on proves nothing)
**Cross-fitting on samples.** Each cell's purified samples are split in two halves with `default_rng(12)` (a cell with 2 samples:
one each). Loci are selected by the rule above from half 1's own sample mean (each sample put on the atlas scale by its source term:
array y − d, sequencing (y − a)/b), with "whole 95 % interval inside the window" replaced by "every half-1 sample inside the window".
Each half-2 sample is then read individually: A = H(mean β over those loci)/H_min. Then swap halves. Every sample is read once, on
loci it did not choose.

- **B1:** ≥ 95 % of all held-out sample readings fall in NORMAL (tier_breakpoints.json v1.5).
- **B2:** every cell's median held-out reading is in NORMAL. A cell that fails is named with its source and platform.
- **B3:** every admitted cell builds a production set (≥ 100 loci) from the v2 posterior.

Reported, not bars: loci per cell; the self-read on the fitted mean (circular, printed for comparison only); overlap between cells'
sets; the atlas interval on each cell's self-read from the 20 posterior draws (V8).

## What is produced
`iamatlas_v2_identity_loci_v1_0.json` — per cell: class, H_min, b*, loci, n_loci, self-read, held-out median — with this file as its
provenance. It is not wired into the chain until the switch-over (V16).
