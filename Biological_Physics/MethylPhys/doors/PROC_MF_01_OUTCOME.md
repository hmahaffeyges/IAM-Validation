# PROC-MF-01 — outcome: NOT COMMISSIONED as written. The full-covariance matched filter ties NNLS; the diagonal control arm detects at 2–10× lower fraction but does not yet estimate.

**Sealed 2026-09-26** against the bars in [`PROC_MF_01_PREREG.md`](PROC_MF_01_PREREG.md), fixed before any spike was
scored. 48 healthy arrays (12 per laboratory, four laboratories, mapped), 1,506 markers present on every array, 768
spikes into real arrays (4 cells × 4 fractions × 48), 160 into constructed blood. Covariance leave-one-laboratory-out,
Ledoit–Wolf shrinkage 0.44–0.57.
Evidence: [`PROC_MF_01.json`](../kit/results/PROC_MF_01.json) · [`PROC_MF_01_null.json`](../kit/results/PROC_MF_01_null.json)
· [`PROC_MF_01.py`](../kit/PROC_MF_01.py) · [`PROC_MF_01.png`](../plates/PROC_MF_01.png)

## Detection limit — smallest fraction detected in ≥ 90 % of real-array spikes at ≤ 1 false positive in 48

| cell | NNLS (chain) | matched filter, full N | **diagonal N (control)** | same-laboratory N (control) |
|---|---|---|---|---|
| Breast | 5 % | 5 % | **1 %** | 0.5 % |
| Colon epithelial | 2 % | 2 % | **0.5 %** | 0.5 % |
| Cortical neurons | 5 % | 5 % | **2 %** | 0.5 % |
| Prostate | 5 % | 5 % | **2 %** | 0.5 % |

## Bars

| bar | result |
|---|---|
| B1 full filter lower than NNLS on ≥ 3 of 4 | **FAILED** — 0 of 4; four ties |
| B2 honest σ: null \|z\| > 2 in 2–10 % | **FAILED** — 94 %. The σ the filter reports is a liar by an order of magnitude |
| B3 unbiased at 2 % and 5 % | **FAILED** — median f̂ − f = −0.021 and −0.025 |
| B4 structured N beats diagonal | **NOT ASSESSABLE** — conditional on B1 |
| B5 same-lab N ≤ 20 % better than leave-one-out | **FAILED** — same-lab reaches 0.5 % on every cell: the covariance memorises its own arrays |
| B6 blood composition unchanged | MET — median Δ 0.0001 |

## What the numbers say, in order

1. **The full covariance cannot be estimated from 36 arrays.** 1,506 markers, 36 noise realisations per fold: the
   estimator is under-determined by a factor of forty, shrinkage sits at one half, and the off-diagonal structure
   that survives is noise that misdirects the filter (B1 tie) and understates its own uncertainty (B2). The same-lab
   arm shows what memorising looks like: 0.5 % everywhere. Cosmology estimates its noise covariance from thousands
   of simulations or a parametric model; we have 48 arrays. **That, not the physics, is why the elegant borrowing
   failed here** — the author's caution of this morning, measured.
2. **Per-locus inverse-variance weighting works, and was in the control arm.** Diagonal N — weight each locus by
   1/variance of the healthy residual there, no off-diagonals, leave-one-laboratory-out — lowers the detection limit
   on all four cells: Breast 5 % → 1 %, colon 2 % → 0.5 %, neurons and prostate 5 % → 2 %. Thirty-six arrays are
   plenty to estimate 1,506 variances. This is the finding, but it is a **control-arm observation, not a
   pre-registered claim**, and it is recorded as such.
3. **The diagonal arm detects but does not estimate.** Read post hoc and labelled so: its healthy-blood readings
   are *negative* (thresholds −0.017 to −0.036) and it under-reads every spike by a constant ≈ 0.03 at both 2 %
   and 5 %. A constant offset independent of fraction is the structured misfit (PROC-COV-01's 0.067) projecting
   onto the template. Detection works because the shift above the null is real; the amount is wrong by a fixed
   amount that the null itself measures.

## Decision, by the pre-registered rule

B1 failed → the matched filter as written is **not adopted**, and the pre-registration's own clause applies:
*"B4 failing → adopt diagonal weighting and record that the structure bought nothing."* B4 could not be assessed
because B1 failed first, so the diagonal arm is **not adopted by this procedure either** — its honesty (σ), bias and
same-lab controls were not bars here, and a detector adopted on a control arm's numbers is a threshold moved after
the results were visible.

**PROC-MF-02 is the follow-up, and it writes itself:** inverse-variance weighted detection, leave-one-laboratory-out,
with (a) the null median subtracted so the estimator is centred, (b) σ from the null's spread rather than from the
weights, (c) the same six bars, all applying to *this* detector. If it meets them it becomes the detection stage
ahead of the per-cell A. Nothing in the chain changes on the strength of MF-01.

## Where this leaves the group that gave up "because of the noise"

Partly vindicated, partly not. The noise that defeats equal-weight component separation at 1–2 % is real and
reproducible. It is also **beatable by the plainest weighting there is** — variance per locus — which lowers the
floor two- to ten-fold on this atlas and platform. The sophisticated tool did not help; the ordinary one did.
