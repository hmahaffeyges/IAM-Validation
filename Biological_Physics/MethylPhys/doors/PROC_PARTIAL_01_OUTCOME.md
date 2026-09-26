# PROC-PARTIAL-01 — outcome: NOT COMMISSIONED. A non-blood class's fidelity score cannot be recovered from whole blood.

**Sealed 2026-09-26** against the bars fixed in [`PROC_PARTIAL_01_PREREG.md`](PROC_PARTIAL_01_PREREG.md)
before any recovery was attempted. 2,862 mixtures estimated from the 5,088 PROC-FOREIGN-01 had already
scored — same hosts, same fractions, same fitted compositions, no new data.
Evidence: [`PROC_PARTIAL_01.json`](../kit/results/PROC_PARTIAL_01.json) ·
[`PROC_PARTIAL_01_landing.json`](../kit/results/PROC_PARTIAL_01_landing.json) ·
scripts [`PROC_PARTIAL_01.py`](../kit/PROC_PARTIAL_01.py) ·
[`PROC_PARTIAL_01_analyse.py`](../kit/PROC_PARTIAL_01_analyse.py).

**No fraction qualifies at any level, and the failure is not close.** B1, B2 and B3 fail at every fraction
for all three classes; B4 is untestable because no reporting floor exists to test; B5 holds.

## The answer, in the only units that matter

A methylation fraction is a number between 0 and 1. Here is where the recovered secretory profile actually
lands, against a true value of **0.732**:

| spiked fraction | median recovered mean β | range across hosts | |
|---|---|---|---|
| **0.02** | **12.37** | +4.36 to +28.66 | 17× outside the possible range |
| 0.05 | 3.16 | +1.96 to +21.47 | impossible |
| 0.10 | 1.48 | +1.21 to +1.96 | impossible |
| 0.20 | 1.013 | +0.91 to +1.17 | at the boundary |
| 0.35 | 0.856 | +0.81 to +0.92 | possible, still biased +0.12 |

At the fractions a blood draw actually presents — 1 to 5 % — the estimator does not return a noisy reading.
It returns a number that cannot be a methylation fraction at all. The A-score built on it is therefore not a
weak measurement to be improved with more samples; it is not a measurement.

## Why, exactly — and the positive control that makes this conclusive

**The estimator's arithmetic is exact.** On a synthetic host built from the atlas itself, it recovers
secretory's score to machine precision at every fraction — error 0.00e+00 at f = 0.02 through 0.35. So
nothing below is a coding defect; the method does what it claims when its assumptions hold.

**The assumption that fails is the composition model.** Against *real* blood, the atlas class means
reconstruct the observed betas at secretory's identity loci with a residual of **+0.067 in β** (median across
hosts, range +0.040 to +0.087). Real blood is not a non-negative mixture of eight atlas class means, and at
these particular loci the model under-predicts it systematically.

**Dividing by f amplifies that misfit, and the misfit wins:**

| f | misfit ÷ f | true signal | |
|---|---|---|---|
| 0.02 | 3.35 | 0.732 | **4.6× the signal** |
| 0.05 | 1.34 | 0.732 | **1.8× the signal** |
| 0.10 | 0.67 | 0.732 | 0.9× |
| 0.20 | 0.34 | 0.732 | 0.5× |

The pre-registration predicted 1/f amplification of *noise*, and expected √n averaging over tens of thousands
of loci to beat it down. That reasoning was wrong in one specific way, and it is worth naming: **the
dominant error is not noise but bias.** Averaging reduces noise by √n; it does not touch a systematic
residual at all. Thirty thousand loci average a 0.067 bias into a 0.067 bias.

## What the naive alternative shows

B3 asked whether the deconvolution earns its place against simply taking the entropy of the raw betas at the
class's loci. It does not — at every fraction, for every class:

| | median \|error\| with deconvolution | without |
|---|---|---|
| secretory @ 0.05 | 0.994 | **0.182** |
| secretory @ 0.20 | 0.994 | **0.148** |
| terminal @ 0.05 | 0.987 | **0.063** |

The naive score is closer to the truth at every point tested. It is still not a usable reading — an error of
0.15 against a band width of 0.021 is seven standard deviations — but it does say the partial-residual step
makes matters worse rather than better, which settles B3 without ambiguity.

## What this closes, and what it does not

**It closes the question as posed.** Scoring a non-blood class's fidelity from an ordinary blood draw is not
available, and the obstacle is not sample size, cohort count, or calibration effort. It is that the reference
model must reconstruct real blood at the identity loci to within roughly f × (the class's own signal) — about
**0.015 in β at f = 0.02**, against the **0.067** it currently achieves. That is a four-fold improvement in
reference accuracy, not a tuning parameter.

**It does not touch detection or quantification.** PROC-SMALL-01 stands: secretory presence is detectable in
unspiked whole blood to about 2 %, and its fraction is quantifiable above about 5 %. Those remain the honest
capabilities, and this procedure's failure is specifically about the *fidelity score*, not about the
composition.

**The one route that would change the answer** is a reference that predicts real blood at these loci far
better than eight class means do — a per-donor or per-cell-type reference rather than a class mean. That is
a different instrument, and whether it is worth building is the author's call, not a next step this procedure
recommends.
