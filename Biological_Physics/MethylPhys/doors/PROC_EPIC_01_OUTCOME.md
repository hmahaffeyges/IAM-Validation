# PROC-EPIC-01 — outcome: the colorectal signal replicates in held-out blood. The breast signal does not.

**Sealed 2026-09-26** against the bars fixed in [`PROC_EPIC_01_PREREG.md`](PROC_EPIC_01_PREREG.md) before any
array was scored. 845 EPIC-Italy arrays scored through the commissioned chain. 516 samples are held out; **313 of them
enter the analysis** — 78 breast, 72 colorectal and 163 female controls. The remaining 203 are the 84
male controls excluded by design (every case is female) and 119 held-out cases of other cancer types,
which no bar tests.
Evidence: [`PROC_EPIC_01.json`](../kit/results/PROC_EPIC_01.json) ·
[`PROC_EPIC_01_scored.json`](../kit/results/PROC_EPIC_01_scored.json) (every array) ·
scripts [`PROC_EPIC_01_score.py`](../kit/PROC_EPIC_01_score.py) ·
[`PROC_EPIC_01_analyse.py`](../kit/PROC_EPIC_01_analyse.py).

**The class read is the immune identity gauge throughout** — the same class as the pre-atlas work. No
non-blood class was scored; on whole blood the chain refuses, and PROC-PARTIAL-01 is the procedure that asks
whether that refusal can be lifted.

## The verdict

| bar | result | |
|---|---|---|
| **B1** breast > 10 y elevated (n = 22) | **d = −0.324, p = 0.93** — the direction *reverses* | **FAILED** |
| **B2** temporal ordering d(>8 y) > d(0–2 y) | +0.046 | met, but **vacuous** — see below |
| **B3** colorectal > 5 y elevated (n = 40) | **d = +0.604, p = 0.0004** | **MET** |
| **B4** healthy controls split at random | median \|d\| = 0.109, p95 = 0.318 | MET (bar < 0.20) |
| **B5** guard not confounded with disease | cases 6/150 (4.0 %), controls 6/163 (3.7 %), ratio **1.09** | MET |
| **B6** instrument unchanged | max \|ΔA\| = **0.000e+00** on 318 published arrays | MET, **substituted source** |

**Decision rule, applied** — the pre-registration's "B3 met but B1 not" branch: *the cross-cancer arm stands
alone and the breast claim is withdrawn pending more cases.*

## What replicates: colorectal, and not where the old analysis said

| lead time | n | d vs controls | |
|---|---|---|---|
| 0–2 y | 7 | +0.25 | inside the noise band |
| **2–5 y** | 25 | **+0.73** | |
| **5–8 y** | 28 | **+0.72** | |
| > 8 y | 12 | +0.33 | inside the noise band |

The signal is **loudest 2 to 8 years before diagnosis** and falls back inside the band beyond 8 years. The
pre-atlas analysis reported the opposite shape — loudest past 10 years. Two things must be said about that
disagreement rather than one: the far-out stratum here holds **12 arrays**, so "quiet" there means *not
demonstrated*, not *shown absent*; and the two analyses read different surfaces, so a pre-atlas elevation and
an identity-gauge elevation are not automatically the same physical statement.

For scale: 0.318 is the 95th percentile of \|d\| when the healthy controls are split at random, so +0.73 is
roughly **twice the largest effect chance produces** in this cohort, while +0.33 is not distinguishable from
it. With three comparisons, B3's p = 0.0004 survives a Bonferroni correction (0.0012).

## What does not replicate: breast

At every lead time the held-out breast cases read **below** controls (−0.19 to −0.46), and at the
pre-registered stratum d = −0.324 with p = 0.93 against the elevation the pre-registration fixed. The
magnitude is inside the random-split band (p95 = 0.318), so this is best read as **no signal**, not as a
depression to be explained.

The likely reason is in the split rather than the biology: the pre-atlas "replication" re-used **146 of its
224 breast cases** from its own discovery set, along with 177 of 424 controls. When the 78 genuinely
held-out cases are analysed alone, nothing survives.

## Two failures of my own procedure, recorded rather than smoothed over

**B2 passed vacuously and should not have been written that way.** It tested the *ordering* of two effect
sizes, not their magnitude — so with every stratum negative, d(>8 y) − d(0–2 y) = +0.046 satisfies it while
meaning nothing. A bar that a null result can pass is not a bar. The correct form would have required the
far stratum to clear the noise band in the pre-specified direction before any ordering was tested.

**B6 could not be run as written.** The pre-registration named PROC-E2E-01's sealed file as the source of the
commissioning readings; that file records exit codes, timings, age and series, and **no per-array readings**.
Naming a source without checking it holds the quantity was an error made while writing the bar. What is
reported is the equivalent check against the readings PROC-BAND-01 *did* publish — 318 arrays, four
laboratories, max \|ΔA_mapped\| exactly zero — and it is labelled a substitution, not the bar as fixed.

## What this does and does not license

It licenses one sentence: **on data the instrument had never seen, in a split with no overlap, the immune
identity gauge separates pre-diagnostic colorectal cases from controls 2–8 years before diagnosis, at roughly
twice the effect size chance produces.**

It does not license a clinical claim. One cohort, one platform, 40 cases at the tested stratum, no
prospective validation, and an effect size that is a group separation rather than a per-patient
discrimination — d = 0.6 means the distributions overlap heavily, and nothing here says any individual
reading is actionable. Disease evidence for the commissioned chain remains Issue 004, after sealed runs.

**The obvious next step is not more analysis of this cohort.** It is a second colorectal cohort with long
lead times, which would settle the > 8 year question that 12 arrays cannot.
