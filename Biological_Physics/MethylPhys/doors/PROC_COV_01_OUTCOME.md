# PROC-COV-01 — outcome: the misfit IS a reproducible, removable bias. Removing it does not rescue fidelity recovery.

**Sealed 2026-09-26** against the bars in [`PROC_COV_01_PREREG.md`](PROC_COV_01_PREREG.md), fixed before any
residual was computed. 318 healthy whole-blood arrays, four laboratories, 149,982 identity loci, everything
estimated leave-one-laboratory-out. Evidence: [`PROC_COV_01.json`](../kit/results/PROC_COV_01.json) ·
script [`PROC_COV_01.py`](../kit/PROC_COV_01.py).

| bar | result | |
|---|---|---|
| **B1** the bias is reproducible across laboratories | reduction **85.8 / 77.2 / 84.6 / 86.5 %** in the four held-out folds | **MET** |
| **B2** it is a bias, not noise | bias vectors from disjoint laboratory pairs: **r = +0.985** | **MET** |
| **B3** it fixes what it was meant to fix | at f = 0.20 the recovered mean β moves 1.003 → **0.659**, but \|A error\| = 0.104 against a 0.042 bar | **FAILED** |
| **B4** the covariance adds beyond the mean | rank-10 factor model explains **23.5 %** of held-out residual variance | MET |
| **B5** the instrument does not move | immune A unchanged, max \|Δ\| = 0 on 318 published arrays | MET |

## What was measured

The atlas under-predicts real blood by a **median +0.0889 β** across identity loci — the same phenomenon
PROC-PARTIAL-01 measured as +0.067 on secretory's loci alone, with the sign fixed in advance and confirmed.
Subtracting a bias estimated **without** the held-out laboratory removes **77–87 %** of it, and the bias
vectors from disjoint laboratory pairs agree at r = 0.985. **This is an instrument constant, not specimen
noise.**

## The ordinary check, run first

- **13,015 loci (10.6 %) have a median residual exceeding 0.3 in β.** These are candidate bad addresses and they were not known before this run. The 1st and 99th percentiles are −0.550 and +0.537.
- **Mean vs median mattered less than feared here** — median gap 0.0034, 99th percentile 0.0257 — but the check was run before any modelling, and the 99th percentile is where a mean would have misled.

## Why B3 still failed, and what it changes

The correction is large and real, and it is not enough:

| f | uncorrected mean β | corrected | true |
|---|---|---|---|
| 0.05 | 3.093 | **0.023** | 0.732 |
| 0.10 | 1.447 | 0.529 | 0.732 |
| 0.20 | 1.003 | **0.659** | 0.732 |
| 0.35 | 0.850 | **0.697** | 0.732 |

Every corrected value now lands **inside the physically possible range**, which the uncorrected ones did not
below f = 0.20. But the residual error is still 2–5× the tolerance, and at f = 0.05 the correction
overshoots into near-zero. The pre-registered decision rule applies: *"B1 met, B3 not: the bias is real and
removable but too small to rescue fidelity recovery at low fractions. The correction still belongs in the
chain as a reconstruction improvement; the PROC-PARTIAL-01 closure stands."*

**The honest reading:** a fixed offset is the first-order term and it is now measured, but what remains is
specimen-dependent structure — consistent with B4, where ten factors are needed to reach 23.5 % and one
reaches only 5.6 %.

## What is licensed

A **reconstruction correction** — measurable per laboratory, removing ~85 % of the reference's systematic
misfit — is available and should be commissioned by a separate procedure that tests it where it is used.
Nothing in the chain changes until then. Fidelity recovery from whole blood remains closed.
