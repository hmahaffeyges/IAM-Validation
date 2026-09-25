# PROC-BAND-01 — outcome: NOT COMMISSIONED. The joint component fails reproducibility on one laboratory in four.

**Sealed 2026-09-25** against the bars fixed in [`PROC_BAND_01_PREREG.md`](PROC_BAND_01_PREREG.md) before any
array was read. 318 healthy whole-blood arrays, four laboratories, scored through the chain's own path.
Evidence: [`PROC_BAND_01_arrays.json`](../kit/results/PROC_BAND_01_arrays.json) (one record per array, with
the age it was scored at) and [`PROC_BAND_01.json`](../kit/results/PROC_BAND_01.json) (the bars).
Scripts: [`PROC_BAND_01_measure.py`](../kit/PROC_BAND_01_measure.py) and
[`PROC_BAND_01_analyse.py`](../kit/PROC_BAND_01_analyse.py).

## The verdict

| bar | result | |
|---|---|---|
| **B1** the component is present | joint fraction ≥ 0.01 on **98.4 %** of 318 arrays, median fraction 0.162 | **MET** |
| **B2** leave-one-laboratory-out coverage in [0.70, 0.90] | UCLA 0.823 · Munich 0.835 · Karolinska 0.763 · **Uppsala 0.646** | **NOT MET** |
| **B3** false alarm ≤ 0.10 per laboratory, ≤ 0.06 pooled | 0.013 · 0.051 · 0.079 · 0.089, pooled **0.0575** | MET |
| **B4** the axis adds something, \|r\| < 0.9 | **r = 0.534** against immune | MET |
| **B5** measured p95 vs chi-square | measured **2.509**, table 2.448, 2.5 % apart | measured, and they agree |
| **B6** the commissioned reading does not move | immune median A″ **1.0005** against the band's own µ = 1.000 | MET |
| **B7** nothing fitted on the arrays that judge it | curve and band from three laboratories per fold | held |

**Decision rule, applied:** *"It fails B2 or B3: it stays unbanded, the statistic stays `|z_immune|`, and the
failing number is published."* So the joint haematopoietic-progenitor component is **not commissioned**, the
departure statistic remains one axis, and its tier stays withheld. Nothing in the chain changes.

## What the failure is, measured

Uppsala's held-out coverage is 0.646 where 0.80 was nominal — its arrays fall outside a band built on the
other three more often than they should. Three measurements say why, and none of them is a property of the
component:

**The joint age curve is not monotone, and the immune one is.** Built here on 234–237 arrays per fold:

| decade | immune c (1,379 arrays) | joint c (this run) | arrays here |
|---|---|---|---|
| 10 | −0.0244 | −0.0464 | 10 |
| 20 | −0.0113 | −0.0100 | 15 |
| 30 | −0.0062 | +0.0066 | 37 |
| 40 | 0.0000 | 0.0000 | 56 |
| 50 | +0.0019 | +0.0117 | 68 |
| 60 | +0.0060 | +0.0003 | 61 |
| 70 | +0.0094 | **−0.0216** | 48 |
| 80 | +0.0121 | **−0.0222** | 21 |

The immune curve rises with age at every decade. The joint curve turns over after 50 and falls, which is what
a decade median does when it is measured on ten to twenty arrays.

**Uppsala is the only cohort that exercises the noisy ends.** Its arrays span 15 to 94 with donors in every
decade; the other three sit between 21 and 92 with their mass in the 30s to 70s. So Uppsala is the only
held-out laboratory scored substantially through the decades where the curve is worst — which is exactly the
fold that failed.

**And Uppsala is genuinely the widest cohort on this surface** (p10–p90 of 0.0491 against Munich's 0.0326),
so it would be the hardest fold even with a perfect curve.

## What this does and does not say

**It does not say the component is unmeasurable.** B1, B3 and B4 all passed, and B4 passed convincingly:
**r = 0.534** means the joint surface is *not* a copy of the immune axis. A second axis of that
independence is worth having — it is precisely the case the Mahalanobis design was chosen for, a specimen
that moves one component while the other holds. The two-axis distance was computed and behaved: measured p95
2.509 against a chi-square table value of 2.448, 2.5 % apart, so the standard threshold would have been
adequate. **All of that is real and none of it is adopted**, because the band underneath it does not
reproduce on a fourth laboratory and a threshold on an unreproducible band is a number that will surprise
someone.

**It does not say the immune reading is affected.** B6 is the check that matters for anything already in
service: immune A″ on this path has median 1.0005 against the band's own µ of 1.000, and the per-laboratory
tails (0.038–0.115 on these 78–80 array subsets) bracket the values the band file records on the full
cohorts (0.044–0.098). The commissioned reading has not moved.

## The named next step — which is a recommendation, not a decision

The plausible cause is a resource difference, not a biological one: **the immune curve was fitted on 1,379
arrays and this one on 234.** The obvious test is to derive the joint age term on the full four cohorts, as
PROC-PANEL-03 did for immune, and re-run this procedure unchanged.

That is **not done here, and must not be.** Re-running with a different curve after seeing which fold failed
is choosing the method to clear the bar, which is the one thing the pre-registration exists to prevent. If it
is to be done it is a new procedure with its own pre-registration, and its bars must be these bars.

One practical obstacle to state now: the full-cohort Stage 1 output does not exist. The Reference tab of every
report already says so — *"the 1,379-array Stage 1 output behind PROC-PANEL-03 was not preserved; 80 arrays
per laboratory are being re-run"* — and those 80-array panels are what this procedure used. Deriving the joint
curve on 1,379 arrays means calibrating roughly 1,400 IDAT pairs from raw, which is a day of compute and
about 15 GB of download, not an afternoon.

**A cheaper intermediate exists and is worth weighing first:** Uppsala alone has 659 arrays in GEO. Adding
that one cohort in full would take the curve's thin decades (10s: 10 arrays, 80s: 21) to something
defensible, at a fifth of the cost. Whether either is worth doing is the author's call.
