# PROC-SYNTH-01 — outcome: the chain recovers what it is handed. But the per-cell A is confounded with the cell's FRACTION.

**Run 2026-09-26** on the author's instruction to verify the chain against synthetic patients before anything
further is built on it, and to run every future test **through the chain** rather than through its internals.
Every specimen below entered via `run_sample.py --betas`, the same entry point a real specimen uses.
Evidence: [`PROC_SYNTH_01.json`](../kit/results/PROC_SYNTH_01.json) ·
[`PROC_SYNTH_01_cells.json`](../kit/results/PROC_SYNTH_01_cells.json) ·
[`PROC_SYNTH_01_fraction.json`](../kit/results/PROC_SYNTH_01_fraction.json).

Truth was **constructed, not estimated**, so every failure below is unambiguous.

## What passed, and it is the architecture the author specified

**Composition recovery is exact.** Four mixtures with known weights, read back with **0.0000 error** on every
component, and no spurious class above 2 %:

| mixture | true | recovered |
|---|---|---|
| immune 0.90 / progenitor 0.07 / stem_adult 0.03 | — | identical |
| immune 0.70 / secretory 0.20 / stromal 0.10 | — | identical |
| immune 0.50 / terminal 0.30 / cycling 0.20 | — | identical |
| secretory 0.60 / cycling 0.25 / stromal 0.15 | — | identical |

**Per-cell scoring is correct to 1e-6.** Hand the chain a pure cell type; it names it, reads its fraction as
1.000, and its A matches `mean_i H(β_i) / H_min[class(cell)]` computed independently — **8 of 9 cells, zero
discrepancies above 1e-6**:

| cell handed in | named | f read | A read | A expected | class whose floor was used |
|---|---|---|---|---|---|
| CD4_T-cells | yes | 1.000 | 0.7447 | 0.7447 | immune |
| Neutrophils_reinius | yes | 1.000 | 0.6366 | 0.6366 | immune |
| HSC | yes | 1.000 | 0.5976 | 0.5976 | stem_adult |
| GMP | yes | 1.000 | 0.5663 | 0.5663 | progenitor |
| **Breast** | yes | 1.000 | 1.0638 | 1.0638 | secretory |
| **Colon_epithelial_cells** | yes | 1.000 | 0.3654 | 0.3654 | cycling |
| fibroblast | yes | 1.000 | 0.2875 | 0.2875 | stromal |
| Cortical_neurons | yes | 1.000 | 0.0099 | 0.0099 | terminal |
| **stem_pluri** | **no — called Cortical_neurons** | **0.000** | 0.3850 | 0.3850 | stem_pluri |

So the deconvolve-then-score-against-the-class-floor path does exactly what it claims, including for the
organ cell types. `Breast` and `Colon_epithelial_cells` are both identified and scored when they dominate a
specimen — which is the regime the author's stool and urine proposal creates.

**One identification failure:** `stem_pluri` is called `Cortical_neurons`. That is the r = +0.817 pairing
from [`ATLAS_READABILITY.md`](ATLAS_READABILITY.md) appearing as a concrete misidentification rather than a
correlation.

## The finding that matters more: A moves with FRACTION, not only with fidelity

The same CD4 profile, byte-identical in every specimen, mixed with a filler at different proportions.
**Fractions were recovered exactly at every level**, so the fit is not at fault:

| true f | f read | **A read** |
|---|---|---|
| 1.00 | 1.000 | **0.7447** |
| 0.80 | 0.800 | 1.0374 |
| 0.50 | 0.500 | **1.1784** |
| 0.30 | 0.300 | 1.1028 |
| 0.20 | 0.200 | 1.0107 |
| 0.10 | 0.100 | 0.8760 |
| 0.05 | 0.050 | 0.7892 |

**Spread 0.4337 — which is 8.3 × the entire immune healthy band width of 0.0524.** Confirmed with a second
filler cell (spread 0.5492, same shape).

The mechanism is not subtle. A cell's discriminative markers are near-deterministic *in that cell*, so a pure
specimen has low entropy there; diluting it moves those addresses toward the filler's values and entropy
peaks near a half-and-half mixture. **A is therefore reading the specimen at the cell's addresses, not the
cell.** It is the same root cause as the absence artefact PROC-CEIL-01 documented, and the same one that
closed PROC-PARTIAL-01.

## Why immune-in-whole-blood works, and what that implies

Immune fraction across 845 real blood arrays: p5 **0.7902**, median 0.9003, p95 **0.9648** — a span of only
**0.1746**. The substrate nearly holds the confound fixed, which is why this is the one commissioned reading.

But "nearly" is not "entirely", and the shortfall is measurable on the commissioned gauge itself:

- **corr(immune fraction, reported A′) = +0.440**, p = 8.05e-26 across 516 held-out arrays — **fraction explains 19.3 % of the variance in the reported reading**
- the natural p5–p95 fraction span moves A′ by **0.0435**, against a band width of **0.0524**

**So the commissioned band is comparable in size to the movement composition alone produces.** That does not
make the gauge wrong, and it does not invalidate a reading — but placement inside a 0.0524 band is partly a
statement about composition, and that has not been stated anywhere until now.

## Consequence for the healthy bands that were about to be measured

A per-cell healthy band **cannot be a single interval**. It has to be conditioned on the cell's fraction, or
computed on a fraction-adjusted residual — otherwise a patient with ordinary cells in unusual proportions
reads as a fidelity departure. This is the single most important thing to fix before per-cell bands are
built, and it was found before they were built, which is what this procedure was for.

## What this does to PROC-EPIC-01's colorectal result

Re-examined because the author doubted it. The composition trend was fitted **on controls only**, so the
adjustment cannot absorb the case effect, and the permutation null was re-run on the adjusted values so an
adjusted effect is not compared against an unadjusted null:

| | Cohen's d | permutation p (5,000 label shuffles) |
|---|---|---|
| as sealed | **+0.6040** | 0.0008 |
| composition-adjusted | **+0.4759** | **0.0062** |

Colorectal cases do have a genuinely higher immune fraction than controls (0.9070 vs 0.8925, d = +0.438,
p = 0.019), so **21 % of the sealed effect was composition.** The remaining effect still clears its own
matched null. **The colorectal finding survives, at a smaller effect size than sealed, and the sealed
document has been qualified rather than left standing alone.**
