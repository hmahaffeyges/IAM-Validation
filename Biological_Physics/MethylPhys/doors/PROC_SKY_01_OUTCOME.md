# PROC-SKY-01 — outcome: SKY WITHHELD. The panel scales are retired; no on-array spread fitted every laboratory.

**Sealed 2026-09-27** against [`PROC_SKY_01_PREREG.md`](PROC_SKY_01_PREREG.md), committed (6ffb045) before any array was read.
Script `kit/PROC_SKY_01.py`; results `kit/results/PROC_SKY_01.json`; 48 arrays, 12 per commissioning laboratory.

| bar | rule | measured | |
|---|---|---|---|
| B1 | constructed atlas specimen quiet | median z +0.0000, \|z\|>2 on 0.00 % | MET |
| B2 | robust SD of z in [0.7, 1.4] on ≥ 40 of 48 | **36 of 48** (GSE111629 12/12, GSE87571 11/12, GSE42861 10/12, GSE125105 3/12); median 0.925 | **FAILED** |
| B3 | no laboratory file read | none | MET |
| B4 | same array twice: Δz = 0; composition drops out of a difference | max Δ 0; median \|Δz\| 0.0000 | MET |
| B5 | same nine panels, same gating | rendered with the chain's plate | MET |

## What failed, exactly

The on-array noise term σ_array²(β) = a + b·β(1 − β) from the 65 SNP probes puts z on the right scale for three laboratories
(robust SD 0.92–1.16) and **over-predicts the scatter on GSE125105 by ~40 %** (robust SD 0.60; its SNP clusters are about
twice as wide as its cg probes' scatter: σ_array(½) 0.10 vs 0.05–0.07 elsewhere). SNP-probe spread is not the cg-probe
spread on every platform/pipeline. One σ model must fit every commissioned laboratory or it is not the instrument's.

## Diagnostic written after the bars (not a bar): the offset the sky now shows

With m = 0 the sky reads a median z of −0.4 to −0.8 on every array. Residual (β − Σ f_c μ_c) by expectation bin, one array
per laboratory:

| laboratory | median residual, all loci | on the identity loci | off them | slope vs (E − ½) |
|---|---|---|---|---|
| GSE111629 | -0.0231 | +0.0371 | -0.0271 | +0.031 |
| GSE125105 | -0.0404 | -0.0016 | -0.0427 | -0.020 |
| GSE42861 | -0.0384 | -0.0051 | -0.0405 | +0.034 |
| GSE87571 | -0.0273 | +0.0098 | -0.0298 | +0.038 |

- **Not compression.** If the array squeezed β toward ½ the residual would be positive at low β and negative at high β
  (slope ≈ −0.06). It is negative at *both* ends (≈ −0.045) and near zero in the middle; the slope has the wrong sign.
  The saturation hypothesis noted after PROC-TARE-01 is therefore not what this is.
- **The pipeline map's fitting range.** On the identity loci — where A is read and where the map was fitted — the median
  residual is ≈ 0. Off them the array sits 0.03–0.04 below the composition expectation. The offset is locus-dependent, not
  β-dependent: the map does not transfer beyond the loci it was fitted on, and/or the atlas carries a source offset at those
  loci. Which of the two was not measured here. **A is unaffected**: it is read on the loci where the map centres.

## What this decides

The four `residual_scale_<lab>.npz` (40-array healthy panels) move to RETIRED: population layers, as pre-registered. The sky
is **withheld** — the Sky tab prints one sentence and no picture — until a σ that is the instrument's fits every laboratory.
Routes recorded, not started: (i) σ_array from the array's control probes rather than the SNP probes; (ii) a pipeline map
fitted on all loci, or per locus stratum, so the zero holds off the identity loci too; (iii) atlas v2's per-source term, where
a locus-dependent atlas offset would be estimated (PLAN 27).
