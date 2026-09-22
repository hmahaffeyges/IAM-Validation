# CPG-VAL-015 — OUTCOME

**Execution date:** 2026-06-07
**Outcome:** **PASS** (all 3 hard pre-registered conditions met)
**Status:** SEALED

## Headline finding

The immune class A-score declines significantly with chronological age in the Hannum cohort, surviving sex stratification and 50/50 cross-validation. The decline is class-specific: A_immune (r=-0.197) and A_stem_pluri (r=-0.184) dominate the aging signal; cycling/stromal/secretory classes show negligible age dependence.

## Pre-registered pass conditions (declared in PREREG.md BEFORE execution)

| Condition | Criterion | Observed | Passed |
|---|---|---|---|
| 1 | A_immune Pearson r < -0.10 AND p < 0.001 | r=**-0.197**, p=**3.69e-07** | ✓ |
| 2 | Same sign + significant in BOTH sex strata | F: r=-0.199 p=2.30e-04; M: r=-0.210 p=1.68e-04 | ✓ |
| 3 | 50/50 split CV: \|Δr\| < 0.05 | \|Δr\| = **0.043** (half_A r=-0.178; half_B r=-0.220) | ✓ |

## Per-class specificity (NOT in pass conditions — diagnostic)

| Class | Pearson r | p | Slope (×10⁻⁴/yr) | Interpretation |
|---|---|---|---|---|
| immune | **-0.197** | 3.69e-07 | **-5.24** | dominant aging signal |
| stem_pluri | -0.184 | 2.02e-06 | -2.89 | secondary aging signal |
| stem_adult | -0.103 | 8.16e-03 | -1.70 | weak aging signal |
| terminal | -0.089 | 2.26e-02 | -1.90 | weak aging signal |
| stromal | -0.068 | 8.36e-02 | -1.48 | not significant |
| progenitor | -0.062 | 1.12e-01 | -1.03 | not significant |
| cycling | -0.016 | 6.84e-01 | -0.30 | flat |
| secretory | +0.008 | 8.45e-01 | +0.15 | flat |

Immune and stem_pluri are an order of magnitude stronger than secretory/cycling. This is class-specific aging, not a global drift.

## Per-decade trajectory

| Decade | n | Median A_immune | ΔA per decade |
|---|---|---|---|
| 10s | 1 | 0.8265 | — |
| 20s | 11 | 0.7999 | -0.027 |
| 30s | 23 | 0.7959 | -0.004 |
| 40s | 74 | 0.7961 | +0.000 |
| 50s | 138 | 0.7921 | -0.004 |
| 60s | 167 | 0.7842 | -0.008 |
| 70s | 142 | 0.7815 | -0.003 |
| 80s | 87 | 0.7664 | **-0.015** |
| 90s | 12 | 0.7693 | +0.003 |
| 100s | 1 | 0.7886 | (n=1) |

Decade-median monotonicity: Spearman ρ = **-0.854**, p = 0.0016. The trend accelerates after age 70 (visible in the 70→80 transition).

## What this means for the immune card

CPG-VAL-015 establishes that the immune class A-score is a quantitative **aging trajectory marker** — not just a "case vs. HC" signal. The slope is small in absolute terms (~5×10⁻⁴ per year) but is highly significant across n=656 samples, survives sex stratification with essentially identical slopes (F=-5.34×10⁻⁴; M=-5.42×10⁻⁴), and survives random 50/50 cross-validation.

This is **cross-cohort by construction**: the markers come from the foundation IAMAtlas, not from Hannum. Hannum's own 71-CpG clock would give r≈0.99 (regression-trained on the same data); our physics-layer signal is r=-0.20, which is the honest cross-cohort signal.

## Connection to VAL-020

VAL-020 (CPG-VAL-020 Hannum anchor reproduction) surfaced this same physics-layer signal as a side-finding when its primary cellular-age inversion saturated. VAL-015 converts it into the primary readout in A-score space: the architectural information loss IS the aging trajectory. Both VALs use the same Hannum 115-cell A-score CSV (sha256 in cohort_manifest.json); the difference is the framing — VAL-020 surfaced the boundary, VAL-015 ratifies the trajectory.

## Cards-don't-carry-runtime-data discipline

This VAL's results live in `results.json`, `stratified_results.json`, etc. — NOT in the immune card JSON. The immune card's `validation_evidence_v1_0_set["CPG-VAL-015"]` block updates to reference SEALED status + headline findings only; the runtime numbers stay here.
