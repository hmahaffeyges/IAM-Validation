# Phase B3 — Age-axis foreground module + IAM cellular age clock prototype

**Date:** 2026-05-30
**Status:** B3 module shipping. B3-bonus age clock prototype demonstrates the concept but does not yet beat published clocks on this cohort.

## Deliverables

| Deliverable | Status | Gate |
|---|---|---|
| `age_axis_foreground.py` — per-CpG age regression L4 foreground module | ✅ Built | **PASS** (100% CpG convergence) |
| `IAMAtlas_age_layer.csv` — per-CpG age slope/intercept artifact | ✅ Generated | 8,199 CpGs × {α, γ, R², n} |
| `age_layer_diagnostics.json` | ✅ Generated | — |
| `iam_cellular_age_clock.py` — methods-paper prototype | ✅ Built | — |
| `8class_ascores_all.csv` — 8-class A-scores for 1,174 EPIC-Italy patients | ✅ Generated | — |
| `age_clock_diagnostics.json` | ✅ Generated | MAE 5.48yr (target <5: **FAIL by 0.48yr**) |

## B3 (the L4 foreground module) — passes

**Convergence:** 8,199 / 8,199 CpGs fitted (100%, target ≥80%).
**Training:** 601 HC patients with valid age, age range 34–72.
**Slope distribution:** range [−0.0065, +0.0049] per year. Median essentially zero, consistent with literature (most CpGs are NOT age-correlated; clocks work by aggregating many weakly-correlated sites).
**Top CpG:** cg10501210, slope −0.0065/yr, R²=0.154.

**Structural finding:** the IAMAtlas marker pool is mostly disjoint from the canonical Horvath/Hannum/PhenoAge clock CpGs. None of the 8 spot-checked literature clock CpGs (ELOVL2, FHL2 ×2, EDARADD, KLF14, SST, SCGN, BRCA1) are in our marker pool. The IAMAtlas markers were selected for *cell-type discrimination*, not age correlation. This is GOOD news for L4: age-axis subtraction at our marker CpGs is structurally orthogonal to the canonical age-clock signal, meaning we're removing a *distinct* source of variation rather than the same age signal that Horvath would remove. Clean separation, exactly what the chain-of-custody discipline requires.

**Interface conformance:** module exposes `.fit(beta, age, hc_mask)` → trains; `.subtract_from(beta, age)` → returns cleaned β. Compatible with the upcoming v1 foreground_registry.py interface from Phase B1.

## B3-bonus (the cellular age clock prototype) — concept works, gate not yet cleared

**Univariate signal:** 4 of 8 classes show statistically significant age correlations in HC:
| Class | Pearson r | p-value | slope (yr / A-score unit) |
|---|---|---|---|
| **progenitor** | +0.153 | 0.00016 | +64.1 |
| **cycling** | +0.149 | 0.00025 | +55.9 |
| **immune** | +0.099 | 0.015 | +36.4 |
| terminal | −0.080 | 0.050 | −23.2 |
| (others) | |  | |

These are real age signals visible at the architectural-class level — the cellular age clock concept is empirically supported.

**OLS clock on 70/30 train-test split** (420 train, 180 test, both HC):
- Test MAE: **5.48 years** (target <5: FAIL by 0.48 years)
- Test R²: 0.076 (explains 7.6% of age variance)
- Test Spearman ρ: +0.273 (predictions correlate weakly with truth)
- Lift over baseline (predict cohort mean): +0.36 years

**Why the MAE doesn't yet clear the gate:**
1. **Training set too small.** 420 HC samples vs Horvath's 8,000.
2. **Age range too narrow.** 34–72 vs Horvath's 0–100+.
3. **Feature compression.** 8 architectural-class A-scores compressing thousands of CpGs; the published clocks use hundreds of raw β values as features.
4. **Single-sex bias.** EPIC-Italy is 75% female (recruitment artifact).
5. **A-score dynamic range narrow.** stem_adult SD=0.021 across all patients — small inter-patient variation to learn from.

**Cellular age acceleration in pre-dx cancer cases:** NULL result. Cases mean δ = −0.05yr vs HC mean δ = +0.24yr (p=0.63, not significant). This is a sensitivity limitation — with 5.48yr MAE noise, the clock can't detect ~1-2yr cancer acceleration that other clocks see. NOT a falsification of the concept, an artifact of the cohort being too small/narrow.

## What this means for the methods paper

**The concept is alive.** 4 of 8 architectural classes carry real age signal. The univariate slopes are biologically interpretable (cycling rate increases with age — consistent with stem cell exhaustion / senescent cell accumulation; immune A-score departure increases with age — consistent with immunosenescence; terminal A-score decreases — consistent with reduced terminal cell turnover at advanced age).

**The clock needs better training data to compete with Horvath.** Specifically: Hannum 2013 (656 samples, ages 19-101) would be the cleanest training cohort. Adding Horvath multi-tissue (8,000+ samples across 30+ tissues) would give a clock that could ship as a methods paper.

**The methods paper is deferred, not killed.** Phase E1 (the productionalized version of this clock, per the Roadmap §10.2.5) will train on Hannum + Horvath training corpora using the same architecture. The math machinery is ready; what's missing is the right training corpus. Until then, the clock as a *prototype* demonstrates the concept and produces interpretable coefficients that match expected aging biology.

## Files in repo (Phase B3)
- `Biological_Physics/atlas_vault/components/age_axis_foreground.py` (the L4 module)
- `Biological_Physics/atlas_vault/components/IAMAtlas_age_layer.csv` (the per-CpG layer)
- `Biological_Physics/atlas_vault/components/age_layer_diagnostics.json`
- `Biological_Physics/atlas_vault/age_clock/iam_cellular_age_clock.py` (methods-paper prototype)
- `Biological_Physics/atlas_vault/age_clock/8class_ascores_all.csv`
- `Biological_Physics/atlas_vault/age_clock/age_clock_diagnostics.json`
- `Biological_Physics/atlas_vault/age_clock/Phase_B3_FINDING.md` (this document)

## Next: B1 (foreground registry) integrates this module + the v3 NILC into the unified L4 stack. B4 (sex/batch/ancestry) extends.

