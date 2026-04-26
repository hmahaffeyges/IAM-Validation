# VAL-096 — TTD-Window Stratification on Loyfer Stage 2 Per-Tile A-Scores (Breast Pre-Diagnostic)

**Status:** PRE-REG (sealed before β-access)
**Card:** breast-epic v0.2 → v0.3 candidate
**Card class:** secretory (primary), cycling and immune (secondary)
**RNG seed:** 20260426
**Date sealed:** 2026-04-26

---

## Background and motivation

VAL-093 ran the Loyfer/Moss 25-cell-type Stage 2 atlas on the >10yr breast pre-dx subset and found `O2_SECRETORY_DISTRIBUTED`: breast tile null (d=+0.20 GSE51057, d=+0.10 GSE51032), pancreatic and cycling tiles concordantly elevated (d≈+0.7-1.0).

VAL-047 Phase 9 (Heath context) reported that at GSE51057, **Stage 1 immune signal is strongest at >10yr (d=+1.78) and weakest at 0-2yr (+0.09 to +0.27)**. This is the inversion of what classical late-stage cancer-progression models predict. We don't have window-stratified per-tile Stage 2 results from VAL-093 itself — VAL-093 ran only the >10yr subset.

**Hypothesis under test:** does the per-tile Stage 2 pattern shift across TTD windows the same way Stage 1 immune does?
- 0-2yr (close to clinical diagnosis)
- 2-5yr
- 5-10yr
- >10yr

For each window, score the same 25 Loyfer tiles. The output is a **window × tile heatmap** showing where signal localizes as time-to-diagnosis shrinks.

This is a **temporal pattern test**, not an outcome test. All data already exist in `/home/claude/run_everything/VAL-093_per_sample.csv` (the per-sample CSV preserves TTD years per sample).

## Cohorts

- **GSE51057** breast pre-dx cases by TTD window (per VAL-093 per-sample CSV):
  - 0-2yr: n=58
  - 2-5yr: n=34
  - 5-10yr: n=43
  - \>10yr: n=11
  - Controls (cancer-free): n≈170 (from VAL-093 group=`control`)
- **GSE51032** breast pre-dx cases by TTD window — to be tabulated from same per-sample CSV

Specimen: buffy-coat whole blood, both cohorts.

## Atlas and method

- **Stage 2 atlas:** Loyfer/Moss 25 cell types, 7,890 array CpGs (PRODUCTION, vault `stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv`)
- **Method:** identical to VAL-093 — per-tile A-score using top-100 specificity CpGs against architecture-class H_min anchor; case vs cancer-free-control Cohen's d per window per cohort
- **Note:** Per-sample A-scores ALREADY computed in VAL-093_per_sample.csv. This VAL re-slices that table by TTD window. No new β-extraction or scoring required. Pure re-analysis.

## Pre-locked decision criteria

**Primary outcome — does the Stage 2 per-tile pattern shift across TTD windows?**

| Outcome | Condition |
|---|---|
| **O1_BREAST_TILE_FIRES_NEAR_DIAGNOSIS** | Breast tile d goes from |d|<0.3 at >10yr to |d|≥0.5 at 0-2yr in BOTH cohorts (signal localizes to breast as diagnosis approaches) |
| **O2_DISTRIBUTED_PERSISTS** | Pancreatic + cycling tile elevation present at all 4 windows; breast tile remains null at all 4 windows (the >10yr pattern is the steady-state pattern, not a window-specific artifact) |
| **O3_PATTERN_INVERTS_AT_NEAR_DX** | Tiles that fire at >10yr (pancreatic, cycling kidney/colon) attenuate at 0-2yr while breast tile rises (classic late-progression localization) |
| **O4_PATTERN_AMPLIFIES_AT_NEAR_DX** | All elevated tiles at >10yr (pancreatic, cycling) show STRONGER d at 0-2yr (more signal closer to diagnosis), but breast tile remains null |
| **O5_DISCORDANT_BETWEEN_COHORTS** | GSE51057 and GSE51032 disagree on the temporal pattern (a cohort-specific finding, not a framework finding) |
| **O6_AD_INSTANCE_TEMPORAL** | Pooled tile signal is null at all windows BUT directional sub-patterns appear at specific windows (rare; flag as suspected, not confirmed without Test 2) |

## Sample-size honesty

The >10yr GSE51057 case window is n=11 — small. d-values from this window have wide CIs. We compute and report 95% bootstrap CIs (1000 iterations, BCa method, RNG seed 20260426). Outcome labels apply to the **direction and replication across cohorts**, not to the absolute d magnitude in any single window.

## CHK-3.2 healthy-baseline check

For each window, compute case-window vs control healthy A. The healthy population is the same across all windows (controls are time-zero). If healthy A differs between cohorts by >1 anchor-SD, flag as cross-cohort mismatch (VAL-093 already passed this check at the >10yr slice; we re-check at all windows for completeness).

## Saturation flag

Each per-sample A reused from VAL-093. Per-window saturation fraction (A ≥ A_ceiling − 0.005) reported per architecture class.

## Test 2 placeholder (CCL-030)

This VAL is Stage 2 cell-of-origin only. Stage 1 immune Test 2 (lymphoid vs myeloid) blocked on OQ-2026-01. No bidirectional cancellation claim possible from this VAL.

## Card-specific routing (heme-LL-001)

This is a SOLID-ORGAN card (breast-epic). Stage 2 elevation on the matching solid organ is positive call. The temporal axis tests whether breast localization arises near clinical diagnosis or is absent across all windows.

## Substrate scope (heme-LL-009)

Single-substrate methyl-only, 450K platform. v1 single-substrate. No cross-tier comparison to Issue 002 5-substrate predictions in this VAL.

## Deliverables

1. `val_096.py` — re-analysis script (slices VAL-093_per_sample.csv by TTD window)
2. `VAL-096_results.json` — per-tile per-window per-cohort d + 95% CI, n
3. `VAL-096_window_tile_heatmap.png` — visualization (4 windows × 25 tiles, dual-cohort)
4. `VAL-096_outcome.md` — outcome label + temporal interpretation

## Rules followed

- CHK-2.1 (decision criteria pre-locked, 6 outcomes covered) ✓
- CHK-2.2 (cross-cohort baseline check at all windows declared) ✓
- CHK-2.3 (saturation flag declared) ✓
- CHK-2.4 (specimen unchanged from VAL-093) ✓
- CHK-2.5 (Test 2 placeholder declared) ✓
- CHK-2.6 (atlas declared: Loyfer/Moss array; same as VAL-093) ✓
- Sample-size honesty (n=11 at >10yr GSE51057) ✓ with bootstrap CI

## Pre-reg seal

This pre-reg is sealed before any window-stratified analysis is run. SHA-256 of this file at seal time committed alongside the VAL outputs.
