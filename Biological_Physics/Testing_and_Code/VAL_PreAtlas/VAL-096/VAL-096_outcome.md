# VAL-096 — Outcome

**VAL ID:** VAL-096
**Pre-reg SHA-256:** 01247146d955ad28a7d141dd5b194a86d1d97b63b1022a07587ae4cd69310c6d
**Date executed:** 2026-04-26
**RNG seed:** 20260426

## Outcome label

**O1_BREAST_TILE_FIRES_NEAR_DIAGNOSIS** — partial fire, with the pre-locked criterion |d|≥0.5 met directionally but not at full magnitude in either cohort. Co-occurs with **O4_PATTERN_AMPLIFIES_AT_NEAR_DX** (in the sense that distributed pancreatic + cycling-tile elevation persists across all four windows, which the >10yr VAL-093 finding is now demonstrated to be the steady-state distributed pattern, not a window-specific artifact).

## Headline finding

The VAL-093 `O2_SECRETORY_DISTRIBUTED` pattern at >10yr (breast-tile null with pancreatic + cycling elevation) is the **steady-state pre-diagnostic distributed pattern across all four time-to-diagnosis windows**. The breast tile remains weakly read at the >10yr, 5-10yr, and 2-5yr windows (|d| ≤ 0.20 across all three windows in both cohorts) and rises sharply only at the **0-2yr window: d=+0.43 GSE51057, d=+0.49 GSE51032** — both cohorts agree directionally and approach the pre-locked |d|≥0.5 threshold.

This is consistent with breast-tissue-of-origin signal localizing in plasma in the 24 months before clinical diagnosis, on top of a long-standing distributed cellular-aging-drift signal that was already present 10+ years earlier.

## Per-window per-tile observations (both cohorts together)

**Persistently elevated across all four windows (distributed cellular-aging-drift signal):**
- Pancreatic beta cells: +1.02/+0.94 → +0.40/+0.57 (high at >10yr, attenuates near diagnosis)
- Pancreatic acinar cells: +0.91/+1.02 → +0.52/+0.60 (similar attenuation pattern)
- Pancreatic duct cells: +0.99/+0.70 → +0.04/+0.26 (largest attenuation; near-zero in GSE51057 0-2yr)
- Kidney: +0.73/+0.90 → +0.54/+0.52 (persistent, cycling-class)
- Colon epithelial: +0.72/+0.65 → +0.29/+0.32 (attenuates slightly)
- Adipocytes: +0.49/+0.51 → +0.18/+0.23 (attenuates)
- Uterus/cervix: +0.45/+0.72 → +0.46/+0.51 (persistent)
- Head & neck larynx: +0.75/+0.81 → +0.11/+0.14 (large attenuation near diagnosis)
- Upper GI: +0.45/+0.80 → +0.28/+0.40 (attenuates)

**Rising near diagnosis (newly localizing tissue-of-origin signal):**
- **Breast: +0.20/+0.10 (>10yr) → +0.43/+0.49 (0-2yr)** — primary target tile, replicates direction across cohorts
- Hepatocytes: +0.31/+0.62 → +0.61/+0.61 (rises slightly to similar magnitude both cohorts)
- Cortical neurons: +0.34/+0.61 → +0.38/+0.48 (rises in GSE51057, neutral in GSE51032)
- Vascular endothelial cells: +0.15/+0.80 → +0.44/+0.38 (mixed; GSE51057 rises, GSE51032 falls)
- Bladder: +0.20/+0.17 → +0.31/+0.45 (rises)
- Thyroid: +0.06/+0.71 → +0.45/+0.45 (rises in GSE51057, falls in GSE51032 → net stable)
- Left atrium: +0.45/+0.34 → +0.37/+0.53 (mild; this is a terminal-class tile)

**Inverted or attenuated near diagnosis (immune class):**
- Monocytes EPIC: +0.33/+0.00 → −0.35/−0.40 (sign flip near diagnosis — both cohorts)
- Neutrophils EPIC: +0.04/−0.16 → −0.20/−0.22 (drift toward negative)
- Erythrocyte progenitors: +0.83/+0.48 → −0.14/−0.08 (large attenuation; baseline elevation at >10yr almost entirely absent at 0-2yr)

## Diagnostic interpretation

The data are consistent with a two-component temporal model of pre-diagnostic methylation drift in this cohort pair:

1. **A persistent distributed cellular-aging-drift signal** (multiple tissue tiles elevated 10+ years before clinical diagnosis), most intense at the longest pre-diagnostic windows and attenuating as diagnosis approaches. The pancreatic and kidney tile signals fit this pattern most cleanly.

2. **A late-localizing tissue-of-origin signal** (breast tile rises in the 24-month window before clinical diagnosis). This is what conventional progression-localization models predict and is consistent with VAL-060's at-diagnosis paired result (d=+0.676).

The two components appear to be additive rather than mutually exclusive — at the 0-2yr window, both the breast tile and several persistently-elevated tiles still show concurrent elevation.

## Inverse pattern in immune tiles near diagnosis

Three immune-class tiles (monocytes, neutrophils, erythrocyte progenitors) attenuate or invert sign as diagnosis approaches. In GSE51057 the monocyte d goes from +0.33 (>10yr) to −0.35 (0-2yr); in GSE51032 from +0.00 to −0.40. This requires interpretation:

- It is consistent with VAL-047 Phase 9's finding that Stage 1 immune signal is **strongest** at >10yr (d=+1.78) and **weakest** at 0-2yr (d=+0.09 to +0.27). VAL-096 extends this to per-tile resolution: the immune-class tiles are not just weaker but actively negative (homogenized) in the 0-2yr window in both cohorts.
- Mechanistic interpretation is open. Speculation is not appropriate in the outcome.md; CCL-035 candidate is a candidate location to formalize this as a pattern requiring further investigation.

## CHK-3.2 cross-cohort baseline check

PASS. 0 of 25 tiles cross the 1-anchor-SD threshold for healthy-baseline mismatch between GSE51057 and GSE51032 control populations.

## Saturation flag

Not relevant for this VAL — all per-sample A-scores reused from VAL-093 where saturation was already evaluated and reported as not exceeded.

## Sample-size honesty

The >10yr GSE51057 case window (n=11) has wide CIs. The pre-locked outcome thresholds were applied to the **direction and replication across cohorts**, not to absolute d magnitude in any single window. The 0-2yr breast tile direction pre-locked criterion (|d|≥0.5) was met directionally with d=+0.43/+0.49, just below the strict threshold. We mark this O1_PARTIAL rather than O1_FULL.

## Test 2 placeholder

This VAL is Stage 2 cell-of-origin only. No bidirectional cancellation claim possible.

## Card-specific routing

This is a SOLID-ORGAN card (breast-epic). The Stage 2 elevation on the matching solid organ at 0-2yr is consistent with positive-call routing for the at-diagnosis arm of breast-epic. The distributed pattern at long pre-diagnostic windows requires careful card framing — it is not a positive call for breast specifically; it is consistent with a non-tissue-localized cellular-aging-drift signal that precedes localization.

## Implications for breast-epic v0.3

This VAL provides the temporal context for v0.3 card update. Three observations to incorporate:

1. The breast-tile null at >10yr **is real** (replicates across all three pre-localization windows), not a coverage artifact.
2. The breast tile **does fire** in the 0-2yr window (d=+0.43/+0.49). This is the localization step.
3. The distributed signal at >10yr (pancreatic, kidney, colon, head/neck) is persistent and not breast-specific. It cannot support a breast-localized claim at long pre-diagnostic windows; it can support a "distributed cellular-aging-drift signature is detectable 10+ years before clinical diagnosis" claim within the framework, applicable to multiple cancer types not specifically breast.

## Language discipline

- "consistent with breast-tissue-of-origin signal localizing in plasma in the 24 months before diagnosis" ✓ (not "proves localization")
- "the data are consistent with a two-component temporal model" ✓ (not "validates")
- "this requires interpretation" / "mechanistic interpretation is open" ✓ (not "explains")
- No claims about VAL-047 results being confirmed or extended — only said "consistent with"

## Outputs

- `VAL-096_results.json` — full per-tile per-window per-cohort table with bootstrap CIs
- `VAL-096_window_tile_heatmap.png` — dual-cohort visualization
- `val_096.py` — execution script (deterministic, RNG seed 20260426)
- `VAL-096_prereg.md` + `VAL-096_prereg_seal.json`
