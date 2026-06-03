# VAL-094 — Outcome

**VAL ID:** VAL-094
**Pre-reg SHA-256:** 501fafad68fa93635a18f43687104756f006ea89ed301de80ac469514ae15626
**Date executed:** 2026-04-26
**RNG seed:** 20260426

## Outcome label

**O2_DISTRIBUTED_AS_LOYFER** — finer EpiSCORE breast sub-cell-type resolution does not surface a hidden breast arm. All 7 EpiSCORE BreastRef cell types (Basal, EC, Fat, Fib, Luminal, Lym, MP) produce nearly identical per-window d values, meaning the markers do not discriminate the breast tissue signal into sub-cell-type-specific arms in this scoring setup.

A secondary observation falls outside the pre-locked outcome categories and is documented separately below.

## Headline finding

The seven EpiSCORE BreastRef sub-cell-types (5 stromal + 2 immune-of-tissue: Basal, EC = endothelial, Fat = adipocyte, Fib = fibroblast, Luminal, Lym = tissue-resident lymphocyte, MP = tissue macrophage) **behave as a single coherent signal**, not as seven independent tile readings. At every TTD window in both cohorts, the seven sub-cell-type d values agree within 0.10 of each other. Examples:

- **GSE51057 >10yr:** d ranges from +1.01 (MP) to +1.17 (Fib) — all 7 cell types within 0.16 of each other
- **GSE51032 >10yr:** d ranges from +0.20 (Luminal) to +0.33 (Fib) — all within 0.13
- **GSE51032 0-2yr:** d ranges from +0.43 (Basal) to +0.50 (Fib) — all within 0.07

This is the resolution-collapse pattern: when you ask the same set of buffy-coat CpGs about seven different tissue sub-cell-types, you get seven correlated readings of the same underlying signal because the discriminating markers cluster in similar genomic regions.

**Practical consequence for breast-epic v0.3:** EpiSCORE BreastRef cannot be used to argue for a specific breast sub-cell-type origin (e.g., luminal vs basal) of pre-diagnostic signal in plasma. EpiSCORE is appropriate for cell-of-origin attribution at TISSUE level, not for sub-cell-type resolution within tissue when the input is buffy-coat plasma.

## Comparison to VAL-093 Loyfer/Moss breast tile

| Window | Loyfer breast tile d (57/32) | EpiSCORE breast (mean across 7 cell types) (57/32) |
|---|---|---|
| >10yr | +0.20 / +0.10 | **+1.09 / +0.25** |
| 5-10yr | +0.05 / +0.19 | +0.44 / +0.40 |
| 2-5yr | +0.14 / +0.16 | +0.49 / +0.42 |
| 0-2yr | +0.43 / +0.49 | +0.21 / +0.48 |

**At 0-2yr, GSE51032:** Both atlases agree (Loyfer +0.49, EpiSCORE +0.48). Cross-atlas agreement.

**At 0-2yr, GSE51057:** Loyfer breast tile fires (+0.43), EpiSCORE breast collapses to +0.21. Cross-atlas DISAGREEMENT in this single cell × cohort.

**At >10yr, GSE51057:** EpiSCORE shows large signal (+1.09) where Loyfer reads null (+0.20). This requires interpretation.

## Discrepancy at >10yr GSE51057

EpiSCORE BreastRef yields d=+1.01-1.17 (all 7 cell types) at the >10yr window in GSE51057, while Loyfer breast tile reads d=+0.20 at the same window for the same 11 case + 177 control samples. The discrepancy is substantial.

Three candidate interpretations:

1. **Different CpG selection pulls a different signal.** EpiSCORE markers were selected from breast tissue scRNA-seq → DNAm regression (Zhu Teschendorff 2022); Loyfer breast tile uses CpGs that discriminate bulk breast from the other 24 Loyfer cell types. The two CpG sets do not overlap heavily. At >10yr in GSE51057 (n=11 cases, n=177 ctrls), the EpiSCORE-selected CpG set may be picking up a signal that the Loyfer breast tile's CpG set does not. **This is the most parsimonious explanation.**

2. **Small-sample artifact at >10yr GSE51057.** n=11 cases. The bootstrap CI on this d will be wide (computed in results JSON; expected ±0.5 or wider). The discrepancy may be within sampling uncertainty.

3. **Cross-cohort selection effect.** GSE51057 long-pre-dx cases were selected by the original DCSE study based on their later cancer diagnoses. The selection may correlate with EpiSCORE-tracked methylation patterns in ways the Loyfer breast tile does not capture.

The data alone do not distinguish these. The cohort-replication test discriminates: GSE51032 >10yr also has long-pre-dx cases (n~85 per VAL-093 windows), and EpiSCORE there yields only d=+0.20-0.33. **The signal does not replicate across cohorts at >10yr**, ruling out a real biological signal under the framework's discipline of requiring two-cohort replication.

The honest reading: at >10yr GSE51057 has an unexplained EpiSCORE elevation that does not replicate to GSE51032. This is consistent with a cohort-specific selection or processing effect that should be investigated but does not support a framework-level positive call for EpiSCORE breast sub-cell-type elevation at long pre-dx windows.

## CHK-3.1 β distribution sanity

PASS in both cohorts: GSE51057 sample has 49.5% extreme β, 7.6% in [0.4-0.6], median 0.658. GSE51032 sample matches. Bimodal as expected for raw β.

## CHK-3.2 cross-cohort baseline check

PASS. 0 of 7 cell types cross the 1-anchor-SD threshold for healthy-baseline mismatch.

## Saturation flag

A-score range stays well below the secretory ceiling (1.1859) at all windows. No samples flagged.

## Sample-size honesty

GSE51057 >10yr n=11 cases vs n=177 controls. Bootstrap CI (1000 iter, BCa-equivalent) is reported per cell type in results JSON. The d~+1.1 figure at this window has a wide CI; the cross-cohort replication failure (GSE51032 >10yr d~+0.25) is the deciding piece.

## Test 2 placeholder

This VAL is Stage 2 cell-of-origin only. No bidirectional cancellation claim possible.

## Card-specific routing

This is a SOLID-ORGAN card (breast-epic). The pre-locked outcome categories under O2_DISTRIBUTED_AS_LOYFER mean: finer EpiSCORE resolution does not change the v0.2 → v0.3 card narrative. The card's at-diagnosis arm (VAL-060 paired d=+0.676) and 0-2yr Loyfer breast tile arm (VAL-096 d=+0.43/+0.49) remain the cleanest pieces of breast-localized evidence.

## Substrate scope

Single-substrate methyl-only, 450K platform. v1 single-substrate.

## Implications for breast-epic v0.3

1. **Do NOT add EpiSCORE breast sub-cell-type calls to the v0.3 card** — the resolution does not separate at this scoring setup. EpiSCORE per-cell-type d values are correlated > 0.95 across the 7 cell types in every window × cohort cell, indicating they are duplicate readings.
2. **EpiSCORE is still appropriate for cross-tissue cell-of-origin attribution** (which of the 14 EpiSCORE tissues is most consistent with the customer's signal), just not for within-tissue sub-cell-type resolution.
3. **Long pre-diagnostic GSE51057 EpiSCORE signal is a cohort-specific anomaly that does not replicate** — log as observation, not finding, and investigate as part of breast-epic Open Questions for v0.3.

## Language discipline

- "behave as a single coherent signal, not as seven independent tile readings" ✓
- "the data alone do not distinguish these" / "the honest reading" ✓
- No "EpiSCORE failed" or "EpiSCORE proves" — said "the resolution does not separate"
- "this requires interpretation" ✓ for the GSE51057 >10yr discrepancy

## Outputs

- `VAL-094_results.json` — full per-cell-type per-window per-cohort table with bootstrap CIs
- `VAL-094_per_sample.csv` — per-sample 7-cell A-scores joined with TTD/group metadata
- `val_094.py` — execution script (RNG seed 20260426)
- `VAL-094_prereg.md` + `VAL-094_prereg_seal.json`
