# PROC-STAGE2D-02 — pre-registration: the foreign-cell detector rebuilt on the held-out finding

**Written 2026-09-27, before any array is scored under these bars.** Follows FINDING_DETECTION_PANEL_HELDOUT.md.
Author's decisions, 2026-09-27: (1) the sky stays drawn as z on the atlas-posterior σ (no panel; the off-identity-loci offset is
drawn as it is and labelled, never re-centred); (2) a detection line measured on arrays known to lack the cell is the
**instrument's noise floor** and is acceptable — whole blood cannot contain colon epithelium, so f̂ there is the detector's
noise, and it says nothing about any cell's A.

## The change (fixed)
1. **Common mode.** Per array, the median f̂ across all templates is subtracted before any template is read. A shift shared by
   every foreign template is no cell; it is this specimen's own number, no population enters.
2. **Lines.** Per template, the line is the 0.99 quantile of common-mode-removed f̂ over all Uppsala arrays the intake gate
   admits (732 minus refusals), plus the Karolinska and UCLA admitted panel arrays for a cross-laboratory check; stated on the
   report as "instrument noise floor, measured on N arrays known to lack the cell". `detection_panel_v2.json` carries N, the
   quantile, and the per-template floor. No line is ever the raw maximum.
3. **Not detectable.** A template whose common-mode-removed f̂ on blood is biased — median > 0 by more than one σ_cm, or
   0.99-quantile FP > 5 % under any line that also keeps 50 % detection at f = 0.02 in a spike — gets **no line** and prints
   "NOT DETECTABLE on this block" with the reason (thin source). Expected from the finding: Colon, Hepatocytes, Lung, Thyroid,
   the two gastric families, Pancreatic acinar/duct.
4. Nothing else in Stage 2d changes: f̂ formula, marker block, blood-only reconstruction, the composition check.

## Bars
- **B1** held-out FP: leave-one-chip-out over the 732 admitted Uppsala arrays, per detectable template FP ≤ 1.5 % (bar allows
  sampling on a 1 % design); arrays with ≥ 10 templates firing ≤ 1 %.
- **B2** cross-laboratory: on the Karolinska and UCLA admitted panel arrays (23), per-template FP ≤ 2 of 23 with the Uppsala
  lines — the floor transfers, or the report says which laboratory it was measured on.
- **B3** detection kept: real-array spikes at f = 0.02 and 0.05 of Breast, Bladder, Kidney, Cortical neurons (full-coverage
  templates) into 12 healthy arrays — ≥ 90 % detected at 0.05, ≥ 50 % at 0.02.
- **B4** the composition check (PROC-FOREIGN-01) is unchanged to 1e-6 on the 48 panel arrays.
- **B5** the thin-source templates print NOT DETECTABLE on every one of the 48 panel arrays; no f̂ or line for them on the page.
- **B6** the kit test ([`test_stage2d_panels.py`](../kit/test_stage2d_panels.py)) is rewritten to this contract and passes; it fails if a v1 line is read anywhere.

## Decision rule
B1–B6 met → adopted; v1 retired with this record. B3 failed → the common mode is removing signal: adopt lines and NOT-DETECTABLE
(B1, B2, B5) but keep the raw f̂ beside the corrected one on the page, and record it. Anything else → not adopted, finding stands,
report keeps its warning.
