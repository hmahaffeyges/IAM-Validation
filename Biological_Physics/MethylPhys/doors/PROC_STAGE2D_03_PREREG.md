# PROC-STAGE2D-03 — pre-registration: foreign-cell detection as one joint fit

**Written 2026-09-27 before the bars are scored.** The exploration that led here is recorded in PROC_STAGE2D_02_OUTCOME.md
(6 quiet hosts, 60 healthy arrays); every bar below is scored on arrays and hosts not used there where the design allows.

## The detector (fixed)
On the 1,506 panel markers (mapped β): one NNLS of the specimen on [the blood reference columns | all 21 foreign templates].
f̂_cell = coefficient_cell / Σ coefficients. No residual projection, no common mode, no centre. Author's decision 2026-09-27:
the line is the instrument's **noise floor**, the 0.99 quantile of f̂_cell over the Uppsala arrays the intake gate admits, stated
on the page with N. A template whose f̂ median on healthy blood exceeds 0.002 carries a standing bias: its floor is still the
0.99 quantile, and the page says "standing bias b on blood" beside it. Gate: composition check verified blood-like.
UNSPECIFIC if ≥ 3 templates fire together. The B7 gate (a non-blood cell in whole blood is scored only when detected) reads this.

## Bars
- **B1** leave-one-chip-out over all admitted Uppsala arrays (732 minus intake refusals; none refused on the first 100): per-template
  FP ≤ 1.5 %; UNSPECIFIC arrays ≤ 1 %.
- **B2** Karolinska + UCLA admitted panel arrays (23) under the Uppsala floors: ≤ 2 fires per template.
- **B3** honest spikes into 12 quiet hosts NOT among the 6 used in exploration, all 21 templates, f = 0.02 / 0.05 / 0.10: at 0.05,
  ≥ 90 % detected for full-coverage templates (neurons, glia) and ≥ 75 % across the thin templates; at 0.10, ≥ 90 % for all.
- **B4** naming: at 0.10 the largest foreign coefficient is the spiked cell in ≥ 80 % of spikes; at 0.05 in ≥ 60 %. Where it is not,
  the page names the group ("epithelial material, cell not resolved") not a cell.
- **B5** composition unchanged: class fractions and the composition check identical to 1e-9 on 12 arrays with the old and new
  stage_2d (the stage writes no composition field; this checks it).
- **B6** the reference report's Glia row: NOT DETECTED, not scored; no non-blood cell scored undetected on 100 healthy arrays.
- **B7** kit test [`test_stage2d_panels.py`](../kit/test_stage2d_panels.py) rewritten to this contract and passing.

## Decision rule
All met → adopted; detection_panel_v3.json replaces v2; v1 and v2 retired with the record. B3 or B4 failed for the thin templates
only → adopted for neurons/glia, thin templates print the measured limit and "cell not resolved". Anything else → not adopted.
