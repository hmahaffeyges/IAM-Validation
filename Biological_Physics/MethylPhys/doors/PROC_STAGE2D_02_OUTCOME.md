# PROC-STAGE2D-02 — outcome: NOT ADOPTED. The detector's design, not its lines, was the defect.

Scored 2026-09-27 against [`PROC_STAGE2D_02_PREREG.md`](PROC_STAGE2D_02_PREREG.md).

| bar | measured | |
|---|---|---|
| B1 held-out FP, leave-one-chip-out, 732 arrays | 8 detectable templates at 1.1–1.2 %; ≥10-fire arrays 0 | MET |
| B2 cross-laboratory (23 Karolinska + UCLA arrays) | 0 fires | MET |
| B3 detection kept (real spikes) | **0 of 8 at 5 % for Breast, Bladder, Kidney, Cortical neurons** | **FAILED** |
| B4 composition unchanged | structurally unchanged (Stage 2d writes no composition field); the shard comparison used a different Stage 1 input and is void | not scored |
| B5 thin-source templates NOT DETECTABLE on every array | 13 of 13 on 47 arrays | MET |
| B6 kit test | not written — superseded | — |
| B7 (added after the author caught a Glia BREACH on a healthy reference array) no non-blood cell scored unless detected | 0 undetected non-blood cells scored on 60 arrays | MET |

## What B3 found, in order
1. My first spike construction dropped every locus where the cell is undefined and was void.
2. **PROC-MF-03's spike was circular**: `v(1−f) + template·f` injected the panel's own filled template and detected it. That is
   where the 0.5–1 % detection limits came from.
3. With honest spikes (the cell's atlas profile at the loci where it is measured; host β elsewhere) the raw detector's
   response to a 5 % spike is **2–13 % of f**; at 20 % it is ~50 %. The blood-only NNLS is fitted *after* the foreign material
   is present and absorbs it. Real detection limit of v1/v2: ~15–20 %.
4. At 20 % a Breast spike moves the uterus and endothelial templates more than Breast: v1/v2 cannot name the epithelium.
5. Common-mode removal (this procedure's change) subtracts a real spike too, because every template rises with any foreign
   material (epithelial–epithelial r = 0.56, epithelial–neural 0.68 on healthy blood). Group-wise estimators do not rescue it.
6. **A joint fit — blood columns and all 21 templates in one NNLS — does not have the defect.** Healthy null median 0 (q99
   0.000–0.021), response 0.6–1.0 of f, 6/6 detected at 5 % for Kidney, Colon, Glia, β-cells with the largest coefficient on the
   right cell; Breast needs 10 %. Glia and Thyroid carry a standing 0.2–0.8 % on healthy blood (the Glia BREACH's origin).

## Correction to the finding of this morning
FINDING_DETECTION_PANEL_HELDOUT said the biased templates were the thin-source family. Coverage does not sort them: 19 of 21
templates are on 1.3–4.5 % of the array; only cortical neurons and glia are full-coverage; the gastric families at 79 % were
among the biased. All 1,506 panel markers are defined for every template. The split was real; the cause I gave was wrong.

## Decision
Not adopted. v2's noise floor and NOT-DETECTABLE list are correct for a detector that should not be used; the B7 gate stays
(a non-blood cell in whole blood is scored only when detected). The detector is rebuilt on the joint fit under PROC-STAGE2D-03.
