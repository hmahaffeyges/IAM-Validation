# DEV-ATLAS-COMMISSION-01 — commissioning IAM-Atlas v2 (atlas_e) as the composition step (STATUS 2b). Bars written 2026-10-09.

**Why it matters.** Whole-blood Met-A needs the neutrophil fraction first: composition decides how much of the blood signal is neutrophil.
Today that is `blood_composition_EPIC_v1.json` (8 purified groups). If atlas v2 measures composition better, it replaces it.

**What is already measured (development, both truth sets seen before these bars):**

| truth set | neutrophils, atlas_e | neutrophils, current | other groups |
|---|---|---|---|
| GSE112618, 6 bloods with FACS counts | mean error **0.016**, max 0.028 | 0.031, max 0.046 | atlas_e better on CD4 T, monocytes; worse on CD8 T, NK |
| GSE182379, 12 constructed mixtures | RMSE **0.014** | 0.019 | atlas_e 6 of 8 groups ≤ bar, current 5 of 8 |

**Independence, checked against the atlas v2 roster.** Atlas v2's blood cells come from Salas 2018 (GSE110554), Salas 2022 (GSE167998) and
Loyfer 2023. GSE182379's mixtures were built by the Salas 2022 laboratory from purified cells and may share donors with GSE167998, so they
are not an independent test of either method (the current method is built from GSE110554 too). GSE112618 (separate bloods, FACS counted)
is independent of the reference cells but small (6).

**Bars (on truth sets not used to build atlas v2; fixed now).**
1. Neutrophil fraction: mean absolute error ≤ 0.02 and every sample within 0.05, on each independent truth set (≥ 2 sets, ≥ 20 samples).
2. Not worse than the current method on neutrophils on any independent set.
3. For use beyond neutrophils (later, per-cell readings): every one of the 8 groups RMSE ≤ 0.03 (DEV-NILC-01 bars).
Bars 1–2 commission atlas_e as the composition step for whole-blood Met-A; then whole-blood Met-A bars are rerun with it and Met-A is
re-commissioned on it. Bar 3 is a separate, later commissioning.

**Independent truth sets to read (to obtain):** GSE122126 in-vitro genomic DNA mixes (EPIC, another laboratory; compositions in the
paper's supplementary table, to be extracted); EPIC whole bloods with flow counts from other laboratories (GEO search to do).
