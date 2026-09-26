# PROC-MF-02 — pre-registration: inverse-variance weighted detection of a foreign cell in blood, as a stage of the chain

**Written 2026-09-26, after PROC-MF-01 sealed and before this detector was scored on any spike with these bars.**
MF-01's diagonal *control arm* lowered the detection limit on all four cells but was not a claim, and read post hoc it
detected without estimating. This procedure makes it the claim, fixes its two defects in advance, and applies the
bars to it. Nothing here moves after results are visible.

## The detector

For each foreign cell **c** and specimen **v** (mapped betas on the solve block's markers):

1. blood-only NNLS fit on the blood columns → reconstruction **b**, residual **r = v − b**
2. template **t = μ_c − b / Σf_blood** (the cell's profile minus this specimen's own blood background)
3. weights **w_i = 1 / var_i**, the per-locus variance of the healthy residual, estimated **leave-one-laboratory-out**
   on the 36 arrays from the other three laboratories
4. raw amplitude **a = Σ w_i t_i r_i / Σ w_i t_i²**
5. **centring:** f̂ = a − median(a on the held-out laboratory's own 12 healthy arrays) — the null median of *that*
   laboratory, which is what a laboratory commissioning panel provides. MF-01 showed a constant −0.03 offset; this
   removes it by measurement, not by fitting.
6. **σ:** the spread (MAD × 1.4826) of a on the *other three* laboratories' healthy arrays — from the null, never
   from the weights, because MF-01 showed the weight-derived σ was wrong by an order of magnitude.
7. **detected** if f̂ exceeds the 47th of 48 ordered null f̂ values (≤ 1 false positive in 48), as in MF-01.

Nothing about the blood composition changes: the blood fractions are the chain's NNLS fractions as today.

## Spikes and null — identical to MF-01

48 healthy arrays (12 per laboratory, same draw); 768 spikes into real arrays (Breast, Colon_epithelial_cells,
Cortical_neurons, Prostate × 0.5, 1, 2, 5 %); 160 into constructed blood (reported, not deciding).

## Bars

| bar | requirement | why |
|---|---|---|
| B1 | detection limit lower than NNLS for **≥ 3 of 4 cells** on real-array spikes | the claim |
| B2 | honest σ: on the unspiked null, fraction of \|f̂/σ\| > 2 in **0.02–0.10** | MF-01's σ lied; this one is from the null and must not |
| B3 | unbiased after centring: median (f̂ − f) within **±0.005** at 2 % and 5 % | a detector that misreads the amount is not an estimator |
| B4 | centred null: median f̂ on the held-out laboratory within **±0.003 of zero** for every cell | the centring must work on a laboratory it did not see |
| B5 | same-laboratory variances (no leave-out) give a limit **no more than 20 % better** than leave-one-out | the gain must not be memorised |
| B6 | blood composition unchanged: median max \|Δ\| over CD4/CD8/B/NK/mono **< 0.005** | detection stage, not a new deconvolver |
| B7 | **false-positive rate on 845 EPIC-Italy whole bloods** (no known CNS or breast disease at draw for the controls; breast/colorectal *cases* excluded) — fraction of control arrays detecting Breast or Cortical_neurons **≤ 0.02** at the 48-array threshold | the threshold must hold on arrays it was not set on, from a fifth laboratory and platform |

**Decision rule.** B1–B7 met → adopted as **Stage 2d, foreign-cell detection**, ahead of the per-cell A, reporting
per foreign cell (f̂, σ, detected yes/no at the pre-registered rate) in the bundle and on the Cells tab, with a
red flag when a foreign cell is detected. Any bar failing → not adopted; the failing bar and its number are recorded,
and whether a further procedure is worth writing is the author's call. B7 not runnable (EPIC block not yet built)
→ recorded NOT ASSESSED and adoption limited to 450K until it is.

## Evidence files (named before they exist)

In `kit/`: **PROC_MF_02.py**. In `kit/results/`: **PROC_MF_02.json**, **PROC_MF_02_null.json**. In `plates/`:
**PROC_MF_02.png**. Linked from the outcome once they exist.
