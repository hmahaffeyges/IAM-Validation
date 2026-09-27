# PROC-SERIAL-01 — pre-registration: serial mode — one person, two or more draws

**Written 2026-09-27 before any pair of draws is scored.** Author, 2026-09-27: "Its always been about the sequence of tests for
individual patients, those will be the real winners. They will have a trajectory from draw to draw that shows progression or not,
and they will create the best sky map as well."

## What serial mode is
`run_sample.py --prior <earlier bundle>`. The specimen is read exactly as today (no stage changes). Then, against the prior draw
of the same person (patient hash must match; array type and pipeline must match, else refused):
1. **Per-cell ΔA** = A_now − A_prior for every cell present in both draws, with the cell's fraction in each; a cell present in
   one draw only is listed as appeared / disappeared, not differenced.
2. **The difference sky**: β_now − β_prior per address, drawn on the same sphere as the sky. No atlas expectation, no σ from any
   population, no composition — two measurements of one person and their difference. Where |Δβ| exceeds the change floor the
   address is marked; the plate reports the fraction of addresses beyond the floor and the median Δβ, both signed.
3. **Trajectory**: with N ≥ 3 draws (each bundle names its prior), a per-cell table A₁ … A_N with dates, the sign of the last
   step, and whether any step exceeded the change floor. No fit, no slope, no forecast — the numbers in order.

## The change floor (measured, not chosen)
- **F0** the same array read twice through the chain: ΔA = 0 and Δβ = 0 at every address exactly (the pipeline is deterministic).
- **F1** technical replicates — the same DNA on two arrays — are the only honest measure of what a draw-to-draw difference is
  when nothing changed. Public cohort with technical replicate pairs required (candidate: GSE55763, 36 pairs, 450K whole blood);
  the floor per cell is the 0.99 quantile of |ΔA| across pairs; the per-address floor is the 0.99 quantile of |Δβ|.
- **F2** until F1 is measured, the floor printed is the SNP-probe noise of the two arrays propagated to A (a + b·β(1−β), summed over
  the cell's identity loci) and is labelled "lower bound — technical-replicate floor not yet measured".

## Bars
- **B1** F0 holds on 10 arrays (exact zeros).
- **B2** F1 measured on ≥ 30 technical-replicate pairs; per-cell floors written to `serial_change_floor_v1.json` (A_Scoring_Module)
  with N and quantile; F2's lower bound is below F1 on every cell (else F2 is wrong and is removed).
- **B3** two different people's draws through serial mode with a forged shared hash are **refused** by the patient-hash check;
  mismatched array type or pipeline refused with one sentence.
- **B4** constructed progression: the reference array with a 2 %, 5 %, 10 % secretory spike as "draw 2": ΔA of the immune cells
  stays inside the floor (the blood did not change) while the composition check and Stage 2d report the foreign material — the
  difference sky marks the marker addresses and nothing else.
- **B5** the report gains a **Trajectory** tab (present only when a prior is given) and the difference-sky plate; the vocabulary
  guard runs on it; nothing on any other tab changes. The OM and the report's reference pages (How-to / Instrument) carry ONE example
  difference sky **drawn after the F1 floor is applied** (author, 2026-09-27: 'have it as an example in the OM and Report on the sky
  tab, with a description of the significance'), captioned with what is significant — the median Δβ of 0.000 over all addresses
  between two draws a decade apart, and the sign on the identity loci — and never on a specimen's own Sky tab, which shows that
  specimen alone. The unfloored plate is not shown: its texture is read-to-read noise.

## Decision rule
B1, B3, B4, B5 met → serial mode adopted with the F2 lower bound printed as such; B2 met → the measured floor replaces it. B4 failed
(immune ΔA outside the floor under a foreign spike) → the fraction confound is entering the difference; the foreign-cell scoring
floor (PROC-FOREIGNSCORE-01) gates which cells are differenced, and B4 is re-scored.
