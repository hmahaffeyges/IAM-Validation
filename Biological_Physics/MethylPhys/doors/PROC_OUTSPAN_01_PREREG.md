# PROC-OUTSPAN-01 — pre-registration: the clean map of what the atlas cannot explain

**Written 2026-09-28, before atlas v2 exists and before any specimen is projected.** Author, 2026-09-28: "Is there anything we could do
to quieten the noise we may get from unclassified cell fractions … something similar to how we do with the sky map when we remove all
the noise and leave only what matters in the difference maps."

## The idea (CMB template removal)
Every mixture of atlas cells, whatever its fractions, lies in the space spanned by the atlas profiles. A specimen's methylation splits
into two parts: **in the span** (anything a mixture of known cells can produce, including every composition error) and **out of the
span** (methylation no mixture of atlas cells can produce — where an unclassified or foreign cell must appear). The out-of-span part is
the clean map. Composition error cannot reach it by construction.

## Construction (atlas only; no population)
- M: atlas v2 cell means on our Stage 1 array scale (stage B), loci × cells. σ_l: expected noise at locus l for this specimen =
  atlas posterior SD of the fitted mixture at l (stage B) combined with the array's own measurement noise from its SNP probes.
- Whiten: x̃ = β/σ, M̃ = M/σ. Projector P⊥ = I − M̃(M̃ᵀM̃)⁻¹M̃ᵀ (computed once per atlas; the σ-dependence is recomputed per specimen).
  Out-of-span residual r⊥ = P⊥x̃; per-address z⊥_l = r⊥_l / √(1 − h_l), h_l the leverage of locus l.
- Outputs per specimen: the out-of-span sky (z⊥ on the existing HEALPix mapping); energy E = Σ z⊥² / (N − K) (K = atlas cells);
  in serial mode, the difference of two draws' out-of-span maps.
- Does not touch A. A stays H/H_min on each cell's identity loci. This feeds detection, the sky and serial mode.

## Simulations (run on the cloud box; constructed specimens only)
- **S0 null.** 1,000 specimens built from atlas v2 posterior draws, blood-like random fractions over atlas cells, array noise at the
  measured SNP-probe level. **Bar:** median E within [0.9, 1.1] and robust SD of z⊥ within [0.9, 1.1]. A miss means the noise map is
  wrong, and the tool does not proceed.
- **S1 composition immunity** (the author's question). S0 specimens with every known cell's fraction perturbed ±50 % relative, nothing
  unknown added. **Bar:** the 95th percentile of E stays inside S0's 95th percentile + 0.02.
- **S2 unknown cells.** Each of the 16 WAIT cells (real cells not in the atlas) spiked at 1, 2, 5 and 10 % into S0 specimens.
  Measured, not barred: each cell's out-of-span fraction ‖P⊥s‖²/‖s‖² — how much of it the atlas cannot absorb (the honest limit:
  a cell that resembles atlas cells leaks into the span). **Bar:** at 5 %, detected above S0's 95th percentile of E in ≥ 12 of 16 cells.
- **S3 serial.** Two constructed draws of one person, the second with a composition change: the difference map must read as S0.
  Then a 5 % WAIT cell added to the second draw: it must be detected in the difference map in ≥ 12 of 16 cells.

## After the bars (reported, not barred)
The out-of-span energy of real whole-blood arrays through Stage 1: each specimen's own number, read against S0. A real specimen above
S0 is atlas incompleteness, model misfit or a real unknown — named by which, not assumed.

## Before this can run
Atlas v2 stage B (cell means and posterior SDs at every locus). S0–S3 are written here and do not move.
