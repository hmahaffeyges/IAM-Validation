# PROC-DECONV-V2-01 — pre-registration: a new composition solver built on atlas v2 alone, tested on real known mixtures

**Written 2026-09-29, before the solver is fitted to anything.** Author, 2026-09-29: "we should create a new updated deconvolver for
v2 … We dont force fit v1 stuff to v2 to make it work!!!" Nothing below reads a v1 file, v1 marker panel, v1 cell name or v1 code.

## The solver (fixed now)
1. **Reference.** For cell c at locus i: μ_ci = v2 posterior mean; its variance v_ci = mu_sd² + donor_sd² (the atlas's own
   uncertainty plus the spread between donors of that cell), both from atlas v2.
2. **Loci.** Only loci where **every** cell in the model is measured (n_obs ≥ 2; ≥ 1 for the four single-pooled-sample cells).
   Nothing is filled. (v1's median fill is what made cells unfindable.)
3. **Markers, per cell, from v2.** For each cell: the loci where its μ differs from the **nearest** other cell by ≥ 0.20 (above or
   below), ranked by that margin divided by the pooled SD; the top 200. The model's marker set is the union. A cell with fewer than 20
   such loci is reported **NOT SEPARABLE on this array** (its nearest cell is named) and is not dropped silently.
4. **Solve.** Fractions f ≥ 0, Σf = 1, minimising Σ_i (β_i − Σ_c f_c μ_ci)² / (Σ_c f_c² v_ci + σ²), with σ = 0.02 (array noise).
   Iterated 5 times (weights recomputed from the current f). All cells in one solve.
5. **Presence.** 200 bootstrap resamples of the marker loci; a cell is PRESENT if its 2.5th-percentile fraction > 0.005.

## The test: 24 real arrays of known composition
Salas 2018 (GSE110554, 12 arrays, 6 cell types) and Salas 2022 (GSE167998, 12 arrays, 12 types): purified leukocytes mixed in known
proportions by the depositing laboratory, run on EPIC, calibrated by our Stage 1 (all 24 pass intake). Their true fractions are the
laboratory's, from GEO. v2 cells are summed to each laboratory category before comparing:
CD4 = cd4 t, naive cd4, memory cd4, t central memory cd4, t effector memory cd4, regulatory t · CD8 = cd8 t, naive cd8, effector
memory cd8, t effector cell cd8 · NK = nk · B = b, naive b, memory b · Mono = monocytes · Neu = neutrophils · Eos · Baso. Salas 2022's
sub-types (naive/memory CD4, naive/memory B, Treg) are compared at that level too, reported.

- **A5 (spec bar):** mean absolute fraction error over the 24 arrays **≤ 0.015** for CD4, CD8 and NK each.
- **B-other:** the total fraction assigned to non-blood cells, per array, reported (these arrays contain none).
- Reported, not bars: error for B, Mono, Neu, Eos, Baso and the Salas 2022 sub-types; how many cells are PRESENT that are not in the
  mixture; separability (item 3) for every cell.

**Known limit, stated now:** the purified cells in these mixtures come from the same two laboratories whose purified samples are in
atlas v2, and possibly the same donors. So this tests the solver on real mixing and real array noise, but it is not fully independent
of the atlas. A fully independent known mixture (a third laboratory) is the next test when one is found.
