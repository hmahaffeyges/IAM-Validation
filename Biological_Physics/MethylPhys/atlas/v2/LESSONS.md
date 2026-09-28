# Atlas v2 — lessons learned

What went wrong or surprised us while building atlas v2, why, and what the build does about it. Read with the v1 lesson,
[IAMAtlas_FLATNESS_LESSON_v1.md](IAMAtlas_FLATNESS_LESSON_v1.md). Numbers are from the files in `records/`.

## The model
1. **Flatness is the failure R-hat cannot see (v1, 2026-05-25).** A class anchor plus parameters for never-observed (cell, locus)
   pairs collapsed every cell toward its class mean. v2 has no class in the model (D3), writes an unobserved pair as NOT MEASURED
   (column `n_obs`; D9), and gates the finished run on distinctness (`15_stageB_run_all.sh`, `distinctness.csv`; FAIL if any pair's
   mean |difference| < 0.005).
2. **The leak that remained.** On the first stage-B block the six bone-marrow cells (450K only) had no data at ~16 % of loci and came
   out 0.0035 apart there — flat fill. Fixed before the full run by `n_obs` and NaN.
3. **Joint fit of source terms and cells did not converge.** Stage A (1,000 loci, 74 cells, 7 sources): cell-mean R-hat max 4.35, ESS
   min 2.1. A sequencing source's slope and offset trade off against the means of every cell only that source measured. Replaced by a
   closed-form Deming fit on cells two sources share ([`12_source_terms.py`](scripts/12_source_terms.py)); stage B then holds the terms fixed and every locus is an
   independent fit.
4. **Parameterisation.** The first smoke test did not converge; the second (log-scale donor SD, a 0.01 beta measurement floor, start
   from the data, longer warm-up) improved parameters but not the cell means; the third (centred cell means — every cell is data-rich)
   gave cell-mean R-hat p99 1.03. With the source terms fixed, stage B block 0: R-hat p99 1.005, ESS p01 753, 0 divergences.
5. **Timing must block on the result.** JAX returns before sampling finishes; early per-locus times were wrong until the timer waited.
6. **Summaries are not a posterior.** G-002 never wrote its chains, so it can be reproduced but not re-sampled. v2 keeps 20 draws per
   (cell, locus) and the per-locus prior (D11).

## The data
7. **A twin test on means is inflated by noise.** With whole-array data a mean difference > 0.2 occurs at thousands of CpGs by chance.
   The test counts a CpG only where the two cells' *samples* do not overlap ([`10_twin_test_sample_level.py`](scripts/10_twin_test_sample_level.py)). Of 23 close pairs one
   met the twin rule: podocyte vs kidney tubule (9 separating CpGs) — same kidneys, no podocyte-gene hypomethylation: sort impurity.
8. **Names must be one convention before anything is counted.** The first roster split one cell under two sources' names.
9. **Mixtures come labelled as cells.** CD3 T cells and granulocytes are mixtures; they are out. The v1 map had ~20 such labels
   (whole_blood, PBMC, Leu, Lym, Mye, organ names) and one cell under five names — the source of v1's duplicates.
10. **Check the genome build, don't assume it.** Lister 2013 (GSE47966) matched ~11,600 of ~895k array CpGs with a flat offset test —
    not hg19. Not used until lifted over.
11. **Raw IDATs or nothing, for arrays.** Our Stage 1 needs IDATs; intensity tables would bring in another scale. Reinius 2012,
    GSE31848, GSE59091, GSE60821 have none. GSE116754 (undifferentiated hESC, found through the Cell Reports Methods 2026 compendium)
    does, and measured the ENCODE stem-line term: slope 1.135, offset −0.087, 4 arrays, 475,964 loci — in line with Loyfer 1.10, Tian 1.12.
12. **A capture panel is not a methylome.** GSE262275 (liver/bile cells) covers 27–59 % of array CpGs at >= 10 reads: held out (D9).
13. **Cells group by lineage, not by class** (exploratory, `scripts/exploratory/`): nearest neighbour shares germ layer 96 %, class 86 %,
    organ 64 %; terminal cells share almost no CpGs across lineages. A class is how a cell fails, not where its identity sits.

## The compute
14. **Cap threads per process.** Eight processes each started their own thread pools: load 143 on 32 cores. One thread per chain,
    set in `15_stageB_run_all.sh`.
15. **The spot quota counts the box you are replacing.** Terminate it (everything harvested first), then launch. A new instance at
    the same address has a new SSH host key — verify its fingerprint from the EC2 console before accepting it.
16. **Image the box before any swap** (AMI `methylphys-cpu-01-2026-09-28-pre-resize`): the environments and all source data survive.
