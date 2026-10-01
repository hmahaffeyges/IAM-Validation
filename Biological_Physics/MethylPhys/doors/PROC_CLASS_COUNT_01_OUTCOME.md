# PROC-CLASS-COUNT-01 — outcome (2026-09-30)

53 cells with ≥ 2 Loyfer WGBS samples, spanning six of the eight classes (no pluripotent or adult-stem cell; one progenitor). Floor-free
measures only; choosing and reading never share a sample for M3.

| measure | classes explain spread? (ratio vs relabelled null) | levels found without labels | match to classes (ARI; null 95 %) |
|---|---|---|---|
| M1 whole-methylome entropy | yes: 1.27 vs 0.09, p < 0.0001 | BIC minimum at 4, but 1 is within 0.11 (a tie); 8 is worse by 36 | 0.09 for the 4-group fit (0.05) |
| M2 entropy profile | yes: 0.99 vs 0.09, p < 0.0001 | 2 | 0.13 (0.04) |
| M3 defining loci, held out | yes: 0.88 vs 0.09, p = 0.0003 | BIC minimum at 6, but 2 and 3 are within 0.16 and 0.32 (ties); 8 is worse by 19 | 0.06 for the 6-group fit (0.05) |

**Where the class signal comes from.** Remove the stromal cells and the single progenitor and it almost disappears: on the remaining four
classes (42 cells) M1 is at the null (p = 0.85) and M3 only marginal (p = 0.03); on cycling, immune and secretory alone, neither (p 0.63, 0.32).

**What it says.**
1. BIC cannot choose between 1 and 4 levels (M1) or between 2, 3 and 6 (M3): differences under 2 are not evidence. It does reject eight on both
   (worse by 36 and 19). The groups the fits find are: a low-entropy group (epithelia, blood cells, most secretory cells), a
   high-entropy group (endothelia, fibroblasts, adipocytes, muscle), with neurons, kidney and macrophages between.
2. Stromal cells are genuinely different. Cycling, immune and secretory are not distinguishable by entropy at any scale measured here.
3. The groups the data form cut across the draft classes (ARI 0.06–0.13): colon epithelium and monocytes read together at one end; striated
   muscle (terminal) and colon fibroblasts (stromal) together at the other.
4. Not tested here: pluripotent and adult stem (no sequencing samples with ≥ 2 replicates in atlas v2).
