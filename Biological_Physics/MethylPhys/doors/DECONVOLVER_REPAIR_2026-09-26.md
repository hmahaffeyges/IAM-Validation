# The deconvolver repair of 2026-09-26 — why Breast read zero in breast tissue, and what was changed

**Found** while answering the author's question of whether breast-cancer patients shed into blood: ten breast-tissue
arrays (GSE69914, tumour, adjacent and normal) all returned **Breast = 0.000**, while plain geometry ranked the atlas's
Breast column nearest to most of them. The solver, not the atlas, was hiding the cell. Every change below was measured
before it was made, on constructed mixtures whose truth is known and on real arrays from four laboratories.

## Four causes, in the order found

| # | cause | measured consequence | change |
|---|---|---|---|
| 1 | cell-type markers were one global top-4,000 by between-cell variance with **no per-cell quota** | a few stomach entries took most markers; 30+ cells had none | per-cell quota (`n_celltype_markers_per_celltype`) |
| 2 | the atlas is **twelve source families defined on 252 to 482,421 loci**; entries were filled with the grand mean where undefined | `erythroblast` (252 loci) absorbed mass from every specimen; Breast (6,105 loci) could never win a marker ranked over 483k | **solve block**: candidates ≥ 1 % coverage, markers only at loci where every candidate is defined; block chosen to maximise the number of cells resolvable together with ≥ 2,000 shared loci |
| 3 | **duplicate cells** — cross-platform copies (`Monocytes_EPIC` vs `CD14_monocytes`) and same-source families the array cannot separate (six stomach entries, five progenitors) were solved as separate columns | `NK-cells_EPIC` alone took 72 % of the "non-blood" mass in healthy blood | twins by **correlation** (r > 0.98 cross-source, > 0.985 same-source, same class): lower-coverage copy dropped; equal-coverage members become a **resolution family** solved as one column, fraction shared to every member |
| 4 | the deconvolver read **raw** stage-1 betas, which sit ~0.07 above the atlas | with tissue columns now solvable, the offset landed on tissue (4.1 % non-blood in healthy blood) | `stage_a_cells` maps first, deconvolves the mapped betas — as the class gauge always did |

Plus the author's request: **uniqueness markers** — for each solve column, block loci where it beats its nearest
rival by > 0.15 β — added as a *union* with the variance set. Alone they cut the worst constructed error from 0.06 to
0.04 but leave the blood background unpinned (9.7 % non-blood); in union they keep the gain at the 0.0 % floor.

## What the repaired solver reads

| specimen | result |
|---|---|
| 48 healthy bloods, 4 laboratories, mapped | non-blood mass median **0.000**, p90 0.023, max 0.061 — the false-positive floor for any shedding claim |
| constructed 10 % Breast in blood — **through the chain** ([`run_sample.py`](../chain/MethylPhys_Interface/run_sample.py), synthetic mixture seed 2, noise σ 0.044) | Breast **0.079**; tier withheld (foreign 0.057) — `MethylPhys_SYNTH_BREAST10_repaired_deconvolver.html` |
| constructed 10 % Breast in blood — solver dissection (fixed mixture, seed 5, block NNLS outside the chain) | Breast 0.102; CD4 0.179 (true 0.20) — an under-read of 2 % on the chain path is the difference between the two backgrounds, not two solvers |
| constructed 5 % Colon / 5 % Prostate in blood | 0.045 / 0.023–0.040 |
| breast tumour tissue (GSE69914) | Breast 0.55–0.65, immune 0.01–0.21 |
| breast normal tissue | Breast 0.09–0.64 with **adipocyte 0.09–0.66** where expected |
| 25 constructed bloods, CD4/CD8/B/NK/mono | max error median 0.056 — the CD4/CD8 pair (r = 0.978, 33 and 8 exclusive loci) is the limit |

## Two questions answered by measurement, not argument

**Are unclassified blood cells causing the leak?** No. The leak was an unmerged NK twin. The nine finer blood subsets
(naïve/memory T, Treg, basophils …) live on a 2,543-locus source that overlaps Breast at 8 % and cannot enter the same
solve; on their own block they take ~9 % of a healthy array, which is a real refinement of the T-cell pool, but the
non-blood floor is already zero without them.

**Would a substrate-only atlas be less noisy?** No, and it loses detection. Blood-only vs full block: constructed
error 0.057 vs 0.056; real-blood fit residual **0.073 vs 0.063** (worse); blood fractions agree within 0.011; and 10 %
spiked Breast has nowhere to go — it lands on neutrophils and CD4 and barely moves the residual. The block rule
already gives the author's intuition its correct form: each source family is solved on the loci it shares.

## What the report now says

Sub-1 %-coverage entries (69 on 450K) print as **not resolvable on this platform**, never as fraction 0. A resolution
family is **one row, one fraction**. Dropped twins are named as the cell they were folded into. Each cell carries its
exclusive-locus count so a reader can see how identifiable the reading is.

## Not done here

Merging the finer blood subsets into a second-block solve; a per-cell A on a family (the A is the same for all members
by construction and should be printed once); and the CD4/CD8 separation, which needs loci this block does not have.
