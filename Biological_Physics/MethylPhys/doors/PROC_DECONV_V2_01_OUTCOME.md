# PROC-DECONV-V2-01 — outcome (2026-09-29): A5 FAILED on CD4 and CD8; NK passes

Run against PROC_DECONV_V2_01_PREREG.md as written; nothing moved. 74 cells, 343,016 loci measured by every cell, 7,445 markers.
Three re-submissions before this run fixed only the test harness (a parallel-call error, then shard file names); the solver and the
bars were not touched.

**A5 (mean absolute fraction error over 24 known mixtures, bar ≤ 0.015): CD4 0.024 FAIL · CD8 0.031 FAIL · NK 0.0075 PASS.**
The v1 figure the spec quotes is ±0.035 (constructed, not real, mixtures).
Other groups: B 0.013 · monocytes 0.015 · neutrophils 0.015 · eosinophils 0.027 · basophils 0.006. Salas 2022 sub-types: naive
CD4 0.097, memory CD4 0.147, Treg 0.031, naive B 0.071, memory B 0.041.
By study: Salas 2022 arrays read closer (CD4 0.026, CD8 0.025, neutrophils 0.006) than Salas 2018 (CD4 0.022, CD8 0.037, neutrophils 0.024).
Non-blood fraction assigned: median 0.013, max 0.029; vascular endothelium called PRESENT on 9 of 24 arrays that contain none.

**The pattern:** CD8 is under-read on 19 of 24 arrays and CD4 over-read on 19 — mass moves from CD8 to CD4. Neutrophils are under-read
on the 2018 arrays by 0.024 on average.

**What the separability report shows:** 24 of 74 cells have fewer than 20 loci separating them by ≥ 0.20 from their nearest
cell (the cell listed is the one most often nearest across loci):
- **16 are in parent/child families or close subsets:** bulk "cd4 t cells" (0 markers; nearest cd8 t), "cd8 t cells" (2; cd4 t),
  "b cells" (1; cd8 t), naive CD4 (8; naive CD8), naive CD8 (13; naive CD4), memory CD4 (1; Treg), T central memory CD4 (3; T effector
  memory CD4), T effector memory CD4 (0; T effector CD8), effector memory CD8 (10; Treg), naive B (6; basophils), and the six
  bone-marrow progenitors against each other (CMP 3, GMP 0, HSC 4, L-MPP 1, MEP 10, MPP 4).
- **8 are single tissue or myeloid cells close to a related cell:** gastric body epithelium (0; gastric fundus), kidney glomerular
  endothelium (4; kidney tubular endothelium), kidney tubular epithelium (7; kidney glomerular epithelium), lung interstitial
  macrophages (1; lung alveolar macrophages), monocytes (8; neutrophils), aorta endothelium (14; striated muscle), colon
  fibroblasts (15; striated muscle), vascular endothelium (3; adipocytes).

The CD8-to-CD4 shift fits the first group: a bulk sorted population is, methylation-wise, a mixture of its own subsets, so the solver
can trade mass between parent and children, and between the two T-cell families, without changing the fit. The second group is
tissue cells from the same organ or lineage that this 0.20 margin does not separate; it bears on tissue specimens more than on these
blood mixtures (monocytes excepted).

Nothing is changed on this result. Whether parent populations stay in the atlas next to their subsets is the author's decision
(a sample-level twin test, V6, measures it).
