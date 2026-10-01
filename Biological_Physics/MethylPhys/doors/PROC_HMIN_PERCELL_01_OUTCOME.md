# PROC-HMIN-PERCELL-01 — outcome (2026-09-30): do cells share one floor?

44 cells with ≥ 3 Loyfer WGBS samples. For each sample, identity loci chosen on the cell's OTHER samples with no floor involved; the floor is
the held-out sample's mean depth-corrected entropy on them. (A first run was void: coverage is stored as uint8 and the depth term overflowed;
fixed and re-run.)

| locus set | median floor | range | spread between cells (SD) | repeatability within a cell (SD) |
|---|---|---|---|---|
| L1 whole methylome | 0.433 | 0.372–0.654 | 0.050 | 0.012 |
| L2 donor-stable loci | 0.365 | 0.211–0.568 | 0.061 | 0.013 |
| L3 donor-stable and distinct | 0.542 | 0.333–0.830 | 0.105 | 0.021 |

- **Not one shared floor.** On every locus set the spread between cells is 4–5 times the repeatability of one cell's floor across its own samples.
  Each cell type has its own floor, and it is reproducible to about ±0.013 bits.
- **But most cells are close.** On donor-stable loci 31 of 44 cells lie between 0.30 and 0.40 bits; 7 below (cortical neurons 0.21, smooth muscle
  0.25, hepatocytes, colon, breast basal, prostate, bladder epithelium) and 6 above (effector T cells, fibroblasts, adipocytes 0.46, erythroid
  progenitors 0.57).
- **The floors do not follow the classes.** Class medians on L2: terminal 0.31, cycling 0.33, secretory 0.36, immune 0.37, stromal 0.38 — inside
  the spread of each class.
- **H of the mean β** reads 0.93–1.00 on these loci for every cell and carries no information: the floor must be the mean of per-locus entropies.

**Consequence for the reading.** A = (the specimen cell's mean per-locus entropy on its identity loci) / (its own floor). In a mixed specimen that
needs each cell's own β at each locus, not the specimen's mixed β. For the majority cell the specimen's β is close; for minor cells a per-locus
separation is required. That is the next design question.
