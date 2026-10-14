# DEV-ARCH-01 · Does the cell's architecture set its healthy held error? (exploration, 2026-10-14)

**Status: exploration, not a test.** No rule was sealed; nothing here is a result. The author asked what sets each cell type's normal
operating point when every cell shares the same temperature and supply; his answer was the architecture, as on a chip.

**What was read.** For the 56 healthy Loyfer cell types, three numbers from each cell's own healthy array-site betas (`data/DEV_ARCH_01/arch_box.py`,
box job c9efb60a): the share of 100 kb domains held loosely (median beta below 0.70), the share of sites fully methylated (beta 0.80 or more), and
the number of identity sites. Set beside the healthy held error from PROC-CHANNEL-01 (`arch_vs_held.csv`, `arch_vs_held.png`; correlations `arch_vs_held_summary.txt`).
Nuclear mitochondrial-gene expression (Human Protein Atlas) showed no relation in a first look; not saved, so not counted here.

| | rank correlation with the held error |
|---|---|
| loosely held share | +0.87 |
| fully methylated share | -0.88 |
| identity sites | +0.05 |

**Limits.** The loosely held share comes from the same Loyfer molecules as the held error, and the held-error windows were not excluded
(the job could not find wgbstools). Part of the match can be built in. PROC-CHANNEL-01 section 7 already found that on the 26,800 sites every
cell keeps fully methylated the held error keeps the same order (rank agreement 0.87; lowest naive CD8 T cells, macrophages, B cells; highest
smooth muscle, heart fibroblasts, erythroid progenitors), which argues against a purely built-in match, but per-cell common-site values were
not set beside the loose share here.

**In plain words.** Cells whose design leaves more of the genome loosely held also hold their fully methylated sites less tightly. Whether
that is one architecture property, read independently, is the next look: the loose share measured on other laboratories' arrays of the same
cell types, and the held error on the common sites, cell by cell.
