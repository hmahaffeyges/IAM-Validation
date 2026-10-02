# DEV-COLON-BLOCKS-02 — colon-epithelium marker regions against every Loyfer cell type (development, 2026-10-02)

**Question.** For the stool route (IAM-A and Met-A on colon cells shed into stool), are there regions where colon epithelium is methylated and every other
cell type is not, so that a qualifying molecule (≥ 6 CpGs, ≥ 80 % methylated) read there can only be a colon-epithelial molecule?

**Run.** Loyfer et al. 2023 WGBS atlas (hg19 .beta, coverage ≥ 10): colon epithelium (left + right, 5 samples) against ALL other cell groups, including
stomach, small intestine, oesophageal squamous, liver, pancreas, blood, fibroblasts and macrophages. Marker CpG: colon mean ≥ 0.80 and every other group
≤ 0.10 (strict) or ≤ 0.20 (loose). Region: ≥ 4 marker CpGs, gaps ≤ 100 bp. Lifted to GRCh38.

**Result.** Strict: 1 region (chr14, 6 CpGs). Loose: 5 regions, 40 CpGs. The earlier build against blood and colon stroma only found many more; most of
those are shared with other gut epithelia.

**What it means.** Methylated-in-colon-only regions are too few to collect enough qualifying molecules from stool DNA for a per-person reading. Next
options: (1) the reverse polarity, regions unmethylated in colon and methylated elsewhere, reading the unmethylated channel only where the instrument
allows; (2) accept "lower gastrointestinal epithelium" as the cell identity (colon + small intestine), which stool sampling largely selects anyway;
(3) use the matched tumour/normal reads already aligned (PROC-TUMOUR-01) to test which regions carry the tumour's copy-error rise.
