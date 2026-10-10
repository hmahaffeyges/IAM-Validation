J=hpc/a38896ab-4a31-40d9-8aec-2bfe19aedcb2; mkdir -p colon && cp $J/colon_epi_M_blocks_all.csv colon/ && cat > colon/DEV_COLON_BLOCKS_02.md <<'EOF'
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
EOF
cd iamrepo && cp ../colon/DEV_COLON_BLOCKS_02.md Biological_Physics/MethylPhys/doors/ && cp ../colon/colon_epi_M_blocks_all.csv Biological_Physics/MethylPhys/doors/data/ && U="https://x-access-token:${GITHUB_TOKEN}@github.com/hmahaffeyges/IAM-Validation.git"; git add Biological_Physics/MethylPhys/doors && git -c user.name="Heath W. Mahaffey" -c user.email="hmahaffeyges@users.noreply.github.com" commit -q -m "DEV-COLON-BLOCKS-02: colon-epithelium methylated-only regions against all Loyfer cell types: 1 strict (6 CpGs), 5 loose (40 CpGs); too few for stool IAM-A as designed" && bash CANON/checked_push.sh -q "$U" HEAD:main 2>&1 | grep -iE "error|rejected|FAILED|blocked"; git log --oneline -1