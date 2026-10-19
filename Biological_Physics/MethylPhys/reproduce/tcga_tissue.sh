#!/bin/bash
# The breast and liver tissue results (DEV-TISSUE-BREAST-01, DEV-TISSUE-LIVER-01) from public files: IAM-Atlas v2 from its Zenodo record (checked
# file by file), the posterior of the cells each sealed rule reads, TCGA open-access 450K IDATs from the NCI GDC (no login; md5 checked by the
# committed fetch script), chain Stage 1, then the sealed scorer unchanged. Usage (by reproduce.py): bash Biological_Physics/MethylPhys/reproduce/tcga_tissue.sh breast|liver
set -euo pipefail; T=${1:?breast|liver}; mkdir -p $WORK/post
case $T in
  breast) D=$MP/doors/data/DEV_TISSUE_BREAST_01; FETCH=fetch_tcga_brca.py; CAL=calib_tcga.py; SC=score_breast_tissue.py
          CELLS=("adipocytes" "b cells" "breast basal epithelium" "breast luminal epithelium" "cd4 t cells" "cd8 t cells" "colon fibroblasts" "colon macrophages" "heart fibroblasts" "vascular endothelium");;
  liver)  D=$MP/doors/data/DEV_TISSUE_LIVER_01; FETCH=fetch_tcga_lihc.py; CAL=calib_tcga_lihc.py; SC=score_liver_tissue.py
          CELLS=("adipocytes" "b cells" "cd4 t cells" "cd8 t cells" "colon fibroblasts" "colon macrophages" "heart fibroblasts" "hepatocyte" "monocytes" "neutrophils" "nk cells" "smooth muscle" "vascular endothelium");;
esac
$PY $MP/atlas/tools/fetch_atlas.py $WORK/atlas_v2 23325865
BLOCKS_DIR=$WORK/atlas_v2/blocks_v2 $PY $MP/atlas/tools/extract_posterior.py $WORK/post "${CELLS[@]}"
rm -rf $WORK/atlas_v2
for w in normal tumour; do $PY $D/$FETCH fetch $WORK $w; $PY $D/$CAL $WORK $w; done
$PY $D/$SC score $WORK/post $WORK/betas $WORK/betas
