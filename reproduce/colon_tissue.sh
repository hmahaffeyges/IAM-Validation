#!/bin/bash
# The colorectal tissue result (DEV-TISSUE-COLON-01) from public files: IAM-Atlas v2 from its Zenodo record (checked file by file), the posterior
# of the nine cells the sealed rule reads, the GSE131013 IDATs from GEO's series archive, chain Stage 1, then the sealed scorer unchanged.
# Run by reproduce.py (REPO, WORK, PY, MP set).
set -euo pipefail; D=$MP/doors/data/DEV_TISSUE_COLON_01; mkdir -p $WORK/idat $WORK/post
$PY $MP/atlas/tools/fetch_atlas.py $WORK/atlas_v2 23325865
BLOCKS_DIR=$WORK/atlas_v2/blocks_v2 $PY $MP/atlas/tools/extract_posterior.py $WORK/post "colon epithelium" "colon fibroblasts" "colon macrophages" \
  "smooth muscle" "vascular endothelium" "cd4 t cells" "cd8 t cells" "b cells" "neutrophils" "adipocytes"
rm -rf $WORK/atlas_v2
[ -s $WORK/GSE131013_RAW.tar ] || curl -sfL --retry 5 -o $WORK/GSE131013_RAW.tar https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131013/suppl/GSE131013_RAW.tar
$PY - "$WORK" "$D" <<'PY'
import sys, os, json, tarfile, pandas as pd
W, D = sys.argv[1:3]; S = pd.read_csv(os.path.join(D, "colon_samples.csv"))
want = set(S[S.tissue.isin(["Mucosa", "Normal"])].gsm) | {p["tumour_gsm"] for p in json.load(open(os.path.join(D, "sealed_colon_tissue.json")))["pairs"]}
n = 0
with tarfile.open(os.path.join(W, "GSE131013_RAW.tar")) as t:
    for m in t.getmembers():
        b = os.path.basename(m.name)
        if b.split("_")[0] in want and ".idat" in b: m.name = b; t.extract(m, os.path.join(W, "idat")); n += 1
print("IDAT files extracted", n, "for", len(want), "arrays")
PY
rm -f $WORK/GSE131013_RAW.tar
$PY $D/calib_colon.py $WORK normal; $PY $D/calib_colon.py $WORK tumour
$PY $D/score_colon_tissue.py score $WORK/post $WORK/betas $WORK/betas
