#!/bin/bash
# DEVELOPMENT - not commissioned. DEV-EPIC-V2-01 (2026-10-04): SeSAMe in its own env on the box, then EPIC v2 / v1 arrays of GSE286313.
set -eo pipefail
W=$(pwd); ENV=/home/ubuntu/data/sesame_env
MM=$(command -v micromamba || true); [ -z "$MM" ] && MM=$(ls /home/ubuntu/bin/micromamba 2>/dev/null || find /home/ubuntu -maxdepth 5 -name micromamba -type f 2>/dev/null | head -1)
echo "micromamba: $MM"
if [ ! -x $ENV/bin/Rscript ]; then
  $MM create -y -p $ENV -c conda-forge -c bioconda r-base=4.3 bioconductor-sesame bioconductor-sesamedata bioconductor-experimenthub > $W/mm_create.log 2>&1 || { tail -40 $W/mm_create.log; exit 1; }
fi
$ENV/bin/Rscript -e 'suppressPackageStartupMessages(library(sesame)); cat("sesame", as.character(packageVersion("sesame")), "\n"); sesameData::sesameDataCache()' > $W/sesame_cache.log 2>&1 || { tail -30 $W/sesame_cache.log; exit 1; }
tail -3 $W/sesame_cache.log
/home/ubuntu/env/bin/python $W/r2_epicv2.py $ENV/bin/Rscript
ls -lh $W/v2out || true
