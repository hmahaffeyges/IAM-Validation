#!/bin/bash
# Work machine job list 01 (2026-10-12). Each result is saved to S3 as soon as it exists; a restarted machine resumes from S3.
#  1. DEV-IAMA-NEURON-01: the 10 Loyfer neuron .pat files read whole (neuron_eps_01.py), one file at a time.
#  2. DEV-TISSUE-COLON-02: the 152 non-cancer GSE199057 arrays fetched (fetch_colon_02.py, tumour lock) and calibrated (calib_colon_02.py).
#  3. DEV-TISSUE-COLON-02 synthetic tests 1-6 (colon02_synthetic.py) on those normals; atlas v2 posteriors from S3 seed/atlpost.
# Environments pinned to the laptop's: Stage Q py3.11 numpy 2.4.6 pandas 2.3.3; Stage 1 py3.11 methylprep 1.7.1 numpy 1.26.4 pandas 1.5.3.
set -u
NAME=$1; B=methylphys-data-945451304272-us-west-2-an; P=results/$NAME; H=~/IAM-Validation/Biological_Physics/MethylPhys; W=~/wb; mkdir -p $W
MM="/mnt/scratch/run2/bin/micromamba -r /mnt/scratch/run2/mamba"
$MM create -y -q -p $W/eq -c conda-forge python=3.11 > $W/env.log 2>&1 && $W/eq/bin/pip -q install numpy==2.4.6 pandas==2.3.3 scipy==1.17.1 pyarrow==25.0.1 boto3 requests >> $W/env.log 2>&1
PY=$W/eq/bin/python
up(){ $PY -c "import boto3,sys;boto3.client('s3').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2"; }
get(){ $PY -c "import boto3,sys;boto3.client('s3').download_file('$B',sys.argv[1],sys.argv[2])" "$1" "$2" 2>/dev/null; }
log(){ echo "$(date -u +%FT%TZ) $*" >> $W/session.log; up $W/session.log $P/session.log; }
echo "$NAME" > $W/s3_check.txt; up $W/s3_check.txt $P/s3_check.txt || { echo "no S3 access" >&2; exit 3; }   # results must be able to leave first
log "START $NAME vcpus=$(nproc) mem_gb=$(free -g | awk '/Mem/{print $2}')"
$MM create -y -q -p $W/e1 -c conda-forge python=3.11 >> $W/env.log 2>&1 && $W/e1/bin/pip -q install methylprep==1.7.1 numpy==1.26.4 pandas==1.5.3 scipy==1.17.1 pyarrow==25.0.1 requests >> $W/env.log 2>&1
up $W/env.log $P/env.log; log "environments ready"
# 1. neurons
get $P/neuron_eps_01.csv $W/neuron_eps_01.csv; mkdir -p $W/neu
for x in GSM5652223_Cortex-Neuron-Z000000TF GSM5652224_Neuron-Z000000TH GSM5652225_Cortex-Neuron-Z0000042F GSM5652226_Cortex-Neuron-Z0000042H GSM5652227_Cortex-Neuron-Z0000042J GSM5652228_Cortex-Neuron-Z0000042M GSM5652229_Cortex-Neuron-Z0000042P GSM5652230_Cortex-Neuron-Z0000042K GSM5652231_Cerebellum-Neuron-Z000000TB GSM5652232_Cortex-Neuron-Z000000TD; do
  grep -q "^$x," $W/neuron_eps_01.csv 2>/dev/null && continue
  g=${x%%_*}; curl -s --retry 5 -o $W/neu/$x.pat.gz "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5652nnn/$g/suppl/$x.pat.gz" || { log "FETCH_FAIL $x"; continue; }
  (cd $H && $PY doors/data/DEV_IAMA_NEURON_01/neuron_eps_01.py $H/chain $W/neu $W/neuron_eps_01.csv) >> $W/neu.log 2>&1 || log "READ_FAIL $x"
  up $W/neuron_eps_01.csv $P/neuron_eps_01.csv; rm -f $W/neu/$x.pat.gz; log "neuron $x done"
done
log "neurons END $(($(wc -l < $W/neuron_eps_01.csv)-1)) rows"
# 2. colon normals
mkdir -p $W/colon/idat $W/colon/betas $W/mph/.methylprep_manifest_files
get seed/methylprep_manifests/MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz $W/mph/.methylprep_manifest_files/MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz
$PY - "$B" "$P/colon_betas/" "$W/colon/betas" <<'PY2'
import boto3, sys, os
s3 = boto3.client("s3"); b, pre, d = sys.argv[1:4]
for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=pre):
    for o in pg.get("Contents", []): s3.download_file(b, o["Key"], os.path.join(d, os.path.basename(o["Key"])))
PY2
(cd $H/doors/data/DEV_TISSUE_COLON_02 && $PY fetch_colon_02.py $W/colon/idat normal > $W/fetch.log 2>&1); log "colon fetch END $(ls $W/colon/idat | wc -l) files"
(cd $H/doors/data/DEV_TISSUE_COLON_02 && HOME=$W/mph $W/e1/bin/python calib_colon_02.py $W/colon/idat $W/colon/betas > $W/calib.log 2>&1) & C=$!
sync_up(){ for f in $W/colon/betas/*.parquet; do [ -e "$f.up" ] || { up "$f" "$P/colon_betas/$(basename $f)" && touch "$f.up"; }; done; }
while kill -0 $C 2>/dev/null; do sync_up; sleep 60; done; sync_up; up $W/calib.log $P/calib.log
log "colon calib END $(ls $W/colon/betas/*.parquet | wc -l) arrays"
# 3. colon synthetic tests (normals only)
mkdir -p $W/atlpost
$PY - "$B" "seed/atlpost/" "$W/atlpost" <<'PY3'
import boto3, sys, os
s3 = boto3.client("s3"); b, pre, d = sys.argv[1:4]
for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=pre):
    for o in pg.get("Contents", []): s3.download_file(b, o["Key"], os.path.join(d, os.path.basename(o["Key"])))
PY3
D2=$H/doors/data/DEV_TISSUE_COLON_02
(cd $D2 && $PY colon02_synthetic.py $W/atlpost $W/colon/betas > $W/synth.log 2>&1) || log "SYNTH_FAIL"
for f in colon02_synthetic_output.csv colon02_synthetic_summary.txt; do [ -f $D2/$f ] && up $D2/$f $P/$f; done; up $W/synth.log $P/synth.log
log "colon synthetic END"
log "DONE"
