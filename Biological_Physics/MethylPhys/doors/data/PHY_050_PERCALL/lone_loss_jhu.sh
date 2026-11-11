#!/bin/bash
# PHY-050 second test (2026-11-10): the six Johns Hopkins tissue samples labeled normal (GSE86340; merged molecule files already in the build bucket,
# spent, no alignment), each counted by the one command (run_sample.py --pat --dev-count-only, chain Stage Q2). ROW (0 to 5) picks the row of
# lone_loss_jhu_samples.csv; the bundle is put in the bucket as written; a sample already counted is skipped.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SD=$R/Biological_Physics/MethylPhys/doors/data/PHY_050_PERCALL
ROOT=${B%%/seq_atlas/*}; O=$B/lone_loss_jhu; mkdir -p $W/ll
IFS=, read -r G PERSON ORGAN PAT < <(sed -n "$((ROW + 2))p" $SD/lone_loss_jhu_samples.csv)
gcloud storage ls "$O/$G.json" > /dev/null 2>&1 && { log "already read: $G"; exit 0; }
gcloud storage cp -q "$ROOT/$PAT" $W/ll/$G.pat.gz > /dev/null 2>&1 || { log "$G: pat not copied"; exit 1; }
(cd $IF && python3 run_sample.py --pat $W/ll/$G.pat.gz --specimen tissue --id $G --dev-count-only --out $W/ll/$G.html --bundle $W/ll/$G.json > $W/ll/$G.log 2>&1) \
  && gcloud storage cp -q $W/ll/$G.json $O/ > /dev/null 2>&1 && log "counted $G" || log "$G FAILED: $(tail -3 $W/ll/$G.log | tr '\n' ' ' | cut -c1-250)"
