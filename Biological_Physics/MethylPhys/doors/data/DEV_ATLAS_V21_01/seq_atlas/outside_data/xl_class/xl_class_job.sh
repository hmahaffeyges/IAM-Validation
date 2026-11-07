#!/bin/bash
# The cross-laboratory rehearsal on spent data with the class floor (DEV_ATLAS_V21_01.md): the 89 outside samples already read, read again by
# the one command with --dev-floor-class. SOURCED by the launcher's job (W, R, B, log, up from the preamble), which sets ROUTE and:
#   ROUTE=merge  GSM, N, SRC (the bucket prefix holding its saved chunks, e.g. seq_atlas/bt3), LST (xl_bt3_samples.csv or xl_jhu_samples.csv):
#                the saved chunks copied in, then split_google.sh's merge (the merged molecules saved as it goes), the bundle to $B/out/
#   ROUTE=bp     FIRST, LAST: those rows of xl_bp_samples.csv through blueprint_sites_read.sh (each bundle uploaded as it is read)
#   ROUTE=lab2   FIRST, LAST: those rows of xl_lab2_samples.csv through xl_lab2_read.sh (each bundle uploaded as it is read)
# A sample whose bundle is already in the bucket is not read again.
set -u; XC=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas/outside_data/xl_class; ROOT=${B%%/seq_atlas/*}
if [ "$ROUTE" = merge ]; then
  if gcloud storage ls "$B/out/${GSM}_bundle.json" > /dev/null 2>&1; then log "already read: $GSM"; return 0; fi
  gcloud storage cp -q "$ROOT/$SRC/chunks/${GSM}_c*_${N}.pat.gz" "$B/chunks/" > /dev/null 2>&1; log "chunks copied: $(gcloud storage ls "$B/chunks/${GSM}_c*_${N}.pat.gz" 2>/dev/null | wc -l) of $N"
  LIST=Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas/outside_data/xl_class/$LST; MODE=merge; K=merge; CKPT_ON=1
  source $R/Biological_Physics/MethylPhys/doors/data/DEV_ERASE_01/split_google.sh
  return 0
fi
if [ "$ROUTE" = bp ]; then SRCL=$XC/xl_bp_samples.csv; else SRCL=$XC/xl_lab2_samples.csv; fi
python3 - "$SRCL" "$FIRST" "$LAST" "$W/rows.csv" <<'PY'
import csv, sys
src, a, b, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
R = list(csv.DictReader(open(src))); w = csv.DictWriter(open(out, "w", newline=""), fieldnames=list(R[0].keys())); w.writeheader()
for r in R[a - 1:b]: w.writerow(r)
PY
python3 - "$W/rows.csv" "$W/todo.csv" "$B" <<'PY'
import csv, sys, subprocess
R = list(csv.DictReader(open(sys.argv[1]))); keep = [r for r in R if subprocess.run(["gcloud", "storage", "ls", f"{sys.argv[3]}/{r['gsm']}_bundle.json"], capture_output=True).returncode != 0]
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=list(R[0].keys())); w.writeheader(); [w.writerow(r) for r in keep]; print(len(keep))
PY
log "rows $FIRST to $LAST: $(($(wc -l < $W/todo.csv) - 1)) still to read"
if [ "$ROUTE" = bp ]; then source $R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas/outside_data/blueprint_sites_read.sh $W/todo.csv
else export -f log up; export W R B L; bash $XC/xl_lab2_read.sh $W/todo.csv; fi   # a child shell sees the preamble's log and up only when exported
log "rows $FIRST to $LAST done"
