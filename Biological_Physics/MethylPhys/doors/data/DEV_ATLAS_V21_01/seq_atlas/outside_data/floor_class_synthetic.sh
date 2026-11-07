#!/bin/bash
# The floor counted at the islands read: its synthetic on a Google machine. SOURCED by the launcher's job (W, R, B, BUCKET, log come from it;
# return, not exit). Saves as it goes: the summary and readings go to the bucket the moment they exist; a relaunch that finds the summary stops.
set -u; O=$B/floor_class
if gcloud storage ls "$O/floor_class_synthetic.txt" > /dev/null 2>&1; then log "already read: the summary is in the bucket"; return 0; fi
gcloud storage cp -q "gs://$BUCKET/seq_atlas/base/GSM5652316.beta" $W/base.beta > /dev/null 2>&1; log "base $(stat -c %s $W/base.beta)"
cd $R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas/outside_data
python3 floor_class_synthetic.py $W/base.beta $W/fc $W/floor_class_synthetic.txt > $W/fc.log 2>&1
for x in $W/floor_class_synthetic.txt $W/fc/floor_class_readings.csv $W/fc.log; do [ -f $x ] && gcloud storage cp -q $x $O/; done
log "synthetic: $(tail -1 $W/floor_class_synthetic.txt 2>/dev/null)"
