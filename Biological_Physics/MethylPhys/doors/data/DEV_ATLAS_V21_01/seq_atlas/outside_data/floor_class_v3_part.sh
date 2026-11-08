#!/bin/bash
# Class-floor synthetic version 3, one part (one atlas donor at one laboratory setting) on a Google machine; SOURCED by the launcher's job
# (W, R, B, log, up given; PART 0 to 17). Saves its part to the bucket after every reading; a part already there is not read again.
set -u; O=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas/outside_data
P=$((PART / 6)); LK=$((PART % 6)); G=(GSM5652316 GSM5652318 GSM5652317); H=(B B A); GSM=${G[$P]}; HALF=${H[$P]}
if gcloud storage ls $B/floor_class_v3_part$PART.done > /dev/null 2>&1; then log "already read: part $PART"; return 0 2>/dev/null || exit 0; fi
U=$(grep "^$GSM," $R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas/seq_atlas_inputs.csv | awk -F, '{print $NF}')
for i in 1 2 3 4 5; do curl -sSfL --retry 5 -o $W/base.beta.gz "$U" && break; sleep 30; done
case "$U" in *.gz) gunzip -f $W/base.beta.gz;; *) mv $W/base.beta.gz $W/base.beta;; esac
log "part $PART: $GSM against half $HALF, laboratory $LK; base $(stat -c %s $W/base.beta) bytes"
( cd $O && python3 floor_class_synthetic_v3.py $W/base.beta $HALF $GSM $LK $W/w $W/floor_class_v3_part$PART.csv > $W/fc3.log 2>&1; echo $? > $W/fc3_rc ) &
while [ ! -f $W/fc3_rc ]; do sleep 120; [ -f $W/floor_class_v3_part$PART.csv ] && gcloud storage cp -q $W/floor_class_v3_part$PART.csv $B/ > /dev/null 2>&1; done
gcloud storage cp -q $W/floor_class_v3_part$PART.csv $W/fc3.log $B/ > /dev/null 2>&1; mv $W/fc3.log $W/fc3_part$PART.log; gcloud storage cp -q $W/fc3_part$PART.log $B/ > /dev/null 2>&1
[ "$(cat $W/fc3_rc)" = 0 ] && { echo done > $W/d; gcloud storage cp -q $W/d $B/floor_class_v3_part$PART.done > /dev/null 2>&1; }
log "part $PART rc $(cat $W/fc3_rc)"
