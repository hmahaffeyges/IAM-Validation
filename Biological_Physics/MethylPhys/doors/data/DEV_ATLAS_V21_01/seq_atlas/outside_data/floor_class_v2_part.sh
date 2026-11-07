#!/bin/bash
# The class-floor synthetic version 2, one part per Google machine (PART 0 to 17: one laboratory setting and person, three runs of the one command). SOURCED
# by the launcher's job, which sets PART (W, R, B, log from the job). Saves as it goes: the part's rows go to the bucket the moment they exist; a
# relaunch that finds them stops. Each run's progress is logged, so the stall check sees work moving.
set -u; O=$B
if gcloud storage ls "$O/floor_class_v2_part${PART}.csv" > /dev/null 2>&1; then log "already read: part $PART is in the bucket"; return 0; fi
gcloud storage cp -q "${B%%/seq_atlas/*}/seq_atlas/base/GSM5652316.beta" $W/base.beta > /dev/null 2>&1; log "base $(stat -c %s $W/base.beta 2>/dev/null || echo missing)"
cd $R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas/outside_data
( python3 floor_class_synthetic_v2.py $W/base.beta $W/fc2 $W/fc_unused.txt $PART > $W/fc2_$PART.log 2>&1; echo $? > $W/fc_rc ) &
while [ ! -f $W/fc_rc ]; do sleep 240; log "part $PART: $(grep -c "'lab'" $W/fc2_$PART.log 2>/dev/null) of 3 runs done"; done
[ -f $W/fc2/floor_class_v2_part${PART}.csv ] && gcloud storage cp -q $W/fc2/floor_class_v2_part${PART}.csv $O/ > /dev/null 2>&1
gcloud storage cp -q $W/fc2_$PART.log $O/ > /dev/null 2>&1; log "part $PART rc $(cat $W/fc_rc): $(ls $W/fc2/floor_class_v2_part${PART}.csv 2>/dev/null || echo 'no rows')"
