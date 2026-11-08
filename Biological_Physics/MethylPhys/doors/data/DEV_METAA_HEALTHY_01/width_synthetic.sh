#!/bin/bash
# The width synthetic, one part (one cell, one rho, five people) on a Google machine; SOURCED by the launcher's job (W, R, B, log, up given; CI,
# RI, FIRST set). Saves its part to the bucket; a part already there is not read again.
set -u; D=$R/Biological_Physics/MethylPhys/doors/data/DEV_METAA_HEALTHY_01; F=width_synthetic_c${CI}_r${RI}_p${FIRST}.json
if gcloud storage ls $B/$F.done > /dev/null 2>&1; then log "already read: $F"; return 0 2>/dev/null || exit 0; fi
( cd $D && python3 width_synthetic.py $W/ws $CI $RI $FIRST > $W/ws.log 2>&1; echo $? > $W/ws_rc ) &
while [ ! -f $W/ws_rc ]; do sleep 240; [ -f $W/ws/$F ] && gcloud storage cp -q $W/ws/$F $B/ > /dev/null 2>&1; done
[ -f $W/ws/$F ] && gcloud storage cp -q $W/ws/$F $B/ > /dev/null 2>&1; mv $W/ws.log $W/ws_c${CI}_r${RI}_p${FIRST}.log; gcloud storage cp -q $W/ws_c${CI}_r${RI}_p${FIRST}.log $B/ > /dev/null 2>&1
[ "$(cat $W/ws_rc)" = 0 ] && [ "$(grep -c '"rc": 0' $W/ws/$F 2>/dev/null)" = 5 ] && { echo done > $W/d; gcloud storage cp -q $W/d $B/$F.done > /dev/null 2>&1; }
log "width synthetic c$CI r$RI p$FIRST rc $(cat $W/ws_rc)"
