#!/bin/bash
# Box Run 2 session 5 (doors/DEV_LINK_IAMA_METAA_02.md): HCT116 decitabine dose series, EM-seq, through the pinned pipeline.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS="SRR25322252 SRR25322251 SRR25322250 SRR25322249 SRR25322248 SRR25322247" \
OUTP=results/BOXRUN2/session5 OUTD=out5dac N_PAIRS=${N_PAIRS:-25000000} THREADS=${THREADS:-120} SKIP_FORMAT=1 bash $HERE/session2.sh
