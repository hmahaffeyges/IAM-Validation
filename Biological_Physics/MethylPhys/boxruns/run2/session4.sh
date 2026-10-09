#!/bin/bash
# Box Run 2 session 4 (doors/DEV_IAMA_WBTARE_01.md): laboratory-G whole blood, 4 donors x Swift and TruSeq (HiSeq X) + repeat libraries.
# Needs the tools and index restored as in session 3 (run session3.sh's tool and restore steps first, or reuse the disk).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS="SRR9888303 SRR9888304 SRR9888308 SRR9888310 SRR9888315 SRR9888317 SRR9888322 SRR9888324 SRR9888309 SRR9888311 SRR9888316 SRR9888318 SRR9888323 SRR9888325" \
OUTP=results/BOXRUN2/session4 OUTD=out4wb N_PAIRS=${N_PAIRS:-25000000} THREADS=${THREADS:-120} SKIP_FORMAT=1 bash $HERE/session2.sh
