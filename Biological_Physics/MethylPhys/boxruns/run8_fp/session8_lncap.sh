#!/bin/bash
# Box Run 8, stage 2 (doors/DEV_FINGERPRINT_01.md): the 5 LNCaP WGBS runs of GSE86833 (SRR4238609-13) through the same pinned pipeline as stage 1
# (session2.sh unchanged, 25,000,000 read pairs per run). Started after both arms were sealed on PrEC (commit f4fce17).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS="SRR4238609 SRR4238610 SRR4238611 SRR4238612 SRR4238613" OUTP=results/BOXRUN8_FINGERPRINT/lncap OUTD=out8lncap \
N_PAIRS=${N_PAIRS:-25000000} THREADS=${THREADS:-40} SKIP_FORMAT=1 bash $HERE/../run2/session2.sh
