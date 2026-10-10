#!/bin/bash
# Box Run 8, stage 1 (doors/DEV_FINGERPRINT_01.md): the 4 PrEC WGBS runs of GSE86833 (normal prostate epithelium; SRR4238614-17) through the
# pinned pipeline of Box Run 2 (session2.sh unchanged: bwa-meth 0.2.0, SAMtools 1.9, sambamba markdup, wgbstools 0.1.0 bam2pat, hg19),
# 25,000,000 read pairs per run as the decitabine series. The bundle's specimen label ("isolated neutrophils") is session2.sh's default and is not
# used: DEV-FINGERPRINT-01 is scored from the .pat files. LNCaP (SRR4238609-13) is stage 2, run only after the windows are sealed on PrEC.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS="SRR4238614 SRR4238615 SRR4238616 SRR4238617" OUTP=results/BOXRUN8_FINGERPRINT/prec OUTD=out8prec \
N_PAIRS=${N_PAIRS:-25000000} THREADS=${THREADS:-40} SKIP_FORMAT=1 bash $HERE/../run2/session2.sh
