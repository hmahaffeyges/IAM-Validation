#!/bin/bash
# Box Run 9, MCF-7 (2026-10-11, after both breast pairs were sealed under rule 02b): cancer side with only T47D skipped; files already in S3
# are skipped by session9.sh. The box shuts itself down at the end.
H=~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run9_fp2
cd ~/IAM-Validation && git pull -q origin main
SKIP_CELLS=T47D SET=cancer THREADS=120 timeout 6h bash $H/session9.sh > /mnt/scratch/run2/s9mcf7.out 2>&1
echo "s9 MCF-7 exit $? $(date -u)" >> ~/s3.log
sleep 120; sudo shutdown -h now
