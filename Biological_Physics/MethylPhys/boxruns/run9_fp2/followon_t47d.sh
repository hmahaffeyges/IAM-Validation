#!/bin/bash
# Box Run 9 follow-on (2026-10-11, author: get the most from each box session): after launch_mcf7.sh's alignment ends, stop its pending shutdown and
# align the 4 T47D files in the same session (files already in S3 skipped), then shut down. Aligning is not reading: no T47D file is read until its
# pair is sealed. Waits at most 5 h.
H=~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run9_fp2
for i in $(seq 1 1800); do grep -q "s9 MCF-7 exit" ~/s3.log && break; sleep 10; done
pkill -f r9m.sh; sudo shutdown -c 2>/dev/null
SKIP_CELLS=NONE SET=cancer THREADS=120 timeout 3h bash $H/session9.sh > /mnt/scratch/run2/s9t47d.out 2>&1
echo "s9 T47D exit $? $(date -u)" >> ~/s3.log
sleep 120; sudo shutdown -h now
