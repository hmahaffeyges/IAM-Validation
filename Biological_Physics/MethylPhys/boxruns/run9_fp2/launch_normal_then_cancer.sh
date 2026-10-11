#!/bin/bash
# Box Run 9 launcher as run 2026-10-11 03:33 UTC (m7a.32xlarge): normal side, then the cancer side ONLY after ~/GO_CANCER exists (created by
# the lead once every DEV-FINGERPRINT-02 pair's seal is committed), else shut down after 60 min. The box shuts itself down at the end.
H=~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run9_fp2
SET=normal THREADS=120 timeout 6h bash $H/session9.sh > /mnt/scratch/run2/s9d.out 2>&1
echo "s9 normal (d) exit $? $(date -u)" >> ~/s3.log
for i in $(seq 1 360); do [ -f ~/GO_CANCER ] && break; sleep 10; done
if [ -f ~/GO_CANCER ]; then
  cd ~/IAM-Validation && git pull -q origin main
  SET=cancer THREADS=120 timeout 8h bash $H/session9.sh > /mnt/scratch/run2/s9cancer.out 2>&1
  echo "s9 cancer exit $? $(date -u)" >> ~/s3.log
else echo "no GO_CANCER within 60 min $(date -u)" >> ~/s3.log; fi
sleep 120; sudo shutdown -h now
