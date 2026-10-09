set -e
DEV=$(lsblk -dpno NAME,SIZE | awk '$2=="500G"{print $1}' | head -1); echo "scratch $DEV"; [ -n "$DEV" ]
sudo blkid "$DEV" >/dev/null 2>&1 || sudo mkfs.ext4 -q -F "$DEV"
sudo mkdir -p /mnt/scratch; mountpoint -q /mnt/scratch || sudo mount "$DEV" /mnt/scratch; sudo chown ubuntu:ubuntu /mnt/scratch; df -h /mnt/scratch | tail -1
/home/ubuntu/env/bin/python -c 'import boto3;print("identity:",boto3.client("sts",region_name="us-west-2").get_caller_identity()["Arn"].split(":")[-1]);boto3.client("s3",region_name="us-west-2").list_objects_v2(Bucket="methylphys-data-945451304272-us-west-2-an",Prefix="results/BOXRUN1/",MaxKeys=1);print("S3 ok")'
cd ~/IAM-Validation && git checkout -q -- . && git pull -q origin main && git log --oneline -1
/home/ubuntu/env/bin/python -c 'import numpy,pandas,scipy,healpy,methylprep,boto3;pandas.read_parquet;print("env ok numpy",numpy.__version__,"pandas",pandas.__version__)'
cat > ~/day1_wrap.sh <<'W'
cd ~/IAM-Validation
( timeout 8h /home/ubuntu/env/bin/python Biological_Physics/MethylPhys/boxruns/run1/run1_driver.py --work /mnt/scratch/boxrun1 --python /home/ubuntu/env/bin/python --workers 8 --only B,E --force B --force E --job-e-prefix downloads/G_chain_tests/jobE_GSE112618/ --job-e-prefix downloads/G_chain_tests/healthy_repeat/GSE182379/ --no-shutdown; echo "run1 driver exit $? at $(date -u)" ) > ~/run1_day1.log 2>&1 &
P1=$!
( timeout 7h bash Biological_Physics/MethylPhys/boxruns/run2/session1.sh; echo "run2 session1 exit $? at $(date -u)" ) > ~/run2_s1.log 2>&1 &
P2=$!
wait $P1; wait $P2
echo "both done at $(date -u); shutting down"
sudo shutdown -h now
W
nohup setsid bash ~/day1_wrap.sh > ~/day1_wrap.log 2>&1 < /dev/null &
sleep 40; tail -3 ~/run1_day1.log; tail -3 /mnt/scratch/run2/out/session1.log 2>/dev/null; echo "launched at $(date -u)"
