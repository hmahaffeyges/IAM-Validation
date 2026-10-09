set -e
DEV=$(lsblk -dpno NAME,SIZE | awk '$2=="600G"{print $1}' | head -1); echo "scratch $DEV"; [ -n "$DEV" ]
sudo mkfs.ext4 -q -F "$DEV"; sudo mkdir -p /mnt/scratch; sudo mount "$DEV" /mnt/scratch; sudo chown ubuntu:ubuntu /mnt/scratch
cd ~/IAM-Validation && git pull -q origin main && git log --oneline -1 | cut -c1-50
cp Biological_Physics/MethylPhys/boxruns/run3/run3.py ~/run3.py
printf '%s\n' 'cd ~' '( /home/ubuntu/env/bin/python run3.py > ~/run3.log 2>&1; echo "run3 exit $?" >> ~/run3.log; rm -rf ~/run3/*/idat ~/run3/*/*.tar ) &' 'P1=$!' '( THREADS=112 bash ~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run2/session3.sh > ~/s3.log 2>&1; echo "s3 exit $?" >> ~/s3.log ) &' 'P2=$!' 'wait $P1; wait $P2' 'echo "both done $(date -u)" >> ~/s3.log' 'sudo shutdown -h now' > ~/w6.sh
nohup setsid bash ~/w6.sh > ~/w6.log 2>&1 < /dev/null &
sleep 30; tail -2 ~/run3.log; tail -2 /mnt/scratch/run2/out3/session3.log 2>/dev/null; echo launched
