set -e
DEV=$(lsblk -dpno NAME,SIZE | awk '$2=="500G"{print $1}' | head -1); sudo mkdir -p /mnt/scratch; mountpoint -q /mnt/scratch || sudo mount "$DEV" /mnt/scratch
df -h /mnt/scratch | tail -1; ls /mnt/scratch/run2 | head -20 | tr '\n' ' '; echo; nproc
cd ~/IAM-Validation && git pull -q origin main && git log --oneline -1 | cut -c1-60
printf '%s\n' 'cd ~/IAM-Validation' '( N_PAIRS=25000000 THREADS=120 timeout 7h bash Biological_Physics/MethylPhys/boxruns/run2/session2.sh; echo "session2 exit $? at $(date -u)" ) > ~/run2_s2.log 2>&1' 'echo "done at $(date -u); shutting down" >> ~/run2_s2.log' 'sudo shutdown -h now' > ~/w5.sh
nohup setsid bash ~/w5.sh > ~/w5.log 2>&1 < /dev/null &
sleep 20; tail -3 /mnt/scratch/run2/out2/session2.log 2>/dev/null; echo launched