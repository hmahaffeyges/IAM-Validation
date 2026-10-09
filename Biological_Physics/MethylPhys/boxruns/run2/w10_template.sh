cd ~/IAM-Validation && git pull -q origin main; cd ~
timeout 9h bash ~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run2/session4.sh > ~/s4.log 2>&1; echo "s4 exit $? $(date -u)" >> ~/s3.log
timeout 5h bash ~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run2/session5.sh > ~/s5.log 2>&1; echo "s5 exit $? $(date -u)" >> ~/s3.log
while ! grep -q CALIB_DONE ~/k450/calib.log 2>/dev/null && pgrep -f "[c]alib450" >/dev/null; do sleep 60; done
sudo shutdown -h now
