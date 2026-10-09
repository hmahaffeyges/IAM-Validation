for p in "[w]9.sh" "[w]10.sh" "boxruns/run2/[s]ession4" "boxruns/run2/[s]ession2" "[f]astq-dump" "[b]wameth" "[s]amtools"; do pkill -f "$p"; done; sleep 3
echo "left: $(pgrep -fa 'run2/[s]ession' | wc -l)"
cd /mnt/scratch/run2
grep -q "Written 25000000 spots" out4wb/SRR9888303_fastq.log 2>/dev/null && touch reads2/SRR9888303.complete
for f in reads2/SRR98883*; do b=$(basename $f); r=${b%%_*}; r=${r%%.*}; [ -f reads2/$r.complete ] || rm -f $f; done; rm -rf sra/*; rm -f out4wb/fastq_done.txt
echo "kept: $(ls reads2 | grep -E 'SRR98883' | tr '\n' ' ')"
cd ~/IAM-Validation && git pull -q origin main && git log --oneline -1 | cut -c1-40; cd ~
cp ~/w10_template.sh ~/w10.sh
nohup setsid bash ~/w10.sh > ~/w10.log 2>&1 < /dev/null &
sleep 25; echo "w10 $(pgrep -f '[w]10.sh' | wc -l) calib $(pgrep -f '[c]alib450' | wc -l)"; tail -2 /mnt/scratch/run2/out4wb/session2.log | cut -c1-150; tail -2 ~/k450/calib.log
