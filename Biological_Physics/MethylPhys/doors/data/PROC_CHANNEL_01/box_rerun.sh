#!/usr/bin/env bash
# PROC-CHANNEL-01 rerun on the box (2026-10-10). Takes every file from origin/main by `git show` into its own folder, so the repo the
# running sessions use is not touched. Outputs to s3://methylphys-data-945451304272-us-west-2-an/results/PROC_CHANNEL_01_RERUN/ with sha256.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/PROC_CHANNEL_01_RERUN; PYB=/home/ubuntu/env/bin/python
W=$HOME/proc_channel_01_rerun; mkdir -p "$W" && cd "$HOME/IAM-Validation" && git fetch -q origin main && REV=$(git rev-parse --short origin/main)
for f in channel.py make_windows.py; do git show origin/main:Biological_Physics/MethylPhys/doors/data/PROC_CHANNEL_01/$f > "$W/$f"; done
git show origin/main:Biological_Physics/MethylPhys/atlas/v2/inputs/roster_samples.csv > "$W/roster_samples.csv"
git show origin/main:Biological_Physics/MethylPhys/atlas/sources/roster_v2/atlas_v2_roster.csv > "$W/atlas_v2_roster.csv"
cd "$W" && echo "repo $REV start $(date -u)" > run.log
$PYB make_windows.py >> run.log
curl -s https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186458/suppl/filelist.txt | awk -F'\t' '$2 ~ /\.pat\.gz$/ && $2 !~ /hg38/ {print $2}' > pat_files.txt
echo "pat files $(wc -l < pat_files.txt)" >> run.log
$PYB channel.py 2>&1 | grep -vE 'Warning|warn' > channel.log; echo "channel exit $? $(date -u)" >> run.log
for f in channel_samples.csv channel_cells.csv channel_summary.json channel.log run.log pat_files.txt windows_hg19_cpgidx.bed; do
  [ -f "$f" ] && $PYB -c "import boto3,sys,hashlib;h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest();boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B','$P/'+sys.argv[1],ExtraArgs={'Metadata':{'sha256':h}})" "$f"
done
echo "UPLOADED $(date -u)" >> run.log; $PYB -c "import boto3;boto3.client('s3',region_name='us-west-2').upload_file('run.log','$B','$P/run.log')"
