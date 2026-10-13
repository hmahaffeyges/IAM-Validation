#!/bin/bash
# DEV-IAMA-CAUSE-01 (box). For each run in cause_runs.csv, three parts of N_PAIRS read pairs, starting at spot 1, at 40 % and at 80 % of the run's
# spots (SRA run info), are written with fastq-dump -N/-X (SRA random access) as r10/fq/<run>p<k>_{1,2}.fq with the .complete marker, then read
# by boxruns/run10_epiqc/session_epiqc.sh UNCHANGED (TAG=cause01), one part per call, through the pinned pipeline. SHUTDOWN is cleared for that call, so only this
# script stops the machine, after the last part. Part p0 must reproduce
# Stage B's own reading of the run (check in decide_cause_01.py). At most 2 parts are on disk at once. Usage: N_PAIRS=12200000 [SHUTDOWN=1] bash cause_01.sh
set -uo pipefail
S=/mnt/scratch/run2; H=~/IAM-Validation/Biological_Physics/MethylPhys; D=$H/doors/data/DEV_IAMA_CAUSE_01; N=${N_PAIRS:-12200000}
MM="$S/bin/micromamba -r $S/mamba"; mkdir -p $S/r10/fq; cd $S
[ -x $S/e_sra/bin/fastq-dump ] || $MM create -y -q -p $S/e_sra -c conda-forge -c bioconda sra-tools=3.0.10 > /tmp/cause_sra_install.log 2>&1 || { echo "sra-tools install failed"; exit 4; }
FQD=$S/e_sra/bin/fastq-dump; $FQD --version | head -2 > /tmp/cause_fqd_version.txt
L=/tmp/cause01_list.csv; echo "kit,lab,line,rep,seq,run" > $L
spots(){ /home/ubuntu/env/bin/python -c "
import urllib.request,sys,csv,io
t=urllib.request.urlopen('https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc='+sys.argv[1],timeout=120).read().decode()
print(next(csv.DictReader(io.StringIO(t)))['spots'])" "$1"; }
stage(){ r=$1; k=$2; a=$3; nm=${r}p$k; [ -f r10/fq/$nm.complete ] && return 0
  $FQD --split-files -N $a -X $((a+N-1)) -O r10/fq $r > /tmp/cause_$nm.log 2>&1 || { echo "$nm fastq-dump failed" >> /tmp/cause_stage.txt; return 1; }
  mv r10/fq/${r}_1.fastq r10/fq/${nm}_1.fq; mv r10/fq/${r}_2.fastq r10/fq/${nm}_2.fq
  n1=$(awk 'END{print NR/4}' r10/fq/${nm}_1.fq); n2=$(awk 'END{print NR/4}' r10/fq/${nm}_2.fq)
  echo "$nm start $a pairs $n1 $n2" >> /tmp/cause_stage.txt; [ "$n1" = "$N" ] && [ "$n2" = "$N" ] && touch r10/fq/$nm.complete; }
JOBS=""
while IFS=, read kit lab line rep seq run role; do
  [ "$run" = "run" ] && continue; s=$(spots $run); [ -n "$s" ] || { echo "no spot count for $run"; exit 5; }
  for k in 0 40 80; do a=$(( k == 0 ? 1 : s * k / 100 )); JOBS="$JOBS $kit,$lab,$line,$rep,$seq,$run,$k,$a"; done
done < $D/cause_runs.csv
# One part per pipeline call, staged BEFORE the call (the pipeline's own download step is never reached for a part name); the next part is
# staged while the current one aligns. A part that cannot be staged is recorded and skipped.
set -- $JOBS; IFS=, read kit lab line rep seq run k a <<< "$1"; stage $run $k $a
while [ $# -gt 0 ]; do
  IFS=, read kit lab line rep seq run k a <<< "$1"; shift
  if [ $# -gt 0 ]; then IFS=, read k2 l2 n2 r2 s2 run2 kk2 a2 <<< "$1"; stage $run2 $kk2 $a2 & BG=$!; else BG=""; fi
  if [ -f r10/fq/${run}p$k.complete ]; then
    printf "kit,lab,line,rep,seq,run\n$kit,$lab,$line,$rep,$seq,${run}p$k\n" > /tmp/cause01_list.csv
    # SHUTDOWN must NOT reach the pipeline: it stops the machine at the end of EVERY part (2026-10-13; it did, after part 1 of 15).
    LIST=/tmp/cause01_list.csv N_PAIRS=$N TAG=cause01 FETCH_PAR=1 MAX_WAIT=99 SHUTDOWN= bash $H/boxruns/run10_epiqc/session_epiqc.sh > /tmp/cause01_${run}p$k.out 2>&1
  else echo "${run}p$k not staged; skipped" >> /tmp/cause_stage.txt; fi
  [ -n "$BG" ] && wait $BG
done
/home/ubuntu/env/bin/python -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file('/tmp/cause_stage.txt','methylphys-data-945451304272-us-west-2-an','results/BOXRUN10_EPIQC/cause01/cause_stage.txt')" || true
cat /tmp/cause_stage.txt
[ -n "${SHUTDOWN:-}" ] && sudo shutdown -h +1
exit 0
