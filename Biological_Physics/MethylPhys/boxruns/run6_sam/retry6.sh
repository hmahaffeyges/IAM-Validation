#!/bin/bash
# Box Run 6 retry: runs whose ENA md5 lookup failed on the box get the md5 committed in GSE77079_runs.csv (looked up 2026-10-10), the same
# download and Trim Galore --rrbs step, and, once session 6 has finished, the same alignment and bam2pat for any run without a .pat in S3.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/BOXRUN6_SAM; S=/mnt/scratch/run2; O=$S/out6sam; R6=$S/r6; T=${THREADS:-60}
PYB=/home/ubuntu/env/bin/python; HERE=~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run6_sam
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_tg/bin:$S/wgbs_tools:$PATH
log(){ echo "$(date -u +%FT%TZ) retry $*" | tee -a $O/session6.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
tail -n +2 $HERE/GSE77079_runs.csv | while IFS=, read -r r rc bc url rest; do
  md5=$(grep "^$r," $HERE/GSE77079_runs.csv | awk -F, '{print $NF}'); f=$R6/fq/$r.fastq.gz
  [ -s $R6/trim/${r}_trimmed.fq.gz ] && continue
  for t in 1 2 3; do [ -s $f ] && [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] && break; curl -s -o $f "https://$url"; done
  [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] || { log "$r MD5_FAIL again"; continue; }
  mkdir -p $R6/trim_tmp; trim_galore --rrbs --cores 4 -o $R6/trim_tmp $f > $O/${r}_trim.log 2>&1 && cp $R6/trim_tmp/${r}.fastq.gz_trimming_report.txt $O/ 2>/dev/null && \
    mv $R6/trim_tmp/${r}_trimmed.fq.gz $R6/trim/   # complete files only: session 6 may be aligning meanwhile
  rm -f $f; log "$r trimmed"
done
for i in $(seq 1 1000); do [ -f $O/status.json ] && break; sleep 30; done
REF=$S/ref/mm10.fa
for r in $(tail -n +2 $HERE/GSE77079_runs.csv | cut -d, -f1); do
  $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').head_object(Bucket='$B',Key=sys.argv[1])" "$P/$r.pat.gz" 2>/dev/null && continue
  t=$R6/trim/${r}_trimmed.fq.gz; [ -s $t ] || { log "$r no trimmed reads"; continue; }
  bwameth.py --threads $T --reference $REF $t 2>> $O/${r}_align.log | samtools view -b -o $R6/bam/$r.u.bam - || { log "$r align FAILED"; continue; }
  samtools sort -@ 16 -m 2G -T $R6/bam/$r.tmp -o $R6/bam/$r.bam $R6/bam/$r.u.bam && rm -f $R6/bam/$r.u.bam && samtools index $R6/bam/$r.bam
  samtools flagstat $R6/bam/$r.bam > $O/${r}_flagstat.txt; up $O/${r}_flagstat.txt "$P/${r}_flagstat.txt"
  wgbstools bam2pat $R6/bam/$r.bam -o $R6/pat --genome mm10 -@ 16 >> $O/${r}_bam2pat.log 2>&1 || { log "$r bam2pat FAILED"; continue; }
  for x in $R6/pat/$r.pat.gz $R6/pat/$r.pat.gz.csi; do [ -f $x ] && up $x "$P/$(basename $x)"; done; rm -f $R6/bam/$r.bam* $t; log "$r done"
done
log "END"; echo '{"status": "DONE"}' > $O/retry_status.json; up $O/session6.log "$P/session6.log"; up $O/retry_status.json "$P/retry_status.json"
