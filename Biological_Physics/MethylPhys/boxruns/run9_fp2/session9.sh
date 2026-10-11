#!/bin/bash
# Box Run 9 (DEV-FINGERPRINT-02): ENCODE HAIB RRBS, single-end 35-50 bp -> .pat files. SET=normal first; SET=cancer only after the
# windows are sealed on the normal files. No copy error is computed here.
# Pipeline (pinned, as Box Run 6 with hg19): ENCODE fastq (md5 checked) -> Trim Galore 0.6.10 --rrbs -> bwa-meth 0.2.0 on hg19 (the Box Run 2
# index, hg19 + lambda + pUC19) -> SAMtools 1.9 sort/index (no duplicate marking: RRBS) -> wgbstools 0.1.0 bam2pat --genome hg19.
# Usage: SET=normal THREADS=60 bash session9.sh     Outputs: s3://<bucket>/results/BOXRUN9_FP2/<SET>/
set -uo pipefail
SET=${SET:-normal}; B=methylphys-data-945451304272-us-west-2-an; P=results/BOXRUN9_FP2/$SET; T=${THREADS:-$(nproc)}
S=${SCR:-/mnt/scratch/run2}; O=$S/out9$SET; W=$S/r9; mkdir -p $O $W/fq $W/trim $W/bam $W/pat; cd $S
PYB=/home/ubuntu/env/bin/python; HERE=~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run9_fp2
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/session9.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
fin(){ log "END $1"; echo "{\"status\": \"$1\"}" > $O/status.json; for f in $O/*; do [ -f "$f" ] && up "$f" "$P/$(basename $f)"; done; exit 0; }
for e in e_bm e_st19 e_tg e_wgbs; do [ -d $S/$e ] || fin FAIL_missing_env_$e; done; [ -d $S/wgbs_tools ] || fin FAIL_missing_wgbstools
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_tg/bin:$S/wgbs_tools:$PATH
REF=$S/ref/hg19_lambda_puc19.fa; [ -s $REF.bwameth.c2t.sa ] || fin FAIL_no_hg19_index
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; trim_galore --version | grep -m1 version; cutadapt --version; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; echo "bwa mem -T30 (synth_align_result.csv)"; } > $O/versions.txt
grep -q "0\.2\.0" $O/versions.txt || fin FAIL_bwameth_not_0.2.0
cp $HERE/${SET}_rrbs_files.csv $O/files.csv; log "START set=$SET threads=$T $(tr '\n' ' ' < $O/versions.txt)"
one_read(){ IFS=, read -r file exp cell rep rl md5 url gb <<< "$1"; f=$W/fq/$file.fastq.gz
  for t in 1 2 3; do [ -s $f ] && [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] && break; curl -sL -o $f "$url"; done
  [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] || { log "$file MD5_FAIL"; return; }
  trim_galore --rrbs --cores 2 -o $W/trim $f > $O/${file}_trim.log 2>&1 && cp $W/trim/${file}.fastq.gz_trimming_report.txt $O/ 2>/dev/null; rm -f $f; log "$file trimmed"; }
export -f one_read log; export W O
tail -n +2 $HERE/${SET}_rrbs_files.csv | xargs -P 8 -d '\n' -I{} bash -c 'one_read "{}"'
for file in $(tail -n +2 $HERE/${SET}_rrbs_files.csv | cut -d, -f1); do
  t=$W/trim/${file}_trimmed.fq.gz; [ -s $t ] || { log "$file no trimmed reads"; continue; }
  # bwa mem -T30: chosen by synth_align.sh under its pre-set rule (synth_align_result.csv; bwa-meth default -T 40 aligns no 36-base read).
  # Passed as one token, after bwa-meth's own -T 40, so bwa takes 30 (LESSONS B1, B1b).
  bwameth.py --threads $T --reference $REF -T30 $t 2>> $O/${file}_align.log | samtools view -b -o $W/bam/$file.u.bam - || { log "$file align FAILED"; continue; }
  grep -q "running: bwa mem .* -T30 " $O/${file}_align.log || fin "FAIL_T30_not_passed_${file}"
  samtools sort -@ 16 -m 2G -T $W/bam/$file.tmp -o $W/bam/$file.bam $W/bam/$file.u.bam && rm -f $W/bam/$file.u.bam && samtools index $W/bam/$file.bam
  samtools flagstat $W/bam/$file.bam > $O/${file}_flagstat.txt
  MR=$(grep -m1 " mapped (" $O/${file}_flagstat.txt | sed -E 's/.*\(([0-9.]+)%.*/\1/'); log "$file mapped ${MR}%"
  awk -v m="$MR" 'BEGIN{exit !(m+0 >= 40)}' || fin "FAIL_mapping_${file}_${MR}pct"   # LESSONS B1: stop the run, do not log 'done' on nothing
  wgbstools bam2pat $W/bam/$file.bam -o $W/pat --genome hg19 -@ 16 >> $O/${file}_bam2pat.log 2>&1 || { log "$file bam2pat FAILED"; continue; }
  [ -s $W/pat/$file.pat.gz ] || fin "FAIL_no_pat_${file}"   # LESSONS B1
  for x in $W/pat/$file.pat.gz $W/pat/$file.pat.gz.csi; do [ -f $x ] && up $x "$P/$(basename $x)"; done; rm -f $W/bam/$file.bam* $t; log "$file done"
done
fin DONE
