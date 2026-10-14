#!/bin/bash
# DEV-ERASE-01 reads (committed before any read). Per sample, in erase_samples.csv order (primary units first): 20 spread blocks via
# fastq-dump -N/-X, bwameth to hg19 (with lambda/pUC19) or mm10, sort, markdup, wgbstools bam2pat, Stage Q counting; results to S3; local
# files removed. A sample whose JSON is already in S3 is skipped (resume). Same tool versions as the cause test (session_epiqc.sh).
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/DEV_ERASE_01/reads; S=/mnt/scratch/run2; W=$S/erase01; mkdir -p $W/fq $W/bam $W/pat; T=$(nproc)
PYB=/home/ubuntu/env/bin/python; HERE=$PWD
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_sra/bin:$S/e_sbb/bin:$S/wgbs_tools:$PATH
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $HERE/erase_reads.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
have(){ $PYB -c "import boto3,sys;r=boto3.client('s3',region_name='us-west-2').list_objects_v2(Bucket='$B',Prefix=sys.argv[1]);sys.exit(0 if r['KeyCount'] else 1)" "$1"; }
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; sambamba 2>&1 | grep -m1 sambamba; fastq-dump --version | grep -m1 dump; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; } > versions.txt
for gsm in $(tail -n +2 erase_samples.csv | cut -d, -f3 | awk '!s[$0]++'); do
  have "$P/${gsm}_stageq.json" && { log "$gsm already read"; continue; }
  gen=$(grep ",$gsm," erase_samples.csv | head -1 | cut -d, -f5); [ "$gen" = "hg19" ] && REF=$S/ref/hg19_lambda_puc19.fa || REF=$S/ref/mm10.fa
  T0=$(date +%s); rm -f $W/fq/*; i=0
  $PYB erase_blocks.py erase_samples.csv $gsm > $W/${gsm}_blocks.txt
  while read r a b; do i=$((i+1)); ( fastq-dump --split-files -N $a -X $b -O $W/fq/b$i $r > /dev/null 2>> $W/${gsm}_fetch.err ) &
    while [ $(jobs -r | wc -l) -ge 6 ]; do sleep 5; done; done < $W/${gsm}_blocks.txt; wait
  cat $W/fq/b*/*_1.fastq > $W/fq/${gsm}_1.fq; cat $W/fq/b*/*_2.fastq > $W/fq/${gsm}_2.fq; rm -rf $W/fq/b*
  n1=$(awk 'END{print NR/4}' $W/fq/${gsm}_1.fq); n2=$(awk 'END{print NR/4}' $W/fq/${gsm}_2.fq); T1=$(date +%s)
  [ "$n1" = "$n2" ] && [ "$n1" -gt 0 ] || { log "$gsm FETCH FAILED pairs $n1/$n2"; continue; }
  bwameth.py --threads $T --reference $REF $W/fq/${gsm}_1.fq $W/fq/${gsm}_2.fq 2>> $W/${gsm}_align.log | samtools view -b -o $W/bam/$gsm.u.bam - || { log "$gsm align FAILED"; continue; }
  rm -f $W/fq/${gsm}_*.fq
  samtools sort -@ 16 -m 2G -T $W/bam/$gsm.tmp -o $W/bam/$gsm.s.bam $W/bam/$gsm.u.bam && rm -f $W/bam/$gsm.u.bam
  sambamba markdup -l 1 -t 16 --sort-buffer-size 16000 --overflow-list-size 10000000 --tmpdir $W/bam $W/bam/$gsm.s.bam $W/bam/$gsm.bam > $W/${gsm}_markdup.log 2>&1 && rm -f $W/bam/$gsm.s.bam || { log "$gsm markdup FAILED"; continue; }
  samtools index $W/bam/$gsm.bam; samtools flagstat $W/bam/$gsm.bam > $W/${gsm}_flagstat.txt
  wgbstools bam2pat $W/bam/$gsm.bam -o $W/pat --genome $gen -@ 16 > $W/${gsm}_bam2pat.log 2>&1 || { log "$gsm bam2pat FAILED"; continue; }
  PAT=$(ls $W/pat/$gsm*.pat.gz | head -1); rm -f $W/bam/$gsm.bam*
  $PYB erase_read_one.py $PAT $gsm > /dev/null && mv ${gsm}_stageq.json $W/ || { log "$gsm stage Q FAILED"; continue; }
  echo "{\"gsm\": \"$gsm\", \"pairs\": $n1, \"fetch_s\": $((T1-T0)), \"total_s\": $(( $(date +%s)-T0 )), \"genome\": \"$gen\"}" > $W/${gsm}_timing.json
  for f in $W/${gsm}_*; do up "$f" "$P/$(basename $f)"; done; up "$PAT" "$P/$gsm.pat.gz"; rm -f $PAT ${PAT}.csi $W/${gsm}_*
  log "$gsm done pairs $n1 $(grep -m1 'mapped (' $W/../erase01/${gsm}_flagstat.txt 2>/dev/null | cut -c1-40)"
done
log "ALL DONE"; up erase_reads.log "$P/erase_reads.log"; up versions.txt "$P/versions.txt"
