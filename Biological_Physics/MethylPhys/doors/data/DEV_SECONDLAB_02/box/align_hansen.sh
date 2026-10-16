#!/bin/bash
# DEV-SECONDLAB-02 step 1 (2026-10-15): ALIGNMENT ONLY. GSE49627 (Hansen lab) resting B cells of three healthy donors, full runs from ENA,
# Trim Galore --paired, Bismark paired-end directional (bowtie2, the hg38 index and tools of downloads/tumour_work/tools.tgz, the same build the
# tumor work used), deduplicate_bismark, samtools sort + index. Nothing is counted here: no held error, no loose share. The count is chain
# Stage Q2 count_bam, run as a separate job only after the second-laboratory rule is written, synthetic-tested and committed.
# Donors run in parallel; each writes a status line and uploads its BAM, index and reports to s3 results/DEV_SECONDLAB_02/bam/.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/DEV_SECONDLAB_02/; W=~/hs; PY=~/env/bin/python; cd $W
st(){ printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> $W/status.txt; $PY -c "import boto3;boto3.client('s3').upload_file('$W/status.txt','$B','${P}status.txt')"; }
up(){ $PY -c "import boto3;boto3.client('s3').upload_file('$1','$B','${P}bam/$(basename $1)')"; }
if [ ! -x $W/mm/bin/bismark ]; then for i in 00 01 02; do $PY -c "import boto3;boto3.client('s3').download_file('$B','downloads/tumour_work/tools.tgz.part_$i','tools.tgz.part_$i')"; done; cat tools.tgz.part_0* | tar xzf - && rm -f tools.tgz.part_0*; fi
export PATH=$W/mm/bin:$PATH; st "tools ready: $(bismark --version | sed -n 2p | xargs)"
one(){ d=$1; srr=$2; u1=$3; u2=$4; mkdir -p $W/$d && cd $W/$d
  [ -f done ] && return
  for k in 1 2; do u=$([ $k = 1 ] && echo $u1 || echo $u2); curl -s --retry 8 --retry-delay 30 -o r$k.fq.gz "https://$u" || { st "$d download $k FAILED"; return; }; done
  st "$d downloaded $(du -sh . | cut -f1)"
  trim_galore -q 20 --paired --cores 4 --gzip r1.fq.gz r2.fq.gz > trim.log 2>&1 && rm -f r1.fq.gz r2.fq.gz
  st "$d trimmed"
  bismark --genome $W/ref --parallel 6 -p 4 --bowtie2 -o . -1 r1_val_1.fq.gz -2 r2_val_2.fq.gz > bismark.log 2>&1 || { st "$d bismark FAILED $(tail -2 bismark.log | tr '\n' ' ')"; return; }
  rm -f r1_val_1.fq.gz r2_val_2.fq.gz; st "$d aligned $(grep -m1 'Mapping efficiency' *_PE_report.txt | xargs)"
  deduplicate_bismark -p --bam r1_val_1_bismark_bt2_pe.bam > dedup.log 2>&1 && rm -f r1_val_1_bismark_bt2_pe.bam
  samtools sort -@ 12 -m 2G -o $srr.bam r1_val_1_bismark_bt2_pe.deduplicated.bam && samtools index -@ 12 $srr.bam && rm -f r1_val_1_bismark_bt2_pe.deduplicated.bam
  for f in $srr.bam $srr.bam.bai; do up $f; done
  cp *_PE_report.txt ${srr}_bismark_PE_report.txt; cp *deduplication_report.txt ${srr}_dedup_report.txt 2>/dev/null; cp trim.log ${srr}_trim.log
  for f in ${srr}_bismark_PE_report.txt ${srr}_dedup_report.txt ${srr}_trim.log; do [ -f $f ] && up $f; done
  touch done; st "$d done: $srr.bam $(du -h $srr.bam | cut -f1)"
}
while IFS=$'\t' read -r d gsm srr u1 u2; do one $d $srr $u1 $u2 & sleep 120; done < <(tail -n +2 $W/samples.tsv); wait   # process substitution: the donors are children of this shell, so wait waits for them
st "ALL DONE"
