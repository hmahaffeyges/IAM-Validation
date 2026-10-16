#!/bin/bash
# DEV-SECONDLAB-02 step 1 (2026-10-15): ALIGNMENT ONLY. GSE49627 (Hansen lab) resting B cells of three healthy donors, full runs from ENA,
# (fetched from NCBI's AWS open copy, fasterq-dump), Trim Galore --paired, Bismark paired-end directional (bowtie2, the hg38 index and tools of downloads/tumour_work/tools.tgz, the same build the
# tumor work used), deduplicate_bismark, samtools sort + index. Nothing is counted here: no held error, no loose share. The count is chain
# Stage Q2 count_bam, run as a separate job only after the second-laboratory rule is written, synthetic-tested and committed.
# Donors run in parallel; each writes a status line and uploads its BAM, index and reports to s3 results/DEV_SECONDLAB_02/bam/.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/DEV_SECONDLAB_02/; W=~/hs; PY=~/env/bin/python; cd $W
st(){ printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> $W/status.txt; $PY -c "import boto3;boto3.client('s3').upload_file('$W/status.txt','$B','${P}status.txt')"; }
up(){ $PY -c "import boto3;boto3.client('s3').upload_file('$1','$B','${P}bam/$(basename $1)')"; }
# Run 3 (2026-10-15 22:40 PDT): the tools bundle is a conda environment built under ~/tum by the tumor work; its Python tools (cutadapt,
# called by Trim Galore) carry that path, so it is unpacked under ~/tum exactly as the tumor work used it (run 2 unpacked it under ~/hs and
# every trim produced nothing). TL holds the tools and the Bismark hg38 index.
TL=/home/ubuntu/tum; mkdir -p $TL
if [ ! -x $TL/mm/bin/bismark ]; then (cd $TL && for i in 00 01 02; do $PY -c "import boto3;boto3.client('s3').download_file('$B','downloads/tumour_work/tools.tgz.part_$i','tools.tgz.part_$i')"; done; cat tools.tgz.part_0* | tar xzf - && rm -f tools.tgz.part_0*); fi
logs(){ for f in "$@"; do [ -f "$f" ] && $PY -c "import boto3;boto3.client('s3').upload_file('$f','$B','${P}logs/$(basename $(dirname $f))_$(basename $f)')"; done; }
if [ ! -x $W/sratk/bin/fasterq-dump ]; then curl -s --retry 5 -o sratk.tgz https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-ubuntu64.tar.gz && mkdir -p sratk && tar xzf sratk.tgz -C sratk --strip-components=1 && rm -f sratk.tgz; fi
export PATH=$TL/mm/bin:$W/sratk/bin:$PATH; vdb-config --restore-defaults > /dev/null 2>&1 || true
st "tools ready: $(bismark --version | grep -m1 -o 'v[0-9.]*') bismark, $(fasterq-dump --version 2>&1 | grep -m1 -o '[0-9]\.[0-9.]*') fasterq-dump"
one(){ d=$1; srr=$2; u1=$3; u2=$4; mkdir -p $W/$d && cd $W/$d
  [ -f done ] && return
  # 2026-10-15 22:00 PDT: ENA served these files at about 0.26 MB/s (33 GB per donor would take days); NCBI's open copy in AWS
  # (s3://sra-pub-run-odp, read anonymously) and NCBI's own converter fasterq-dump give the same reads
  $PY -c "import boto3;from botocore import UNSIGNED;from botocore.config import Config;boto3.client('s3',config=Config(signature_version=UNSIGNED),region_name='us-east-1').download_file('sra-pub-run-odp','sra/$srr/$srr','$srr.sra')" || { st "$d sra download FAILED"; return; }
  st "$d downloaded $(du -h $srr.sra | cut -f1)"
  fasterq-dump --split-files -e 8 -t $W/$d/tmp -o r $srr.sra > fqd.log 2>&1 || { st "$d fasterq-dump FAILED $(tail -2 fqd.log | tr '\n' ' ')"; return; }
  rm -f $srr.sra; st "$d converted $(ls -s --block-size=G r_1.fastq | cut -d' ' -f1) per mate"
  trim_galore -q 20 --paired --cores 4 --gzip r_1.fastq r_2.fastq > trim.log 2>&1
  if [ ! -s r_1_val_1.fq.gz ] || [ ! -s r_2_val_2.fq.gz ]; then logs $W/$d/trim.log; st "$d trim FAILED $(tail -3 trim.log | tr '\n' ' ' | cut -c1-200)"; return; fi
  rm -f r_1.fastq r_2.fastq; mv r_1_val_1.fq.gz r1_val_1.fq.gz; mv r_2_val_2.fq.gz r2_val_2.fq.gz
  st "$d trimmed"
  bismark --genome $TL/ref --parallel 6 -p 4 --bowtie2 -o . -1 r1_val_1.fq.gz -2 r2_val_2.fq.gz > bismark.log 2>&1 || { st "$d bismark FAILED $(tail -2 bismark.log | tr '\n' ' ')"; return; }
  logs $W/$d/bismark.log $W/$d/trim.log; rm -f r1_val_1.fq.gz r2_val_2.fq.gz; st "$d aligned $(grep -m1 'Mapping efficiency' *_PE_report.txt | xargs)"
  deduplicate_bismark -p --bam r1_val_1_bismark_bt2_pe.bam > dedup.log 2>&1 && rm -f r1_val_1_bismark_bt2_pe.bam
  samtools sort -@ 12 -m 2G -o $srr.bam r1_val_1_bismark_bt2_pe.deduplicated.bam && samtools index -@ 12 $srr.bam && rm -f r1_val_1_bismark_bt2_pe.deduplicated.bam
  for f in $srr.bam $srr.bam.bai; do up $f; done
  cp *_PE_report.txt ${srr}_bismark_PE_report.txt; cp *deduplication_report.txt ${srr}_dedup_report.txt 2>/dev/null; cp trim.log ${srr}_trim.log
  for f in ${srr}_bismark_PE_report.txt ${srr}_dedup_report.txt ${srr}_trim.log; do [ -f $f ] && up $f; done
  touch done; st "$d done: $srr.bam $(du -h $srr.bam | cut -f1)"
}
# preflight: 100,000 read pairs of donor 1 through trim and Bismark; stop everything if any output is missing
mkdir -p $W/pre && cd $W/pre && $PY -c "import boto3;from botocore import UNSIGNED;from botocore.config import Config;boto3.client('s3',config=Config(signature_version=UNSIGNED),region_name='us-east-1').download_file('sra-pub-run-odp','sra/SRR948845/SRR948845','p.sra')" \
 && fastq-dump --split-files -X 100000 -O . p.sra > fqd.log 2>&1 && trim_galore -q 20 --paired --cores 4 --gzip p_1.fastq p_2.fastq > trim.log 2>&1 \
 && bismark --genome $TL/ref -p 4 --bowtie2 -o . -1 p_1_val_1.fq.gz -2 p_2_val_2.fq.gz > bismark.log 2>&1 && [ -s p_1_val_1_bismark_bt2_pe.bam ] \
 && st "preflight ok: $(grep -m1 'Mapping efficiency' *_PE_report.txt | xargs)" || { logs $W/pre/fqd.log $W/pre/trim.log $W/pre/bismark.log; st "preflight FAILED: $(tail -2 trim.log bismark.log 2>/dev/null | tr '\n' ' ' | cut -c1-250)"; st "ALL DONE"; exit 1; }
rm -rf $W/pre/*.sra; cd $W
while IFS=$'\t' read -r d gsm srr u1 u2; do one $d $srr $u1 $u2 & sleep 120; done < <(tail -n +2 $W/samples.tsv); wait   # process substitution: the donors are children of this shell, so wait waits for them
st "ALL DONE"
