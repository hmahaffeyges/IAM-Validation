#!/bin/bash
# DEV-IAMA-GRAND-01 on a self-terminating machine (MACHINE=A or B): for each of its CD8 samples (samples.csv), the first spots_cap read pairs
# (40 billion bases) are aligned exactly as DEV-SECONDLAB-02's Hansen donors were (align_hansen.sh: Trim Galore, Bismark on the bundle's hg38,
# deduplicate, sort, index), except that poly-G trimming is left ON: these runs are NovaSeq 6000 (two-colour chemistry), where Hansen's HiSeq 2000
# was not. Each alignment is then counted by chain Stage Q2 (count_grand.py) on this machine. Nothing is scored here. Up to 4 samples at once.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/DEV_IAMA_GRAND_01/; W=~/gr; PY=~/env/bin/python; M=${MACHINE:?}; cd $W
st(){ printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> $W/status_$M.txt; $PY -c "import boto3;boto3.client('s3').upload_file('$W/status_$M.txt','$B','${P}status_$M.txt')"; }
up(){ $PY -c "import boto3;boto3.client('s3').upload_file('$1','$B','${P}$2/$(basename $1)')"; }
TL=/home/ubuntu/tum; mkdir -p $TL
if [ ! -x $TL/mm/bin/bismark ]; then (cd $TL && for i in 00 01 02; do $PY -c "import boto3;boto3.client('s3').download_file('$B','downloads/tumour_work/tools.tgz.part_$i','tools.tgz.part_$i')"; done; cat tools.tgz.part_0* | tar xzf - && rm -f tools.tgz.part_0*); fi
logs(){ for f in "$@"; do [ -f "$f" ] && $PY -c "import boto3;boto3.client('s3').upload_file('$f','$B','${P}logs/$M/$(basename $(dirname $f))_$(basename $f)')"; done; }
if [ ! -x $W/sratk/bin/fasterq-dump ]; then curl -s --retry 5 -o sratk.tgz https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-ubuntu64.tar.gz && mkdir -p sratk && tar xzf sratk.tgz -C sratk --strip-components=1 && rm -f sratk.tgz; fi
export PATH=$TL/mm/bin:$W/sratk/bin:$PATH; vdb-config --restore-defaults > /dev/null 2>&1 || true
tar xzf code.tgz || { st "code unpack FAILED"; exit 1; }
FA=$(find $TL/ref -maxdepth 2 \( -name '*.fa' -o -name '*.fasta' \) | head -1); [ -s "$FA" ] || { st "reference FAILED: no fasta in the bundle's ref/"; exit 1; }
[ -s "$FA.fai" ] || $PY -c "import pysam;pysam.faidx('$FA')"
st "tools ready on $(nproc) cores, code $(cat REPO_COMMIT): $(bismark --version | grep -m1 -o 'v[0-9.]*') bismark, reference $FA"
fq(){ b=$1; for f in $b.fq.gz $b.fq; do if [ -s $f ]; then if [ $f = $b.fq ] && gzip -t $f 2>/dev/null; then mv $f $b.fq.gz; f=$b.fq.gz; fi; echo $f; return 0; fi; done; return 1; }
TG="trim_galore -q 20 --paired --cores 4 --gzip"
dl(){ $PY -c "import boto3;from botocore import UNSIGNED;from botocore.config import Config;boto3.client('s3',config=Config(signature_version=UNSIGNED),region_name='us-east-1').download_file('sra-pub-run-odp','sra/$1/$1','$2')"; }
cut8(){ srr=$1; cap=$2; k=8; step=$(( (cap + k - 1) / k )); for i in $(seq 0 $((k - 1))); do a=$((i * step + 1)); z=$(( (i + 1) * step < cap ? (i + 1) * step : cap ))
  mkdir -p c$i; (cd c$i && fastq-dump --split-3 -N $a -X $z -O . ../$srr.sra > fqd.log 2>&1) & done; wait
  for m in 1 2; do : > r_$m.fastq; for i in $(seq 0 $((k - 1))); do cat c$i/${srr}_$m.fastq >> r_$m.fastq; done; done; rm -rf c?; }
one(){ gsm=$1; sid=$2; sub=$3; typ=$4; srr=$5; cap=$6; mkdir -p $W/$srr && cd $W/$srr
  [ -f done ] && return
  dl $srr $srr.sra || { st "$srr download FAILED"; return; }; st "$srr ($sub) downloaded $(du -h $srr.sra | cut -f1)"
  cut8 $srr $cap; rm -f $srr.sra; n=$(( $(wc -l < r_1.fastq) / 4 )); st "$srr converted: first $n of $cap capped read pairs"
  [ "$n" -eq "$cap" ] || st "$srr NOTE: pairs $n differ from the cap $cap (singletons are left out by --split-3)"
  $TG r_1.fastq r_2.fastq > trim.log 2>&1
  R1=$(fq r_1_val_1) && R2=$(fq r_2_val_2) || { logs $W/$srr/trim.log; st "$srr trim FAILED $(tail -3 trim.log | tr '\n' ' ' | cut -c1-200)"; return; }
  rm -f r_1.fastq r_2.fastq; e1=${R1#r_1_val_1}; e2=${R2#r_2_val_2}; mv $R1 r1_val_1$e1; mv $R2 r2_val_2$e2; st "$srr trimmed"
  bismark --genome $TL/ref --parallel 4 -p 4 --bowtie2 -o . -1 r1_val_1$e1 -2 r2_val_2$e2 > bismark.log 2>&1 || { logs $W/$srr/bismark.log; st "$srr bismark FAILED $(tail -2 bismark.log | tr '\n' ' ')"; return; }
  logs $W/$srr/bismark.log $W/$srr/trim.log; rm -f r1_val_1$e1 r2_val_2$e2; st "$srr aligned $(grep -m1 'Mapping efficiency' *_PE_report.txt | xargs)"
  deduplicate_bismark -p --bam r1_val_1_bismark_bt2_pe.bam > dedup.log 2>&1 && rm -f r1_val_1_bismark_bt2_pe.bam
  samtools sort -@ 8 -m 2G -o $srr.bam r1_val_1_bismark_bt2_pe.deduplicated.bam && samtools index -@ 8 $srr.bam && rm -f r1_val_1_bismark_bt2_pe.deduplicated.bam
  cp *_PE_report.txt ${srr}_bismark_PE_report.txt; cp *deduplication_report.txt ${srr}_dedup_report.txt 2>/dev/null; cp trim.log ${srr}_trim.log
  for f in ${srr}_bismark_PE_report.txt ${srr}_dedup_report.txt ${srr}_trim.log; do [ -f $f ] && up $f reports; done
  mkdir -p out && cp -r $W/chain . 2>/dev/null; OMP_NUM_THREADS=1 $PY $W/count_grand.py $srr.bam "$FA" $gsm $sid $srr $sub "$typ" 16 >> count.log 2>&1 || st "$srr count FAILED $(tail -2 count.log | tr '\n' ' ' | cut -c1-200)"
  [ -s out/$srr.json ] && up out/$srr.json out && up $srr.bam bam && up $srr.bam.bai bam && touch done && st "$srr counted"
  [ -f done ] || st "$srr no count written"
}
mkdir -p $W/pre && cd $W/pre && first=$($PY -c "import csv;R=list(csv.DictReader(open('$W/samples.csv')));print(([r['runs'] for r in R if r['machine']=='$M'] or [R[0]['runs']])[0])") && dl $first p.sra \
 && fastq-dump --split-3 -X 100000 -O . p.sra > fqd.log 2>&1 && mv p_1.fastq q_1.fastq && mv p_2.fastq q_2.fastq && $TG q_1.fastq q_2.fastq > trim.log 2>&1 && P1=$(fq q_1_val_1) && P2=$(fq q_2_val_2) \
 && bismark --genome $TL/ref -p 4 --bowtie2 -o . -1 $P1 -2 $P2 > bismark.log 2>&1 && ls *_bismark_bt2_pe.bam > /dev/null 2>&1 \
 && st "preflight ok ($first): $(grep -m1 'Mapping efficiency' *_PE_report.txt | xargs); poly-G: $(grep -i -m1 'poly' trim.log | cut -c1-120)" || { logs $W/pre/fqd.log $W/pre/trim.log $W/pre/bismark.log; st "preflight FAILED: $(tail -2 trim.log bismark.log 2>/dev/null | tr '\n' ' ' | cut -c1-240)"; st "ALL DONE"; exit 1; }
rm -rf $W/pre; cd $W
# Run 3 (2026-10-16): both run-2 spot machines were reclaimed by AWS (no capacity); only SRR11007281 was counted. Scheduling only changes:
# with QUEUE=1 every machine works one shared list. A sample is skipped when its count is already in S3 (out/<run>.json); otherwise a machine
# claims it (claims/<run>, holding the machine name and time) and runs it. A claim older than 6.5 h with no count is a dead machine's and is
# taken over. The alignment, the count (count_grand.py) and the sealed rule are unchanged.
q(){ $PY - "$@" <<'EOQ'
import sys, time, boto3
s = boto3.client("s3"); B = "methylphys-data-945451304272-us-west-2-an"; P = "results/DEV_IAMA_GRAND_01/"; op, srr, me = sys.argv[1:4]
def has(k):
    try: return s.get_object(Bucket=B, Key=P + k)["Body"].read().decode()
    except s.exceptions.NoSuchKey: return None
if has(f"out/{srr}.json") is not None: print("done"); sys.exit()
c = has(f"claims/{srr}")
if c and not c.startswith(me) and time.time() - float(c.split()[1]) < 6.5 * 3600: print("taken"); sys.exit()
if op == "claim": s.put_object(Bucket=B, Key=P + f"claims/{srr}", Body=f"{me} {time.time()}".encode())
print("go")
EOQ
}
if [ "${QUEUE:-0}" = 1 ]; then
  for pass in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36; do
    left=0
    while IFS=, read -r gsm sid sub typ lay srr gb spots cap mach; do
      while [ "$(jobs -rp | wc -l)" -ge ${PAR:-4} ]; do sleep 60; done
      r=$(q claim $srr $M); [ "$r" = done ] && continue; left=1; [ "$r" = go ] || continue
      [ -d $W/$srr ] && [ ! -f $W/$srr/done ] && rm -rf $W/$srr   # a half-finished local copy is started again
      st "$srr claimed by $M"; one "$gsm" "$sid" "$sub" "$typ" "$srr" "$cap" & sleep 300
    done < <(tail -n +2 $W/samples.csv); wait
    [ $left = 0 ] && break; sleep 900
  done
else
while IFS=, read -r gsm sid sub typ lay srr gb spots cap mach; do
  [ "$mach" = "$M" ] || continue
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 60; done
  one "$gsm" "$sid" "$sub" "$typ" "$srr" "$cap" & sleep 300
done < <(tail -n +2 $W/samples.csv); wait
fi
st "ALL DONE $(ls $W/*/done 2>/dev/null | wc -l) samples"
