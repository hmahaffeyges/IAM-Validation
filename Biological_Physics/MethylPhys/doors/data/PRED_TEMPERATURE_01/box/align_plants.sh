#!/bin/bash
# PRED-TEMPERATURE-01 stage T3 on self-terminating spot machines working one shared queue (claims in S3, as DEV-IAMA-GRAND-01 run 3). For each
# sample of t3_pairs_runs.csv: every run's FASTQ from ENA, Trim Galore (HiSeq 2000: no poly-G trim), Bismark (Bowtie 2) on TAIR10 (Ensembl Plants,
# with the chloroplast), deduplicate, sort, index, then chain stage TL counts it (count_plant.py). Only the chain-stamped count, the Bismark reports
# and the logs are kept in S3; the alignment is rebuilt from the public reads with this script. Nothing is scored here.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/PRED_TEMPERATURE_01/T3/; W=~/tl; PY=~/env/bin/python; M=${MACHINE:?}; cd $W
st(){ printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> $W/status_$M.txt; $PY -c "import boto3;boto3.client('s3').upload_file('$W/status_$M.txt','$B','${P}status_$M.txt')"; }
up(){ $PY -c "import boto3;boto3.client('s3').upload_file('$1','$B','${P}$2/$(basename $1)')"; }
TL=/home/ubuntu/tum; mkdir -p $TL
if [ ! -x $TL/mm/bin/bismark ]; then (cd $TL && for i in 00 01 02; do $PY -c "import boto3;boto3.client('s3').download_file('$B','downloads/tumour_work/tools.tgz.part_$i','tools.tgz.part_$i')"; done; cat tools.tgz.part_0* | tar xzf - && rm -f tools.tgz.part_0*); fi
export PATH=$TL/mm/bin:$PATH
tar xzf code.tgz || { st "code unpack FAILED"; exit 1; }
mkdir -p $W/ref && cd $W/ref
if [ ! -d Bisulfite_Genome ]; then
  curl -s --retry 5 -o tair10.fa.gz https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz && gunzip -f tair10.fa.gz \
   && bismark_genome_preparation --bowtie2 --parallel 8 . > prep.log 2>&1 && $PY -c "import pysam;pysam.faidx('tair10.fa')" || { st "reference FAILED $(tail -2 prep.log | tr '\n' ' ')"; st "ALL DONE"; exit 1; }
fi
FA=$W/ref/tair10.fa; cd $W
st "tools ready on $(nproc) cores, code $(cat REPO_COMMIT): $(bismark --version | grep -m1 -o 'v[0-9.]*') bismark, TAIR10 $(cut -f1 $FA.fai | tr '\n' ' ')"
fq(){ b=$1; for f in $b.fq.gz $b.fq; do [ -s $f ] && { echo $f; return 0; }; done; return 1; }
one(){ gsm=$1; acc=$2; temp=$3; runs=$4; urls=$5; mkdir -p $W/$gsm && cd $W/$gsm; t0=$(date +%s)
  : > r_1.fastq.gz; : > r_2.fastq.gz
  for u in $(echo "$urls" | tr '|' ' '); do for m in 1 2; do curl -s --retry 8 -C - "https://${u}_$m.fastq.gz" -o part_$m.fq.gz || { st "$gsm download FAILED $u"; return; }; cat part_$m.fq.gz >> r_$m.fastq.gz; rm -f part_$m.fq.gz; done; done
  st "$gsm ($acc $temp) downloaded $(du -ch r_?.fastq.gz | tail -1 | cut -f1)"
  trim_galore -q 20 --paired --cores 2 --gzip r_1.fastq.gz r_2.fastq.gz > trim.log 2>&1; rm -f r_1.fastq.gz r_2.fastq.gz
  R1=$(fq r_1_val_1) && R2=$(fq r_2_val_2) || { st "$gsm trim FAILED"; up $W/$gsm/trim.log logs; return; }
  bismark --genome $W/ref --parallel 2 -p 2 --bowtie2 -o . -1 $R1 -2 $R2 > bismark.log 2>&1 || { st "$gsm bismark FAILED $(tail -2 bismark.log | tr '\n' ' ')"; up $W/$gsm/bismark.log logs; return; }
  rm -f $R1 $R2; deduplicate_bismark -p --bam r_1_val_1_bismark_bt2_pe.bam > dedup.log 2>&1 && rm -f r_1_val_1_bismark_bt2_pe.bam
  samtools sort -@ 4 -m 2G -o $gsm.bam r_1_val_1_bismark_bt2_pe.deduplicated.bam && samtools index $gsm.bam && rm -f r_1_val_1_bismark_bt2_pe.deduplicated.bam
  cp *_PE_report.txt ${gsm}_bismark_PE_report.txt; cp *deduplication_report.txt ${gsm}_dedup_report.txt 2>/dev/null; cp trim.log ${gsm}_trim.log
  for f in ${gsm}_bismark_PE_report.txt ${gsm}_dedup_report.txt ${gsm}_trim.log; do [ -f $f ] && up $f reports; done
  ln -sf $W/chain chain; OMP_NUM_THREADS=1 $PY $W/count_plant.py $gsm.bam $FA $gsm $acc $temp "$runs" 4 >> count.log 2>&1 || st "$gsm count FAILED $(tail -2 count.log | tr '\n' ' ' | cut -c1-200)"
  if [ -s out/$gsm.json ]; then up out/$gsm.json out; up out/$gsm.json.stamp.json out 2>/dev/null; st "$gsm counted in $(( ($(date +%s) - t0) / 60 )) min: $(tail -1 count.log | cut -c1-160)"; cd $W && rm -rf $W/$gsm; else st "$gsm no count written"; fi
}
q(){ $PY - "$@" <<'EOQ'
import sys, time, boto3
s = boto3.client("s3"); B = "methylphys-data-945451304272-us-west-2-an"; P = "results/PRED_TEMPERATURE_01/T3/"; op, gsm, me = sys.argv[1:4]
def has(k):
    try: return s.get_object(Bucket=B, Key=P + k)["Body"].read().decode()
    except s.exceptions.NoSuchKey: return None
if has(f"out/{gsm}.json") is not None: print("done"); sys.exit()
c = has(f"claims/{gsm}")
if c and not c.startswith(me) and time.time() - float(c.split()[1]) < 4 * 3600: print("taken"); sys.exit()
if op == "claim": s.put_object(Bucket=B, Key=P + f"claims/{gsm}", Body=f"{me} {time.time()}".encode())
print("go")
EOQ
}
$PY -c "
import csv, collections; R=list(csv.DictReader(open('t3_pairs_runs.csv'))); G=collections.OrderedDict()
for r in R: G.setdefault(r['gsm'], [r['acc'], r['temp'], [], []]); G[r['gsm']][2].append(r['run_accession']); G[r['gsm']][3].append(r['fastq_ftp'].split(';')[0].rsplit('_',1)[0])
open('samples.txt','w').write(''.join(f\"{g} {v[0]} {v[1]} {';'.join(v[2])} {'|'.join(v[3])}\n\" for g, v in G.items()))"
st "queue: $(wc -l < samples.txt) samples"
for pass in $(seq 1 30); do
  left=0
  while read -r gsm acc temp runs urls; do
    while [ "$(jobs -rp | wc -l)" -ge ${PAR:-8} ]; do sleep 30; done
    r=$(q claim $gsm $M); [ "$r" = done ] && continue; left=1; [ "$r" = go ] || continue
    rm -rf $W/$gsm; st "$gsm claimed by $M"; one "$gsm" "$acc" "$temp" "$runs" "$urls" & sleep 20
  done < samples.txt; wait
  [ $left = 0 ] && break; sleep 600
done
st "ALL DONE"
