#!/bin/bash
# Box Run 2, session 2 (JOBS.md): IAM-A on another laboratory's healthy purified neutrophils (GSE128731: 2 donors x 4 kit/sequencer runs),
# each run cut to the same N_PAIRS read pairs, through the pinned Loyfer 2023 pipeline (bwa-meth 0.2.0 -> SAMtools 1.9 sort -> Sambamba 0.6.5
# markdup -> wgbstools 0.1.0 bam2pat defaults, -F 1796 -q 10, hg19), then alignment_qc.py -> run_sample.py (Stage Q0, Stage Q).
# Needs session 1's environments and index on /mnt/scratch/run2 (or restores the index and references from S3).
# Usage: N_PAIRS=50000000 THREADS=120 bash session2.sh      Outputs: s3://<bucket>/results/BOXRUN2/session2/
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/BOXRUN2/session2
N_PAIRS=${N_PAIRS:-50000000}; T=${THREADS:-$(nproc)}
S=/mnt/scratch/run2; O=$S/out2; mkdir -p $O $S/reads2 $S/bam2 $S/pat2; cd $S
PYB=/home/ubuntu/env/bin/python; REPO=~/IAM-Validation; HERE=$REPO/Biological_Physics/MethylPhys/boxruns/run2
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_sra/bin:$S/e_sbb/bin:$S/wgbs_tools:$PATH
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/session2.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
fin(){ log "END $1"; echo "{\"status\": \"$1\"}" > $O/status.json; for f in $O/*; do [ -f "$f" ] && up "$f" "$P/$(basename $f)"; done; exit 0; }

log "START N_PAIRS=$N_PAIRS threads=$T vcpus=$(nproc)"
REF=$S/ref/hg19_lambda_puc19.fa
[ -s $REF.bwameth.c2t.sa ] || fin FAIL_no_index_run_session1_first
for t in bwameth.py samtools sambamba wgbstools fastq-dump; do command -v $t >/dev/null || fin FAIL_missing_$t; done
samtools --version | head -1 | grep -q " 1\.9" || fin FAIL_samtools_not_1.9
sambamba --version 2>&1 | grep -q "0\.6\.5" || fin FAIL_sambamba_not_0.6.5
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; sambamba --version 2>&1 | grep -m1 sambamba; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; } > $O/versions.txt
cd $REPO && git pull -q origin main && git log --oneline -1 > $O/repo_commit.txt; cd $S

log "STEP format check first (session 1's 1M-pair BAM of SRR9888333; session 1 could not run it: output folder and script path)"
mkdir -p $S/pat1; [ -s $S/test.sorted.bam ] || fin FAIL_no_session1_bam
wgbstools bam2pat $S/test.sorted.bam -o $S/pat1 --genome hg19 -f >> $O/format_bam2pat.log 2>&1 || fin FAIL_format_bam2pat
P1=$(ls $S/pat1/*.pat.gz | head -1); cp "$P1" $O/SRR9888333_1M.pat.gz
curl -s https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5652nnn/GSM5652313/suppl/GSM5652313_Blood-Granulocytes-Z000000TZ.pat.gz | zcat 2>/dev/null | head -200000 > $S/loyfer_head.pat
python3 $HERE/format_check.py $O/SRR9888333_1M.pat.gz $S/loyfer_head.pat $S/wgbs_tools/references/hg19/CpG.bed.gz $O/format_check.json >> $O/session2.log 2>&1
grep -q '"pass": true' $O/format_check.json || fin FAIL_FORMAT_CHECK
log "format check passed"

RUNS="SRR9888330 SRR9888331 SRR9888332 SRR9888333 SRR9888334 SRR9888335 SRR9888336 SRR9888337"
log "STEP reads: first $N_PAIRS pairs of each run, 8 downloads in parallel"
for r in $RUNS; do ( [ -s reads2/${r}_2.fastq.gz ] || fastq-dump -X $N_PAIRS --split-files --gzip -O reads2 $r > $O/${r}_fastq.log 2>&1; echo "$r $?" >> $O/fastq_done.txt ) & done

for r in $RUNS; do
  while ! grep -q "^$r " $O/fastq_done.txt 2>/dev/null; do sleep 30; done
  grep -q "^$r 0" $O/fastq_done.txt || { log "$r reads FAILED"; continue; }
  [ -s $O/${r}_bundle.json ] && { log "$r already read"; continue; }
  log "$r align"
  T0=$(date +%s)
  bwameth.py --threads $T --reference $REF reads2/${r}_1.fastq.gz reads2/${r}_2.fastq.gz 2>> $O/${r}_align.log \
    | samtools view -b -o bam2/$r.unsorted.bam - || { log "$r align FAILED"; continue; }
  T1=$(date +%s)
  samtools sort -@ 16 -m 2G -T bam2/$r.tmp -o bam2/$r.sorted.bam bam2/$r.unsorted.bam && rm -f bam2/$r.unsorted.bam
  sambamba markdup -l 1 -t 16 --sort-buffer-size 16000 --overflow-list-size 10000000 --tmpdir bam2 bam2/$r.sorted.bam bam2/$r.bam >> $O/${r}_markdup.log 2>&1 \
    && rm -f bam2/$r.sorted.bam || { log "$r markdup FAILED"; continue; }
  samtools index bam2/$r.bam
  python3 $HERE/alignment_qc.py bam2/$r.bam $REF $O/${r}_alignment_qc.json >> $O/${r}_qc.log 2>&1 || log "$r alignment_qc failed"
  samtools flagstat bam2/$r.bam > $O/${r}_flagstat.txt
  wgbstools bam2pat bam2/$r.bam -o pat2 --genome hg19 -@ 16 >> $O/${r}_bam2pat.log 2>&1 || { log "$r bam2pat FAILED"; continue; }
  PAT=$(ls pat2/$r*.pat.gz | head -1); T2=$(date +%s)
  echo "{\"run\": \"$r\", \"read_pairs\": $N_PAIRS, \"align_s\": $((T1-T0)), \"to_pat_s\": $((T2-T1)), \"threads\": $T}" > $O/${r}_timing.json
  log "$r Stage Q0 + Q"
  $PYB $REPO/Biological_Physics/MethylPhys/chain/MethylPhys_Interface/run_sample.py --pat $PAT --specimen "isolated neutrophils" --seq-cell neutrophils \
     --alignment-qc $O/${r}_alignment_qc.json --id GSE128731_$r --out $O/${r}.html --ledger $O/ledger.jsonl >> $O/${r}_read.log 2>&1 || log "$r read FAILED"
  cp $PAT $O/ 2>/dev/null; up bam2/$r.bam "$P/bam/$r.bam"; rm -f bam2/$r.bam* reads2/${r}_*.fastq.gz
  log "$r done: $(python3 -c "import json;b=json.load(open('$O/${r}_bundle.json'));q=b.get('iam_a',{});print(q.get('A'),q.get('state'),q.get('eps'),q.get('refusal_code') or q.get('refusal','')[:80])" 2>&1)"
done
fin DONE
