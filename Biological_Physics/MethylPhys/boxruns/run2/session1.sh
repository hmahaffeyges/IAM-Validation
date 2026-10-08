#!/bin/bash
# Box Run 2, session 1 (JOBS.md): the pinned Loyfer pipeline (bwa-meth 0.2.0, SAMtools 1.9, wgbstools 0.1.0, hg19), the hg19 + lambda
# + pUC19 bwa-meth index saved to S3, and 1M read pairs of SRR9888333 (Sample6 neutrophils, TruSeq) through the whole path, with a format
# check against a Loyfer Blood-Granulocytes PAT file and the measured alignment speed. Outputs: s3://<bucket>/results/BOXRUN2/session1/.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/BOXRUN2/session1
S=/mnt/scratch/run2; O=$S/out; mkdir -p $O $S/ref $S/reads; cd $S
PYB=/home/ubuntu/env/bin/python; HERE=$(cd "$(dirname "$0")" && pwd)
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/session1.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
fin(){ log "END $1"; echo "{\"status\": \"$1\"}" > $O/status.json; for f in $O/*; do [ -f "$f" ] && up "$f" "$P/$(basename $f)"; done; exit 0; }

log "STEP tools"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba || fin FAIL_micromamba
MM="$S/bin/micromamba -r $S/mamba"
# bwameth 0.2.0 is built for Python <= 3.6 only, so it gets its own environment; wgbstools runs on Python 3.8
$MM create -y -q -p $S/e_bm -c conda-forge -c bioconda python=3.6 bwameth=0.2.0 bwa >> $O/install.log 2>&1 || fin FAIL_install_bwameth_0.2.0
$MM create -y -q -p $S/e_wgbs -c conda-forge -c bioconda python=3.8 numpy pandas scipy htslib bedtools >> $O/install.log 2>&1 || fin FAIL_install_wgbs_env
$MM create -y -q -p $S/e_st19 -c conda-forge -c bioconda samtools=1.9 >> $O/install.log 2>&1 || fin FAIL_install_samtools_1.9
$MM create -y -q -p $S/e_sra -c conda-forge -c bioconda sra-tools >> $O/install.log 2>&1 || fin FAIL_install_sra_tools
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_sra/bin:$PATH
command -v g++ >/dev/null || sudo apt-get install -y -q build-essential >> $O/install.log 2>&1
[ -d wgbs_tools ] || git clone -q https://github.com/nloyfer/wgbs_tools.git
(cd wgbs_tools && git checkout -q 0.1.0 && python setup.py >> $O/install.log 2>&1)
export PATH=$S/wgbs_tools:$PATH
{ $MM list -p $S/e_bm | grep -E "bwameth|^ *bwa "; $MM list -p $S/e_wgbs | grep -E "^ *python "; samtools --version | head -1; echo "wgbstools $(git -C wgbs_tools describe --tags)"; fastq-dump --version | grep -o "[0-9][0-9.]*" | head -1; } > $O/versions.txt 2>&1
samtools --version | head -1 | grep -q " 1\.9" || fin FAIL_samtools_not_1.9
grep -q "0\.2\.0" $O/versions.txt || fin FAIL_bwameth_not_0.2.0
log "versions: $(tr '\n' ';' < $O/versions.txt)"

log "STEP genome (wgbstools init_genome hg19)"
# UCSC redirects; wgbstools 0.1.0 calls curl without -L, so the FASTA is fetched here and passed in (bgzipped, as init_genome asks)
curl -sL -o ref/hg19.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz || fin FAIL_hg19_download
gunzip -f ref/hg19.fa.gz && bgzip -@ 16 -f ref/hg19.fa || fin FAIL_hg19_bgzip
wgbstools init_genome hg19 --fasta_path $S/ref/hg19.fa.gz -@ 16 -f >> $O/init_genome.log 2>&1 || fin FAIL_init_genome
R=$S/wgbs_tools/references/hg19
N=$(zcat $R/CpG.bed.gz | wc -l); echo "$N" > $O/cpg_count.txt; log "CpG sites in the hg19 dictionary: $N (Loyfer hg19: 28217448)"

log "STEP spike-ins and bwa-meth index"
E="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=fasta&id"
{ curl -s "$E=NC_001416.1" | sed 's/^>.*/>lambda/'; curl -s "$E=L09137.2" | sed 's/^>.*/>pUC19/'; } > ref/spikes.fa
[ "$(grep -c '>' ref/spikes.fa)" = 2 ] || fin FAIL_spikes
zcat ref/hg19.fa.gz > ref/hg19_lambda_puc19.fa && cat ref/spikes.fa >> ref/hg19_lambda_puc19.fa
T0=$(date +%s); bwameth.py index ref/hg19_lambda_puc19.fa >> $O/index.log 2>&1 || fin FAIL_index; log "index built in $(( $(date +%s)-T0 )) s"
for f in ref/hg19_lambda_puc19.fa*; do up "$f" "reference/hg19_bwameth/$(basename $f)"; done
tar czf $O/wgbstools_hg19_references.tgz -C $S/wgbs_tools/references hg19
log "index and wgbstools references saved to S3"

log "STEP 1M read pairs of SRR9888333"
fastq-dump -X 1000000 --split-files --gzip -O reads SRR9888333 >> $O/reads.log 2>&1 || fin FAIL_reads
RL=$(zcat reads/SRR9888333_1.fastq.gz | head -2 | tail -1 | tr -d '\n' | wc -c)
T0=$(date +%s)
bwameth.py --threads 24 --reference ref/hg19_lambda_puc19.fa reads/SRR9888333_1.fastq.gz reads/SRR9888333_2.fastq.gz 2>> $O/align.log | samtools view -b -o test.bam - || fin FAIL_align
SEC=$(( $(date +%s)-T0 ))
echo "{\"run\": \"SRR9888333\", \"read_pairs\": 1000000, \"read_length\": $RL, \"threads\": 24, \"vcpus\": $(nproc), \"align_seconds\": $SEC}" > $O/align_timing.json
log "aligned 1M pairs (read length $RL) in $SEC s on 24 threads"
samtools sort -@ 8 -o test.sorted.bam test.bam && samtools index test.sorted.bam && samtools flagstat test.sorted.bam > $O/flagstat.txt
wgbstools bam2pat test.sorted.bam -o $S/pat --genome hg19 >> $O/bam2pat.log 2>&1 || fin FAIL_bam2pat
PAT=$(ls $S/pat/*.pat.gz | head -1); cp "$PAT" $O/SRR9888333_1M.pat.gz

log "STEP format check against Loyfer GSM5652313 Blood-Granulocytes"
curl -s https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5652nnn/GSM5652313/suppl/GSM5652313_Blood-Granulocytes-Z000000TZ.pat.gz | zcat 2>/dev/null | head -200000 > loyfer_head.pat
python $HERE/format_check.py $O/SRR9888333_1M.pat.gz loyfer_head.pat $R/CpG.bed.gz $O/format_check.json >> $O/session1.log 2>&1
grep -q '"pass": true' $O/format_check.json && fin DONE_FORMAT_PASS || fin DONE_FORMAT_FAIL
