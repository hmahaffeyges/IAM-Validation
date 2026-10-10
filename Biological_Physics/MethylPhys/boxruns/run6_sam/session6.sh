#!/bin/bash
# Box Run 6 (DEV-SAM-LEVER-01): GSE77079 mouse liver RRBS, 19 mice (38 runs, single-end 50 bp) -> .pat files. No copy error is computed here.
# Pipeline (pinned, as Box Run 2 with the mouse genome): ENA fastq (md5 checked) -> Trim Galore 0.6.10 --rrbs (cutadapt) -> bwa-meth 0.2.0 on
# mm10 -> SAMtools 1.9 sort/index (no duplicate marking: RRBS reads start at MspI sites by design) -> wgbstools 0.1.0 bam2pat --genome mm10.
# Usage: THREADS=60 bash session6.sh     Outputs: s3://<bucket>/results/BOXRUN6_SAM/
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/BOXRUN6_SAM; T=${THREADS:-$(nproc)}
S=${SCR:-/mnt/scratch/run2}; O=$S/out6sam; mkdir -p $O $S/r6/fq $S/r6/trim $S/r6/bam $S/r6/pat $S/ref; cd $S
PYB=/home/ubuntu/env/bin/python; HERE=~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/run6_sam
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/session6.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
fin(){ log "END $1"; echo "{\"status\": \"$1\"}" > $O/status.json; for f in $O/*; do [ -f "$f" ] && up "$f" "$P/$(basename $f)"; done; exit 0; }
MM="$S/bin/micromamba -r $S/mamba"
[ -x $S/bin/micromamba ] || { curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba || fin FAIL_micromamba; }
[ -d $S/e_bm ]   || $MM create -y -q -p $S/e_bm -c conda-forge -c bioconda python=3.6 bwameth=0.2.0 bwa >> $O/install.log 2>&1 || fin FAIL_install_bwameth
[ -d $S/e_wgbs ] || $MM create -y -q -p $S/e_wgbs -c conda-forge -c bioconda python=3.8 numpy pandas scipy htslib bedtools >> $O/install.log 2>&1 || fin FAIL_install_wgbs
[ -d $S/e_st19 ] || $MM create -y -q -p $S/e_st19 -c conda-forge -c bioconda samtools=1.9 >> $O/install.log 2>&1 || fin FAIL_install_samtools
[ -d $S/e_tg ]   || $MM create -y -q -p $S/e_tg -c conda-forge -c bioconda trim-galore=0.6.10 >> $O/install.log 2>&1 || fin FAIL_install_trimgalore
[ -d $S/wgbs_tools ] || { git clone -q https://github.com/nloyfer/wgbs_tools.git $S/wgbs_tools && git -C $S/wgbs_tools checkout -q v0.1.0 && (cd $S/wgbs_tools && PATH=$S/e_wgbs/bin:$PATH python setup.py >> $O/install.log 2>&1) || fin FAIL_wgbstools; }
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_tg/bin:$S/wgbs_tools:$PATH
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; trim_galore --version | grep -m1 version; cutadapt --version; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; } > $O/versions.txt
grep -q "0\.2\.0" $O/versions.txt || fin FAIL_bwameth_not_0.2.0
log "START threads=$T vcpus=$(nproc) $(tr '\n' ' ' < $O/versions.txt)"
REF=$S/ref/mm10.fa
# --- genome (in the background; reads are fetched and trimmed meanwhile)
( if [ ! -s $REF.bwameth.c2t.sa ]; then
    curl -sL -o ref/mm10.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz && gunzip -f ref/mm10.fa.gz && \
    bgzip -@ 8 -c ref/mm10.fa > ref/mm10.fa.gz && wgbstools init_genome mm10 --fasta_path $S/ref/mm10.fa.gz -@ 8 -f >> $O/init_genome.log 2>&1 && \
    T0=$(date +%s) && bwameth.py index $REF >> $O/index.log 2>&1 && echo "index $(( $(date +%s)-T0 )) s" >> $O/index.log; fi
  echo "$(zcat $S/wgbs_tools/references/mm10/CpG.bed.gz | wc -l)" > $O/cpg_count_mm10.txt; touch $S/ref/GENOME_DONE ) &
G=$!
# --- reads: ENA fastq with md5, Trim Galore --rrbs (removes adapter, and 2 bp at the 3' end of adapter-trimmed reads: the MspI fill-in)
RUNS=$(tail -n +2 $HERE/GSE77079_runs.csv | cut -d, -f1)
one_read(){ r=$1; f=$S/r6/fq/$r.fastq.gz
  m=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$r&result=read_run&fields=fastq_md5,fastq_ftp" | tail -1)
  md5=$(echo "$m" | cut -f2); url=$(echo "$m" | cut -f3)
  for t in 1 2 3; do [ -s $f ] && [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] && break; curl -s -o $f "https://$url"; done
  [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] || { log "$r MD5_FAIL"; return; }
  trim_galore --rrbs --cores 2 -o $S/r6/trim $f > $O/${r}_trim.log 2>&1 && cp $S/r6/trim/${r}.fastq.gz_trimming_report.txt $O/ 2>/dev/null; rm -f $f; log "$r trimmed"; }
export -f one_read log; export S O HERE
echo "$RUNS" | xargs -P 8 -I{} bash -c 'one_read {}'
wait $G; [ -f $S/ref/GENOME_DONE ] && [ -s $REF.bwameth.c2t.sa ] || fin FAIL_genome
for f in $REF $REF.bwameth.c2t*; do up "$f" "reference/mm10_bwameth/$(basename $f)"; done
# --- align, sort, bam2pat
for r in $RUNS; do
  t=$S/r6/trim/${r}_trimmed.fq.gz; [ -s $t ] || { log "$r no trimmed reads"; continue; }
  bwameth.py --threads $T --reference $REF $t 2>> $O/${r}_align.log | samtools view -b -o $S/r6/bam/$r.u.bam - || { log "$r align FAILED"; continue; }
  samtools sort -@ 16 -m 2G -T $S/r6/bam/$r.tmp -o $S/r6/bam/$r.bam $S/r6/bam/$r.u.bam && rm -f $S/r6/bam/$r.u.bam && samtools index $S/r6/bam/$r.bam
  samtools flagstat $S/r6/bam/$r.bam > $O/${r}_flagstat.txt
  wgbstools bam2pat $S/r6/bam/$r.bam -o $S/r6/pat --genome mm10 -@ 16 >> $O/${r}_bam2pat.log 2>&1 || { log "$r bam2pat FAILED"; continue; }
  for x in $S/r6/pat/$r.pat.gz $S/r6/pat/$r.pat.gz.csi; do [ -f $x ] && up $x "$P/$(basename $x)"; done; rm -f $S/r6/bam/$r.bam* $t; log "$r done"
done
fin DONE
