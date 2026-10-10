#!/bin/bash
# Box Run 7 (2026-10-10): reference check of the withdrawn cross-species reading. 16 RRBS runs (GSE195869; mouse, rat, dog, rabbit; liver and
# heart; realign_4species_runs.csv) aligned to each species' own genome with the pinned pipeline of Box Run 6 (Trim Galore 0.6.10 --rrbs,
# bwa-meth 0.2.0, SAMtools 1.9, wgbstools 0.1.0 bam2pat), then ε by Stage Q's rule (boxruns/run2/eps_of.py, unchanged). Compared with the
# reference-free reading of the same runs. Usage: bash session7_realign.sh    Outputs: s3://<bucket>/results/BOXRUN7_XSP_REALIGN/
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/BOXRUN7_XSP_REALIGN; S=/mnt/scratch/run2; O=$S/out7xsp; X=$S/r7; mkdir -p $O $X/trim $X/bam $X/pat $S/ref; cd $S
PYB=/home/ubuntu/env/bin/python; REPO=~/IAM-Validation; HERE=$REPO/Biological_Physics/MethylPhys/boxruns/xspecies
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_tg/bin:$S/e_sra/bin:$S/wgbs_tools:$PATH
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/session7.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
fin(){ log "END $1"; echo "{\"status\": \"$1\"}" > $O/status.json; for f in $O/*; do [ -f "$f" ] && up "$f" "$P/$(basename $f)"; done; exit 0; }
log "START $(nproc) vcpus"
for i in $(seq 1 60); do [ -x $S/e_tg/bin/trim_galore ] && break; sleep 30; done   # installed by session 6
genome(){ g=$1; F=$S/ref/$g.fa
  if [ ! -s $F.bwameth.c2t.sa ]; then
    curl -sL -o $S/ref/$g.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/$g/bigZips/$g.fa.gz && gunzip -f $S/ref/$g.fa.gz && bgzip -@ 4 -c $F > $F.gz && \
    wgbstools init_genome $g --fasta_path $F.gz -@ 4 -f > $O/init_$g.log 2>&1 && bwameth.py index $F > $O/index_$g.log 2>&1 || { log "$g genome FAILED"; return; }
  fi; touch $S/ref/$g.DONE; log "$g genome ready"; }
for g in rn6 canFam3 oryCun2; do genome $g & done
for i in $(seq 1 720); do [ -f $S/ref/GENOME_DONE ] && [ -s $S/ref/mm10.fa.bwameth.c2t.sa ] && break; sleep 30; done; touch $S/ref/mm10.DONE   # mm10 from session 6
tail -n +2 $HERE/realign_4species_runs.csv | while IFS=, read -r r sp ti ind tb g key rest; do
  [ -f $X/trim/${r}_trimmed.fq.gz ] && continue
  $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').download_file('$B',sys.argv[1],sys.argv[2])" "$key" $X/$r.sra && \
  fastq-dump -Z $X/$r.sra 2>/dev/null | gzip -1 > $X/$r.fastq.gz && rm -f $X/$r.sra && \
  trim_galore --rrbs --cores 4 -o $X/trim $X/$r.fastq.gz > $O/${r}_trim.log 2>&1 && cp $X/trim/$r.fastq.gz_trimming_report.txt $O/ && rm -f $X/$r.fastq.gz; log "$r trimmed"
done
wait
tail -n +2 $HERE/realign_4species_runs.csv | while IFS=, read -r r sp ti ind tb g key rest; do
  for i in $(seq 1 480); do [ -f $S/ref/$g.DONE ] && break; sleep 30; done; F=$S/ref/$g.fa; [ -s $F.bwameth.c2t.sa ] || { log "$r no $g index"; continue; }
  bwameth.py --threads 24 --reference $F $X/trim/${r}_trimmed.fq.gz 2>> $O/${r}_align.log | samtools view -b -o $X/bam/$r.u.bam - && \
  samtools sort -@ 8 -m 2G -T $X/bam/$r.tmp -o $X/bam/$r.bam $X/bam/$r.u.bam && rm -f $X/bam/$r.u.bam && samtools index $X/bam/$r.bam && \
  samtools flagstat $X/bam/$r.bam > $O/${r}_flagstat.txt && wgbstools bam2pat $X/bam/$r.bam -o $X/pat --genome $g -@ 8 >> $O/${r}_bam2pat.log 2>&1 || { log "$r FAILED"; continue; }
  up $X/pat/$r.pat.gz "$P/$r.pat.gz"; rm -f $X/bam/$r.bam*; log "$r done"
done
$PYB $REPO/Biological_Physics/MethylPhys/boxruns/run2/eps_of.py $REPO/Biological_Physics/MethylPhys/chain $O/eps_aligned.json $X/pat/*.pat.gz > $O/eps.log 2>&1
fin DONE
