#!/bin/bash
# DEV-ERASE-01 recount on GitHub (to-do step 14, 2026-10-22; written so the recount does not need AWS). One sample (GSM, argument 1) from
# public SRA to .pat by the committed pipeline of erase_reads.sh: the same 20 spread blocks of 10,000,000 read pairs (erase_blocks.py), but
# each block read remotely by fastq-dump -N/-X instead of from a whole-run copy (a GitHub machine cannot hold a 30-60 GB run); bwa-meth
# 0.2.0 on UCSC mm10 (or hg19 with lambda and pUC19), SAMtools 1.9 sort, Sambamba 0.6.5 markdup, wgbstools 0.1.0 bam2pat. Then both
# counters on the same .pat: Stage Q (pat_site_table, the count the committed verdict used, as a check that the pipeline is the same) and
# the commissioned counter, chain Stage Q2 (count_pat). Writes $OUT/<gsm>_stageq.json, <gsm>_q2.json, <gsm>_timing.json and the .pat.
# Environment: MP (the MethylPhys folder), WORK, OUT, NW (aligner threads).
set -uo pipefail; GSM=$1; S=$WORK; T=${NW:-2}; mkdir -p $S/fq $S/bam $S/pat $S/ref $S/logs $OUT; E=$MP/doors/data/DEV_ERASE_01
MM="$S/bin/micromamba -r $S/mamba"; log(){ echo "$(date -u +%H:%M:%S) $*"; }
[ -x $S/bin/micromamba ] || /usr/bin/curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba || exit 11
[ -d $S/e_bm ]   || $MM create -y -q -p $S/e_bm -c conda-forge -c bioconda python=3.6 bwameth=0.2.0 bwa > $S/logs/install.log 2>&1 || exit 12
[ -d $S/e_wgbs ] || $MM create -y -q -p $S/e_wgbs -c conda-forge -c bioconda python=3.8 numpy pandas scipy htslib bedtools >> $S/logs/install.log 2>&1 || exit 13
[ -d $S/e_st19 ] || $MM create -y -q -p $S/e_st19 -c conda-forge -c bioconda samtools=1.9 >> $S/logs/install.log 2>&1 || exit 14
[ -d $S/e_sbb ]  || $MM create -y -q -p $S/e_sbb -c conda-forge -c bioconda sambamba=0.6.5 >> $S/logs/install.log 2>&1 || exit 15
[ -d $S/e_sra ]  || $MM create -y -q -p $S/e_sra -c conda-forge -c bioconda sra-tools >> $S/logs/install.log 2>&1 || exit 16
[ -d $S/wgbs_tools ] || { git clone -q https://github.com/nloyfer/wgbs_tools.git $S/wgbs_tools && git -C $S/wgbs_tools checkout -q 0.1.0 && (cd $S/wgbs_tools && PATH=$S/e_wgbs/bin:$PATH python setup.py >> $S/logs/install.log 2>&1); } || exit 17
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_sbb/bin:$S/e_sra/bin:$S/wgbs_tools:$PATH
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; sambamba 2>&1 | grep -m1 sambamba; fastq-dump --version | grep -m1 dump; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; } | tee $OUT/versions.txt
gen=$(awk -F, -v g=$GSM '$3==g{print $5; exit}' $E/erase_samples.csv); [ -n "$gen" ] || { echo "$GSM not in erase_samples.csv"; exit 19; }
[ "$gen" = "mm10" ] || { echo "hg19 with lambda and pUC19: not yet built for GitHub"; exit 20; }
REF=$S/ref/mm10.fa; T0=$(date +%s)
if [ ! -s $REF.bwameth.c2t.sa ]; then
  /usr/bin/curl -fsSL --retry 5 --retry-delay 30 -o $S/ref/mm10.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz || exit 18
  gunzip -f $S/ref/mm10.fa.gz && bgzip -@ 4 -c $REF > $S/ref/mm10.fa.gz || exit 23
  wgbstools init_genome mm10 --fasta_path $S/ref/mm10.fa.gz -@ 4 -f > $S/logs/init_genome.log 2>&1 || exit 25
  bwameth.py index $REF > $S/logs/index.log 2>&1 || exit 26
fi
TI=$(date +%s); log "index ready after $((TI-T0)) s"; df -h $S | tail -1
python3 $E/erase_blocks.py $E/erase_samples.csv $GSM > $S/${GSM}_blocks.txt; rm -rf $S/fq/*; i=0
while read r a b; do i=$((i+1)); ( for t in 1 2 3 4 5; do fastq-dump --split-files -N $a -X $b -O $S/fq/b$i $r > /dev/null 2>> $S/logs/${GSM}_fetch.err && break; rm -rf $S/fq/b$i; sleep $((30*t)); done ) &
  while [ $(jobs -r | wc -l) -ge 10 ]; do sleep 2; done; done < $S/${GSM}_blocks.txt; wait
cat $S/fq/b*/*_1.fastq > $S/fq/${GSM}_1.fq; cat $S/fq/b*/*_2.fastq > $S/fq/${GSM}_2.fq; rm -rf $S/fq/b*
n1=$(awk 'END{print NR/4}' $S/fq/${GSM}_1.fq); n2=$(awk 'END{print NR/4}' $S/fq/${GSM}_2.fq); TF=$(date +%s); log "fetched pairs $n1/$n2 in $((TF-TI)) s"
[ "$n1" = "$n2" ] && [ "$n1" -gt 0 ] || { echo "$GSM FETCH FAILED pairs $n1/$n2"; exit 30; }
bwameth.py --threads $T --reference $REF $S/fq/${GSM}_1.fq $S/fq/${GSM}_2.fq 2> $S/logs/${GSM}_align.log | samtools view -b -o $S/bam/$GSM.u.bam - || exit 31
rm -f $S/fq/${GSM}_*.fq; TA=$(date +%s); log "aligned in $((TA-TF)) s"
samtools sort -@ $T -m 1G -T $S/bam/$GSM.tmp -o $S/bam/$GSM.s.bam $S/bam/$GSM.u.bam && rm -f $S/bam/$GSM.u.bam || exit 32
sambamba markdup -l 1 -t $T --sort-buffer-size 4000 --overflow-list-size 10000000 --tmpdir $S/bam $S/bam/$GSM.s.bam $S/bam/$GSM.bam > $S/logs/${GSM}_markdup.log 2>&1 && rm -f $S/bam/$GSM.s.bam || exit 33
samtools index $S/bam/$GSM.bam; samtools flagstat $S/bam/$GSM.bam > $OUT/${GSM}_flagstat.txt
wgbstools bam2pat $S/bam/$GSM.bam -o $S/pat --genome $gen -@ $T > $S/logs/${GSM}_bam2pat.log 2>&1 || exit 34
PAT=$(ls $S/pat/$GSM*.pat.gz | head -1); rm -f $S/bam/$GSM.bam*; cp $PAT $OUT/; TP=$(date +%s); log "pat in $((TP-TA)) s"
( cd $MP/chain && python3 $E/erase_read_one.py $PAT $GSM > /dev/null && mv ${GSM}_stageq.json $OUT/ ) || exit 35
python3 $E/erase_q2_one.py $PAT $GSM $OUT $gen || exit 36
echo "{\"gsm\": \"$GSM\", \"pairs\": $n1, \"index_s\": $((TI-T0)), \"fetch_s\": $((TF-TI)), \"align_s\": $((TA-TF)), \"to_pat_s\": $((TP-TA)), \"threads\": $T, \"genome\": \"$gen\"}" > $OUT/${GSM}_timing.json
cat $OUT/${GSM}_timing.json; cat $OUT/${GSM}_stageq.json; cat $OUT/${GSM}_q2.json
