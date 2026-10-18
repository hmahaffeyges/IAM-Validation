#!/bin/bash
# The mouse prediction's reads, from ENA to .pat files, with the pinned pipeline of the committed run (boxruns/run6_sam/session6.sh,
# without its storage steps): ENA fastq (md5 checked) -> Trim Galore 0.6.10 --rrbs -> bwa-meth 0.2.0 on UCSC mm10 -> SAMtools 1.9 sort/index
# -> wgbstools 0.1.0 bam2pat. Tools come from bioconda through micromamba into $WORK/tools. Run by reproduce.py (REPO, WORK, PY, MP set).
set -uo pipefail; S=$WORK/seq; T=${NW:-$(nproc)}; mkdir -p $S/fq $S/trim $S/bam $S/pat $S/ref $S/logs; cd $S
RUNS_CSV=$MP/boxruns/run6_sam/GSE77079_runs.csv; MM="$S/bin/micromamba -r $S/mamba"
[ -x $S/bin/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba || exit 11
[ -d $S/e_bm ]   || $MM create -y -q -p $S/e_bm -c conda-forge -c bioconda python=3.6 bwameth=0.2.0 bwa > logs/install.log 2>&1 || exit 12
[ -d $S/e_wgbs ] || $MM create -y -q -p $S/e_wgbs -c conda-forge -c bioconda python=3.8 numpy pandas scipy htslib bedtools >> logs/install.log 2>&1 || exit 13
[ -d $S/e_st19 ] || $MM create -y -q -p $S/e_st19 -c conda-forge -c bioconda samtools=1.9 >> logs/install.log 2>&1 || exit 14
[ -d $S/e_tg ]   || $MM create -y -q -p $S/e_tg -c conda-forge -c bioconda trim-galore=0.6.10 >> logs/install.log 2>&1 || exit 15
[ -d $S/wgbs_tools ] || { git clone -q https://github.com/nloyfer/wgbs_tools.git $S/wgbs_tools && git -C $S/wgbs_tools checkout -q v0.1.0 && (cd $S/wgbs_tools && PATH=$S/e_wgbs/bin:$PATH python setup.py >> $S/logs/install.log 2>&1); } || exit 16
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_tg/bin:$S/wgbs_tools:$PATH
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; trim_galore --version | grep -m1 version; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; } | tee logs/versions.txt
grep -q "0\.2\.0" logs/versions.txt || exit 17
REF=$S/ref/mm10.fa
if [ ! -s $REF.bwameth.c2t.sa ]; then
  curl -sL -o ref/mm10.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz && gunzip -f ref/mm10.fa.gz && bgzip -@ 8 -c $REF > ref/mm10.fa.gz \
   && wgbstools init_genome mm10 --fasta_path $S/ref/mm10.fa.gz -@ 8 -f > logs/init_genome.log 2>&1 && bwameth.py index $REF > logs/index.log 2>&1 || exit 18
fi
for r in $(tail -n +2 $RUNS_CSV | cut -d, -f1); do
  [ -s pat/$r.pat.gz ] && continue; f=fq/$r.fastq.gz
  m=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$r&result=read_run&fields=fastq_md5,fastq_ftp" | tail -1); md5=$(echo "$m" | cut -f2); url=$(echo "$m" | cut -f3)
  for t in 1 2 3 4 5; do [ -s $f ] && [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] && break; curl -s -o $f "https://$url"; sleep 5; done
  [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] || { echo "$r MD5_FAIL"; exit 19; }
  trim_galore --rrbs --cores 2 -o trim $f > logs/${r}_trim.log 2>&1 && rm -f $f
  bwameth.py --threads $T --reference $REF trim/${r}_trimmed.fq.gz 2>> logs/${r}_align.log | samtools view -b -o bam/$r.u.bam - || exit 20
  samtools sort -@ 8 -m 1G -T bam/$r.tmp -o bam/$r.bam bam/$r.u.bam && rm -f bam/$r.u.bam && samtools index bam/$r.bam
  wgbstools bam2pat bam/$r.bam -o pat --genome mm10 -@ 8 > logs/${r}_bam2pat.log 2>&1 || exit 21
  rm -f bam/$r.bam* trim/${r}_trimmed.fq.gz; echo "$r done"
done
