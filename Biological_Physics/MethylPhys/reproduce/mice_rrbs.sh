#!/bin/bash
# The mouse prediction's reads, from ENA to .pat files, with the pinned pipeline of the committed run (boxruns/run6_sam/session6.sh,
# without its storage steps): ENA fastq (md5 checked) -> Trim Galore 0.6.10 --rrbs -> bwa-meth 0.2.0 on UCSC mm10 -> SAMtools 1.9 sort/index
# -> wgbstools 0.1.0 bam2pat. Tools come from bioconda through micromamba into $WORK/tools. Run by reproduce.py (REPO, WORK, PY, MP set).
# 2026-10-18: the wgbs_tools tag is 0.1.0 (no v); the committed run's box image already held wgbs_tools 0.1.0, so its checkout line never ran.
# 2026-10-18 run 8: the PATH puts the tool environments first, and bwa-meth's environment carries its own old curl (7.87) that cannot find
# the system's certificates (curl: (60)); every download uses the system curl.
set -uo pipefail; S=$WORK/seq; T=${NW:-$(nproc)}; mkdir -p $S/fq $S/trim $S/bam $S/pat $S/ref $S/logs; cd $S
RUNS_CSV=$MP/boxruns/run6_sam/GSE77079_runs.csv; MM="$S/bin/micromamba -r $S/mamba"
[ -x $S/bin/micromamba ] || /usr/bin/curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba || exit 11
[ -d $S/e_bm ]   || $MM create -y -q -p $S/e_bm -c conda-forge -c bioconda python=3.6 bwameth=0.2.0 bwa > logs/install.log 2>&1 || exit 12
[ -d $S/e_wgbs ] || $MM create -y -q -p $S/e_wgbs -c conda-forge -c bioconda python=3.8 numpy pandas scipy htslib bedtools >> logs/install.log 2>&1 || exit 13
[ -d $S/e_st19 ] || $MM create -y -q -p $S/e_st19 -c conda-forge -c bioconda samtools=1.9 >> logs/install.log 2>&1 || exit 14
[ -d $S/e_tg ]   || $MM create -y -q -p $S/e_tg -c conda-forge -c bioconda trim-galore=0.6.10 >> logs/install.log 2>&1 || exit 15
[ -d $S/wgbs_tools ] || { git clone -q https://github.com/nloyfer/wgbs_tools.git $S/wgbs_tools && git -C $S/wgbs_tools checkout -q 0.1.0 && (cd $S/wgbs_tools && PATH=$S/e_wgbs/bin:$PATH python setup.py >> $S/logs/install.log 2>&1); } || exit 16
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_tg/bin:$S/wgbs_tools:$PATH
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; trim_galore --version | grep -m1 version; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; } | tee logs/versions.txt
grep -q "wgbstools 0\.1\.0" logs/versions.txt || exit 22
grep -q "0\.2\.0" logs/versions.txt || exit 17
REF=$S/ref/mm10.fa
if [ ! -s $REF.bwameth.c2t.sa ]; then
  # 2026-10-18 run 7 stopped here within a minute with no log: each part now logs and has its own exit code (18 download, 23 unpack, 24 bgzip,
  # 25 init_genome, 26 index); the download fails on an HTTP error instead of saving an error page
  /usr/bin/curl -fsSL --retry 5 --retry-delay 30 -o ref/mm10.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz 2> logs/genome_download.log || exit 18
  ls -l ref/mm10.fa.gz >> logs/genome_download.log; gunzip -f ref/mm10.fa.gz 2>> logs/genome_download.log || exit 23
  command -v bgzip >> logs/genome_download.log; bgzip -@ 8 -c $REF > ref/mm10.fa.gz 2>> logs/genome_download.log || exit 24
  wgbstools init_genome mm10 --fasta_path $S/ref/mm10.fa.gz -@ 8 -f > logs/init_genome.log 2>&1 || exit 25
  bwameth.py index $REF > logs/index.log 2>&1 || exit 26
fi
for r in $(tail -n +2 $RUNS_CSV | cut -d, -f1); do
  [ -s pat/$r.pat.gz ] && continue; f=fq/$r.fastq.gz
  # 2026-10-19 run 12: ENA's lookup service failed (HTTP 500) and the run stopped at 5 of 38. The address and MD5 of every file are in the
  # committed run list (GSE77079_runs.csv, columns fastq_ftp and fastq_md5), so they are read from there; downloads retry with growing pauses.
  url=$(awk -F, -v r=$r 'NR==1{for(i=1;i<=NF;i++){if($i=="fastq_ftp")a=i;if($i=="fastq_md5")b=i}} $1==r{print $a}' $RUNS_CSV)
  md5=$(awk -F, -v r=$r 'NR==1{for(i=1;i<=NF;i++){if($i=="fastq_ftp")a=i;if($i=="fastq_md5")b=i}} $1==r{print $b}' $RUNS_CSV)
  [ -n "$md5" ] && [ -n "$url" ] || { echo "$r not in the run list"; exit 19; }
  for t in 1 2 3 4 5 6; do [ -s $f ] && [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] && break
    rm -f $f; /usr/bin/curl -sf --retry 3 -o $f "https://$url" || echo "$r download failed (try $t)"; sleep $((30 * t)); done
  [ -s $f ] && [ "$(md5sum $f | cut -d' ' -f1)" = "$md5" ] || { echo "$r MD5_FAIL"; exit 19; }
  trim_galore --rrbs --cores 2 -o trim $f > logs/${r}_trim.log 2>&1 && rm -f $f
  bwameth.py --threads $T --reference $REF trim/${r}_trimmed.fq.gz 2>> logs/${r}_align.log | samtools view -b -o bam/$r.u.bam - || exit 20
  samtools sort -@ 8 -m 1G -T bam/$r.tmp -o bam/$r.bam bam/$r.u.bam && rm -f bam/$r.u.bam && samtools index bam/$r.bam
  wgbstools bam2pat bam/$r.bam -o pat --genome mm10 -@ 8 > logs/${r}_bam2pat.log 2>&1 || exit 21
  rm -f bam/$r.bam* trim/${r}_trimmed.fq.gz; echo "$r done"
done
