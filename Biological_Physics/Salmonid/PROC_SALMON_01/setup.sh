#!/bin/bash
# PROC-SALMON-01 setup: tools via micromamba (bioconda), O. mykiss Omyk_1.0 reference, Bismark index.
set -euo pipefail
S=/home/ubuntu/data/salmon; mkdir -p $S/ref && W0=$(pwd) && cd $S
if [ ! -x $S/mm/bin/bismark ] && python3 $W0/s3io.py getsplit tools $S/tools.tgz; then tar xzf $S/tools.tgz -C $S && rm -f $S/tools.tgz && touch $S/.cached && echo RESTORED_FROM_S3; fi
if [ ! -x $S/mm/bin/bismark ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba
  $S/bin/micromamba create -y -q -p $S/mm -c conda-forge -c bioconda bismark=0.24 bowtie2 trim-galore cutadapt samtools pigz pysam pandas pyarrow python=3.11 >/dev/null
fi
export PATH=$S/mm/bin:$PATH; bismark --version | head -2 | tail -1; bowtie2 --version | head -1
cd ref
if [ ! -s Omyk_1.0.fa ]; then
  curl -s -o g.fna.gz https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/002/163/495/GCF_002163495.1_Omyk_1.0/GCF_002163495.1_Omyk_1.0_genomic.fna.gz
  pigz -dc g.fna.gz > Omyk_1.0.fa && rm g.fna.gz && samtools faidx Omyk_1.0.fa
fi
grep -c ">" Omyk_1.0.fa; awk '{s+=$2} END {print s/1e9" Gb"}' Omyk_1.0.fa.fai
if [ ! -d Bisulfite_Genome/GA_conversion ] || [ ! -s Bisulfite_Genome/GA_conversion/BS_GA.rev.2.bt2l ] && [ ! -s Bisulfite_Genome/GA_conversion/BS_GA.rev.2.bt2 ]; then
  bismark_genome_preparation --parallel 60 --bowtie2 . > prep.log 2>&1
fi
ls Bisulfite_Genome/*/ | head
if [ ! -f $S/.cached ]; then cd $S && tar cf - mm ref | $S/mm/bin/pigz -p 32 > $S/tools.tgz && python3 $W0/s3io.py putsplit $S/tools.tgz downloads/salmon_work/tools.tgz && touch $S/.cached && rm -f $S/tools.tgz* && echo CACHED_TO_S3; fi
echo SETUP_DONE
