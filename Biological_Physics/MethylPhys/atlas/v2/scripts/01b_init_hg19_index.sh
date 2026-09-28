#!/bin/bash
set -e
cd /home/ubuntu/data/wgbs_tools
export PATH=/home/ubuntu/env/bin:$PATH
F=/home/ubuntu/data/hg19.fa.gz; B=/home/ubuntu/data/hg19.bgz.fa.gz
if [ ! -s "$B" ]; then gunzip -c "$F" | bgzip -@ 32 -c > "$B"; fi
echo "bgzipped: $(stat -c%s $B) bytes"
rm -rf references/hg19
./wgbstools init_genome hg19 --fasta_path "$B" -f 2>&1 | tail -4
ls -la references/hg19/ | head -12
zcat references/hg19/CpG.bed.gz | wc -l | awk '{print "hg19 CpG index sites: "$1}'
zcat references/hg19/CpG.bed.gz | head -3
