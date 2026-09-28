#!/bin/bash
# wgbstools (Loyfer lab's own reader for .beta files) + the hg19 CpG index it needs, then read ONE sample at the array CpGs.
set -e
sudo apt-get install -y -q g++ make zlib1g-dev libbz2-dev liblzma-dev tabix bedtools samtools >/dev/null 2>&1
cd /home/ubuntu/data
[ -d wgbs_tools ] || git clone -q https://github.com/nloyfer/wgbs_tools.git
cd wgbs_tools && /home/ubuntu/env/bin/python setup.py >/dev/null 2>&1 && echo "wgbstools built"
[ -f references/hg19/CpG.bed.gz ] || /home/ubuntu/env/bin/python wgbstools.py init_genome hg19 2>&1 | tail -3
ls -la references/hg19/ | head
zcat references/hg19/CpG.bed.gz | wc -l | awk '{print "hg19 CpG index sites: "$1}'
B=$(ls /home/ubuntu/data/atlas_sources/loyfer2023/beta/*.beta | head -1); echo "test file: $(basename $B) $(stat -c%s $B) bytes"
/home/ubuntu/env/bin/python - "$B" <<'PY'
import sys, numpy as np, gzip
b=np.fromfile(sys.argv[1],dtype=np.uint8).reshape(-1,2); print("beta rows:", len(b), "| covered:", int((b[:,1]>0).sum()), "| median depth:", int(np.median(b[b[:,1]>0,1])))
PY
