#!/bin/bash
# The library floor by molecule position on the already-aligned molecule files in the build bucket (no alignment; spent data): every .pat
# piece under seq_atlas/jhu_cd4, seq_atlas/bt3 and seq_atlas/jhu_tissue, shard SHARD of NSH, four pieces at a time (one per core), each read
# by run_sample.py --build seq-floor-position; each piece's bundle is put in the bucket as it finishes and skipped on a relaunch.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; BR=${B%/*}; O=$B/floor_position; mkdir -p $W/fp
for P in jhu_cd4 bt3 jhu_tissue; do gcloud storage ls "$BR/$P/**.pat.gz" 2>/dev/null; done | sort > $W/all.txt
awk -v s=$SHARD -v n=$NSH '(NR-1)%n==s' $W/all.txt > $W/mine.txt; log "pieces: $(wc -l < $W/mine.txt) of $(wc -l < $W/all.txt)"
gcloud storage ls "$O/" 2>/dev/null | sed 's#.*/##' > $W/done.txt
one(){ u=$1; f=$(basename $u .pat.gz); grep -qx "$f.json" $W/done.txt && return 0; gcloud storage cp -q $u $W/fp/$f.pat.gz > /dev/null 2>&1 || return 1
  (cd $IF && python3 run_sample.py --build seq-floor-position --build-input $W/fp/$f.pat.gz --bundle $W/fp/$f.json > $W/fp/$f.log 2>&1) && gcloud storage cp -q $W/fp/$f.json $O/ > /dev/null 2>&1; rm -f $W/fp/$f.pat.gz; }
export -f one; export W O IF
cat $W/mine.txt | xargs -P 4 -I{} bash -c 'one {}'
log "shard $SHARD done: $(ls $W/fp/*.json 2>/dev/null | wc -l) read"
