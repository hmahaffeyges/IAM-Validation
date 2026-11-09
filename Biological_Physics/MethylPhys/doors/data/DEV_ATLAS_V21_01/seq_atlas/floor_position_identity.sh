#!/bin/bash
# The molecule-position split at each library's identity sites (2026-11-09; development, spent, already-aligned molecule files, no alignment):
# every .pat piece of the 18 sorted libraries under seq_atlas/jhu_cd4 and seq_atlas/bt3, shard SHARD of NSH, four at a time (one per core),
# each read by run_sample.py --build seq-floor-position --build-cell CELL, CELL the library's main cell as the IAM-Deconvolver found it
# (identity_position_cells.csv, from its own earlier bundle; never the label). Each bundle is put in the bucket as it finishes; already read are skipped.
set -u; IF=$R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface; SDIR=$R/Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/seq_atlas
BR=${B%/*}; O=$B/floor_position_identity; mkdir -p $W/fi
for P in jhu_cd4 bt3; do gcloud storage ls "$BR/$P/**.pat.gz" 2>/dev/null; done | sort > $W/all.txt
awk -v s=$SHARD -v n=$NSH '(NR-1)%n==s' $W/all.txt > $W/mine.txt; log "pieces: $(wc -l < $W/mine.txt) of $(wc -l < $W/all.txt)"
gcloud storage ls "$O/" 2>/dev/null | sed 's#.*/##' > $W/done.txt
one(){ u=$1; f=$(basename $u .pat.gz); grep -qx "$f.json" $W/done.txt && { echo "already read: $f"; return 0; }
  g=$(echo $f | grep -o "GSM[0-9]*" | head -1); c=$(awk -F, -v g=$g '$1==g{print $2}' $SDIR/identity_position_cells.csv); [ -z "$c" ] && return 0
  gcloud storage cp -q $u $W/fi/$f.pat.gz > /dev/null 2>&1 || return 1
  (cd $IF && python3 run_sample.py --build seq-floor-position --build-input $W/fi/$f.pat.gz --build-cell "$c" --bundle $W/fi/$f.json > $W/fi/$f.log 2>&1) && gcloud storage cp -q $W/fi/$f.json $O/ > /dev/null 2>&1; rm -f $W/fi/$f.pat.gz; }
export -f one; export W O IF SDIR
cat $W/mine.txt | xargs -P 4 -I{} bash -c 'one {}'
log "shard $SHARD done: $(ls $W/fi/*.json 2>/dev/null | wc -l) read"
