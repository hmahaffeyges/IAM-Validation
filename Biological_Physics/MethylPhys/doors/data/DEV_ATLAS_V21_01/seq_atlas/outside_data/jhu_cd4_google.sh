#!/bin/bash
# The Johns Hopkins healthy CD4 T-cell test (GSE86340) on a Google spot machine: one job per machine, run by the launcher after its preamble
# (gcp.py: $R the repository, $W the work folder, $B the bucket folder, log and up). MODE=chunk aligns chunk K of N of sample GSM by the committed
# pipeline (erase_gh.sh with CHUNK and PATONLY, every read pair of every run, as split_align.yml does on GitHub) and uploads its molecules;
# MODE=merge downloads all N chunks, merges them (merge_pats.py) and reads the sample once by the one command with the list's flags.
MP=$R/Biological_Physics/MethylPhys; E=$MP/doors/data/DEV_ERASE_01; L_=$MP/doors/data/DEV_ATLAS_V21_01/seq_atlas/outside_data/jhu_cd4_samples.csv
export INDEX_TAG=hg19-bwameth-index-v1 GITHUB_REPOSITORY=hmahaffeyges/IAM-Validation MP WORK=$W/ew OUT=$W/eout NW=4 SAMPLES=$L_ MAXPAIRS=999999999
mkdir -p $W/eout
if [ "$MODE" = chunk ]; then
  CHUNK=$K/$N PATONLY=1 bash $E/erase_gh.sh $GSM > $W/run.log 2>&1; rc=$?
  f=$W/eout/${GSM}_c${K}_${N}.pat.gz; log "chunk $K/$N rc $rc: $(grep -E 'fetched pairs|aligned in|pat in' $W/run.log | tr '\n' ' ' | cut -c1-250)"
  [ -s "$f" ] && gcloud storage cp -q $f $B/chunks/ > /dev/null 2>&1 && log "uploaded $(basename $f)"
  cp $W/run.log $W/${GSM}_c${K}_${N}_run.log; up $W/${GSM}_c${K}_${N}_run.log
else
  mkdir -p $W/chunks; gcloud storage cp -q "$B/chunks/${GSM}_c*_${N}.pat.gz" $W/chunks/ > /dev/null 2>&1
  n=$(ls $W/chunks/${GSM}_c*_${N}.pat.gz | wc -l); log "chunks with molecules: $n of $N"
  [ "$n" = "$N" ] || { log "MISSING CHUNKS: merge refused ($n of $N)"; exit 0; }
  . $W/v/bin/activate; python3 $E/merge_pats.py $W/$GSM.pat.gz $(ls $W/chunks/${GSM}_c*_${N}.pat.gz | sort -V)
  FULL=1 PATFILE=$W/$GSM.pat.gz bash $E/erase_gh.sh $GSM > $W/run.log 2>&1; rc=$?
  log "merge rc $rc: $(tail -3 $W/run.log | tr '\n' ' ' | cut -c1-300)"
  for x in $W/eout/${GSM}_bundle.json $W/eout/${GSM}*_metaa_sites.csv.gz $W/eout/${GSM}*_whole_sites.csv.gz $W/eout/${GSM}*.html; do [ -f "$x" ] && gcloud storage cp -q $x $B/out/ > /dev/null 2>&1; done
  cp $W/run.log $W/${GSM}_merge_run.log; up $W/${GSM}_merge_run.log
fi
