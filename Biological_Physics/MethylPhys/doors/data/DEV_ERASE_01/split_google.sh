#!/bin/bash
# The committed split alignment (split_align.yml) on a Google spot machine, for chunks GitHub's 6-hour limit cannot finish (2026-11-05: the
# last chunk of a run stored in genome order holds the run's unplaced reads; it downloaded in 2 h 10 min and was still aligning at 6 h). Same
# steps as split_align.yml: MODE=chunk runs erase_gh.sh with CHUNK=K/N and PATONLY on every read pair of the range and uploads the molecules;
# MODE=merge takes all N chunks from the bucket folder ($B/chunks: GitHub's chunks are copied there as they are), merges them (merge_pats.py)
# and reads the sample once by the one command with the list's flags. LIST is the repository-relative sample list.
MP=$R/Biological_Physics/MethylPhys; E=$MP/doors/data/DEV_ERASE_01; L_=$R/$LIST   # the committed sample list, given by the launcher (repository-relative path)
export INDEX_TAG=hg19-bwameth-index-v1 GITHUB_REPOSITORY=hmahaffeyges/IAM-Validation MP WORK=$W/ew OUT=$W/eout NW=4 SAMPLES=$L_ MAXPAIRS=999999999
mkdir -p $W/eout
if [ "$MODE" = chunk ]; then
  # 2026-11-06: stage checkpoints (erase_gh.sh, CKPT): a reclaimed machine's relaunch resumes from the saved reads or the saved alignment
  # the job switches checkpoints on with CKPT_ON=1 (the Google launcher refuses a chunk job without it); a job without it runs as before
  CKPT=$([ "${CKPT_ON:-0}" = 1 ] && echo $B/ckpt/${GSM}_c${K}_${N}) CHUNK=$K/$N PATONLY=1 bash $E/erase_gh.sh $GSM > $W/run.log 2>&1; rc=$?
  f=$W/eout/${GSM}_c${K}_${N}.pat.gz; log "chunk $K/$N rc $rc: $(grep -E 'fetched pairs|aligned in|pat in' $W/run.log | tr '\n' ' ' | cut -c1-250)"
  [ -s "$f" ] && gcloud storage cp -q $f $B/chunks/ > /dev/null 2>&1 && log "uploaded $(basename $f)" && { gcloud storage rm -r -q "$B/ckpt/${GSM}_c${K}_${N}" > /dev/null 2>&1; log "checkpoints cleared"; }
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
