#!/bin/bash
# plan 60.24: the platelet test of the megakaryocyte column, on a Google machine; SOURCED by the job (W, R, B and log/up come from the launcher; return, not exit). Saves as it goes: each
# stage's output is put in the bucket the moment it exists and a relaunch takes it back (already read) instead of redoing it.
set -u; MP=$R/Biological_Physics/MethylPhys; IF=$MP/chain/MethylPhys_Interface; O=$B/mk_platelet; mkdir -p $W/p56 $W/pmk $W/cpg
have(){ gcloud storage ls "$O/$1" > /dev/null 2>&1; }
if have mk_platelet_compare.json; then log "already read: the comparison is in the bucket"; return 0; fi
# the hg19 CpG dictionary from the repository's index release (its checked size), then the platelet track from GEO
if have platelets.lbeta; then gcloud storage cp -q $O/platelets.lbeta $W/platelets.lbeta; log "already read: platelets.lbeta taken back"
else
  # the hg19 CpG dictionary from the repository's index release, in parts, checked against its SHA256SUMS stream line (as cov_read.yml)
  U=https://github.com/hmahaffeyges/IAM-Validation/releases/download/hg19-bwameth-index-v1; mkdir -p $W/s; curl -fsSL --retry 5 -o $W/s/sums $U/SHA256SUMS
  for q in $(awk '$2=="parts"{for(i=3;i<=NF;i++) print $i}' $W/s/sums); do curl -fsSL --retry 5 "$U/$q" || return 1; done \
    | tee >(sha256sum | cut -d' ' -f1 > $W/s/got) | zstd -dc -T4 | tar -C $W/s -xf - --wildcards '*references/hg19/CpG.bed.gz'; sleep 3
  [ "$(cat $W/s/got)" = "$(awk '$2=="stream"{print $1}' $W/s/sums)" ] || { log "index stream checksum failed"; return 1; }
  cp "$(find $W/s -name CpG.bed.gz | head -1)" $W/cpg/CpG.bed.gz
  [ "$(zcat $W/cpg/CpG.bed.gz | wc -l)" = 28217448 ] || { log "CpG dictionary incomplete"; return 1; }
  curl -sSfL --retry 5 -o $W/platelets.bedGraph.gz "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE206nnn/GSE206818/suppl/GSE206818_platelets_merged.bedGraph.gz"
  log "platelet track fetched: $(stat -c %s $W/platelets.bedGraph.gz) bytes, sha256 $(sha256sum $W/platelets.bedGraph.gz | cut -c1-16)"
  cd $IF && python3 run_sample.py --build seq-table --build-input $W/platelets.bedGraph.gz --build-out $W/platelets.lbeta --build-samples "level-bedgraph:hg19:$W/cpg/CpG.bed.gz" --bundle $W/platelets_convert.json > $W/convert.log 2>&1 || { log "convert failed: $(tail -2 $W/convert.log)"; return 1; }
  gcloud storage cp -q $W/platelets.lbeta $W/platelets_convert.json $O/; log "converted: $(tail -1 $W/convert.log | cut -c1-200)"
fi
for P0 in $(seq 0 250000 28217447); do P1=$((P0 + 250000)); [ $P1 -gt 28217448 ] && P1=28217448; f=piece_$(printf %08d $P0)_$(printf %08d $P1).parquet
  gcloud storage cp -q $B/pieces/$f $W/p56/ > /dev/null 2>&1; done
for P0 in $(seq 0 1000000 28217447); do P1=$((P0 + 1000000)); [ $P1 -gt 28217448 ] && P1=28217448   # the megakaryocyte pieces hold 1,000,000 CpGs each
  gcloud storage cp -q $B/pieces_mk/piece_$(printf %08d $P0)_$(printf %08d $P1).parquet $W/pmk/ > /dev/null 2>&1; done
[ "$(ls $W/p56 | wc -l)" -eq 113 ] && [ "$(ls $W/pmk | wc -l)" -eq 29 ] || { log "pieces missing: $(ls $W/p56 | wc -l) of 113, $(ls $W/pmk | wc -l) of 29"; return 1; }
cd $IF && python3 run_sample.py --build seq-compare --build-input $W/platelets.lbeta --build-samples $W/p56,$W/pmk --bundle $W/mk_platelet_compare.json > $W/compare.log 2>&1 || { log "compare failed: $(tail -2 $W/compare.log)"; return 1; }
gcloud storage cp -q $W/mk_platelet_compare.json $O/; log "compared: $(tail -1 $W/compare.log | cut -c1-300)"
