#!/bin/bash
# The megakaryocyte column into the atlas (v1.1) and the globe's new version on Zenodo; SOURCED by the Google launcher (W, R, B, log and up come from it;
# return, not exit). Saves as it goes: zenodo_mk_v11.py takes back every output already saved ('already read') and skips files already in a draft.
# Usage: MK_ZA=<atlas draft files URL> MK_ZG=<globe draft files URL> GBK=<bucket name>, then source this file.
apt-get install -y -qq zstd > /dev/null 2>&1; mkdir -p $W/cpg $W/s
if gcloud storage cp -q gs://$GBK/seq_atlas/mk_v11/CpG.bed.gz $W/cpg/CpG.bed.gz > /dev/null 2>&1; then log "already read: CpG dictionary"; else
  U=https://github.com/hmahaffeyges/IAM-Validation/releases/download/hg19-bwameth-index-v1; curl -fsSL --retry 5 -o $W/s/sums $U/SHA256SUMS
  for q in $(awk '$2=="parts"{for(i=3;i<=NF;i++) print $i}' $W/s/sums); do curl -fsSL --retry 5 "$U/$q" || return 1; done \
    | tee >(sha256sum | cut -d' ' -f1 > $W/s/got) | zstd -dc -T4 | tar -C $W/s -xf - --wildcards '*references/hg19/CpG.bed.gz'; sleep 3
  [ "$(cat $W/s/got)" = "$(awk '$2=="stream"{print $1}' $W/s/sums)" ] || { log "index stream checksum FAILED"; return 1; }
  cp "$(find $W/s -name CpG.bed.gz | head -1)" $W/cpg/CpG.bed.gz; [ "$(zcat $W/cpg/CpG.bed.gz | wc -l)" = 28217448 ] || { log "CpG dictionary incomplete"; return 1; }
  gcloud storage cp -q $W/cpg/CpG.bed.gz gs://$GBK/seq_atlas/mk_v11/ > /dev/null 2>&1; log "CpG dictionary checked"
fi
cd $R && python3 Biological_Physics/MethylPhys/deposit/zenodo_mk_v11.py $GBK $MK_ZA $MK_ZG $W/cpg/CpG.bed.gz $W/mk > $W/mk.log 2>&1; rc=$?
log "mk deposit rc $rc: $(tail -3 $W/mk.log | tr '\n' ' ' | cut -c1-300)"; up $W/mk.log
