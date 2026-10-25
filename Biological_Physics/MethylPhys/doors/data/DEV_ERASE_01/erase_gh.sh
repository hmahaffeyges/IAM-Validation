#!/bin/bash
# DEV-ERASE-01 recount on GitHub (to-do step 14, 2026-10-22; written so the recount does not need AWS). One sample (GSM, argument 1) from
# public SRA to .pat by the committed pipeline of erase_reads.sh: the same 20 spread blocks of 10,000,000 read pairs (erase_blocks.py), but
# each block read remotely by fastq-dump -N/-X instead of from a whole-run copy (a GitHub machine cannot hold a 30-60 GB run); bwa-meth
# 0.2.0 on UCSC mm10 (or hg19 with lambda and pUC19), SAMtools 1.9 sort, Sambamba 0.6.5 markdup, wgbstools 0.1.0 bam2pat. Then both
# counters on the same .pat: Stage Q (pat_site_table, the count the committed verdict used, as a check that the pipeline is the same) and
# the commissioned counter, chain Stage Q2 (count_pat). Writes $OUT/<gsm>_stageq.json, <gsm>_q2.json, <gsm>_timing.json and the .pat.
# Environment: MP (the MethylPhys folder), WORK, OUT, NW (aligner threads).
# 2026-10-23: optional SAMPLES (a sample list in erase_samples.csv's format; default erase_samples.csv), SPECIMEN and FULL=1: the one command
# reads the molecules in full (IAM-A, and with --dev-seq-values 1 the IAM-Deconvolver, Met-A, territories, globe and sphere) instead of the
# count alone. Used by DEV-ONE-DRAW-01 plasma_seq_01 (real plasma). The recount's defaults are unchanged.
CHAINPY=$(command -v python3)   # 2026-10-23: the chain's own Python (the one preflight checked), fixed before the tools' environments go on PATH
set -uo pipefail; GSM=$1; S=$WORK; T=${NW:-2}; mkdir -p $S/fq $S/bam $S/pat $S/ref $S/logs $OUT; E=$MP/doors/data/DEV_ERASE_01
MM="$S/bin/micromamba -r $S/mamba"; log(){ echo "$(date -u +%H:%M:%S) $*"; }
[ -x $S/bin/micromamba ] || /usr/bin/curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $S bin/micromamba || exit 11
[ -d $S/e_bm ]   || $MM create -y -q -p $S/e_bm -c conda-forge -c bioconda python=3.6 bwameth=0.2.0 bwa > $S/logs/install.log 2>&1 || exit 12
[ -d $S/e_wgbs ] || $MM create -y -q -p $S/e_wgbs -c conda-forge -c bioconda python=3.8 numpy pandas scipy htslib bedtools >> $S/logs/install.log 2>&1 || exit 13
[ -d $S/e_st19 ] || $MM create -y -q -p $S/e_st19 -c conda-forge -c bioconda samtools=1.9 >> $S/logs/install.log 2>&1 || exit 14
[ -d $S/e_sbb ]  || $MM create -y -q -p $S/e_sbb -c conda-forge -c bioconda sambamba=0.6.5 >> $S/logs/install.log 2>&1 || exit 15
[ -d $S/e_sra ]  || $MM create -y -q -p $S/e_sra -c conda-forge -c bioconda sra-tools >> $S/logs/install.log 2>&1 || exit 16
[ -d $S/wgbs_tools ] || { git clone -q https://github.com/nloyfer/wgbs_tools.git $S/wgbs_tools && git -C $S/wgbs_tools checkout -q 0.1.0 && (cd $S/wgbs_tools && PATH=$S/e_wgbs/bin:$PATH python setup.py >> $S/logs/install.log 2>&1); } || exit 17
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_sbb/bin:$S/e_sra/bin:$S/wgbs_tools:$PATH
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; sambamba 2>&1 | grep -m1 sambamba; fastq-dump --version | grep -m1 dump; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; } | tee $OUT/versions.txt
gen=$(awk -F, -v g=$GSM '$3==g{print $5; exit}' ${SAMPLES:-$E/erase_samples.csv}); [ -n "$gen" ] || { echo "$GSM not in erase_samples.csv"; exit 19; }
# 2026-10-25: SPECIMEN from the sample list's specimen column when the list has one and SPECIMEN is not set (the leukemia study's samples:
# leukemias "not stated", cord-blood B cells "sorted b cells", the Reh line "cell line"); the full reading once fell back to plasma;
# placed before the saved-molecules branch, which the merge takes (it was below it at first, so the merge never read it)
[ -z "${SPECIMEN:-}" ] && SPECIMEN=$(python3 -c "import csv,sys; r=[x for x in csv.DictReader(open(sys.argv[1])) if x['gsm']==sys.argv[2]]; print(r[0].get('specimen') or '' if r else '')" ${SAMPLES:-$E/erase_samples.csv} $GSM); export SPECIMEN; log "specimen: ${SPECIMEN:-plasma (default)}"
if [ -n "${PATFILE:-}" ]; then   # 2026-10-23: count the molecules an earlier run saved, without aligning again
  [ -s "$PATFILE" ] || { echo "PATFILE $PATFILE missing"; exit 20; }; PAT=$PATFILE; n1=0; T0=$(date +%s); TI=$T0; TF=$T0; TA=$T0; TP=$T0; log "count only, on the saved molecules $PAT"
else
T0=$(date +%s)
if [ "$gen" = "mm10" ]; then
  REF=$S/ref/mm10.fa
  if [ ! -s $REF.bwameth.c2t.sa ]; then
    /usr/bin/curl -fsSL --retry 5 --retry-delay 30 -o $S/ref/mm10.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz || exit 18
    gunzip -f $S/ref/mm10.fa.gz && bgzip -@ 4 -c $REF > $S/ref/mm10.fa.gz || exit 23
    wgbstools init_genome mm10 --fasta_path $S/ref/mm10.fa.gz -@ 4 -f > $S/logs/init_genome.log 2>&1 || exit 25
    bwameth.py index $REF > $S/logs/index.log 2>&1 || exit 26
  fi
else   # 2026-10-22: hg19 with lambda and pUC19, built as the committed pipeline built it (boxruns/run2/session1.sh: UCSC hg19, NCBI NC_001416.1, L09137.2)
  REF=$S/ref/hg19_lambda_puc19.fa
  # 2026-10-24: the index (and wgbstools' hg19 dictionary) built once by build_index.yml through this same block, unchanged, and kept as
  # release files; each job streams them in and checks the stream's SHA-256 instead of rebuilding (about 6,000 s per job). INDEX_TAG unset,
  # or a failed check: built here, as before.
  if [ -n "${INDEX_TAG:-}" ] && [ ! -s $REF.bwameth.c2t.sa ]; then
    U="https://github.com/$GITHUB_REPOSITORY/releases/download/$INDEX_TAG"; /usr/bin/curl -fsSL --retry 5 -o $S/idx_sums "$U/SHA256SUMS" \
      && for q in $(awk '$2=="parts"{for(i=3;i<=NF;i++) print $i}' $S/idx_sums); do /usr/bin/curl -fsSL --retry 5 "$U/$q" || exit 1; done \
         | tee >(sha256sum | cut -d' ' -f1 > $S/idx_got) | zstd -dc -T4 | tar -C $S -xf - \
      && sleep 2 && [ "$(cat $S/idx_got)" = "$(awk '$2=="stream"{print $1}' $S/idx_sums)" ] \
      && log "index fetched from release $INDEX_TAG, stream checksum verified" || { log "index fetch failed: building"; rm -rf $S/ref/* $S/wgbs_tools/references/hg19; }
  fi
  if [ ! -s $REF.bwameth.c2t.sa ]; then
    /usr/bin/curl -fsSL --retry 5 --retry-delay 30 -o $S/ref/hg19.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz || exit 18
    gunzip -f $S/ref/hg19.fa.gz && bgzip -@ 4 -f $S/ref/hg19.fa || exit 23
    wgbstools init_genome hg19 --fasta_path $S/ref/hg19.fa.gz -@ 4 -f > $S/logs/init_genome.log 2>&1 || exit 25
    N=$(zcat $S/wgbs_tools/references/hg19/CpG.bed.gz | wc -l); log "CpG sites in the hg19 dictionary: $N (must be 28217448)"; [ "$N" = 28217448 ] || exit 27
    EU="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=fasta&id"
    { /usr/bin/curl -fsSL --retry 5 "$EU=NC_001416.1" | sed 's/^>.*/>lambda/'; /usr/bin/curl -fsSL --retry 5 "$EU=L09137.2" | sed 's/^>.*/>pUC19/'; } > $S/ref/spikes.fa
    [ "$(grep -c '>' $S/ref/spikes.fa)" = 2 ] || exit 28
    zcat $S/ref/hg19.fa.gz > $REF && cat $S/ref/spikes.fa >> $REF
    bwameth.py index $REF > $S/logs/index.log 2>&1 || exit 26
  fi
fi
TI=$(date +%s); log "index ready after $((TI-T0)) s"; df -h $S | tail -1
if [ "${INDEXONLY:-0}" = 1 ]; then   # 2026-10-24 (build_index.yml): stop once the index is built
  tar -C $S -cf - ref wgbs_tools/references/hg19 | zstd -T4 -6 | tee >(sha256sum | cut -d' ' -f1 > $OUT/stream_sha) | split -b 1900M - $OUT/hg19idx.tar.zst.part
  sleep 2; { echo "$(cat $OUT/stream_sha) stream"; echo "x parts $(cd $OUT && ls hg19idx.tar.zst.part* | tr '\n' ' ')"; } > $OUT/SHA256SUMS; cat $OUT/SHA256SUMS; exit 0
fi
python3 $E/erase_blocks.py ${SAMPLES:-$E/erase_samples.csv} $GSM ${CHUNK:-${MAXPAIRS:-10000000}} > $S/${GSM}_blocks.txt   # CHUNK=k/n: one contiguous range (split alignment, 2026-10-24); the comment once swallowed the two commands before it (fixed 2026-10-24)
fetch_all(){ rm -rf $S/fq/*; i=0   # --split-3 (2026-10-25): only true pairs go to _1 and _2; a spot with one empty read goes to a third file, unused
  while read r a b; do i=$((i+1)); ( for t in 1 2 3 4 5; do fastq-dump --split-3 -N $a -X $b -O $S/fq/b$i $r > /dev/null 2>> $S/logs/${GSM}_fetch.err && break; rm -rf $S/fq/b$i; sleep $((30*t)); done ) &
    while [ $(jobs -r | wc -l) -ge 10 ]; do sleep 2; done; done < $S/${GSM}_blocks.txt; wait
  # 2026-10-25: a block whose read-1 and read-2 files differ in length (seen on the last chunk of cord-blood control 3: 7,060,310 against
  # 7,217,487, no error) keeps the pairs that are true pairs, record by record by read name, up to the shorter file; it fails if the names do
  # not match there. What was dropped is logged. The fetch's error log is copied into the run log on any mismatch.
  for d in $S/fq/b*; do f1=$(ls $d/*_1.fastq 2>/dev/null); f2=$(ls $d/*_2.fastq 2>/dev/null); [ -n "$f1" ] && [ -n "$f2" ] || continue
    c1=$(awk 'END{print NR/4}' $f1); c2=$(awk 'END{print NR/4}' $f2); [ "$c1" = "$c2" ] && continue
    m=$(( c1 < c2 ? c1 : c2 )); x1=$(awk -v n=$((4*m-3)) 'NR==n{sub(/\/[12]$/,"",$1); print $1; exit}' $f1); x2=$(awk -v n=$((4*m-3)) 'NR==n{sub(/\/[12]$/,"",$1); print $1; exit}' $f2)
    log "block $(basename $d): read 1 $c1, read 2 $c2 records; name at record $m: $x1 / $x2"; tail -5 $S/logs/${GSM}_fetch.err 2>/dev/null | sed 's/^/  fetch.err: /'
    if [ -n "$x1" ] && [ "$x1" = "$x2" ]; then head -n $((4*m)) $f1 > $f1.t && mv $f1.t $f1; head -n $((4*m)) $f2 > $f2.t && mv $f2.t $f2; log "block $(basename $d): kept $m true pairs, dropped $(( c1 + c2 - 2*m )) unpaired records"; fi
  done
  cat $S/fq/b*/*_1.fastq > $S/fq/${GSM}_1.fq; cat $S/fq/b*/*_2.fastq > $S/fq/${GSM}_2.fq; rm -rf $S/fq/b*
  n1=$(awk 'END{print NR/4}' $S/fq/${GSM}_1.fq); n2=$(awk 'END{print NR/4}' $S/fq/${GSM}_2.fq); }
fetch_all; TF=$(date +%s); log "fetched pairs $n1/$n2 in $((TF-TI)) s"
if [ "$n1" != "$n2" ] || [ "$n1" = 0 ]; then log "short fetch ($n1/$n2): fetching once more"; fetch_all; TF=$(date +%s); log "fetched pairs $n1/$n2 in $((TF-TI)) s"; fi   # 2026-10-25: a read file came back short with no error
[ "$n1" = "$n2" ] && [ "$n1" -gt 0 ] || { echo "$GSM FETCH FAILED pairs $n1/$n2"; exit 30; }
bwameth.py --threads $T --reference $REF $S/fq/${GSM}_1.fq $S/fq/${GSM}_2.fq 2> $S/logs/${GSM}_align.log | samtools view -b -o $S/bam/$GSM.u.bam - || exit 31
rm -f $S/fq/${GSM}_*.fq; TA=$(date +%s); log "aligned in $((TA-TF)) s"
samtools sort -@ $T -m 1G -T $S/bam/$GSM.tmp -o $S/bam/$GSM.s.bam $S/bam/$GSM.u.bam && rm -f $S/bam/$GSM.u.bam || exit 32
sambamba markdup -l 1 -t $T --sort-buffer-size 4000 --overflow-list-size 10000000 --tmpdir $S/bam $S/bam/$GSM.s.bam $S/bam/$GSM.bam > $S/logs/${GSM}_markdup.log 2>&1 && rm -f $S/bam/$GSM.s.bam || exit 33
samtools index $S/bam/$GSM.bam; samtools flagstat $S/bam/$GSM.bam > $OUT/${GSM}_flagstat.txt
wgbstools bam2pat $S/bam/$GSM.bam -o $S/pat --genome $gen -@ $T > $S/logs/${GSM}_bam2pat.log 2>&1 || {
  # 2026-10-24: with the index fetched from the release, bam2pat stopped (exit 34) in every job; its log is kept, the genome dictionary is
  # rebuilt from the fetched hg19 FASTA the committed way (init_genome, the CpG count checked), and bam2pat is run once more
  cp $S/logs/${GSM}_bam2pat.log $OUT/ 2>/dev/null; ls -la $S/wgbs_tools/references/$gen > $OUT/${GSM}_references_ls.txt 2>&1; tail -20 $S/logs/${GSM}_bam2pat.log
  log "bam2pat failed; rebuilding the $gen dictionary and retrying once"
  wgbstools init_genome $gen --fasta_path $S/ref/$gen.fa.gz -@ $T -f > $S/logs/init_genome_retry.log 2>&1 || { cp $S/logs/init_genome_retry.log $OUT/; exit 25; }
  if [ "$gen" = hg19 ]; then N=$(zcat $S/wgbs_tools/references/hg19/CpG.bed.gz | wc -l); log "CpG sites in the hg19 dictionary: $N (must be 28217448)"; [ "$N" = 28217448 ] || exit 27; fi
  rm -rf $S/pat/*; wgbstools bam2pat $S/bam/$GSM.bam -o $S/pat --genome $gen -@ $T > $S/logs/${GSM}_bam2pat.log 2>&1 || { cp $S/logs/${GSM}_bam2pat.log $OUT/; exit 34; }
  log "bam2pat succeeded after the dictionary was rebuilt"; }
PAT=$(ls $S/pat/$GSM*.pat.gz | head -1); rm -f $S/bam/$GSM.bam*; cp $PAT $OUT/; TP=$(date +%s); log "pat in $((TP-TA)) s"
if [ "${PATONLY:-0}" = 1 ]; then   # 2026-10-24 (split alignment): one chunk's molecules only; the merge job counts them all once
  C=$(echo "${CHUNK:-0/1}" | tr / _); mv $OUT/$(basename $PAT) $OUT/${GSM}_c${C}.pat.gz
  echo "{\"gsm\": \"$GSM\", \"chunk\": \"${CHUNK:-}\", \"pairs\": $n1, \"fetch_s\": $((TF-TI)), \"align_s\": $((TA-TF)), \"to_pat_s\": $((TP-TA))}" > $OUT/${GSM}_c${C}_timing.json; exit 0
fi
fi
# 2026-10-22 (author: "if the chain didnt do it, it didnt happen"): the count is the one command's, run_sample.py --dev-count-only, flagged; the
# older Stage Q cross-check (erase_read_one.py) called chain pieces directly and is no longer run; the .pat file is kept, so it can be rerun
if [ "${FULL:-0}" = 1 ]; then   # 2026-10-23: the full reading by the one command (plasma_seq_01)
( cd $MP/chain/MethylPhys_Interface && "$CHAINPY" run_sample.py --pat $PAT --specimen "${SPECIMEN:-plasma}" --id $GSM --dev-seq-values 1 \
    --out $OUT/${GSM}_report.html --bundle $OUT/${GSM}_bundle.json > $OUT/${GSM}_run_sample.log 2>&1 ) || exit 35
echo "{\"gsm\": \"$GSM\", \"pairs\": $n1, \"index_s\": $((TI-T0)), \"fetch_s\": $((TF-TI)), \"align_s\": $((TA-TF)), \"to_pat_s\": $((TP-TA)), \"threads\": $T, \"genome\": \"$gen\"}" > $OUT/${GSM}_timing.json; cat $OUT/${GSM}_timing.json; exit 0
fi
( cd $MP/chain/MethylPhys_Interface && "$CHAINPY" run_sample.py --pat $PAT --specimen "cultured cells" --seq-cell "$GSM" --id $GSM --dev-count-only \
    --dev-count-genome $gen --dev-count-tiles $OUT/${GSM}_tiles.csv.gz --out $OUT/${GSM}_report.html --bundle $OUT/${GSM}_bundle.json > $OUT/${GSM}_run_sample.log 2>&1 ) || exit 35
python3 -c "import json,sys; B=json.load(open(sys.argv[1])); c=B['iam_a_count_only']; json.dump(dict(gsm=sys.argv[2], genome=c['genome'], counter='run_sample.py --dev-count-only (chain Stage Q2)', **c['count']), open(sys.argv[3], 'w'), indent=1)" \
    $OUT/${GSM}_bundle.json $GSM $OUT/${GSM}_q2.json || exit 36
echo "{\"gsm\": \"$GSM\", \"pairs\": $n1, \"index_s\": $((TI-T0)), \"fetch_s\": $((TF-TI)), \"align_s\": $((TA-TF)), \"to_pat_s\": $((TP-TA)), \"threads\": $T, \"genome\": \"$gen\"}" > $OUT/${GSM}_timing.json
cat $OUT/${GSM}_timing.json; cat $OUT/${GSM}_q2.json
