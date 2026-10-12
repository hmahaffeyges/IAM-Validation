#!/bin/bash
# Box Run 10 (2026-10-11): DEV-IAMA-COMMISSION-01 on EpiQC identical DNA. Reads the libraries in LIST (columns kit,lab,line,rep,seq,run,...):
# the first N_PAIRS read pairs of each run streamed from ENA FASTQ (only those bytes are downloaded), through the pinned Loyfer 2023 pipeline of
# Box Run 2 (bwa-meth 0.2.0 -> SAMtools 1.9 sort -> Sambamba 0.6.5 markdup -> wgbstools 0.1.0 bam2pat defaults, hg19 + lambda + pUC19), then
# alignment_qc.py --both (conversion from human chr1 CHH and from lambda) and samtools flagstat. Stage Q, Q0 and the planted readings run on the
# laptop from the .pat files. Each library's outputs go to S3 as soon as it is done; a library already in S3 is skipped (restart-safe).
# Stage B settings (2026-10-11): FETCH_PAR parallel downloads, MAX_WAIT downloaded libraries waiting (disk), finishing overlapped.
# Fetch: first N pairs streamed from ENA, up to 8 libraries at once, with lookup retries (fixed 2026-10-11 after the first start).
# Usage: LIST=... N_PAIRS=10000000 TAG=stageA [SHUTDOWN=1] bash session_epiqc.sh     Outputs: s3://<bucket>/results/BOXRUN10_EPIQC/$TAG/
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/BOXRUN10_EPIQC/${TAG}; N=${N_PAIRS:-10000000}; T=$(nproc)
S=/mnt/scratch/run2; O=$S/out10_${TAG}; mkdir -p $O $S/r10/fq $S/r10/bam $S/r10/pat; cd $S
PYB=/home/ubuntu/env/bin/python; REPO=~/IAM-Validation; QC=$REPO/Biological_Physics/MethylPhys/boxruns/run2/alignment_qc.py
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/e_bm/bin:$S/e_sra/bin:$S/e_sbb/bin:$S/wgbs_tools:$PATH
REF=$S/ref/hg19_lambda_puc19.fa
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $O/session.log; }
up(){ $PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" "$1" "$2" || log "upload failed $1"; }
have(){ $PYB -c "import boto3,sys;r=boto3.client('s3',region_name='us-west-2').list_objects_v2(Bucket='$B',Prefix=sys.argv[1]);sys.exit(0 if r['KeyCount'] else 1)" "$1"; }
fin(){ log "END $1"; echo "{\"status\": \"$1\"}" > $O/status.json; for f in $O/*.log $O/*.json $O/*.txt; do [ -f "$f" ] && up "$f" "$P/$(basename $f)"; done
       [ -n "${SHUTDOWN:-}" ] && sudo shutdown -h +1; exit 0; }
log "START TAG=$TAG N_PAIRS=$N threads=$T"
# The aligner environments were on the retired scratch disk (2026-10-11); rebuild them with Box Run 2's pinned specs (session1.sh, session3.sh)
MM="$S/bin/micromamba -r $S/mamba"
[ -x $S/e_bm/bin/bwameth.py ] || $MM create -y -q -p $S/e_bm -c conda-forge -c bioconda python=3.6 bwameth=0.2.0 bwa >> $O/install.log 2>&1 || fin FAIL_install_bwameth
[ -x $S/e_wgbs/bin/python ] || { $MM create -y -q -p $S/e_wgbs -c conda-forge -c bioconda python=3.8 numpy pandas scipy htslib bedtools >> $O/install.log 2>&1 && (cd $S/wgbs_tools && git checkout -q 0.1.0 && $S/e_wgbs/bin/python setup.py >> $O/install.log 2>&1); } || fin FAIL_install_wgbs
[ -x $S/e_st19/bin/samtools ] || $MM create -y -q -p $S/e_st19 -c conda-forge -c bioconda samtools=1.9 >> $O/install.log 2>&1 || fin FAIL_install_samtools
[ -x $S/e_sbb/bin/sambamba ] || $MM create -y -q -p $S/e_sbb -c conda-forge -c bioconda sambamba=0.6.5 >> $O/install.log 2>&1 || fin FAIL_install_sambamba
log "environments ready"
# Results must be able to leave the machine before any work is done (2026-10-12: a spot copy started without the S3 role).
echo "$TAG $(date -u +%FT%TZ)" > $O/s3_check.txt
$PYB -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'$B',sys.argv[2])" $O/s3_check.txt "$P/s3_check.txt" || { echo "no S3 access" >&2; exit 3; }
[ -s $REF.bwameth.c2t.sa ] || fin FAIL_no_index
for t in bwameth.py samtools sambamba wgbstools; do command -v $t >/dev/null || fin FAIL_missing_$t; done
samtools --version | head -1 | grep -q " 1\.9" || fin FAIL_samtools_not_1.9
SBV=$(sambamba 2>&1 || true); echo "$SBV" | grep -q "sambamba 0\.6\.5" || fin FAIL_sambamba_not_0.6.5
{ bwameth.py --version 2>&1 | head -1; samtools --version | head -1; echo "$SBV" | grep -m1 sambamba; echo "wgbstools $(git -C $S/wgbs_tools describe --tags)"; } > $O/versions.txt
cd $REPO && git pull -q origin main && git log --oneline -1 > $O/repo_commit.txt; cd $S
RUNS=$(tail -n +2 $LIST | cut -d, -f6)
url(){ $PYB - "$1" <<'PY2'
import sys, time, urllib.request
a = sys.argv[1]
for i in range(8):                      # ENA's file report sometimes returns nothing on the first try (2026-10-11)
    try:
        t = urllib.request.urlopen(f"https://www.ebi.ac.uk/ena/portal/api/filereport?accession={a}&result=read_run&fields=fastq_ftp&format=tsv", timeout=60).read().decode()
        f = t.strip().split("\n")[1].split("\t")[1].split(";")
        if len(f) == 2: print(" ".join("https://" + x for x in f)); break
    except Exception: pass
    time.sleep(10 * (i + 1))
PY2
}
fetch(){ r=$1; [ -f r10/fq/$r.complete ] && return 0; U=($(url $r)); T0=$(date +%s)
  [ ${#U[@]} -eq 2 ] || { echo "$r pairs 0 0 fetch_s 0 no_url" >> $O/fetch.txt; return 1; }
  # plain FASTQ (bwa-meth reads it); about 0.7 MB/s per stream from ENA, so up to 8 libraries are fetched at once
  ( set +o pipefail; curl -s --retry 5 "${U[0]}" | zcat 2>/dev/null | head -n $((4*N)) > r10/fq/${r}_1.fq ) &
  ( set +o pipefail; curl -s --retry 5 "${U[1]}" | zcat 2>/dev/null | head -n $((4*N)) > r10/fq/${r}_2.fq ) & wait
  n1=$(awk 'END{print NR/4}' r10/fq/${r}_1.fq); n2=$(awk 'END{print NR/4}' r10/fq/${r}_2.fq)
  echo "$r pairs $n1 $n2 fetch_s $(( $(date +%s)-T0 ))" >> $O/fetch.txt
  [ "$n1" = "$n2" ] && [ "$n1" = "$N" ] && touch r10/fq/$r.complete; }
finish(){ r=$1; T0=$2; T1=$3
  samtools sort -@ 16 -m 2G -T r10/bam/$r.tmp -o r10/bam/$r.s.bam r10/bam/$r.u.bam && rm -f r10/bam/$r.u.bam
  sambamba markdup -l 1 -t 16 --sort-buffer-size 16000 --overflow-list-size 10000000 --tmpdir r10/bam r10/bam/$r.s.bam r10/bam/$r.bam >> $O/${r}_markdup.log 2>&1 && rm -f r10/bam/$r.s.bam || { log "$r markdup FAILED"; return 1; }
  samtools index r10/bam/$r.bam; samtools flagstat r10/bam/$r.bam > $O/${r}_flagstat.txt
  $PYB $QC r10/bam/$r.bam $REF $O/${r}_alignment_qc.json --both >> $O/${r}_qc.log 2>&1 || log "$r alignment_qc failed"
  wgbstools bam2pat r10/bam/$r.bam -o r10/pat --genome hg19 -@ 16 >> $O/${r}_bam2pat.log 2>&1 || { log "$r bam2pat FAILED"; return 1; }
  PAT=$(ls r10/pat/$r*.pat.gz | head -1); T2=$(date +%s)
  echo "{\"run\": \"$r\", \"read_pairs\": $N, \"align_s\": $((T1-T0)), \"to_pat_s\": $((T2-T1)), \"threads\": $T}" > $O/${r}_timing.json
  for f in $O/${r}_*; do up "$f" "$P/$(basename $f)"; done; up "$PAT" "$P/$r.pat.gz"
  rm -f r10/bam/$r.bam* $PAT ${PAT}.csi; log "$r done $(grep -m1 'mapped (' $O/${r}_flagstat.txt | cut -c1-60)"
}
# A rerun must not read an earlier attempt's failure: drop fetch.txt lines of every run without a complete download (2026-10-12, LESSONS C7).
[ -f $O/fetch.txt ] && for r in $RUNS; do [ -f r10/fq/$r.complete ] || sed -i "/^$r pairs/d" $O/fetch.txt; done
( for r in $RUNS; do have "$P/$r.pat.gz" && continue; while [ $(jobs -r | wc -l) -ge ${FETCH_PAR:-8} ] || [ $(ls r10/fq/*_1.fq 2>/dev/null | wc -l) -ge ${MAX_WAIT:-8} ]; do sleep 10; done; fetch $r & sleep 2; done; wait ) &
for r in $RUNS; do
  have "$P/$r.pat.gz" && { log "$r already in S3"; continue; }
  while [ ! -f r10/fq/$r.complete ]; do sleep 15; grep -q "^$r pairs" $O/fetch.txt 2>/dev/null && [ ! -f r10/fq/$r.complete ] && break; done
  [ -f r10/fq/$r.complete ] || { log "$r fetch FAILED"; continue; }
  log "$r align"; T0=$(date +%s)
  bwameth.py --threads $T --reference $REF r10/fq/${r}_1.fq r10/fq/${r}_2.fq 2>> $O/${r}_align.log | samtools view -b -o r10/bam/$r.u.bam - || { log "$r align FAILED"; continue; }
  T1=$(date +%s); rm -f r10/fq/${r}_*.fq r10/fq/$r.complete
  # finishing steps (sort, markdup, QC, bam2pat, upload) run alongside the next alignment, at most 2 at once (2026-10-11: they use few cores)
  while [ $(jobs -rp | wc -l) -ge $((2 + 1)) ]; do sleep 10; done
  finish "$r" "$T0" "$T1" &
done
wait; fin DONE
