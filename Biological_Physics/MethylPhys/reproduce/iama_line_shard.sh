#!/bin/bash
# IAM-A's self-reference line, split for GitHub's machines (2026-10-21, author: "split up the mouse and IAM-A runs to run on Github").
# The same counting as reproduce/iama_line.sh: the committed driver DEV_CHAIN_Q2_01/box_q2_line.py and chain Stage Q2, unchanged; only the
# sample list each job is handed is shorter. The final job compares every count, the line and the shape reference with the committed ones
# (CLEANROOM_IAMA_01/cleanroom_iama.py compare), as the one-machine run does.
#   bash iama_line_shard.sh loyfer K N WORK   samples K, K+N, K+2N, ... of loyfer_samples.csv
#   bash iama_line_shard.sh encode ACC WORK   one ENCODE file (downloaded, counted, deleted)
#   bash iama_line_shard.sh compare COUNTS    every job's counts gathered in COUNTS
set -euo pipefail
R="$(cd "$(dirname "$0")/../../.." && pwd)"; MP=$R/Biological_Physics/MethylPhys; Q=$MP/doors/data/DEV_CHAIN_Q2_01; NW=$(nproc)
# the reproduction's pinned environment (reproduce/setup.sh), as reproduce.py uses; the system python has none of the packages (first split run, 2026-10-21)
PY=${PY:-$MP/reproduce/.venv/bin/python}; [ -x "$PY" ] || { echo "run reproduce/setup.sh first: $PY not found"; exit 2; }
prep() { mkdir -p "$1" && cd "$1" && rm -rf chain && cp -r $MP/chain ./chain && cp $Q/box_q2_line.py . ; }
case "$1" in
  loyfer)
    prep "$4"; $PY - "$Q/loyfer_samples.csv" "$2" "$3" <<'PY'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1]))); k, n = int(sys.argv[2]), int(sys.argv[3])
w = csv.DictWriter(open("loyfer_samples.csv", "w", newline=""), fieldnames=list(r[0])); w.writeheader(); w.writerows(r[k::n]); print(len(r[k::n]), "samples")
PY
    want=$(( $(wc -l < loyfer_samples.csv) - 1 ))
    for pass in 1 2 3 4 5 6; do   # GEO answers 503 when busy; the driver skips samples already counted
      $PY box_q2_line.py loyfer 4; n=$(ls out/GSM*.json 2>/dev/null | wc -l); echo "pass $pass: $n of $want"; [ "$n" -ge "$want" ] && break; sleep $(( pass * 120 ))
    done; [ "$n" -ge "$want" ] ;;
  encode)
    prep "$3"; head -1 $Q/encode_samples.tsv > encode_samples.tsv; grep -P "\t$2\t" $Q/encode_samples.tsv >> encode_samples.tsv
    url=$(tail -1 encode_samples.tsv | cut -f5)
    curl -sfL --retry 5 -o ref.fa.gz "https://www.encodeproject.org/files/GRCh38_no_alt_analysis_set_GCA_000001405.15/@@download/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz" && gunzip -f ref.fa.gz
    $PY -c "import pysam; pysam.faidx('ref.fa')"
    curl -sfL --retry 5 -o $2.bam "https://encode-public.s3.amazonaws.com/${url#s3://encode-public/}" && $PY -c "import pysam; pysam.index('$2.bam')"
    df -h . | tail -1; OMP_NUM_THREADS=1 $PY box_q2_line.py encode $NW; rm -f $2.bam $2.bam.bai ref.fa ref.fa.fai; ls out ;;
  compare)
    $PY $MP/doors/data/CLEANROOM_IAMA_01/cleanroom_iama.py compare $R "$2" ;;
esac
