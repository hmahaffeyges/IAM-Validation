#!/bin/bash
# DEV-FINGERPRINT-02 step 10, NORMAL side only (no cancer file is read): for one normal cell type, its Box Run 9 .pat files from S3 (sha256
# recorded), then with the DEV-FINGERPRINT-01 tools unchanged:
#   a) Stage Q copy error per file, whole file (boxruns/run2/eps_of.py)                       -> normal_<tag>/eps_per_file.json
#   b) Stage Q readability per file: molecules with >= 6 calls, and the share Stage Q reads    -> normal_<tag>/readability.csv
#   c) Stage Q's measured response to a planted loss on the largest file (insilico_loss_02.py)  -> normal_<tag>/insilico_<file>.csv
#   d) run-loss simulation on the two largest files (runloss_01.py sim)                        -> normal_<tag>/runloss_sim.txt
# Usage: bash prep_normal.sh "<ENCODE cell name>" WORKDIR        (needs AWS read access; python with pandas, numpy)
set -euo pipefail
CELL="$1"; W="$2"; HERE=$(cd "$(dirname "$0")" && pwd); MP=$(cd "$HERE/../../.." && pwd); B=methylphys-data-945451304272-us-west-2-an
TAG=$(echo "$CELL" | tr -c 'A-Za-z0-9\n' '_' | sed 's/__*/_/g; s/_$//'); O="$HERE/normal_$TAG"; mkdir -p "$O" "$W/pat"
FILES=$(python3 -c "import pandas as pd,sys;d=pd.read_csv('$MP/boxruns/run9_fp2/normal_rrbs_files.csv');print(' '.join(d[d.cell==sys.argv[1]].file))" "$CELL")
[ -n "$FILES" ] || { echo "no files for $CELL"; exit 2; }
: > "$O/pat_sha256.txt"
for f in $FILES; do
  [ -s "$W/pat/$f.pat.gz" ] || aws s3 cp --quiet "s3://$B/results/BOXRUN9_FP2/normal/$f.pat.gz" "$W/pat/$f.pat.gz"
  (cd "$W/pat" && shasum -a 256 "$f.pat.gz") >> "$O/pat_sha256.txt"
done
python3 "$MP/boxruns/run2/eps_of.py" "$MP/chain" "$O/eps_per_file.json" $(for f in $FILES; do echo "$W/pat/$f.pat.gz"; done) > /dev/null
python3 - "$O/readability.csv" $(for f in $FILES; do echo "$W/pat/$f.pat.gz"; done) <<'PY'
import sys, gzip, pandas as pd
rows = []
for p in sys.argv[2:]:
    n = m6 = rd = 0
    for l in gzip.open(p, "rt"):
        x = l.rstrip("\n").split("\t"); s = x[2]; c = int(x[3]); k = sum(ch in "CT" for ch in s); n += c
        if k >= 6:
            m6 += c; rd += c * (s.count("C") >= 0.8 * k)
    rows.append(dict(file=p.rsplit("/", 1)[-1][:11], molecules=n, ge6_calls=m6, read_by_stageQ=rd, share_ge6=round(m6 / n, 5), share_read_of_ge6=round(rd / max(m6, 1), 5)))
pd.DataFrame(rows).to_csv(sys.argv[1], index=False); print(pd.DataFrame(rows).to_string(index=False))
PY
BIG=($(for f in $FILES; do echo "$(stat -f%z "$W/pat/$f.pat.gz" 2>/dev/null || stat -c%s "$W/pat/$f.pat.gz") $f"; done | sort -rn | awk '{print $2}'))
python3 "$MP/doors/data/DEV_LINK_IAMA_METAA_02/insilico_loss_02.py" "$MP/chain" "$W/pat/${BIG[0]}.pat.gz" "$O/insilico_${BIG[0]}.csv" > /dev/null
python3 "$MP/doors/data/DEV_RUNLOSS_01/runloss_01.py" sim "$W/pat/${BIG[0]}.pat.gz" "$W/pat/${BIG[1]}.pat.gz" > "$O/runloss_sim.txt"
echo "done $CELL -> $O"; cat "$O/eps_per_file.json"; echo; cat "$O/runloss_sim.txt"
