#!/bin/bash
# BLUEPRINT's spent healthy libraries read again by the one command with the per-site tables written (--dev-metaa-sites), for the laboratory
# term at the identity sites (development; see Biological_Physics/MethylPhys/doors/DEV_ATLAS_V21_01.md). Sourced inside a machine job after the
# launcher's preamble ($W, $R, log, up). Each row: both GRCh38 bigWig tracks downloaded, then read with the row's own flags (rsflags) plus
# --dev-metaa-sites, specimen from the row. Usage (inside a machine job): . blueprint_sites_read.sh SAMPLES.csv
mkdir -p $W/out; cd $R/Biological_Physics/MethylPhys/chain/MethylPhys_Interface
python3 -c "import csv,sys; [print('|'.join([r['gsm'], r['specimen'], r['rsflags'], r['call_bw'], r['cov_bw']])) for r in csv.DictReader(open(sys.argv[1]))]" "$1" > $W/bp_list.txt
while IFS='|' read -r G SPEC FL CB VB; do
  [ -n "$G" ] || continue
  ok=1; for u in "$CB" "$VB"; do f=$W/$(basename "$u"); for t in 1 2 3 4 5; do curl -fsSL --retry 5 --retry-delay 20 -o "$f" "$u" && break; rm -f "$f"; sleep 30; done; [ -s "$f" ] || ok=0; done
  [ $ok = 1 ] || { log "$G download FAILED"; continue; }
  python3 run_sample.py --dev-seq-beta "$W/$(basename "$CB"),$W/$(basename "$VB")" --dev-seq-values 1 --specimen "$SPEC" --id $G $FL --dev-metaa-sites \
    --out $W/out/$G.html --bundle $W/out/${G}_bundle.json > $W/$G.r.log 2>&1
  up $W/out/${G}_bundle.json $W/out/${G}_metaa_sites.csv.gz $W/$G.r.log; log "$G read: $(tail -1 $W/$G.r.log | cut -c1-150)"; rm -f "$W/$(basename "$CB")" "$W/$(basename "$VB")"
done < $W/bp_list.txt
