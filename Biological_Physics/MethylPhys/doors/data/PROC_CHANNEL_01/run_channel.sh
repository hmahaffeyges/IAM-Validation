#!/usr/bin/env bash
# PROC-CHANNEL-01, as run 2026-09-30 (job jCH4) on the box. Rebuilt 2026-10-09 from the session record (creation + four patches, replayed).
# Inputs: atlas/v2/inputs/roster_samples.csv, atlas/sources/roster_v2/atlas_v2_roster.csv (repo); pat_files.txt (GEO GSE186458 file list,
# hg19 .pat.gz); windows_hg19_cpgidx.bed (make_windows.py). Needs python with pandas, numpy, pysam; tabix reads the .pat files remotely.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); R=$(cd "$HERE/../../.." && pwd); W=${1:-$HOME/proc_channel_01}; mkdir -p "$W" && cd "$W"
cp "$HERE/channel.py" . && cp "$R/atlas/v2/inputs/roster_samples.csv" "$R/atlas/sources/roster_v2/atlas_v2_roster.csv" .
python3 "$HERE/make_windows.py"
curl -s https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186458/suppl/filelist.txt | awk -F'\t' '$2 ~ /\.pat\.gz$/ && $2 !~ /hg38/ {print $2}' > pat_files.txt
wc -l pat_files.txt windows_hg19_cpgidx.bed
python3 channel.py 2>&1 | grep -vE 'Warning|warn' | tee channel.log | tail -90     # writes channel_samples.csv, channel_summary.json
