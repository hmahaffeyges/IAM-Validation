#!/usr/bin/env bash
# DEV-IAMA-TRIM-01 calibration on the box (written 2026-10-13, BEFORE it runs): the chemistry diagnostic d of head_diag_reads_01.py,
# unchanged, on every one of the 71 EpiQC Stage B libraries (epiqc_stageB_runs.csv). No alignment, no IAM-A, no reading of the
# physics: only the raw reads' non-CpG C fraction in 20 windows per run, and the trim each run would get under trim_rule_01.py
# with K_SD and MAX_TRIM unchanged. 16 runs at a time, one output folder per run, each uploaded to S3 as it finishes so a stop
# loses nothing. Usage: bash calib_trim_all_01.sh FASTQ_DUMP PYTHON OUTROOT S3PREFIX
set -uo pipefail
FQD=$1; PY=$2; OUT=$3; S3=$4; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
tail -n +2 "$HERE/../DEV_IAMA_COMMISSION_01/epiqc_stageB_runs.csv" | cut -d, -f6 | sort -u > "$OUT/runs.txt"
echo "runs: $(wc -l < "$OUT/runs.txt")"
one() {
  run=$1; o="$OUT/$run"
  [ -s "$o/head_diag_reads_01_decision.json" ] && { echo "$run done already"; return 0; }
  mkdir -p "$o"; "$PY" "$HERE/head_diag_reads_01.py" "$o" "$FQD" "$run" > "$o/log.txt" 2>&1
  for f in head_diag_reads_01.csv head_diag_reads_01_decision.json log.txt; do
    [ -f "$o/$f" ] && "$PY" -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').upload_file(sys.argv[1],'methylphys-data-945451304272-us-west-2-an',sys.argv[2])" "$o/$f" "$S3/$run/$f"
  done
  echo "$run $(grep -c '' "$o/head_diag_reads_01.csv" 2>/dev/null) rows"
}
export -f one; export OUT FQD PY HERE S3
xargs -P 16 -I{} bash -c 'one {}' < "$OUT/runs.txt"
echo "END $(ls "$OUT"/*/head_diag_reads_01_decision.json 2>/dev/null | wc -l) of $(wc -l < "$OUT/runs.txt") runs decided"
