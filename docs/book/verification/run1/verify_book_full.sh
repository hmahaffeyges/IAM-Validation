#!/bin/bash
# The book's check record on a full clone (2026-10-19): verify_book.py needs every committed data file, which a sparse checkout lacks.
# Usage: bash verify_book_full.sh <commit>. Writes verify_book_output.txt and .json to S3 results/BOOK_CHECK/<commit8>/.
set -u; C=${1:?commit}; S=${C:0:8}; W=$HOME/vb; rm -rf $W; mkdir -p $W; cd $W; B=methylphys-data-945451304272-us-west-2-an; K=results/BOOK_CHECK/$S
PYB=$HOME/env/bin/python; put(){ $PYB -c "import boto3,sys;boto3.client('s3').upload_file(sys.argv[1],'$B','$K/'+sys.argv[2])" "$1" "$2"; }
st(){ echo "$(date -u +%FT%TZ) $*" >> $W/status.txt; put $W/status.txt status.txt; }
st "clone $S"; git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && git -C repo checkout -q $C || { st "FAILED clone"; exit 1; }
cd repo/docs/book && st "verify_book"; $PYB verify_book.py > $W/vb.log 2>&1; rc=$?
for f in verification/verify_book_output.txt verification/verify_book_output.json verification/VERIFY_BOOK_INVENTORY.md; do [ -f $f ] && put $f $(basename $f); done
put $W/vb.log vb.log; st "finished rc $rc: $(grep -m1 '^SUMMARY' $W/vb.log)"
