#!/bin/bash
# The researcher package and the book on Zenodo (2026-10-19, OTH-077): run on an AWS machine, never the laptop. Clones the public repository at
# one commit, packages it, TESTS THE PACKAGE AS A RESEARCHER RECEIVES IT (unzip, git init, a new Python 3.11 environment from
# chain/requirements.txt only, the chain v4 release check), runs the book's own checks on the full clone, then uploads to the two drafts.
# Nothing uploads unless the archive passes the release check; the book uploads only if every book check passes. Both stay drafts: the
# author publishes. Usage: bash zenodo_v4_box.sh <commit>   (drafts and their buckets: zenodo_v4_drafts.json beside this script)
set -u; C=${1:?commit}; S=${C:0:8}; W=$HOME/zv4; rm -rf $W; mkdir -p $W/up/package $W/up/book; cd $W
B=methylphys-data-945451304272-us-west-2-an; K=results/ZENODO_V4; PYB=$HOME/env/bin/python
put(){ $PYB -c "import boto3,sys;boto3.client('s3').upload_file(sys.argv[1],'$B','$K/'+sys.argv[2])" "$1" "$2"; }
st(){ echo "$(date -u +%FT%TZ) $*" >> $W/status.txt; put $W/status.txt status.txt; }
st "start commit $S"
git clone -q https://github.com/hmahaffeyges/IAM-Validation.git repo && git -C repo checkout -q $C || { st "FAILED clone"; exit 1; }
st "archive"; git -C repo archive --format=zip --prefix=IAM-Validation-$S/ -o up/package/IAM-Validation-$S.zip $C || { st "FAILED archive"; exit 1; }
st "archive $(du -m up/package/IAM-Validation-$S.zip | cut -f1) MB; testing it as received"
mkdir t && cd t && unzip -q ../up/package/IAM-Validation-$S.zip && cd IAM-Validation-$S && git init -q && git add -A && git -c user.name=archive -c user.email=archive@example.org commit -qm "archive $S" || { st "FAILED unpack"; exit 1; }
curl -LsSf https://astral.sh/uv/install.sh | sh > $W/uv.log 2>&1; export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.11 $W/venv >> $W/uv.log 2>&1 && uv pip install -q -p $W/venv/bin/python -r Biological_Physics/MethylPhys/chain/requirements.txt >> $W/uv.log 2>&1 || { st "FAILED environment"; exit 1; }
$W/venv/bin/python Biological_Physics/MethylPhys/chain/release_check_v4.py --json $W/up/package/release_check_v4_from_archive.json > $W/release_check.log 2>&1; rc=$?
put $W/release_check.log release_check.log; st "release check on the archive rc $rc: $(tail -1 $W/release_check.log)"
[ $rc -eq 0 ] || { st "FAILED release check on the archive; nothing uploaded"; exit 1; }
cd $W/repo/docs/book && $W/venv/bin/python verify_book.py > $W/up/book/verify_book_output.txt 2>&1; put $W/up/book/verify_book_output.txt verify_book_output.txt
BS=$(grep "^SUMMARY" $W/up/book/verify_book_output.txt | tail -1); st "book checks: $BS"; cd $W
git -C repo archive --format=zip --prefix=IAMs_Law_and_Order_source/ -o up/book/IAMs_Law_and_Order_LaTeX_source_v2.0.0.zip $C docs/book
for f in pdf/IAMs_Law_and_Order.pdf:IAMs_Law_and_Order_v2.0.0.pdf epub/IAMs_Law_and_Order.epub:IAMs_Law_and_Order_v2.0.0.epub; do
  curl -sfL --retry 5 -o up/book/${f#*:} https://hmahaffeyges.github.io/IAM-Validation/${f%%:*} || { st "FAILED fetching ${f%%:*}"; exit 1; }; done
R=repo; M=$R/Biological_Physics/MethylPhys
cp $R/QUESTIONS_AND_ANSWERS.md $M/START_HERE.md $M/sop/MethylPhys_CPG_SOP_v4.md $M/manual/MethylPhys_CPG_Operations_Manual.pdf $R/reproduce/README.md up/package/
mv up/package/README.md up/package/REPRODUCE_README.md; cp $R/reproduce/verified.json up/package/reproduce_verified.json
cp $M/doors/COMMISSIONING_NOTE_*.md up/package/; cp $M/doors/data/REPORT_EXAMPLES_V4/final/*_researcher.pdf up/package/
for d in package book; do (cd up/$d && sha256sum * > SHA256SUMS.txt); done
st "uploading"; $PYB $(dirname $(readlink -f $0))/zenodo_v4_upload.py $(dirname $(readlink -f $0))/zenodo_v4_drafts.json $W/up "$BS" >> $W/upload.log 2>&1; rc=$?
put $W/upload.log upload.log; put $W/up/package/SHA256SUMS.txt package_SHA256SUMS.txt; put $W/up/book/SHA256SUMS.txt book_SHA256SUMS.txt
st "finished rc $rc: $(tail -1 $W/upload.log)"; exit $rc
