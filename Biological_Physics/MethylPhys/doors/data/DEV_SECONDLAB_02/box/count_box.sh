#!/bin/bash
# DEV-SECONDLAB-02 step 3 on a self-terminating machine: waits for the three deduplicated BAMs in S3, then counts each with chain Stage Q2
# (count_hansen.py), on the Bismark hg38 reference the alignment used. Nothing is scored here; scoring is score_secondlab02.py, once, after.
set -uo pipefail
B=methylphys-data-945451304272-us-west-2-an; P=results/DEV_SECONDLAB_02/; PY=~/env/bin/python; W=~/cnt; mkdir -p $W/out && cd $W
st(){ printf '%s %s\n' "$1" "$(date -u +%FT%TZ)" >> cstatus.txt; $PY -c "import boto3;boto3.client('s3').upload_file('cstatus.txt','$B','${P}count_status.txt')"; }
$PY -c "import boto3;s=boto3.client('s3')
for f in ('count_hansen.py','samples.tsv','code.tgz'): s.download_file('$B','${P}code/count3/'+f,f)" && tar xzf code.tgz || { st "code download FAILED"; exit 1; }
st "started on $(nproc) cores, code $(cat REPO_COMMIT)"
for i in 00 01 02; do $PY -c "import boto3;boto3.client('s3').download_file('$B','downloads/tumour_work/tools.tgz.part_$i','tools.tgz.part_$i')"; done
cat tools.tgz.part_0* | tar xzf - --wildcards --exclude='*Bisulfite_Genome*' '*ref/*' 2>/dev/null; rm -f tools.tgz.part_0*
FA=$(ls $(find . -path '*ref/*' \( -name '*.fa' -o -name '*.fasta' \) | head -1)); [ -s "$FA" ] || { st "reference FAILED: no fasta in the bundle's ref/"; exit 1; }
[ -s "$FA.fai" ] || $PY -c "import pysam;pysam.faidx('$FA')"; st "reference $FA ready"
for k in $(seq 1 90); do n=$($PY -c "import boto3;print(sum(o['Key'].endswith('.bam') for o in boto3.client('s3').list_objects_v2(Bucket='$B',Prefix='${P}bam/').get('Contents',[])))"); [ "$n" -ge 3 ] && break; sleep 120; done
[ "$n" -ge 3 ] || { st "waited 3 h, only $n BAMs"; exit 1; }
tail -n +2 samples.tsv | while IFS=$'\t' read -r d gsm srr rest; do
  st "$d $srr counting"; $PY -c "import boto3;s=boto3.client('s3');s.download_file('$B','${P}bam/$srr.bam','$srr.bam')
[s.download_file('$B','${P}bam/$srr.bam.bai','$srr.bam.bai') for o in s.list_objects_v2(Bucket='$B',Prefix='${P}bam/$srr.bam.bai').get('Contents',[])]"
  OMP_NUM_THREADS=1 $PY count_hansen.py $srr.bam "$FA" $d $gsm $srr 30 >> count.log 2>&1 || st "$d count FAILED $(tail -2 count.log | tr '\n' ' ' | cut -c1-200)"
  $PY -c "import boto3,os;f='out/$srr.json';os.path.exists(f) and boto3.client('s3').upload_file(f,'$B','${P}out/$srr.json')"; [ -s out/$srr.json ] || st "$d no count written"; rm -f $srr.bam $srr.bam.bai; [ -s out/$srr.json ] && st "$d $srr counted"
done
$PY -c "import boto3;boto3.client('s3').upload_file('count.log','$B','${P}count.log')"; st "ALL COUNTED $(ls out/*.json 2>/dev/null | wc -l) of 3"
