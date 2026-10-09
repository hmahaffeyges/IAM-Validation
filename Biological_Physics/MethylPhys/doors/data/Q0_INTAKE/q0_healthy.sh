#!/bin/bash
# Stage Q0 on the three whole healthy Loyfer granulocyte files (hg19) and, as a real wrong-build control, the hg38 copy of one of them.
set -uo pipefail
W=/mnt/scratch/q0; mkdir -p $W && cd $W
git -C ~/IAM-Validation fetch -q origin main
for f in chain/stage_q0_intake.py "chain/Runtime Matrices/IAM_A_Positions/hg19_cpg_chrom_ranges.json"; do
  mkdir -p "$W/$(dirname "$f")"; git -C ~/IAM-Validation show "origin/main:Biological_Physics/MethylPhys/$f" > "$W/$f"; done
G=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5652nnn
for x in GSM5652313_Blood-Granulocytes-Z000000TZ GSM5652314_Blood-Granulocytes-Z000000UD GSM5652315_Blood-Granulocytes-Z000000UT; do
  g=${x%%_*}; [ -s $x.pat.gz ] || curl -sfL -o $x.pat.gz $G/$g/suppl/$x.pat.gz || echo "download failed $x"; done
x=GSM5652313_Blood-Granulocytes-Z000000TZ; [ -s $x.hg38.pat.gz ] || curl -sfL -o $x.hg38.pat.gz $G/GSM5652313/suppl/$x.hg38.pat.gz
ls -la *.pat.gz
for f in *.pat.gz; do ( PYTHONPATH=$W/chain /home/ubuntu/env/bin/python -c "
import json,sys,stage_q0_intake as Q0
r=Q0.intake(sys.argv[1],'blood granulocytes'); m=r['measured']
json.dump(r,open(sys.argv[1]+'.q0.json','w'),indent=1)
print(sys.argv[1], r['decision'], r.get('refusal_code'), 'lines',m['lines'],'molecules',m['molecules'],'share_ge6',m['share_ge6_calls'],'out_of_range',m['out_of_range_lines'],'autosomes',sum(1 for c in Q0.AUTOSOMES if m['per_chrom_molecules'].get(c)))
" $f > $f.q0.txt 2>&1 & ) ; done
wait; sleep 1; while pgrep -f stage_q0_intake >/dev/null || pgrep -f "Q0.intake" >/dev/null; do sleep 20; done
cat *.q0.txt
/home/ubuntu/env/bin/python -c "
import boto3,glob,os
s3=boto3.client('s3',region_name='us-west-2')
for f in glob.glob('*.q0.json')+glob.glob('*.q0.txt'): s3.upload_file(f,'methylphys-data-945451304272-us-west-2-an','results/DEV_Q0_HEALTHY_01/'+os.path.basename(f))
print('uploaded')"
