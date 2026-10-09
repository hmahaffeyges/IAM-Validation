# DEV-IAMA-KIT-01: Swift (SRR9888332) vs TruSeq (SRR9888333), same donor and sequencer: bam2pat with 0 / 10 / 15 bp clipped from both read ends, M-bias table, Stage Q.
set -uo pipefail
S=/mnt/scratch/run2; W=/mnt/scratch/clip; mkdir -p $W && cd $W
export PATH=$S/e_st19/bin:$S/e_wgbs/bin:$S/wgbs_tools:$PATH
PYB=/home/ubuntu/env/bin/python; CH=~/IAM-Validation/Biological_Physics/MethylPhys/chain
for r in SRR9888332 SRR9888333; do
  [ -s $r.bam ] || $PYB -c "import boto3;boto3.client('s3',region_name='us-west-2').download_file('methylphys-data-945451304272-us-west-2-an','results/BOXRUN2/session2/bam/$r.bam','$r.bam')"
  samtools index $r.bam
  for c in 0 10 15; do
    mkdir -p pat_c$c; mb=""; [ $c = 0 ] && mb="--mbias"
    wgbstools bam2pat $r.bam -o pat_c$c --genome hg19 -@ 8 --clip $c $mb -f > pat_c$c/$r.log 2>&1 || echo "$r clip $c bam2pat FAILED"
    P=$(ls pat_c$c/$r*.pat.gz | head -1)
    $PYB $CH/MethylPhys_Interface/run_sample.py --pat $P --specimen "isolated neutrophils" --seq-cell neutrophils --id ${r}_clip$c --out $W/${r}_clip$c.html > $W/${r}_clip$c.read.log 2>&1
    $PYB -c "import json,sys;b=json.load(open(sys.argv[1]));a=b.get('iam_a',{});print(sys.argv[2],sys.argv[3],'A',a.get('A'),'eps',a.get('eps'),'opp',a.get('opportunities'),'C',(a.get('cscore') or {}).get('C'),a.get('refusal_code') or '')" $W/${r}_clip${c}_bundle.json $r $c 2>&1 | tee -a $W/clip_summary.txt
  done
done
ls pat_c0/*mbias* 2>/dev/null | head
$PYB -c "
import boto3,glob,os
s3=boto3.client('s3',region_name='us-west-2')
for f in glob.glob('$W/*.json')+glob.glob('$W/clip_summary.txt')+glob.glob('$W/pat_c0/*mbias*')+glob.glob('$W/pat_c0/*.txt'): s3.upload_file(f,'methylphys-data-945451304272-us-west-2-an','results/DEV_IAMA_KIT_01/'+os.path.basename(f))
print('uploaded')"
