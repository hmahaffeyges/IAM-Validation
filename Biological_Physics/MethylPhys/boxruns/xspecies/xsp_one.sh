r=$1; k=$2; D=/mnt/scratch/xsp; S=/mnt/scratch/run2
[ -s $D/out/$r.json ] && exit 0
/home/ubuntu/env/bin/python -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').download_file('methylphys-data-945451304272-us-west-2-an',sys.argv[1],sys.argv[2])" $k $D/$r.sra || { echo "{\"run\":\"$r\",\"error\":\"download\"}" > $D/out/$r.json; exit 0; }
$S/e_sra/bin/fastq-dump -Z $D/$r.sra 2>/dev/null | /home/ubuntu/env/bin/python ~/IAM-Validation/Biological_Physics/MethylPhys/boxruns/xspecies/rrbs_iama.py > $D/out/$r.tmp && \
  /home/ubuntu/env/bin/python -c "import json,sys;d=json.load(open(sys.argv[1]));d['run']=sys.argv[2];json.dump(d,open(sys.argv[3],'w'))" $D/out/$r.tmp $r $D/out/$r.json
rm -f $D/$r.sra $D/out/$r.tmp
