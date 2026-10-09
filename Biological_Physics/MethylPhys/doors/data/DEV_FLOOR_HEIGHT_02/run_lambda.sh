S=/mnt/scratch/run2; D=/mnt/scratch/lam; mkdir -p $D; export PATH=$S/e_st19/bin:$PATH
for r in SRR9888333 SRR9888330; do /home/ubuntu/env/bin/python -c "import boto3,sys;boto3.client('s3',region_name='us-west-2').download_file('methylphys-data-945451304272-us-west-2-an','results/BOXRUN2/session2/bam/'+sys.argv[1]+'.bam',sys.argv[2])" $r $D/$r.bam && samtools index -@ 4 $D/$r.bam; done
/home/ubuntu/env/bin/python lambda_nc.py $S/ref/hg19_lambda_puc19.fa $D/lambda.json $D/SRR9888333.bam $D/SRR9888330.bam
rm -f $D/*.bam $D/*.bai
