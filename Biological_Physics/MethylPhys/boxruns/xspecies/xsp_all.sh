D=/mnt/scratch/xsp; mkdir -p $D/out; cp xsp_one.sh runs.tsv $D/; cd $D
nohup setsid bash -c "awk -F'\t' '{print \$1\" \"\$2}' runs.tsv | xargs -P 32 -n 2 bash xsp_one.sh; cat out/*.json > xsp_results.jsonl; /home/ubuntu/env/bin/python -c \"import boto3;boto3.client('s3',region_name='us-west-2').upload_file('xsp_results.jsonl','methylphys-data-945451304272-us-west-2-an','results/DEV_XSPECIES_TEMP_01/xsp_results.jsonl')\"; echo XSP_DONE > done.txt" > xsp.log 2>&1 < /dev/null &
sleep 5; echo started; ls out | wc -l
