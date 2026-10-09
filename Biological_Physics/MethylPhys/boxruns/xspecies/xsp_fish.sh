D=/mnt/scratch/xsp; mkdir -p $D/out; cp runs_fish.tsv $D/; cd $D
nohup setsid bash -c "awk -F'\t' '{print \$1\" \"\$2}' runs_fish.tsv | xargs -P 32 -n 2 bash xsp_one.sh; for r in \$(cut -f1 runs_fish.tsv); do cat out/\$r.json; echo; done > xsp_fish.jsonl; /home/ubuntu/env/bin/python -c \"import boto3;boto3.client('s3',region_name='us-west-2').upload_file('xsp_fish.jsonl','methylphys-data-945451304272-us-west-2-an','results/DEV_XSPECIES_TEMP_01/xsp_fish.jsonl')\"; echo FISH_DONE > fish_done.txt" > xsp_fish.log 2>&1 < /dev/null &
sleep 3; echo started
