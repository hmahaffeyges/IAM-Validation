#!/bin/bash
shutdown -h +1440
sudo -u ubuntu -H bash -c 'mkdir -p ~/tl && cd ~/tl && ~/env/bin/python -c "
import boto3;s=boto3.client(\"s3\");B=\"methylphys-data-945451304272-us-west-2-an\"
for f in (\"align_plants.sh\",\"t3_pairs_runs.csv\",\"count_plant.py\",\"code.tgz\"): s.download_file(B,\"results/PRED_TEMPERATURE_01/T3/code/run3/\"+f,f)" && MACHINE=__M__ PAR=__PAR__ bash align_plants.sh > align.log 2>&1; ~/env/bin/python -c "import boto3;boto3.client(\"s3\").upload_file(\"/home/ubuntu/tl/align.log\",\"methylphys-data-945451304272-us-west-2-an\",\"results/PRED_TEMPERATURE_01/T3/align___M__.log\")"'
shutdown -h now
