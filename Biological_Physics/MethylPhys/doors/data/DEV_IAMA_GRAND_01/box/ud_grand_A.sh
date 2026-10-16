#!/bin/bash
shutdown -h +1200
sudo -u ubuntu -H bash -c 'mkdir -p ~/gr && cd ~/gr && ~/env/bin/python -c "
import boto3;s=boto3.client(\"s3\");B=\"methylphys-data-945451304272-us-west-2-an\"
for f in (\"align_grand.sh\",\"samples.csv\",\"count_grand.py\",\"code.tgz\"): s.download_file(B,\"results/DEV_IAMA_GRAND_01/code/run2_A/\"+f,f)" && MACHINE=A bash align_grand.sh > align_A.log 2>&1; ~/env/bin/python -c "import boto3;boto3.client(\"s3\").upload_file(\"align_A.log\",\"methylphys-data-945451304272-us-west-2-an\",\"results/DEV_IAMA_GRAND_01/align_A.log\")"'
shutdown -h now
