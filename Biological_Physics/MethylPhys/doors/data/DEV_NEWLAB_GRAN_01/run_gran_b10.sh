set -uo pipefail
df -h $HOME | tail -1; cd ~/IAM-Validation && git pull -q origin main && git log --oneline -1 | cut -c1-50; cd - >/dev/null
W=$HOME/gran; mkdir -p $W/idat && cp newlab_gran.py $W/ && cd $W
curl -s -o gsm.txt "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE226298&targ=gsm&form=text&view=brief"
[ -s RAW.tar ] || curl -sfL -o RAW.tar https://ftp.ncbi.nlm.nih.gov/geo/series/GSE226nnn/GSE226298/suppl/GSE226298_RAW.tar
tar xf RAW.tar -C idat && ls idat | wc -l
/home/ubuntu/env/bin/python newlab_gran.py $W ~/IAM-Validation/Biological_Physics/MethylPhys/chain 12 2>&1 | grep -v -i warn | tail -5
/home/ubuntu/env/bin/python -c "
import boto3
s3=boto3.client('s3',region_name='us-west-2')
for f in ('gran_rows.csv','gran_summary.json'): s3.upload_file('$W/'+f,'methylphys-data-945451304272-us-west-2-an','results/DEV_NEWLAB_GRAN_01_B10/'+f)
print('uploaded')"
rm -rf $W/idat $W/RAW.tar
