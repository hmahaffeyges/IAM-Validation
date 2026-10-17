#!/bin/bash
shutdown -h +180
sudo -u ubuntu -H bash -c 'cd ~ && mkdir -p zb && cd zb && ~/env/bin/python - <<"PY" > bundle.log 2>&1
import boto3, os, tarfile, hashlib, json, shutil
s=boto3.client("s3"); B="methylphys-data-945451304272-us-west-2-an"; P="results/ATLAS_V21/fit/"; OUT="results/ATLAS_V21/zenodo_bundles/"
st=lambda m: s.put_object(Bucket=B,Key=OUT+"status.txt",Body=m.encode())
man={}
for k in range(10):
    lo,hi=70*k,70*k+69; name=f"iam_atlas_v21_fit_blocks_{lo:05d}-{hi:05d}.tar"; os.makedirs("blocks",exist_ok=True)
    keys=[o["Key"] for pg in s.get_paginator("list_objects_v2").paginate(Bucket=B,Prefix=P+"blocks/") for o in pg.get("Contents",[]) if lo<=int(os.path.basename(o["Key"])[6:11])<=hi]
    with tarfile.open(name,"w") as T:
        for key in sorted(keys):
            f="blocks/"+os.path.basename(key); s.download_file(B,key,f); T.add(f,arcname=f); os.remove(f)
    h=hashlib.sha256(); [h.update(c) for c in iter(lambda fh=open(name,"rb"): fh.read(1<<24), b"")]
    s.upload_file(name,B,OUT+name); man[name]=dict(files=len(keys),bytes=os.path.getsize(name),sha256=h.hexdigest()); os.remove(name)
    st(f"{k+1} of 10 bundles: {name}")
for key in ("code/","run_v21.log","stageB_v21_summary.json","distinctness_v21.csv"):
    pass
json.dump(man,open("bundles_manifest.json","w"),indent=1); s.upload_file("bundles_manifest.json",B,OUT+"bundles_manifest.json"); st("ALL DONE")
PY
~/env/bin/python -c "import boto3;boto3.client(\"s3\").upload_file(\"/home/ubuntu/zb/bundle.log\",\"methylphys-data-945451304272-us-west-2-an\",\"results/ATLAS_V21/zenodo_bundles/bundle.log\")"'
shutdown -h now
