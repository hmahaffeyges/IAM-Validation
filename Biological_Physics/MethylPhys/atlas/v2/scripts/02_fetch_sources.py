#!/usr/bin/env python3
"""Pulls atlas v2 candidate sources to the box's persistent disk (/home/ubuntu/data/atlas_sources). Resumable, checksums by size.
  loyfer2023/beta/   hg19 .beta per sorted-cell sample (GSE186458; wgbstools format: uint8 [meth, cov] per CpG in hg19 CpG-index order)
  moss2018/idats/    GSE122126_RAW.tar extracted (450K + EPIC IDATs of the sorted cells behind the 25-cell array atlas)
Writes ./fetch_manifest.csv and ./fetch_summary.json into the job workdir."""
import os, json, time, tarfile, urllib.request, concurrent.futures as cf
R="/home/ubuntu/data/atlas_sources"; LB=f"{R}/loyfer2023/beta"; MI=f"{R}/moss2018/idats"
for p in (LB,MI): os.makedirs(p,exist_ok=True)
rows=json.load(open("loyfer_beta_list.json")); t0=time.time()
def get(r):
    dst=f"{LB}/{r['file']}"
    if os.path.exists(dst) and os.path.getsize(dst)==r["bytes"]: return r["gsm"],"exists"
    for k in range(4):
        try:
            urllib.request.urlretrieve(r["url"],dst+".part")
            if os.path.getsize(dst+".part")==r["bytes"]: os.replace(dst+".part",dst); return r["gsm"],"ok"
        except Exception as e: err=type(e).__name__
        time.sleep(10*(k+1))
    return r["gsm"],"failed"
with cf.ThreadPoolExecutor(12) as ex: res=list(ex.map(get,rows))
print(f"loyfer: {sum(s in ('ok','exists') for _,s in res)}/{len(rows)} in {time.time()-t0:.0f}s",flush=True)
t1=time.time(); tar=f"{R}/moss2018/GSE122126_RAW.tar"
if len([f for f in os.listdir(MI) if "idat" in f])<10:
    if not os.path.exists(tar): urllib.request.urlretrieve("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122126/suppl/GSE122126_RAW.tar",tar)
    with tarfile.open(tar) as tf: tf.extractall(MI)
    os.remove(tar)
nm=len([f for f in os.listdir(MI) if "idat" in f.lower()]); print(f"moss: {nm} idat files in {time.time()-t1:.0f}s",flush=True)
import csv
with open("fetch_manifest.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["gsm","status","cell_type","tissue","file"])
    for (g,s),r in zip(res,rows): w.writerow([g,s,r["cell_type"],r["tissue"],r["file"]])
json.dump({"loyfer_ok":sum(s in ('ok','exists') for _,s in res),"loyfer_total":len(rows),"loyfer_failed":[g for g,s in res if s=="failed"],"moss_idat_files":nm,"seconds":round(time.time()-t0),"root":R},open("fetch_summary.json","w"),indent=1)
print("DONE",flush=True)
