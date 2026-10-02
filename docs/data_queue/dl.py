#!/usr/bin/env python3
"""Resumable ENA -> S3 transfer, in manifest order (stool, coho x3, 580-species fish, 580-species other).
Each file: curl from ENA (md5-free, retried 4x), size-checked against ENA bytes, posted to S3 downloads/<track>/<study>/<run>/<file>
(split into 4 GB parts if larger), deleted locally. done.txt (S3 downloads/dlqueue/done.txt) is re-read at start and re-posted every 25 files."""
import csv, os, subprocess, json, time, threading, sys
from concurrent.futures import ThreadPoolExecutor
W="/home/ubuntu/data/dl"; os.makedirs(W,exist_ok=True)
G=json.load(open("s3_get.json")); PAR=int(os.environ.get("PAR","24"))
subprocess.run(["curl","-sS","-f","-o","done.txt",G["done"]])
done=set(open("done.txt").read().split()) if os.path.exists("done.txt") else set()
rows=[r for r in csv.DictReader(open("manifest.tsv"),delimiter="\t") if r["url"] not in done]
print("to do",len(rows),"already",len(done),flush=True)
lock=threading.Lock(); n=[0]; gb=[0.0]; t0=time.time()
def one(r):
    fn=r["url"].rsplit("/",1)[1]; p=f"{W}/{r['run']}_{fn}"; key=f"downloads/{r['track']}/{r['study']}/{r['run']}/{fn}"
    for k in range(4):
        subprocess.run(["curl","-sS","-L","--retry","3","-o",p,r["url"]])
        if os.path.exists(p) and os.path.getsize(p)==int(r["bytes"]): break
        time.sleep(20*(k+1))
    else:
        os.path.exists(p) and os.remove(p); return f"FAIL {r['url']}"
    big=int(r["bytes"])>4*1024**3
    ok=subprocess.run(["python3","s3io.py","putsplit" if big else "put",p,key]).returncode==0
    for f in os.listdir(W):
        if f.startswith(os.path.basename(p)): os.remove(f"{W}/{f}")
    if not ok: return f"FAILPUT {r['url']}"
    with lock:
        done.add(r["url"]); n[0]+=1; gb[0]+=int(r["bytes"])/1e9
        open("done.txt","w").write("\n".join(sorted(done)))
        if n[0]%25==0:
            subprocess.run(["python3","s3io.py","put","done.txt","downloads/dlqueue/done.txt"])
            print(f"{n[0]} files {gb[0]:.0f} GB {gb[0]/((time.time()-t0)/3600):.0f} GB/h  last {r['track']}",flush=True)
    return "ok"
with ThreadPoolExecutor(PAR) as ex:
    res=list(ex.map(one,rows))
subprocess.run(["python3","s3io.py","put","done.txt","downloads/dlqueue/done.txt"])
bad=[x for x in res if x!="ok"]; print("DONE ok",len(res)-len(bad),"failed",len(bad)); print("\n".join(bad[:50]))
