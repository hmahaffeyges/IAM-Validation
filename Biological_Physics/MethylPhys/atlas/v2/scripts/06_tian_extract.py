#!/usr/bin/env python3
"""Tian 2023 human brain single-cell methylome (Science 382:eadf5357), MajorType pseudobulk allc (figshare 28438499, hg38,
strands not merged, CG + CH). Non-neuron types only. Each file is STREAMED (curl | gzip) and only CG rows at array CpGs kept;
nothing but the result is written. Array CpG hg38 positions from Zhou's InfiniumAnnotation manifests (HM450 + EPIC, hg38):
CpG_beg is the 0-based start of the CG, so the + strand C is at CpG_beg+1 and the - strand C at CpG_beg+2 (1-based, allc).
Both strands are summed. Pseudobulk pools three donors: ONE profile per type, no donor variance - recorded as such.
Out: /home/ubuntu/data/atlas_sources/tian2023/{type}_array.parquet (probe, mc, cov, beta); job output tian_extract_report.json."""
import os, sys, json, gzip, io, subprocess, time, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd
OUT="/home/ubuntu/data/atlas_sources/tian2023"; os.makedirs(OUT,exist_ok=True)
TYPES={"ASC":"astrocytes","MGC":"microglia","ODC":"oligodendrocytes","OPC":"oligodendrocyte precursors","VLMC":"vascular leptomeningeal cells","PC":"pericytes","EC":"endothelial (major-type label EC)"}
F={x["name"].split(".")[0]:x for x in json.load(open("/home/ubuntu/data/tian_majortype_files.json"))}
man=[]
for u in ("https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/HM450/HM450.hg38.manifest.tsv.gz",
          "https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/EPIC/EPIC.hg38.manifest.tsv.gz"):
    d=pd.read_csv(io.BytesIO(urllib.request.urlopen(u,timeout=300).read()),sep="\t",compression="gzip",usecols=["CpG_chrm","CpG_beg","Probe_ID"],dtype={"CpG_chrm":str})
    man.append(d)
m=pd.concat(man).dropna().drop_duplicates("Probe_ID"); m=m[m.Probe_ID.str.startswith("cg")]
m["CpG_beg"]=m.CpG_beg.astype(np.int64)
print("array CpGs with hg38 positions:", len(m), flush=True)
pos={}
for pid,c,b in zip(m.Probe_ID,m.CpG_chrm,m.CpG_beg):
    pos[(c,b+1)]=pid; pos[(c,b+2)]=pid
def one(t):
    t0=time.time(); url=F[t]["url"]
    if os.path.exists(f"{OUT}/{t}_array.parquet"):
        d=pd.read_parquet(f"{OUT}/{t}_array.parquet")
        return t,{"cell":TYPES[t],"rows_read":None,"array_cpgs":int(len(d)),"median_cov":float(d["cov"].median()),"frac_cov_ge10":round(float((d["cov"]>=10).mean()),4),"mean_beta":round(float(d["beta"].mean()),4),"seconds":0}
    p=subprocess.Popen(f"curl -sL --retry 5 '{url}' | gzip -dc", shell=True, stdout=subprocess.PIPE, bufsize=1<<20)
    mc={}; cov={}; n=0
    for line in io.TextIOWrapper(p.stdout, encoding="ascii", errors="ignore"):
        n+=1
        f=line.split("\t",6)
        if len(f)<6 or not f[3].startswith("CG"): continue
        k=(f[0],int(f[1])); pid=pos.get(k)
        if pid is None: continue
        mc[pid]=mc.get(pid,0)+int(f[4]); cov[pid]=cov.get(pid,0)+int(f[5])
    p.wait()
    d=pd.DataFrame({"mc":pd.Series(mc),"cov":pd.Series(cov)})
    d.to_parquet(f"{OUT}/{t}_array_counts.parquet")            # counts saved first, so a later error cannot lose the stream
    d["beta"]=d["mc"]/d["cov"].where(d["cov"]>0)
    d.to_parquet(f"{OUT}/{t}_array.parquet")
    return t,{"cell":TYPES[t],"rows_read":n,"array_cpgs":int(len(d)),"median_cov":float(d["cov"].median()) if len(d) else None,
              "frac_cov_ge10":round(float((d["cov"]>=10).mean()),4) if len(d) else None,"mean_beta":round(float(d["beta"].mean()),4) if len(d) else None,"seconds":round(time.time()-t0)}
rep={}
with cf.ThreadPoolExecutor(len(TYPES)) as ex:
    for t,r in ex.map(one, [t for t in TYPES if t in F]):
        rep[t]=r; print(t, r, flush=True)
json.dump({"source":"Tian et al. 2023 Science, figshare 28438499 (MajorType, hg38)","array_cpgs_hg38":int(len(m)),"donors":"3 adult male brains, pooled per type (no donor variance)","types":rep},open("tian_extract_report.json","w"),indent=1)
print("DONE", flush=True)
