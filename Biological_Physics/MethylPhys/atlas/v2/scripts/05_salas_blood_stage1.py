#!/usr/bin/env python3
"""Sorted-blood references behind v1's immune panels, raw IDATs -> the chain's own Stage 1 -> one shard per array.
  GSE35069  Reinius 2012   450K  60 (10 types x 6: neutrophils, eosinophils, granulocytes, CD4, CD8, NK, B, monocytes, PBMC, whole blood)
  GSE110554 Salas 2018     EPIC  49 (Neu, Mono, B, CD4T, CD8T, NK x ~6; 12 mixes)
  GSE167998 Salas 2022     EPIC  68 (12 types incl. naive/memory CD4 & CD8, Treg, naive/memory B, basophils, eosinophils; 12 mixes)
Shards: /home/ubuntu/data/atlas_sources/blood/<GSE>/shards. Writes blood_manifest.csv (gse, gsm, label, call_rate)."""
import os, re, gzip, json, glob, tarfile, urllib.request, multiprocessing as mp, sys
import pandas as pd
sys.path.insert(0, os.getcwd())
ROOT="/home/ubuntu/data/atlas_sources/blood"
SERIES=["GSE35069","GSE110554","GSE167998"]
def labels(gse):
    t=gzip.decompress(urllib.request.urlopen(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/matrix/{gse}_series_matrix.txt.gz",timeout=180).read()).decode("utf-8","replace")
    g=re.search(r"^!Sample_geo_accession\t(.+)$",t,re.M).group(1).replace('"','').split("\t")
    src=re.search(r"^!Sample_source_name_ch1\t(.+)$",t,re.M).group(1).replace('"','').split("\t")
    chs=[[x.replace('"','') for x in l.split("\t")[1:]] for l in re.findall(r"^!Sample_characteristics_ch1\t.+$",t,re.M)]
    ct=next((r for r in chs if any(k in r[0].lower() for k in ("cell type","celltype","cell_type"))),None)
    lab=[x.split(":",1)[1].strip() for x in ct] if ct else src
    if gse=="GSE110554":   # its cell-type row holds sentrix ids for two samples; fall back to source name there
        lab=[s if l.startswith("2018") else l for l,s in zip(lab,src)]
    return dict(zip(g,lab))
def fetch(gse):
    d=f"{ROOT}/{gse}/idats"; os.makedirs(d,exist_ok=True); os.makedirs(f"{ROOT}/{gse}/shards",exist_ok=True)
    if len(glob.glob(f"{d}/*_Grn.idat*"))<10:
        tar=f"{ROOT}/{gse}/{gse}_RAW.tar"
        urllib.request.urlretrieve(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{gse[:-3]}nnn/{gse}/suppl/{gse}_RAW.tar",tar)
        with tarfile.open(tar) as tf: tf.extractall(d)
        os.remove(tar)
    for f in glob.glob(f"{d}/*.idat.gz"):
        if not os.path.exists(f[:-3]):
            with gzip.open(f) as s, open(f[:-3],"wb") as o: o.write(s.read())
    return gse, len(glob.glob(f"{d}/*_Grn.idat"))
jobs=[]
for gse in SERIES:
    g,n=fetch(gse); lab=labels(gse); print(f"{gse}: {n} arrays on disk, {len(lab)} labelled",flush=True)
    for grn in glob.glob(f"{ROOT}/{gse}/idats/*_Grn.idat"):
        b=os.path.basename(grn)[:-9]; jobs.append((gse,b,grn,grn.replace("_Grn.idat","_Red.idat"),lab.get(b.split("_")[0],"?")))
def calib(j):
    gse,b,g,r,lab=j; sh=f"{ROOT}/{gse}/shards/{b}.parquet"
    try:
        from stage_1_idat_calibration import calibrate_idat_to_beta
        if os.path.exists(sh): return gse,b,lab,"exists",None
        beta,meta=calibrate_idat_to_beta(g,r,verbose=False); beta=beta.iloc[:,0] if hasattr(beta,"columns") else beta
        beta.to_frame(b).to_parquet(sh); d=meta.get("detection") or {}
        return gse,b,lab,"ok",d.get("n_detected",0)/max(d.get("n_probes",1),1)
    except Exception as e: return gse,b,lab,"error:"+type(e).__name__+":"+str(e)[:100],None
# one array per platform first (manifest download), then the pool
seen=set(); first=[]
for j in jobs:
    if j[0] not in seen: seen.add(j[0]); first.append(calib(j))
rest=[j for j in jobs if (j[0],j[1]) not in {(f[0],f[1]) for f in first}]
with mp.get_context("fork").Pool(28) as p: rows=first+list(p.imap_unordered(calib,rest))
M=pd.DataFrame(rows,columns=["gse","base","label","status","call_rate"]); M["gsm"]=M.base.str.split("_").str[0]
M.to_csv("blood_manifest.csv",index=False)
print(M.status.str.split(":").str[0].value_counts().to_dict(),flush=True)
S=M[~M.label.str.contains("mix|MIX|PBMC|Whole blood",regex=True)]
print(S.groupby(["gse","label"]).call_rate.agg(n="size",median="median",below=lambda s:int((s<0.93).sum())).round(3).to_string())
