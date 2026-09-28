#!/usr/bin/env python3
"""Moss 2018 (GSE122126) raw 450K/EPIC arrays -> the chain's own Stage 1 -> one parquet shard per array under
/home/ubuntu/data/atlas_sources/moss2018/shards. Sample labels from the GEO series matrices. Writes moss_manifest.csv."""
import os, re, gzip, json, glob, urllib.request, multiprocessing as mp, sys
import pandas as pd
R="/home/ubuntu/data/atlas_sources/moss2018"; I=f"{R}/idats"; SH=f"{R}/shards"; os.makedirs(SH,exist_ok=True); os.makedirs("meta",exist_ok=True)
sys.path.insert(0,os.getcwd())
lab={}
for m in ("GSE122126-GPL13534","GSE122126-GPL21145"):
    t=gzip.decompress(urllib.request.urlopen(f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE122nnn/GSE122126/matrix/{m}_series_matrix.txt.gz",timeout=120).read()).decode("utf-8","replace")
    g=re.search(r"^!Sample_geo_accession\t(.+)$",t,re.M).group(1).replace('"','').split("\t")
    s=re.search(r"^!Sample_source_name_ch1\t(.+)$",t,re.M).group(1).replace('"','').split("\t")
    for a,b in zip(g,s): lab[a]=(b,m.split("-")[1])
for f in glob.glob(f"{I}/*.idat.gz"):
    out=f[:-3]
    if not os.path.exists(out):
        with gzip.open(f) as src, open(out,"wb") as dst: dst.write(src.read())
pairs={}
for g in glob.glob(f"{I}/*_Grn.idat"):
    b=os.path.basename(g)[:-9]; pairs[b]=(g,g.replace("_Grn.idat","_Red.idat"))
print("arrays:",len(pairs),flush=True)
def calib(b):
    g,r=pairs[b]; sh=f"{SH}/{b}.parquet"; gsm=b.split("_")[0]
    if os.path.exists(sh): return b,gsm,"exists",None
    try:
        from stage_1_idat_calibration import calibrate_idat_to_beta
        beta,meta=calibrate_idat_to_beta(g,r,verbose=False); beta=beta.iloc[:,0] if hasattr(beta,"columns") else beta
        beta.to_frame(b).to_parquet(sh); d=meta.get("detection") or {}
        json.dump({k:v for k,v in meta.items() if isinstance(v,(int,float,str,bool,type(None),dict,list))},open(f"meta/{b}.json","w"),default=str)
        return b,gsm,"ok",d.get("n_detected",0)/max(d.get("n_probes",1),1)
    except Exception as e: return b,gsm,"error:"+type(e).__name__+":"+str(e)[:100],None
bs=sorted(pairs); first=calib(bs[0]); rows=[first]
with mp.get_context("fork").Pool(24) as p: rows+=list(p.imap_unordered(calib,bs[1:]))
M=pd.DataFrame(rows,columns=["base","gsm","status","call_rate"]); M["label"]=M.gsm.map(lambda x:lab.get(x,("?",""))[0]); M["platform"]=M.gsm.map(lambda x:lab.get(x,("",""))[1])
M.to_csv("moss_manifest.csv",index=False); print(M.status.str.split(":").str[0].value_counts().to_dict(),flush=True)
print(M[~M.label.str.startswith("cfDNA")&~M.label.str.contains("mix")].groupby("label").call_rate.agg(["size","median"]).round(3).to_string())
