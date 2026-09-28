#!/usr/bin/env python3
"""GSE63409 (sorted normal bone-marrow HSC, MPP, L-MPP, CMP, GMP, MEP x 5 donors; AML CD34+38-, CD34+38+, CD34- sorted; 450K):
raw tar -> IDAT pairs -> the chain's own Stage 1 -> one shard per array under /home/ubuntu/data/atlas_sources/hsc_gse63409/shards.
Normal arrays are atlas-v2 candidates (adult stem, progenitor); AML arrays are kept for PROC-BLOODCANCER-01 and never enter the atlas.
Writes hsc_manifest.csv (gsm, base, label, subject, status, call_rate)."""
import os, re, gzip, tarfile, urllib.request, glob, json, multiprocessing as mp, sys
import pandas as pd
D="/home/ubuntu/data/atlas_sources/hsc_gse63409"; ID=f"{D}/idats"; SH=f"{D}/shards"
for p in (ID,SH,"meta"): os.makedirs(p,exist_ok=True)
sys.path.insert(0,os.getcwd())
U="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63409/"
t=gzip.decompress(urllib.request.urlopen(U+"matrix/GSE63409_series_matrix.txt.gz",timeout=300).read()).decode("utf-8","replace")
row=lambda k: re.search(rf"^!{k}\t(.+)$",t,re.M).group(1).replace('"','').split("\t")
gsm=row("Sample_geo_accession"); src=row("Sample_source_name_ch1")
chs=[[x.replace('"','') for x in l.split("\t")[1:]] for l in re.findall(r"^!Sample_characteristics_ch1\t.+$",t,re.M)]
getc=lambda key:[next((x.split(":",1)[1].strip() for r in chs for x in [r[i]] if x.lower().startswith(key)),"") for i in range(len(gsm))]
lab=dict(zip(gsm,src)); subj=dict(zip(gsm,getc("subject id"))); stat=dict(zip(gsm,getc("subject status")))
tarp=f"{D}/GSE63409_RAW.tar"
if len(glob.glob(f"{ID}/*_Red.idat*"))<len(gsm):
    if not os.path.exists(tarp): urllib.request.urlretrieve(U+"suppl/GSE63409_RAW.tar",tarp)
    with tarfile.open(tarp) as tf:
        for m in tf.getmembers():
            if ".idat" in m.name: tf.extract(m,ID)
    os.remove(tarp)
pairs={}
for g in glob.glob(f"{ID}/*_Grn.idat*"):
    b=os.path.basename(g); pairs[b.split("_")[0]]=(g,g.replace("_Grn","_Red"))
print("pairs:",len(pairs),"| samples:",len(gsm),flush=True)
def calib(k):
    g,r=pairs[k]; sh=f"{SH}/{k}.parquet"
    if os.path.exists(sh) and os.path.exists(f"meta/{k}.json"): return k,"exists",json.load(open(f"meta/{k}.json")).get("call_rate")
    try:
        from stage_1_idat_calibration import calibrate_idat_to_beta
        beta,meta=calibrate_idat_to_beta(g,r,verbose=False); beta=beta.iloc[:,0] if hasattr(beta,"columns") else beta
        beta.to_frame(k).to_parquet(sh+".tmp"); os.replace(sh+".tmp",sh)
        det=meta.get("detection") or {}; cr=det.get("n_detected",0)/max(det.get("n_probes",1),1)
        json.dump({"call_rate":cr},open(f"meta/{k}.json","w")); return k,"ok",cr
    except Exception as e: return k,"error:"+type(e).__name__+":"+str(e)[:100],None
keys=sorted(pairs); first=calib(keys[0]); print("warm-up:",first,flush=True)
with mp.get_context("fork").Pool(24,maxtasksperchild=10) as pool: rows=[first]+list(pool.imap_unordered(calib,keys[1:]))
M=pd.DataFrame(rows,columns=["gsm","status","call_rate"]); M["label"]=M.gsm.map(lab); M["subject"]=M.gsm.map(subj); M["subject_status"]=M.gsm.map(stat)
M.to_csv("hsc_manifest.csv",index=False)
print(M.status.str.split(":").str[0].value_counts().to_dict(),flush=True)
print(M.groupby(["subject_status","label"]).call_rate.agg(n="size",median="median",below=lambda s:int((s<0.93).sum())).round(3).to_string(),flush=True)
