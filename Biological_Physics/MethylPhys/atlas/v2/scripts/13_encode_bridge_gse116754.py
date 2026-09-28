#!/usr/bin/env python3
"""Bridge for the ENCODE embryonic stem lines (H1, HUES64; WGBS), which share no cell with our arrays (2026-09-28).
GSE116754 (450K, raw IDATs; found via the Cell Reports Methods 2026 compendium): UNDIFFERENTIATED hESC arrays only (source name 'Human Embryonic Stem Cells', title not naming a derivative) -> raw IDATs
-> the chain's own Stage 1 -> shards. The array-scale mean over those lines vs the ENCODE mean (depth >= 10) at every shared CpG gives
ENCODE's a, b by Deming regression (errors in both), the same estimator as source_terms.py. Line-to-line variation stays in the residual,
as donor variation does for every other cell. Cross-check: ENCODE and Loyfer are both WGBS, so ENCODE's b should sit near Loyfer's 1.10.
Assumption stated, not hidden: GSE116754's own array-lab offset is not measurable (it shares no cell with another array lab) and is taken
as 0; the measured array-lab offsets so far are Salas 0.000 and Moss -0.005 (via Loyfer)."""
import os, re, gzip, tarfile, urllib.request, glob, json, sys, multiprocessing as mp, numpy as np, pandas as pd
D="/home/ubuntu/data/atlas_sources/es_gse116754"; ID=f"{D}/idats"; SH=f"{D}/shards"
for p in (ID,SH,"meta"): os.makedirs(p,exist_ok=True)
sys.path.insert(0,os.getcwd())
U="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE116nnn/GSE116754/"
t=gzip.decompress(urllib.request.urlopen(U+"matrix/GSE116754_series_matrix.txt.gz",timeout=300).read()).decode("utf-8","replace")
row=lambda k: re.search(rf"^!{k}\t(.+)$",t,re.M).group(1).replace('"','').split("\t")
gsm=row("Sample_geo_accession"); src=row("Sample_source_name_ch1"); lab=dict(zip(gsm,src))
tit=row("Sample_title"); lab=dict(zip(gsm,[f"{s} | {t}" for s,t in zip(src,tit)]))
keep={g for g,s,t in zip(gsm,src,tit) if s.strip()=="Human Embryonic Stem Cells" and not re.search(r"meso|splanch|MC|SM|deriv|diff",t,re.I)}
print("kept titles:",sorted(tit[gsm.index(g)] for g in keep),flush=True)
print("undifferentiated ES arrays:",len(keep),"| lines:",len({re.sub(r'\.passage.*','',lab[g]) for g in keep}),flush=True)
tarp=f"{D}/GSE116754_RAW.tar"
if len(glob.glob(f"{ID}/*_Grn.idat*"))<len(keep):
    if not os.path.exists(tarp): urllib.request.urlretrieve(U+"suppl/GSE116754_RAW.tar",tarp)
    with tarfile.open(tarp) as tf:
        names=tf.getnames(); print("tar members:",len(names),"| idat:",sum(".idat" in n.lower() for n in names),flush=True)
        for m in tf.getmembers():
            if ".idat" in m.name.lower() and m.name.split("_")[0] in keep: tf.extract(m,ID)
    os.remove(tarp)
pairs={}
for g in glob.glob(f"{ID}/*_Grn.idat*"): pairs[os.path.basename(g).split("_")[0]]=(g,g.replace("_Grn","_Red"))
print("pairs:",len(pairs),flush=True)
if not pairs: json.dump({"status":"NO IDATS in GSE116754_RAW.tar"},open("es_bridge.json","w")); print("NO IDATS"); sys.exit(0)
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
Mf=pd.DataFrame(rows,columns=["gsm","status","call_rate"]); Mf["label"]=Mf.gsm.map(lab); Mf.to_csv("es_manifest.csv",index=False)
ok=Mf[(Mf.call_rate.fillna(0)>=0.93)].gsm.tolist(); print("pass intake:",len(ok),"of",len(Mf),flush=True)
A=pd.concat([pd.read_parquet(f"{SH}/{g}.parquet").iloc[:,0] for g in ok],axis=1).mean(axis=1)
E=[pd.read_parquet(p) for p in sorted(glob.glob("/home/ubuntu/data/atlas_sources/encode_stem/*_array.parquet"))]
Em=pd.concat([d["beta"].where(d["cov"]>=10) for d in E],axis=1).mean(axis=1)
X=pd.DataFrame({"x":A,"y":Em}).dropna()
def deming(x,y):
    sx,sy=np.var(x),np.var(y); sxy=np.cov(x,y)[0,1]; b=(sy-sx+np.sqrt((sy-sx)**2+4*sxy**2))/(2*sxy); return float(np.mean(y)-b*np.mean(x)),float(b)
a,b=deming(X.x.values,X.y.values); rng=np.random.default_rng(20260928)
bs=np.array([deming(*X.sample(len(X),replace=True,random_state=int(rng.integers(1e9)))[["x","y"]].values.T) for _ in range(500)])
out={"source":"ENCODE","status":"IDENTIFIED VIA GSE116754 (undifferentiated ES lines on 450K, our Stage 1)","arrays_used":len(ok),"loci":int(len(X)),"a":a,"b":b,
     "a_ci95_loci":np.percentile(bs[:,0],[2.5,97.5]).tolist(),"b_ci95_loci":np.percentile(bs[:,1],[2.5,97.5]).tolist(),"r":float(np.corrcoef(X.x,X.y)[0,1]),
     "resid_sd":float((X.y-(a+b*X.x)).std()),"note":"CI resamples loci only (one cell on each side); GSE116754 array-lab offset taken as 0 (not measurable)"}
json.dump(out,open("es_bridge.json","w"),indent=1); print(json.dumps(out),flush=True); print("DONE",flush=True)
