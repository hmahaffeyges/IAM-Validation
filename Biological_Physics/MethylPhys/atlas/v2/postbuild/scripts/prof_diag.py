#!/usr/bin/env python3
import json, glob, numpy as np, pandas as pd
exec(open("sweep2.py").read().split("def run(cfg):")[0])
ST=json.load(open("source_terms_v1.json")); ST.pop("_meta",None)
chosen=dict(markers="hybrid", margin=0.10, pair_margin=0.15, per_pair=60, k_near=8, top=600, sigma=0.02)
D=DeconvV2(None, pooled_one_sample=POOL1, A=ATL, **chosen); loci=pd.Index(D.loci); ci={c:k for k,c in enumerate(D.cells)}
S=pd.read_csv("roster_samples.csv")
def raw(cell, src, sub):
    g=S[(S.cell==cell)&(S.source==src)&(S.qc==True)]; V=[]
    for s in g["sample"]:
        p=glob.glob(f"/home/ubuntu/data/atlas_sources/{sub}/{s}*.parquet")
        if p: b=pd.read_parquet(p[0]).iloc[:,0]; b.index=b.index.astype(str); V.append(b.reindex(loci)-ST[src]["d"])
    return pd.concat(V,axis=1)
out={}
for cell,src,sub in (("eosinophils","Salas2022","blood/GSE167998/shards"),("neutrophils","Salas2022","blood/GSE167998/shards"),("vascular endothelium","Moss2018","moss2018/shards")):
    R=raw(cell,src,sub); rm=R.mean(axis=1).values; am=D.mu[:,ci[cell]]
    out[cell]=dict(n_raw=R.shape[1], mean_abs_atlas_minus_raw=float(np.nanmean(np.abs(am-rm))), mean_atlas_minus_raw=float(np.nanmean(am-rm)),
                   corr=float(pd.Series(am).corr(pd.Series(rm))))
    # distinctiveness at markers: |atlas eos - atlas neu| vs |raw eos - atlas neu|
    out[cell]["raw_samples_solved"]=[]
    for j in range(R.shape[1]):
        o=D.deconvolve(R.iloc[:,j], n_boot=0); fr=sorted(o["fractions"].items(), key=lambda x:-x[1])[:5]
        out[cell]["raw_samples_solved"].append([(c,round(v,3)) for c,v in fr])
# vascular endothelium's atlas profile solved against every OTHER cell
Dx=DeconvV2(None, pooled_one_sample=POOL1, A=ATL.drop(columns=[c for c in ATL.columns if c.startswith("vascular endothelium_")]), **chosen)
vp=pd.Series(D.mu[:,ci["vascular endothelium"]], index=loci)
o=Dx.deconvolve(vp, n_boot=0); out["vascular_endothelium_profile_as_mix_of_others"]=sorted([(c,round(v,3)) for c,v in o["fractions"].items() if v>0.01], key=lambda x:-x[1])
json.dump(out, open("prof_diag.json","w"), indent=1); print(json.dumps(out, indent=1))
