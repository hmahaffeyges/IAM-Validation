#!/usr/bin/env python3
"""Twin test at the sample level (2026-09-28). The chain's rule (twin_family_thresholds_v1.json): two cells are one if r > twin_r
AND fewer than min_separating_loci CpGs separate them by > sep_delta. v1 applied it to atlas MEANS on marker panels. With whole-array
data a mean difference > 0.2 happens at thousands of CpGs by sampling noise alone, so a CpG counts as separating only when the two
cells' SAMPLES do not overlap: every sample of one lies > sep_delta beyond every sample of the other. Noise cannot manufacture that.
Same platform only (no transfer fitted). Cells need >= 2 QC samples. Runs on every pair with r >= 0.975 from twins.csv."""
import json, numpy as np, pandas as pd
R="/home/ubuntu/data/atlas_sources"; TW=json.load(open("twin_family_thresholds_v1.json")); SD, SL = TW["sep_delta"], TW["min_separating_loci"]
S=pd.read_csv("roster_samples.csv"); S=S[S.qc]
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
def samples(cell):
    g=S[S.cell==cell]; w=g[g.source=="Loyfer2023"]["sample"].tolist()
    if len(w)>=2: return "WGBS", B[w].where(C[w]>=10)
    arr=g[g.platform=="array"]
    if len(arr)>=2:
        cols=[]
        for _,r in arr.iterrows():
            sub={"Moss2018":"moss2018/shards","Salas2018":"blood/GSE110554/shards","Salas2022":"blood/GSE167998/shards"}[r.source]
            cols.append(pd.read_parquet(f"{R}/{sub}/{r['sample']}.parquet").iloc[:,0].rename(r["sample"]))
        return "array", pd.concat(cols,axis=1)
    return None, None
T=pd.read_csv("twins.csv"); T=T[T.r>=0.975]
out=[]
for _,p in T.iterrows():
    pa,Xa=samples(p.a); pb,Xb=samples(p.b)
    if Xa is None or Xb is None or pa!=pb: out.append(dict(a=p.a,b=p.b,r=p.r,platform=f"{pa}/{pb}",n_a=None,n_b=None,shared=None,nonoverlap_sep=None,twin=None,note="not testable (need >=2 samples each, same platform)")); continue
    j=Xa.dropna(thresh=Xa.shape[1]).index.intersection(Xb.dropna(thresh=Xb.shape[1]).index)
    a=Xa.loc[j].values; b=Xb.loc[j].values
    sep=int(((a.min(1)-b.max(1))>SD).sum()+((b.min(1)-a.max(1))>SD).sum())
    out.append(dict(a=p.a,b=p.b,r=p.r,platform=pa,n_a=Xa.shape[1],n_b=Xb.shape[1],shared=len(j),nonoverlap_sep=sep,twin=(p.r>TW["twin_r"] and sep<SL),note=""))
    print(out[-1],flush=True)
pd.DataFrame(out).to_csv("twin_nonoverlap.csv",index=False); print("DONE",flush=True)
