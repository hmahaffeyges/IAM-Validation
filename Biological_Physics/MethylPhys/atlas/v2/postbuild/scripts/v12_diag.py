#!/usr/bin/env python3
"""PROC-V12-IDENTITY-02, test B1/B2: cross-fitting on samples (doors/PROC_V12_IDENTITY_02_PREREG.md).
Each cell's samples split in two with default_rng(12); identity loci chosen from half 1's own samples (every half-1 sample within
b* +/- 0.05, each put on the atlas scale by its source term), half-2 samples read individually; then swapped."""
import json, numpy as np, pandas as pd
from scipy.optimize import brentq
R="/home/ubuntu/data/atlas_sources"
ST=json.load(open("source_terms_v1.json")); ST.pop("_meta")
S=pd.read_csv("roster_samples.csv"); RO=pd.read_csv("roster.csv",index_col=0); H_=pd.read_csv("hsc_manifest.csv"); H_=H_[H_.subject_status=="normal"]
S=pd.concat([S,pd.DataFrame({"cell":H_.label.str.replace(" of normal bone marrow","",regex=False)+" (bone marrow)","source":"GSE63409","platform":"array","sample":H_.gsm,"qc":H_.call_rate>=0.93,"metric":H_.call_rate})])
adm=RO.index[RO.v2_status.str.startswith("IN")].tolist(); S=S[(S.qc==True)&S.cell.isin(adm)].reset_index(drop=True)
HMIN=json.load(open("hmin.json")); H=lambda b: -(b*np.log2(b)+(1-b)*np.log2(1-b))
BSTAR={k:brentq(lambda b:H(b)-h,0.5+1e-9,1-1e-9) for k,h in HMIN.items()}
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
loci=np.sort(B.index[(C>=10).mean(axis=1)>0.9].values)
SUB={"Moss2018":"moss2018/shards","Salas2018":"blood/GSE110554/shards","Salas2022":"blood/GSE167998/shards","GSE63409":"hsc_gse63409/shards"}
def vec(r):
    t=ST[r.source]
    if r.source=="Loyfer2023": y=B[r["sample"]].reindex(loci).where(C[r["sample"]].reindex(loci)>=5)
    elif r.source=="Tian2023": d=pd.read_parquet(f"{R}/tian2023/{r['sample']}_array.parquet"); y=d["beta"].reindex(loci).where(d["cov"].reindex(loci)>=5)
    elif r.source=="ENCODE": d=pd.read_parquet(r["sample"]); y=d["beta"].reindex(loci).where(d["cov"].reindex(loci)>=5)
    else: return (pd.read_parquet(f"{R}/{SUB[r.source]}/{r['sample']}.parquet").iloc[:,0].reindex(loci)-t["d"]).clip(1e-3,1-1e-3).values
    return ((y-t["a"])/t["b"]).clip(1e-3,1-1e-3).values
out=[]
for cell,g in S.groupby("cell"):
    k=RO.loc[cell,"class_by_draft_rule"]; bs=BSTAR[k]; hm=HMIN[k]; n=len(g)
    if n<4: continue
    X=np.vstack([vec(r) for _,r in g.iterrows()])
    for i in range(n):
        Xs=np.delete(X,i,axis=0); fin=np.all(np.isfinite(Xs),axis=0)
        plain=fin&np.all(np.abs(Xs-bs)<=0.05,axis=0); stable=plain&(np.nanstd(Xs,axis=0)<=0.03)
        for var,ok in (("L-plain",plain),("L-stable",stable)):
            v=X[i,ok]; v=v[np.isfinite(v)]
            A=float(H(np.clip(v.mean(),1e-9,1-1e-9))/hm) if len(v)>=100 else None
            out.append(dict(cell=cell,klass=k,n_choose=n-1,variant=var,sample=g.iloc[i]["sample"],source=g.iloc[i]["source"],n_loci=int(ok.sum()),A=A))
    print(cell,n,flush=True)
O=pd.DataFrame(out); O.to_csv("v12_diag_loo.csv",index=False)
O["dev"]=(O.A-1).abs(); O["inn"]=O.A.between(0.95,1.05,inclusive="left")
M=O.groupby(["cell","variant"]).agg(n_choose=("n_choose","first"),med_dev=("dev","median"),frac_normal=("inn","mean"),median_A=("A","median"),loci=("n_loci","median")).reset_index()
M.to_csv("v12_diag_cells.csv",index=False)
Z=O.groupby(["variant","n_choose"]).agg(readings=("A","size"),med_dev=("dev","median"),frac_normal=("inn","mean")).reset_index()
print(Z.round(4).to_string(index=False)); print(O.groupby("variant").agg(med_dev=("dev","median"),frac_normal=("inn","mean"),unreadable=("A",lambda x:int(x.isna().sum()))).round(4).to_string())
