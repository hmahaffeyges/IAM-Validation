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

ARRAY={"Moss2018","Salas2018","Salas2022","GSE63409"}; SEQ={"Loyfer2023","Tian2023","ENCODE"}
print("loading samples", flush=True)
Xc={}; Src={}
for cell,g in S.groupby("cell"):
    Xc[cell]=np.vstack([vec(r) for _,r in g.iterrows()]).astype(np.float32); Src[cell]=g.source.values
shared=[c for c in Xc if any(s in ARRAY for s in Src[c]) and any(s=="Loyfer2023" for s in Src[c])]
print("cells with Loyfer AND array samples:", len(shared), flush=True)
def cmean(c, which):
    m=np.isin(Src[c], list(which)); return np.nanmean(Xc[c][m],axis=0) if m.any() else None
# per-locus Loyfer->array residual for each shared cell (after the linear source term already applied in vec)
DEL={c: cmean(c,ARRAY)-cmean(c,{"Loyfer2023"}) for c in shared}
def delta_excluding(c):
    M=np.vstack([DEL[k] for k in shared if k!=c]); return np.nanmedian(M,axis=0)
def pick(Xs, bs):
    fin=np.all(np.isfinite(Xs),axis=0); return fin & np.all(np.abs(Xs-bs)<=0.05,axis=0)
def read(v, ok, hm):
    v=v[ok]; v=v[np.isfinite(v)]; return float(H(np.clip(v.mean(),1e-9,1-1e-9))/hm) if len(v)>=100 else None
rows=[]
rng=np.random.default_rng(12)
for c in Xc:
    k=RO.loc[c,"class_by_draft_rule"]; bs=BSTAR[k]; hm=HMIN[k]; arr=np.where(np.isin(Src[c],list(ARRAY)))[0]
    # T1: array cells, choose on half of the array samples, read the other half (what a patient array sees)
    if len(arr)>=2:
        p=rng.permutation(arr); hs=[p[:len(p)//2],p[len(p)//2:]]
        for a,b in ((0,1),(1,0)):
            ok=pick(Xc[c][hs[a]],bs)
            for i in hs[b]: rows.append(dict(test="T1_array_to_array",cell=c,klass=k,source=Src[c][i],n_loci=int(ok.sum()),A=read(Xc[c][i],ok,hm)))
    # T2/T3: sequencing -> array transfer, only for shared cells
    if c in shared:
        loy=np.where(Src[c]=="Loyfer2023")[0]
        for tag,corr in (("T2_loyfer_linear_to_array",0.0),("T3_loyfer_perlocus_to_array",delta_excluding(c))):
            Xs=np.clip(Xc[c][loy]+corr,1e-3,1-1e-3); ok=pick(Xs,bs)
            for i in arr: rows.append(dict(test=tag,cell=c,klass=k,source=Src[c][i],n_loci=int(ok.sum()),A=read(Xc[c][i],ok,hm)))
    print(c, flush=True)
O=pd.DataFrame(rows); O["in_normal"]=O.A.between(0.95,1.05,inclusive="left"); O.to_csv("v12b_samples.csv",index=False)
Z=O.groupby("test").agg(readings=("A","size"),unreadable=("A",lambda x:int(x.isna().sum())),frac_normal=("in_normal","mean"),median_A=("A","median"),p05=("A",lambda x:x.quantile(.05)),p95=("A",lambda x:x.quantile(.95)))
print(Z.round(4).to_string())
C=O.groupby(["test","cell"]).agg(n=("A","size"),median_A=("A","median"),frac_normal=("in_normal","mean"),loci=("n_loci","median")).reset_index(); C.to_csv("v12b_cells.csv",index=False)
print(C[C.test!="T1_array_to_array"].round(3).to_string(index=False))
# the per-locus correction from ALL shared cells, for the production build of sequencing-only cells
np.save("loyfer_perlocus_delta.npy", np.nanmedian(np.vstack([DEL[k] for k in shared]),axis=0)); np.save("loci.npy", loci)
json.dump(dict(shared=shared, summary=Z.reset_index().to_dict("records")), open("v12b.json","w"), indent=1, default=float)
