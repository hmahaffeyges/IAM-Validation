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

LOYM={c: cmean(c,{"Loyfer2023"}) for c in shared}; ARRM={c: cmean(c,ARRAY) for c in shared}
def corr_regress(c, lam):
    ks=[k for k in shared if k!=c]; Xl=np.vstack([LOYM[k] for k in ks]); Ya=np.vstack([ARRM[k] for k in ks])
    m=np.isfinite(Xl)&np.isfinite(Ya); n=m.sum(0).clip(1)
    xb=np.where(m,Xl,0).sum(0)/n; yb=np.where(m,Ya,0).sum(0)/n
    Sxx=np.where(m,(Xl-xb)**2,0).sum(0); Sxy=np.where(m,(Xl-xb)*(Ya-yb),0).sum(0)
    b=(Sxy+lam)/(Sxx+lam); a0=yb-b*xb
    return lambda X: a0+b*X
def corr_kernel(c, bw):
    ks=[k for k in shared if k!=c]; Xl=np.vstack([LOYM[k] for k in ks]); Dl=np.vstack([DEL[k] for k in ks])
    def f(X):
        out=np.empty_like(X)
        for r in range(X.shape[0]):
            w=np.exp(-(Xl-X[r])**2/(2*bw*bw)); w=np.where(np.isfinite(Dl)&np.isfinite(Xl),w,0)
            out[r]=X[r]+np.where(w.sum(0)>0,(w*np.nan_to_num(Dl)).sum(0)/w.sum(0).clip(1e-9),0)
        return out
    return f

# ---- PRODUCTION BUILD (2026-09-30): array-first; sequencing via per-locus regression learned on ALL 17 shared cells (lam 0.02)
ks=shared; Xl=np.vstack([LOYM[k] for k in ks]); Ya=np.vstack([ARRM[k] for k in ks]); m=np.isfinite(Xl)&np.isfinite(Ya); n=m.sum(0).clip(1)
xb=np.where(m,Xl,0).sum(0)/n; yb=np.where(m,Ya,0).sum(0)/n; Sxx=np.where(m,(Xl-xb)**2,0).sum(0); Sxy=np.where(m,(Xl-xb)*(Ya-yb),0).sum(0)
bb=(Sxy+0.02)/(Sxx+0.02); aa=yb-bb*xb; CORR=lambda X: np.clip(aa+bb*X,1e-3,1-1e-3)
T1={}
try:
    import pandas as _pd; _C=_pd.read_csv("v12b_cells.csv"); T1={r.cell:r.median_A for r in _C[_C.test=="T1_array_to_array"].itertuples()}
except Exception: pass
OUT={}; rep=[]
for c in Xc:
    k=RO.loc[c,"class_by_draft_rule"]; bs=BSTAR[k]; hm=HMIN[k]; src=Src[c]; arr=np.isin(src,list(ARRAY)); seq=~arr; flags=[]
    if arr.sum()>=2: Xs=Xc[c][arr]; path="array samples"
    else:
        parts=[]
        if arr.sum()==1: parts.append(Xc[c][arr]); flags.append("ONE-ARRAY-SAMPLE")
        if seq.any():
            parts.append(CORR(Xc[c][seq])); flags.append("SEQUENCING-CORRECTED-PER-LOCUS")
            if any(s in ("Tian2023","ENCODE") for s in src[seq]): flags.append("WGBS-CORRECTION-LEARNED-ON-LOYFER")
        Xs=np.vstack(parts); path="array + corrected sequencing" if arr.sum()==1 else "corrected sequencing"
    if Xs.shape[0]==1: flags.append("ONE-SAMPLE")
    ok=pick(Xs,bs); sel=loci[ok]
    selfA=read(np.nanmean(Xs,axis=0),ok,hm)
    if c in T1 and not (0.95<=T1[c]<1.05): flags.append(f"HELD-OUT-MEDIAN-{T1[c]:.3f}")
    rep.append(dict(cell=c,klass=k,path=path,n_samples=int(Xs.shape[0]),n_loci=int(ok.sum()),A_self=selfA,flags=";".join(flags)))
    if ok.sum()>=100: OUT[c]=dict(klass=k,H_min=hm,b_star=bs,path=path,n_loci=int(ok.sum()),loci=[str(x) for x in sel],A_self=selfA,flags=flags)
P=pd.DataFrame(rep).sort_values("n_loci"); P.to_csv("v12_prod_build.csv",index=False)
json.dump({"_provenance":{"built":"2026-09-30","atlas":"atlas v2","rule":"array-first: cells with >=2 array samples choose on array samples; otherwise array sample + sequencing samples mapped to the array scale by a per-locus regression learned on the 17 cells both platforms measured (ridge lam 0.02); every contributing sample within b*(class)+/-0.05; >=100 loci","H_min":"G-002","class":"draft rule","tests":"v12b/v12c: array->array 87.7% of held-out readings in NORMAL, 28/29 cell medians; Loyfer->array with this correction 73.1%, 14/17 cell medians"},"cells":OUT},open("iamatlas_v2_identity_loci_v1_1.json","w"))
print("built",len(OUT),"of",len(P)); print(P.groupby("path").agg(cells=("cell","size"),loci_min=("n_loci","min"),loci_med=("n_loci","median")).to_string()); print(P[P.n_loci<1000].to_string(index=False))
print("A_self range", round(P.A_self.min(),4), round(P.A_self.max(),4))
raise SystemExit
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
        d0=delta_excluding(c)
        variants=[("T2_loyfer_linear_to_array",lambda X:X),("T3_loyfer_perlocus_to_array",lambda X,d0=d0:X+d0)]
        for lam in (0.02,0.1,0.5): variants.append((f"T4_regress_lam{lam}",corr_regress(c,lam)))
        for bw in (0.05,0.10,0.20): variants.append((f"T5_kernel_bw{bw}",corr_kernel(c,bw)))
        for tag,fn in variants:
            Xs=np.clip(fn(Xc[c][loy]),1e-3,1-1e-3); ok=pick(Xs,bs)
            for i in arr: rows.append(dict(test=tag,cell=c,klass=k,source=Src[c][i],n_loci=int(ok.sum()),A=read(Xc[c][i],ok,hm)))
    print(c, flush=True)
O=pd.DataFrame(rows); O["in_normal"]=O.A.between(0.95,1.05,inclusive="left"); O.to_csv("v12c_samples.csv",index=False)
Z=O.groupby("test").agg(readings=("A","size"),unreadable=("A",lambda x:int(x.isna().sum())),frac_normal=("in_normal","mean"),median_A=("A","median"),p05=("A",lambda x:x.quantile(.05)),p95=("A",lambda x:x.quantile(.95)))
print(Z.round(4).to_string())
C=O.groupby(["test","cell"]).agg(n=("A","size"),median_A=("A","median"),frac_normal=("in_normal","mean"),loci=("n_loci","median")).reset_index(); C.to_csv("v12c_cells.csv",index=False)
print(C[C.test!="T1_array_to_array"].pivot(index="cell",columns="test",values="median_A").round(3).to_string())
# the per-locus correction from ALL shared cells, for the production build of sequencing-only cells
np.save("loyfer_perlocus_delta.npy", np.nanmedian(np.vstack([DEL[k] for k in shared]),axis=0)); np.save("loci.npy", loci)
json.dump(dict(v="v12c",shared=shared, summary=Z.reset_index().to_dict("records")), open("v12c.json","w"), indent=1, default=float)
