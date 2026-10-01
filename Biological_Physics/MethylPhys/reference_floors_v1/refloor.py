#!/usr/bin/env python3
"""Met-A reference floors, v1 (development build, 2026-10-01; no prediction declared).
Reference floor(cell, platform) = mean over purified healthy specimens of H(mean beta at the cell's v2 identity loci), H = binary entropy in bits.
Sources: arrays through our Stage 1 (Moss 2018 purified tissue cells, Salas 2018/2022 purified blood cells); Loyfer 2023 purified-cell WGBS
projected onto array probe positions (coverage >= 10). Per cell/platform: leave-one-specimen-out A = m_s / mean(m of the others).
Cross-platform: can a WGBS floor stand in for an array floor? Single factor k = array/WGBS, fitted leaving the predicted cell out."""
import os, glob, json, numpy as np, pandas as pd, multiprocessing as mp
R="/home/ubuntu/data/atlas_sources"; t0=__import__("time").time()
H=lambda b: float(-(b*np.log2(b)+(1-b)*np.log2(1-b))) if 0<b<1 else 0.0
ID=json.load(open("iamatlas_v2_identity_loci_v1_1.json"))["cells"]
S=pd.read_csv("roster_samples.csv"); S=S[(S.qc==True)&S.cell.isin(ID)]
DIR={"Moss2018":f"{R}/moss2018/shards","Salas2018":f"{R}/blood/GSE110554/shards","Salas2022":f"{R}/blood/GSE167998/shards"}
A=S[S.source.isin(DIR)].reset_index(drop=True)
def read_arr(i):
    r=A.iloc[i]; f=glob.glob(f"{DIR[r.source]}/{r['sample'].split('_')[0]}_*.parquet")
    if not f: return None
    b=pd.read_parquet(f[0]).iloc[:,0]; plat="EPIC" if len(b)>700000 else "450K"
    loci=[l for l in ID[r.cell]["loci"] if l in b.index]; v=b.reindex(loci).dropna()
    return dict(cell=r.cell,source=r.source,platform=plat,sample=r["sample"],n_loci=len(v),mean_beta=float(v.mean()),m=H(float(v.mean())),vec=v.astype("float32"))
with mp.get_context("fork").Pool(16) as P: rows=[x for x in P.map(read_arr,range(len(A))) if x]
print("arrays read",len(rows),f"{__import__('time').time()-t0:.0f}s",flush=True)
L=S[S.source=="Loyfer2023"]
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
for r in L.itertuples():
    if r.sample not in B.columns: continue
    loci=[l for l in ID[r.cell]["loci"] if l in B.index]; b=B.loc[loci,r.sample].where(C.loc[loci,r.sample]>=10).dropna()
    if len(b)<50: continue
    rows.append(dict(cell=r.cell,source="Loyfer2023",platform="WGBS",sample=r.sample,n_loci=len(b),mean_beta=float(b.mean()),m=H(float(b.mean())),vec=b.astype("float32")))
# two channels: split each cell's identity loci by its own pooled purified pattern (all its specimens, any platform)
pool={}
for r in rows: pool.setdefault(r["cell"],[]).append(r["vec"])
SPLIT={c:pd.concat(v,axis=1).mean(axis=1) for c,v in pool.items()}
ERO=(0.0,0.02,0.05,0.10)   # blur toward 0.5: beta' = beta + s*(0.5-beta)
for r in rows:
    sp=SPLIT[r["cell"]].reindex(r["vec"].index); hi=r["vec"][sp>0.5]; lo=r["vec"][sp<0.5]
    r["n_meth"]=len(hi); r["n_unmeth"]=len(lo)
    for e in ERO:
        r[f"mM_{e}"]=H(float((hi+e*(0.5-hi)).mean())) if len(hi) else np.nan
        r[f"mU_{e}"]=H(float((lo+e*(0.5-lo)).mean())) if len(lo) else np.nan
        r[f"m1_{e}"]=H(float((r["vec"]+e*(0.5-r["vec"])).mean()))
    v=r["vec"]; ex=v[(sp>0.8)|(sp<0.2)]; r["n_extreme"]=len(ex)
    Hv=lambda x: np.nanmean(-(x*np.log2(np.clip(x,1e-6,1))+(1-x)*np.log2(np.clip(1-x,1e-6,1))))
    for e in ERO:
        r[f"mP_{e}"]=float(Hv((v+e*(0.5-v)).clip(1e-6,1-1e-6)))
        r[f"mX_{e}"]=float(Hv((ex+e*(0.5-ex)).clip(1e-6,1-1e-6))) if len(ex) else np.nan
        r[f"mXm_{e}"]=H(float((ex[sp.reindex(ex.index)>0.5]+e*(0.5-ex[sp.reindex(ex.index)>0.5])).mean())) if (sp.reindex(ex.index)>0.5).sum() else np.nan
    r["mM"]=r["mM_0.0"]; r["mU"]=r["mU_0.0"]; del r["vec"]
X=pd.DataFrame(rows); print("specimens",len(X),X.platform.value_counts().to_dict(),flush=True)
loo=[]
for (c,p),g in X.groupby(["cell","platform"]):
    if len(g)<2: continue
    for i in g.index:
        d=dict(cell=c,platform=p,sample=g["sample"][i],A_loo=g.m[i]/g.m.drop(i).mean(),AM_loo=g.mM[i]/g.mM.drop(i).mean(),AU_loo=g.mU[i]/g.mU.drop(i).mean())
        for e in ERO[1:]:
            d[f"A1_ero{e}"]=g[f"m1_{e}"][i]/g.m.drop(i).mean(); d[f"AM_ero{e}"]=g[f"mM_{e}"][i]/g.mM.drop(i).mean(); d[f"AU_ero{e}"]=g[f"mU_{e}"][i]/g.mU.drop(i).mean()
        for f_ in ("P","X","Xm"):
            base=g[f"m{f_}_0.0"].drop(i).mean(); d[f"A{f_}_loo"]=g[f"m{f_}_0.0"][i]/base
            for e in ERO[1:]: d[f"A{f_}_ero{e}"]=g[f"m{f_}_{e}"][i]/base
        loo.append(d)
LOO=pd.DataFrame(loo)
F=X.groupby(["cell","platform"]).agg(n=("m","size"),floor=("m","mean"),sd=("m","std"),floor_meth=("mM","mean"),floor_unmeth=("mU","mean"),n_meth=("n_meth","median"),n_unmeth=("n_unmeth","median"),mean_beta=("mean_beta","mean"),n_loci=("n_loci","median"),sources=("source",lambda s:",".join(sorted(set(s))))).reset_index()
inN=lambda a:float(((a>=0.95)&(a<=1.05)).mean())
q=LOO.groupby(["cell","platform"]).agg(loo_sd=("A_loo","std"),loo_in_normal=("A_loo",inN),AM_sd=("AM_loo","std"),AM_in=("AM_loo",inN),AU_sd=("AU_loo","std"),AU_in=("AU_loo",inN),
   AM_ero05=("AM_ero0.05","median"),AU_ero05=("AU_ero0.05","median"),A1_ero05=("A1_ero0.05","median")).reset_index()
F=F.merge(q,on=["cell","platform"],how="left")
W=F[F.platform=="WGBS"].set_index("cell").floor; cp=[]
for p in ("450K","EPIC"):
    Ar=F[F.platform==p].set_index("cell").floor; both=sorted(set(Ar.index)&set(W.index))
    for c in both:
        k=np.median([Ar[o]/W[o] for o in both if o!=c]) if len(both)>2 else np.nan
        cp.append(dict(platform=p,cell=c,floor_array=Ar[c],floor_wgbs=W[c],ratio=Ar[c]/W[c],k_loo=k,pred=W[c]*k,A_if_projected=Ar[c]/(W[c]*k) if k==k else np.nan))
CP=pd.DataFrame(cp)
X.to_csv("refloor_specimens.csv",index=False); LOO.to_csv("refloor_loo.csv",index=False); F.to_csv("refloor_table.csv",index=False); CP.to_csv("refloor_crossplatform.csv",index=False)
out={"version":"reference_floors_v1","built":"2026-10-01","definition":"H(mean beta at the cell's v2 identity loci) on purified healthy specimens, our Stage 1; status 'measured' = from specimens of that platform",
     "floors":{}}
for r in F.itertuples(): out["floors"].setdefault(r.cell,{})[r.platform]=dict(floor_meth=round(r.floor_meth,6),floor_unmeth=round(r.floor_unmeth,6),floor_single=round(r.floor,6),n=int(r.n),sd=None if r.sd!=r.sd else round(r.sd,5),sources=r.sources,status="measured")
json.dump(out,open("reference_floors_v1.json","w"),indent=1)
pd.set_option("display.width",220)
print(F[F.platform!="WGBS"][["cell","platform","n","floor","floor_meth","floor_unmeth","n_meth","n_unmeth","AM_sd","AM_in","AU_sd","AU_in","A1_ero05","AM_ero05","AU_ero05"]].round(4).to_string(index=False))
ar=LOO[LOO.platform!="WGBS"]
print("ARRAY LOO all specimens: single in-Normal %.3f | meth %.3f | unmeth %.3f"%(inN(ar.A_loo),inN(ar.AM_loo),inN(ar.AU_loo)))
for e in ERO[1:]: print(f"blur {e}: median A single {ar[f'A1_ero{e}'].median():.4f} | meth {ar[f'AM_ero{e}'].median():.4f} | unmeth {ar[f'AU_ero{e}'].median():.4f} | share outside Normal: single {1-inN(ar[f'A1_ero{e}']):.2f} meth {1-inN(ar[f'AM_ero{e}']):.2f} unmeth {1-inN(ar[f'AU_ero{e}']):.2f}"); print(F[F.platform=="WGBS"].loo_in_normal.describe().round(3).to_dict())
print(CP.round(4).to_string(index=False)); SN=[]
for (p_,lab),sub in ((("arrays","arrays"),LOO[LOO.platform!="WGBS"]),(("WGBS","WGBS"),LOO[LOO.platform=="WGBS"])):
    for f_,nm in (("A","single H(mean)"),("AM","meth channel H(mean)"),("AP","per-site mean H"),("AX","extreme sites, per-site H"),("AXm","extreme meth sites, H(mean)")):
        col=f"{f_}_loo" if f_!="A" else "A_loo"; e05=f"{f_}_ero0.05" if f_!="A" else "A1_ero0.05"; e10=f"{f_}_ero0.1" if f_!="A" else "A1_ero0.1"
        x=sub[col].replace([np.inf,-np.inf],np.nan).dropna(); sd=float(x.std())
        SN.append(dict(data=p_,form=nm,n=len(x),loo_in_normal=inN(x),loo_sd=sd,shift05=float(sub[e05].median()-1),shift10=float(sub[e10].median()-1),snr05=float((sub[e05].median()-1)/sd),snr10=float((sub[e10].median()-1)/sd)))
SN=pd.DataFrame(SN); SN.to_csv("refloor_snr.csv",index=False); print(SN.round(4).to_string(index=False)); print("DONE",flush=True)
