#!/usr/bin/env python3
"""Met-A reference floors v1.1 (development, 2026-10-01): identity sites chosen per platform from purified arrays.
Sites for a cell (chosen on the specimens NOT being read): across-donor SD <= 0.05 and own mean beta <= 0.10 (unmethylated channel) or >= 0.90
(methylated channel); 'distinct' variant also requires |own mean - median of the other cells on that platform| >= 0.30. Up to 3000 per channel
(most stable first). Forms read on the held-out specimen: per-site mean H, H(mean) per channel. Known blur toward 0.5 (2, 5, 10 %) for sensitivity."""
import glob, json, numpy as np, pandas as pd, multiprocessing as mp, time
R="/home/ubuntu/data/atlas_sources"; t0=time.time(); ERO=(0.0,0.02,0.05,0.10)
S=pd.read_csv("roster_samples.csv"); S=S[S.qc==True]
DIR={"Moss2018":f"{R}/moss2018/shards","Salas2018":f"{R}/blood/GSE110554/shards","Salas2022":f"{R}/blood/GSE167998/shards"}
A=S[S.source.isin(DIR)].reset_index(drop=True)
def rd(i):
    r=A.iloc[i]; f=glob.glob(f"{DIR[r.source]}/{r['sample'].split('_')[0]}_*.parquet")
    if not f: return None
    b=pd.read_parquet(f[0]).iloc[:,0].astype("float32"); return (r.cell,r["sample"],"EPIC" if len(b)>700000 else "450K",b,r.source)
with mp.get_context("fork").Pool(16) as P: SP=[x for x in P.map(rd,range(len(A))) if x]
print("specimens",len(SP),f"{time.time()-t0:.0f}s",flush=True)
Hb=lambda x: -(x*np.log2(x)+(1-x)*np.log2(1-x))
def forms(v,hi,lo):
    out={}
    for e in ERO:
        def bl(idx): x=v.reindex(idx).dropna().values; return np.clip(x+e*(0.5-x),1e-6,1-1e-6)
        h,l=bl(hi),bl(lo); al=np.concatenate([h,l])
        out[e]=dict(P=float(Hb(al).mean()) if len(al) else np.nan, M=float(Hb(h.mean())) if len(h) else np.nan, U=float(Hb(l.mean())) if len(l) else np.nan)
    return out
res=[]
for plat in ("EPIC","450K"):
    SPp=[x for x in SP if x[2]==plat]
    if not SPp: continue
    common=sorted(set.intersection(*[set(x[3].index) for x in SPp]))
    Mx=pd.DataFrame({x[1]:x[3].reindex(common) for x in SPp}); cell_of={x[1]:x[0] for x in SPp}; src_of={x[1]:x[4] for x in SPp}
    cells=sorted(set(cell_of.values())); print(plat,"cells",len(cells),"probes",len(common),flush=True)
    cmean={c:Mx[[s for s in Mx if cell_of[s]==c]].mean(axis=1) for c in cells}
    def job(args):
        c,held,scope=args; mine=[s for s in Mx if cell_of[s]==c and s!=held and (scope=="all" or src_of[s]==src_of[held])]
        if len(mine)<2: return []
        mu=Mx[mine].mean(axis=1); sd=Mx[mine].std(axis=1)
        oth=pd.concat([cmean[o] for o in cells if o!=c],axis=1).median(axis=1) if len(cells)>1 else mu*np.nan
        rows=[]
        for var,(hiR,loR) in (("extreme",((0.90,1.0),(0.0,0.10))),("moderate",((0.80,0.95),(0.05,0.20)))):
            ok=sd<=0.05
            hi=sd[ok&(mu>=hiR[0])&(mu<=hiR[1])].sort_values().index[:3000]; lo=sd[ok&(mu>=loR[0])&(mu<=loR[1])].sort_values().index[:3000]
            base=[forms(Mx[s],hi,lo)[0.0] for s in mine]; bP=np.nanmean([b["P"] for b in base]); bM=np.nanmean([b["M"] for b in base]); bU=np.nanmean([b["U"] for b in base])
            f=forms(Mx[held],hi,lo)
            for e in ERO: rows.append(dict(platform=plat,cell=c,held=held,source=src_of[held],scope=scope,variant=var,n_meth=len(hi),n_unmeth=len(lo),blur=e,A_site=f[e]["P"]/bP,A_meth=f[e]["M"]/bM,A_unmeth=f[e]["U"]/bU,floor_site=bP,floor_meth=bM,floor_unmeth=bU))
        return rows
    tasks=[(cell_of[s],s,sc) for s in Mx for sc in ("all","same_study")]
    with mp.get_context("fork").Pool(32) as P:
        for rr in P.imap_unordered(job,tasks): res+=rr
    print(plat,"done",f"{time.time()-t0:.0f}s",flush=True)
X=pd.DataFrame(res); X.to_csv("refloor3_loo.csv",index=False)
inN=lambda a:float(((a>=0.95)&(a<=1.05)).mean())
T=[]
for (p,sc,v),g in X.groupby(["platform","scope","variant"]):
    g0=g[g.blur==0]
    for f in ("A_site","A_meth","A_unmeth"):
        x=g0[f].replace([np.inf,-np.inf],np.nan).dropna(); sd=float(x.std()); row=dict(platform=p,scope=sc,variant=v,form=f,n=len(x),in_normal=inN(x),sd=sd,median=float(x.median()))
        for e in ERO[1:]:
            y=g[g.blur==e][f].replace([np.inf,-np.inf],np.nan).dropna(); row[f"shift{int(e*100)}"]=float(y.median()-1); row[f"outside{int(e*100)}"]=1-inN(y)
        row["snr10"]=row["shift10"]/sd if sd else np.nan; T.append(row)
T=pd.DataFrame(T); T.to_csv("refloor3_snr.csv",index=False); pd.set_option("display.width",250); print(T.round(4).to_string(index=False))
print(X[X.blur==0].groupby(["platform","scope","variant","cell"])[["n_meth","n_unmeth"]].median().reset_index().groupby(["platform","scope","variant"])[["n_meth","n_unmeth"]].median().to_string())
print("DONE",flush=True)
