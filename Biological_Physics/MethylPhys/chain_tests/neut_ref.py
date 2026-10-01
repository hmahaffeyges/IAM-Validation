#!/usr/bin/env python3
"""Neutrophil reference for chain v3 (2026-10-01) + PROC-WB-NEUT-01 W2 (solver fractions; pre-registered sha 36e8620f).
Freezes, at the 6,000 frozen neutrophil identity sites: (1) mean beta of every Salas EPIC purified blood cell type (profiles for the
composition-matched expectation); (2) per-site healthy spread of H(beta) among the 12 neutrophils (shrunk, k = 10) and genomic order (for the
Met-A C-score); (3) the healthy clustering baseline (leave-one-out neutrophil maps). Then W2: the 24 Salas mixtures, composition from the
atlas v2 solver (whole-blood cell set), expectation with profiles from the OTHER study."""
import json, glob, os, urllib.request, numpy as np, pandas as pd
R="/home/ubuntu/data/atlas_sources"; D={"Salas2018":f"{R}/blood/GSE110554/shards","Salas2022":f"{R}/blood/GSE167998/shards"}; G={"Salas2018":"GSE110554","Salas2022":"GSE167998"}
H=lambda x: -(np.clip(x,1e-6,1-1e-6)*np.log2(np.clip(x,1e-6,1-1e-6))+(1-np.clip(x,1e-6,1-1e-6))*np.log2(1-np.clip(x,1e-6,1-1e-6)))
F=json.load(open("metA_floors_v1_2.json"))["platforms"]["EPIC"]["neutrophils"]; S=pd.Index(F["sites"])
PMAP={"t central memory cd4":"memory cd4 t cells","t effector memory cd4":"memory cd4 t cells","t effector cell cd8":"effector memory cd8 t cells"}   # atlas subtypes with no Salas EPIC profile -> nearest purified profile
St=pd.read_csv("roster_samples.csv"); St=St[St.source.isin(D)&(St.qc==True)].copy(); St["gsm"]=St["sample"].str.split("_").str[0]
rd=lambda src,g: pd.read_parquet(glob.glob(f"{D[src]}/{g}_*.parquet")[0]).iloc[:,0].astype("float64")
X={r.gsm:(r.cell,r.source,rd(r.source,r.gsm).reindex(S)) for _,r in St.iterrows()}
cells=sorted({c for c,_,_ in X.values()})
prof=lambda srcs: {c:pd.concat([v for cc,s,v in X.values() if cc==c and s in srcs],axis=1).mean(1) for c in cells if any(cc==c and s in srcs for cc,s,_ in X.values())}
P_all=prof(set(D))
NE=[g for g,(c,_,_) in X.items() if c=="neutrophils"]; HN=pd.concat([H(X[g][2]) for g in NE],axis=1)
sd=HN.std(1); sp=np.sqrt(np.nanmedian(sd**2)); s_sh=np.sqrt(((len(NE)-1)*sd**2+10*sp**2)/(len(NE)-1+10))
EP=pd.read_csv("/home/ubuntu/data/EPIC.hg38.manifest.gencode.v36.tsv.gz",sep="\t",usecols=["probeID","CpG_chrm","CpG_beg"]).dropna()
EP=EP[EP.CpG_chrm.str.match(r"^chr([0-9]+|X|Y)$")].copy(); EP["ck"]=EP.CpG_chrm.str[3:].replace({"X":"23","Y":"24"}).astype(int)
EP=EP.sort_values(["ck","CpG_beg"]).reset_index(drop=True); order=pd.Series(np.arange(len(EP)),index=EP.probeID)
So=[s for s in S if s in order.index]; So=sorted(So,key=lambda s:order[s])
def clus(z,w=50):
    o=z.reindex(So).dropna().values; nb=len(o)//w; b=o[:nb*w].reshape(nb,w).mean(1)*np.sqrt(w); return float(np.var(b)/np.var(o))
base=[]
for g in NE:
    oth=[x for x in NE if x!=g]; m=pd.concat([H(X[x][2]) for x in oth],axis=1).mean(1); base.append(clus((H(X[g][2])-m)/s_sh))
REF={"version":"neutrophil_reference_v1","date":"2026-10-01","platform":"EPIC","sites_ordered":So,
     "profiles_mean_beta":{c:[None if np.isnan(v) else round(float(v),5) for v in P_all[c].reindex(So)] for c in P_all},
     "neutrophil_H_mean":[round(float(v),6) for v in HN.mean(1).reindex(So)],"neutrophil_H_sd_shrunk":[round(float(v),6) for v in s_sh.reindex(So)],
     "clustering_block":50,"healthy_clustering_LOO":[round(b,4) for b in base],"healthy_clustering_median":round(float(np.median(base)),4),
     "n_neutrophils":len(NE),"sources":["GSE110554","GSE167998"]}
REF["profile_map"]=PMAP
json.dump(REF,open("neutrophil_reference_v1.json","w")); print("ref saved; healthy clustering LOO median %.3f range %.3f-%.3f"%(np.median(base),min(base),max(base)),flush=True)
# ---- W2
U=json.load(open("urls.json")); AP="/home/ubuntu/data/IAMAtlas_v2.parquet"
if not os.path.exists(AP): urllib.request.urlretrieve(U["atlas"],AP)
import stage_a_composition_v2 as SA
SOLV=SA.solver(AP,"whole blood")
T=pd.read_csv("salas_mixture_truth.csv"); out=[]
for _,r in T.iterrows():
    src=[k for k,v in G.items() if v==r.gse][0]; other=[k for k in D if k!=src]; Po=prof(set(other))
    full=pd.read_parquet(glob.glob(f"{D[src]}/{r.gsm}_*.parquet")[0]).iloc[:,0].astype("float64"); full.index=full.index.astype(str)
    comp=SOLV.deconvolve(full,n_boot=20)["fractions"]
    cm={}
    for c,v in comp.items():
        if v>0: k=PMAP.get(c,c); cm[k]=cm.get(k,0)+v
    f={c:v for c,v in cm.items() if c in Po}; un=sum(v for c,v in cm.items() if c not in Po)
    if r.gsm in ("GSM5121359","GSM2998066"): print(r.gsm,"solver:",{c:round(v,3) for c,v in sorted(comp.items(),key=lambda x:-x[1]) if v>0.005},flush=True); tot=sum(f.values()); f={k:v/tot for k,v in f.items()}
    x=full.reindex(So); e=sum(v*Po[k].reindex(So) for k,v in f.items()); ok=x.notna()&e.notna()
    sc=100.0 if r[["cd4t","cd8t","bcell","nk","mono","neu"]].sum()>2 else 1.0
    out.append(dict(gsm=r.gsm,gse=r.gse,f_neu_true=r.neu/sc,f_neu_solver=comp.get("neutrophils",0.0),unmatched_fraction=un,A_solver=float(H(x[ok]).mean()/H(e[ok]).mean())))
W=pd.DataFrame(out).sort_values("f_neu_true"); W.to_csv("w2_solver_readings.csv",index=False)
b=W[W.f_neu_true>=0.5]; inN=lambda a:((a>=0.95)&(a<=1.05))
print("W2 healthy, solver fractions, Normal: %d/%d (>= 80%%)"%(inN(b.A_solver).sum(),len(b)))
pd.set_option("display.width",200); print(W.round(4).to_string(index=False)); print("DONE")
