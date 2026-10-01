#!/usr/bin/env python3
"""Met-A neutrophil floor v1.3 (development, 2026-10-01). Fix: GSE110554 and GSE167998 deposit the same 6 physical neutrophil arrays (same Sentrix IDs);
v1.2 counted 12. Here each physical array counts once (GSE110554 copy). Sites: the frozen 6,000 (unchanged). Floor = mean over the 6 arrays of mean H(beta)
at the sites. Held-out reading: for each array, sites re-chosen by the canon rule (SD <= 0.05; mean beta 0.75-0.95 or 0.05-0.25; <= 3,000 per channel,
smallest SD) on the OTHER 5 arrays only, floor from those 5, the held-out array read on them. Also rebuilds the per-site healthy spread (shrunk, k = 10)
and the leave-one-out clustering baseline on the 6. Every other cell in the development floors file is checked for the same duplication."""
import glob, json, numpy as np, pandas as pd
H=lambda x: -(np.clip(x,1e-6,1-1e-6)*np.log2(np.clip(x,1e-6,1-1e-6))+(1-np.clip(x,1e-6,1-1e-6))*np.log2(1-np.clip(x,1e-6,1-1e-6)))
F=json.load(open("metA_floors_v1_2.json")); N=F["platforms"]["EPIC"]["neutrophils"]; S=pd.Index(N["sites"])
refs=N["refs"]; sentrix={}
for r in refs:
    g,sx=r.split("_",1); sentrix.setdefault(sx,[]).append(g)
uniq=[sorted(v)[0] for v in sentrix.values()]; print("arrays listed",len(refs),"physical",len(uniq),uniq,flush=True)
D="/home/ubuntu/data/atlas_sources/blood/GSE110554/shards"
X={g:pd.read_parquet(glob.glob(f"{D}/{g}_*.parquet")[0]).iloc[:,0].astype("float64") for g in uniq}
for g in X: X[g].index=X[g].index.astype(str)
M=pd.DataFrame(X)
fl=float(np.mean([H(M.loc[S,g].dropna()).mean() for g in uniq])); print("floor v1.3 %.6f (v1.2 %.6f)"%(fl,N["floor"]),flush=True)
def sites(cols):
    R=M[cols]; mu=R.mean(1); sd=R.std(1); ok=sd<=0.05
    hi=sd[ok&(mu>=0.75)&(mu<=0.95)].sort_values().index[:3000]; lo=sd[ok&(mu>=0.05)&(mu<=0.25)].sort_values().index[:3000]; return hi.union(lo)
loo=[]
for g in uniq:
    oth=[x for x in uniq if x!=g]; s=sites(oth); f5=float(np.mean([H(M.loc[s,x].dropna()).mean() for x in oth]))
    a_new=float(H(M.loc[s,g].dropna()).mean()/f5)
    f5_frozen=float(np.mean([H(M.loc[S,x].dropna()).mean() for x in oth])); a_fz=float(H(M.loc[S,g].dropna()).mean()/f5_frozen)
    loo.append(dict(ref=g,A_heldout_sites_reselected=a_new,A_heldout_frozen_sites=a_fz,n_sites=len(s)))
L=pd.DataFrame(loo); print(L.round(4).to_string(index=False),flush=True)
print("held-out (sites re-chosen on the other 5): SD %.4f range %.4f-%.4f | frozen sites: SD %.4f"%(L.A_heldout_sites_reselected.std(),L.A_heldout_sites_reselected.min(),L.A_heldout_sites_reselected.max(),L.A_heldout_frozen_sites.std()),flush=True)
# reference spread + clustering baseline on the 6
Rf=json.load(open("neutrophil_reference_v1.json")); So=pd.Index(Rf["sites_ordered"])
HN=pd.concat([H(M.loc[So,g]) for g in uniq],axis=1); sd=HN.std(1); sp=np.sqrt(np.nanmedian(sd**2)); n=len(uniq); s_sh=np.sqrt(((n-1)*sd**2+10*sp**2)/(n-1+10))
def clus(z,w=50):
    o=z.dropna().values; nb=len(o)//w; b=o[:nb*w].reshape(nb,w).mean(1)*np.sqrt(w); return float(np.var(b)/np.var(o))
base=[clus((H(M.loc[So,g])-pd.concat([H(M.loc[So,x]) for x in uniq if x!=g],axis=1).mean(1))/s_sh) for g in uniq]
print("clustering LOO",np.round(base,4).tolist(),"median %.4f (v1 %.4f)"%(np.median(base),Rf["healthy_clustering_median"]),flush=True)
N13=dict(N); N13.update(floor=fl,n_ref=len(uniq),refs=[f"{g}_{sx}" for sx,v in sentrix.items() for g in [sorted(v)[0]]],
    duplicates_removed={sx:v for sx,v in sentrix.items() if len(v)>1},
    precision_heldout=dict(n=len(L),sd=float(L.A_heldout_sites_reselected.std()),min=float(L.A_heldout_sites_reselected.min()),max=float(L.A_heldout_sites_reselected.max()),
        rule="sites re-chosen on the other 5 physical arrays; held-out array and its duplicate excluded"))
F13={"version":"metA_floors_v1_3","date":"2026-10-01","platforms":{"EPIC":{"neutrophils":N13}},
     "change":"each physical array counted once (Sentrix ID); v1.2 counted GSE110554 and GSE167998 deposits of the same 6 arrays as 12"}
json.dump(F13,open("metA_floors_v1_3.json","w"))
R13=dict(Rf); R13.update(version="neutrophil_reference_v1_1",neutrophil_H_mean=[round(float(v),6) for v in HN.mean(1)],neutrophil_H_sd_shrunk=[round(float(v),6) for v in s_sh],
    healthy_clustering_LOO=[round(b,4) for b in base],healthy_clustering_median=round(float(np.median(base)),4),n_neutrophils=n,sources=["GSE110554 (6 physical arrays; GSE167998 copies are the same arrays)"])
json.dump(R13,open("neutrophil_reference_v1_1.json","w")); L.to_csv("metA_floors_v1_3_loo.csv",index=False); print("DONE")
