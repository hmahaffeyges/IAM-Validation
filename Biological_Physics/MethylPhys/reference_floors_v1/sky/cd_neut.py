#!/usr/bin/env python3
"""Genome-distance correlation C(d) for single neutrophil arrays (2026-10-02). Six physical Salas EPIC arrays (GSE110554), Stage 1 betas.
Per array: pairs of EPIC CpGs on the same chromosome at genomic distance d (log bins 10 bp - 10 Mb), up to 300,000 random pairs per bin.
C_beta(d) = Pearson correlation of beta across pairs (co-methylation within one specimen).
C_z(d): correlation of the residual z (H(beta) minus the other five arrays' mean H, over their shrunk SD), at sites passing the chain's map rule;
for the healthy array and the same array with 5 % blur in 10 regions."""
import glob, json, numpy as np, pandas as pd
R="/home/ubuntu/data/atlas_sources/blood/GSE110554/shards"; rng=np.random.default_rng(7)
F=json.load(open("metA_floors_v1_3.json"))["platforms"]["EPIC"]["neutrophils"]; ARR=F["refs"]
B=pd.concat([pd.read_parquet(glob.glob(f"{R}/{a}*.parquet")[0]).iloc[:,0].rename(a.split("_")[0]) for a in ARR],axis=1).astype("float64")
M=pd.read_csv("/home/ubuntu/data/EPIC.hg38.manifest.gencode.v36.tsv.gz",sep="\t",usecols=["probeID","CpG_chrm","CpG_beg"]).dropna()
M=M[M.CpG_chrm.str.match(r"^chr([0-9]+)$")].set_index("probeID")
B=B.loc[B.index.intersection(M.index)].dropna(); M=M.loc[B.index]
edges=np.unique(np.round(np.logspace(1,7,25)).astype(int))
def Hb(x): x=np.clip(x,1e-6,1-1e-6); return -(x*np.log2(x)+(1-x)*np.log2(1-x))
def pairs(idx):
    m=M.loc[idx]; out=[]
    for c,g in m.groupby("CpG_chrm"):
        g=g.sort_values("CpG_beg"); out.append((g.index.values,g.CpG_beg.values.astype(np.int64)))
    P={}
    for lo,hi in zip(edges[:-1],edges[1:]):
        I=[];J=[]
        for ids,pos in out:
            n=len(pos); a=rng.integers(0,n,size=max(1,int(300000*n/len(idx))))
            l=np.searchsorted(pos,pos[a]+lo); h=np.searchsorted(pos,pos[a]+hi); ok=h>l
            a=a[ok]; l=l[ok]; h=h[ok]; j=l+(rng.random(len(a))*(h-l)).astype(int)
            I.append(ids[a]); J.append(ids[j])
        P[(lo,hi)]=(np.concatenate(I),np.concatenate(J))
    return P
def corr(v,P):
    rows=[]
    for (lo,hi),(i,j) in P.items():
        x=v.reindex(i).values; y=v.reindex(j).values; k=np.isfinite(x)&np.isfinite(y)
        rows.append((lo,hi,int(k.sum()),float(np.corrcoef(x[k],y[k])[0,1]) if k.sum()>100 else np.nan))
    return rows
cols=list(B.columns); allp=pairs(B.index); out=[]
for a in cols:
    for lo,hi,n,r in corr(B[a],allp): out.append(dict(array=a,quantity="beta",d_lo=lo,d_hi=hi,n_pairs=n,C=r))
h=cols[0]; ref=cols[1:]; mu=B[ref].mean(1); sd=B[ref].std(1)
idx=B.index[(sd<=0.05)&(((mu>=0.75)&(mu<=0.95))|((mu>=0.05)&(mu<=0.25)))]
Hr=Hb(B.loc[idx,ref]); m=Hr.mean(1); s=Hr.std(1); sp=np.sqrt(np.nanmedian(s**2)); s2=np.sqrt(((5-1)*s**2+10*sp**2)/(5-1+10))
x=B.loc[idx,h]; z=(Hb(x)-m)/s2
o=M.loc[idx].sort_values(["CpG_chrm","CpG_beg"]).index; L=len(o); bl=int(0.005*L); st=rng.choice(L-bl,10,replace=False)
loc=pd.Index(np.unique(np.concatenate([o[s_:s_+bl] for s_ in st]))); xl=x.copy(); xl.loc[loc]=xl.loc[loc]+0.05*(0.5-xl.loc[loc])
zl=(Hb(xl)-m)/s2; zp=pairs(idx)
for nm,v in (("z_healthy",z),("z_local5pct",zl)):
    for lo,hi,n,r in corr(v,zp): out.append(dict(array=h,quantity=nm,d_lo=lo,d_hi=hi,n_pairs=n,C=r))
T=pd.DataFrame(out); T.to_csv("cd_neut.csv",index=False)
print(T.pivot_table(index=["d_lo"],columns=["quantity"],values="C",aggfunc="median").round(3).to_string()); print("DONE")
