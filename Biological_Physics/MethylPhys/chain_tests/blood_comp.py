#!/usr/bin/env python3
"""Chain v3 Stage A' (2026-10-01): blood composition for the composition-matched expectation, solved on the SAME platform as the profiles.
Groups (Salas EPIC purified): NEU, EOS, BASO, MONO, B (b/naive/memory), NK, CD4T (cd4/naive/memory/treg), CD8T (cd8/naive/effector memory).
Markers per group: loci (NOT neutrophil identity sites) with within-group SD <= 0.05 and margin >= 0.25 against every other group's mean;
top 150 per group by margin. Composition: NNLS on the markers, normalised to sum 1. Profiles at the neutrophil sites = group means.
Test: the 24 Salas mixtures with markers/profiles from the OTHER study (independent), and the frozen both-study version (what the chain ships)."""
import json, glob, numpy as np, pandas as pd
from scipy.optimize import nnls
R="/home/ubuntu/data/atlas_sources"; D={"Salas2018":f"{R}/blood/GSE110554/shards","Salas2022":f"{R}/blood/GSE167998/shards"}; G={"Salas2018":"GSE110554","Salas2022":"GSE167998"}
H=lambda x: -(np.clip(x,1e-6,1-1e-6)*np.log2(np.clip(x,1e-6,1-1e-6))+(1-np.clip(x,1e-6,1-1e-6))*np.log2(1-np.clip(x,1e-6,1-1e-6)))
GRP={"neutrophils":"NEU","eosinophils":"EOS","basophils":"BASO","monocytes":"MONO","b cells":"B","naive b cells":"B","memory b cells":"B","nk cells":"NK",
     "cd4 t cells":"CD4T","naive cd4 t cells":"CD4T","memory cd4 t cells":"CD4T","regulatory t cells":"CD4T","cd8 t cells":"CD8T","naive cd8 t cells":"CD8T","effector memory cd8 t cells":"CD8T"}
NS=pd.Index(json.load(open("metA_floors_v1_2.json"))["platforms"]["EPIC"]["neutrophils"]["sites"])
St=pd.read_csv("roster_samples.csv"); St=St[St.source.isin(D)&(St.qc==True)&St.cell.isin(GRP)].copy(); St["gsm"]=St["sample"].str.split("_").str[0]
rd=lambda src,g: pd.read_parquet(glob.glob(f"{D[src]}/{g}_*.parquet")[0]).iloc[:,0].astype("float64")
B={r.gsm:rd(r.source,r.gsm) for _,r in St.iterrows()}; grp={r.gsm:GRP[r.cell] for _,r in St.iterrows()}; src={r.gsm:r.source for _,r in St.iterrows()}
common=None
for v in B.values(): common=v.dropna().index if common is None else common.intersection(v.dropna().index)
M=pd.DataFrame({g:B[g].reindex(common) for g in B})
def build(srcs):
    cols=[g for g in M if src[g] in srcs]; X=M[cols]; groups=sorted({grp[g] for g in cols})
    mu=pd.DataFrame({k:X[[g for g in cols if grp[g]==k]].mean(1) for k in groups}); sd=pd.DataFrame({k:X[[g for g in cols if grp[g]==k]].std(1).fillna(0) for k in groups})
    cand=mu.index.difference(NS); mk={}
    for k in groups:
        oth=mu.loc[cand,[c for c in groups if c!=k]]; margin=(mu.loc[cand,k].values[:,None]-oth.values); margin=np.abs(margin).min(1)
        s=pd.Series(margin,index=cand)[(sd.loc[cand,k]<=0.05).values]; mk[k]=list(s[s>=0.25].sort_values(ascending=False).index[:150])
    markers=sorted(set(sum(mk.values(),[])))
    prof_ns={k:pd.concat([B[g].reindex(NS) for g in cols if grp[g]==k],axis=1).mean(1) for k in groups}
    return dict(groups=groups,markers=markers,mu_markers=mu.loc[markers],prof_ns=prof_ns,n_markers={k:len(v) for k,v in mk.items()})
def solve(beta,Rf):
    y=beta.reindex(Rf["markers"]); ok=y.notna(); f,_=nnls(Rf["mu_markers"][ok].values,y[ok].values); f=f/f.sum() if f.sum()>0 else f
    return dict(zip(Rf["groups"],f))
def metA(beta,f,Rf):
    x=beta.reindex(NS); e=sum(v*Rf["prof_ns"][k] for k,v in f.items()); ok=x.notna()&e.notna(); return float(H(x[ok]).mean()/H(e[ok]).mean())
T=pd.read_csv("salas_mixture_truth.csv"); REF={s:build({s}) for s in D}; FULL=build(set(D))
print("markers per group (both studies):",FULL["n_markers"],"total",len(FULL["markers"]),flush=True)
out=[]
for _,r in T.iterrows():
    s=[k for k,v in G.items() if v==r.gse][0]; o=[k for k in D if k!=s][0]; b=rd(s,r.gsm); sc=100.0 if r[["cd4t","cd8t","bcell","nk","mono","neu"]].sum()>2 else 1.0
    fi=solve(b,REF[o]); ff=solve(b,FULL)
    out.append(dict(gsm=r.gsm,gse=r.gse,f_neu_true=r.neu/sc,f_neu_indep=fi.get("NEU",0),A_indep=metA(b,fi,REF[o]),f_neu_frozen=ff.get("NEU",0),A_frozen=metA(b,ff,FULL)))
X=pd.DataFrame(out).sort_values("f_neu_true"); X.to_csv("blood_comp_mixtures.csv",index=False)
inN=lambda a:((a>=0.95)&(a<=1.05)); b=X[X.f_neu_true>=0.5]
print("blood-like mixtures (n=%d): independent Normal %d/%d (A %.3f-%.3f); frozen Normal %d/%d (A %.3f-%.3f)"%(len(b),inN(b.A_indep).sum(),len(b),b.A_indep.min(),b.A_indep.max(),inN(b.A_frozen).sum(),len(b),b.A_frozen.min(),b.A_frozen.max()))
print("neutrophil fraction error (independent): median %.3f"%((X.f_neu_indep-X.f_neu_true).abs().median()))
pd.set_option("display.width",200); print(X.round(4).to_string(index=False))
FROZ={"version":"blood_composition_EPIC_v1","date":"2026-10-01","groups":FULL["groups"],"markers":FULL["markers"],
      "mu_markers":{k:[round(float(v),5) for v in FULL["mu_markers"][k]] for k in FULL["groups"]},
      "profiles_at_neutrophil_sites":{k:[None if np.isnan(v) else round(float(v),5) for v in FULL["prof_ns"][k].reindex(NS)] for k in FULL["groups"]},
      "neutrophil_sites":list(NS),"group_of_cell":GRP,"rule":"markers: not neutrophil sites; within-group SD <= 0.05; margin >= 0.25 vs every other group; top 150/group; NNLS, sum 1"}
json.dump(FROZ,open("blood_composition_EPIC_v1.json","w")); print("DONE")
