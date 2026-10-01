#!/usr/bin/env python3
"""Development diagnostic (2026-10-01): whole-blood neutrophil Met-A with the neutrophil fraction re-estimated on the neutrophil sites.
Composition from the EPIC NNLS (blood_composition_EPIC_v1 rule, other-study profiles). Then, keeping the non-neutrophil cells in their NNLS
proportions, f_neu is re-fitted by least squares on the neutrophil sites in BETA space (mixing is linear in beta; pattern loss is a pull toward
0.5, which is not along the mixing direction). Expectation e = f mu_neu + (1-f) mu_other; Met-A = mean H(beta)/mean H(e).
Checks on mixtures with >= 50 % neutrophils: healthy in Normal; 2 % neutrophil blur (beta' = beta + f_true*0.02*(0.5-mu_neu)) above 1.05;
the fraction the fit absorbs from the damage."""
import json, glob, numpy as np, pandas as pd
exec(open("blood_comp.py").read().split("T=pd.read_csv(\"salas_mixture_truth.csv\")")[0])
T=pd.read_csv("salas_mixture_truth.csv"); REF={s:build({s}) for s in D}
def fit_f(beta,f0,Rf):
    mn=Rf["prof_ns"]["NEU"]; oth={k:v for k,v in f0.items() if k!="NEU"}; t=sum(oth.values())
    mo=sum((v/t)*Rf["prof_ns"][k] for k,v in oth.items()) if t>0 else mn*0
    x=beta.reindex(NS); ok=x.notna()&mn.notna()&mo.notna(); d=(mn-mo)[ok]; f=float(((x[ok]-mo[ok])*d).sum()/(d*d).sum())
    f=min(max(f,0.0),1.0); return f, f*mn+(1-f)*mo
def A(x,e): ok=x.notna()&e.notna(); return float(H(x[ok]).mean()/H(e[ok]).mean())
out=[]
for _,r in T.iterrows():
    s=[k for k,v in G.items() if v==r.gse][0]; o=[k for k in D if k!=s][0]; b=rd(s,r.gsm); sc=100.0 if r[["cd4t","cd8t","bcell","nk","mono","neu"]].sum()>2 else 1.0; ft=r.neu/sc
    Rf=REF[o]; f0=solve(b,Rf); mn=Rf["prof_ns"]["NEU"]
    f1,e1=fit_f(b,f0,Rf); x=b.reindex(NS)
    ix=NS.intersection(b.index); bd=b.copy(); bd.loc[ix]=(b.loc[ix]+ft*0.02*(0.5-mn.reindex(ix))).values; f2,e2=fit_f(bd,f0,Rf); xd=bd.reindex(NS)
    e0=sum(v*Rf["prof_ns"][k] for k,v in f0.items())
    out.append(dict(gsm=r.gsm,f_true=ft,f_nnls=f0.get("NEU",0),f_fit=f1,f_fit_damaged=f2,A_nnls=A(x,e0),A_fit=A(x,e1),A_fit_damaged=A(xd,e2),A_nnls_damaged=A(xd,e0)))
X=pd.DataFrame(out).sort_values("f_true"); X.to_csv("selfconsist.csv",index=False)
b=X[X.f_true>=0.5]; inN=lambda a:((a>=0.95)&(a<=1.05))
print("n=%d | healthy Normal: nnls %d, fit %d | 2%% damage >1.05: nnls %d, fit %d | median f error: nnls %.3f fit %.3f | f absorbed by damage %.4f"%(
 len(b),inN(b.A_nnls).sum(),inN(b.A_fit).sum(),(b.A_nnls_damaged>1.05).sum(),(b.A_fit_damaged>1.05).sum(),(b.f_nnls-b.f_true).abs().median(),(b.f_fit-b.f_true).abs().median(),(b.f_fit_damaged-b.f_fit).median()))
pd.set_option("display.width",200); print(X.round(4).to_string(index=False)); print("DONE")
