mkdir -p remote_jobs/wbneut2 && cd remote_jobs/wbneut2 && cp ../deconv2/salas_mixture_truth.csv ../deconv2/deconv_v2.py ../amlserial/roster_samples.csv . && cat > PROC_WB_NEUT_01_PREREG.md <<'EOF'
# PROC-WB-NEUT-01 — pre-registration (written 2026-10-01, before any mixture is read on this reading)

**Question.** In a blood specimen, does Met-A of the neutrophil read Normal when healthy, and detect a known 2 % loss of the neutrophil's pattern,
if it is read against the healthy expectation for that specimen's own composition (no cohort)?

**Reading (fixed now).** Sites: neutrophil identity sites (Met-A v1.2 rule: SD ≤ 0.05; mean 0.75–0.95 or 0.05–0.25; ≤ 3,000 per channel).
Expected healthy β at each site for the specimen: e_i = Σ_c f_c μ_{c,i} (f = the specimen's fractions, μ = purified healthy profiles).
Met-A_blood = mean_i H(β_i) / mean_i H(e_i). Normal 0.95–1.05.
**Independence.** Sites and profiles for the Salas 2018 mixtures come only from Salas 2022 purified arrays, and vice versa.

**Data.** The 24 Salas DNA mixtures of purified healthy cells (GSE110554, GSE167998), known fractions (salas_mixture_truth.csv); our Stage 1.
Two fraction sources: (a) the known fractions, (b) the atlas v2 solver's fractions (deconv_v2).
**Damage.** The neutrophil share is blurred 2 % toward 0.5 in silico: β' = β + f_neu · 0.02 · (0.5 − μ_neu).

**Predictions** (on mixtures with neutrophil fraction ≥ 0.50, the blood-like range):
- W1: healthy, known fractions: all in Normal.
- W2: healthy, solver fractions: ≥ 80 % in Normal.
- W3: 2 % damage, known fractions: ≥ 80 % above 1.05.
Descriptive: every mixture vs its neutrophil fraction; the raw reading (no expectation) for comparison; repeat-reading agreement is tested
separately on real replicate pairs.
**Stated limits now.** DNA mixtures, not real blood; few mixtures with ≥ 50 % neutrophils; damage is simulated.
EOF
shasum -a 256 PROC_WB_NEUT_01_PREREG.md | cut -c1-16
cat > wb_neut2.py <<'EOF'
#!/usr/bin/env python3
"""PROC-WB-NEUT-01 (pre-registered 2026-10-01). Neutrophil Met-A in blood against the composition-matched healthy expectation."""
import pandas as pd, numpy as np, glob, os
R="/home/ubuntu/data/atlas_sources"; D={"GSE110554":f"{R}/blood/GSE110554/shards","GSE167998":f"{R}/blood/GSE167998/shards"}
SRC={"Salas2018":"GSE110554","Salas2022":"GSE167998"}
H=lambda x: -(np.clip(x,1e-6,1-1e-6)*np.log2(np.clip(x,1e-6,1-1e-6))+(1-np.clip(x,1e-6,1-1e-6))*np.log2(1-np.clip(x,1e-6,1-1e-6)))
rd=lambda gse,g: pd.read_parquet(glob.glob(f"{D[gse]}/{g}_*.parquet")[0]).iloc[:,0].astype("float64")
S=pd.read_csv("roster_samples.csv"); S=S[S.source.isin(SRC)&(S.qc==True)].copy(); S["gse"]=S.source.map(SRC); S["gsm"]=S["sample"].str.split("_").str[0]
T=pd.read_csv("salas_mixture_truth.csv")
MAP={"neu":["neutrophils"],"mono":["monocytes"],"nk":["nk cells"],"bcell":["b cells","naive b cells","memory b cells"],"cd4t":["cd4 t cells","naive cd4 t cells","memory cd4 t cells"],
     "cd8t":["cd8 t cells","naive cd8 t cells","effector memory cd8 t cells"],"treg":["regulatory t cells"],"cd4nv":["naive cd4 t cells"],"cd4mem":["memory cd4 t cells"],
     "bnv":["naive b cells"],"bmem":["memory b cells"],"eos":["eosinophils"]}
def refs(gse):
    s=S[S.gse==gse]; B=pd.concat({r.gsm:rd(gse,r.gsm) for _,r in s.iterrows()},axis=1); return B, dict(zip(s.gsm,s.cell))
REF={g:refs(g) for g in D}
def sites(B,cell):
    X=B[[k for k,c in cell.items() if c=="neutrophils"]]; m=X.mean(1); sd=X.std(1); ok=(sd<=0.05)&X.notna().all(1)
    return list(sd[ok&m.between(0.75,0.95)].sort_values().index[:3000])+list(sd[ok&m.between(0.05,0.25)].sort_values().index[:3000])
out=[]
for _,r in T.iterrows():
    other=[g for g in D if g!=r.gse][0]; B,cell=REF[other]; St=sites(B,cell)
    prof={c:B[[k for k,v in cell.items() if v==c]].mean(1) for c in set(cell.values())}
    sc=100.0 if r[["cd4t","cd8t","bcell","nk","mono","neu"]].sum()>2 else 1.0
    f={}
    for col,cells in MAP.items():
        if col in r and pd.notna(r[col]) and r[col]>0:
            c=[x for x in cells if x in prof]
            if c: f[c[0]]=f.get(c[0],0)+r[col]/sc
    tot=sum(f.values()); f={k:v/tot for k,v in f.items()}
    x=rd(r.gse,r.gsm).reindex(St); e=sum(v*prof[k].reindex(St) for k,v in f.items())
    ok=x.notna()&e.notna(); fn=f.get("neutrophils",0)
    A=float(H(x[ok]).mean()/H(e[ok]).mean()); Araw=float(H(x[ok]).mean()/H(prof["neutrophils"].reindex(St)[ok]).mean())
    xd=x+fn*0.02*(0.5-prof["neutrophils"].reindex(St)); Ad=float(H(xd[ok]).mean()/H(e[ok]).mean())
    out.append(dict(gsm=r.gsm,gse=r.gse,f_neu=fn,A_known=A,A_known_damaged=Ad,A_raw=Araw,n_sites=int(ok.sum())))
X=pd.DataFrame(out).sort_values("f_neu"); X.to_csv("wb_neut2_readings.csv",index=False)
inN=lambda a:((a>=0.95)&(a<=1.05))
b=X[X.f_neu>=0.5]; print("mixtures with f_neu>=0.5:",len(b))
print("W1 healthy known-fraction Normal: %d/%d"%(inN(b.A_known).sum(),len(b)))
print("W3 2%% damage above 1.05: %d/%d"%((b.A_known_damaged>1.05).sum(),len(b)))
pd.set_option("display.width",200); print(X.round(4).to_string(index=False)); print("DONE (W2 solver-fraction run follows in a separate step)")
EOF
python3 -c "import ast;ast.parse(open('wb_neut2.py').read());print('ok')"