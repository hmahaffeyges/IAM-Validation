#!/usr/bin/env python3
# INSTRUMENT-TEST: PROC-STAGE2D-03 - joint-fit foreign-cell detector. Bar results only. Own output dir; polls STOP.
import os, sys, json, glob, time, numpy as np, pandas as pd
from scipy.optimize import nnls
W=os.path.dirname(os.path.abspath(__file__)); CH=os.path.join(W,"iamrepo/Biological_Physics/MethylPhys/chain"); sys.path.insert(0,CH); sys.path.insert(0,CH+"/Synthetic_Patient_Generator")
import cpg_conductor as C, synthetic_patient_generator as SPG
# detection_panel_v1/v2 were retired 2026-09-27 (PROC-STAGE2D-02/03); this runner reads them as the record it was built from
_RET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RETIRED_2026-09', 'detection_panel_v1_v2')
_RETIRED_PANEL_1 = os.path.join(_RET, 'detection_panel_v1.json'); _RETIRED_PANEL_2 = os.path.join(_RET, 'detection_panel_v2.json')
OUT=os.path.join(W,"results/stage2d03"); os.makedirs(OUT,exist_ok=True); open(os.path.join(OUT,"PID"),"w").write(str(os.getpid()))
P=json.load(open(_RETIRED_PANEL_2)); M=P["markers"]; Ab0=np.array([P["blood_ref"][c] for c in P["blood_columns"]]).T
cells=list(P["foreign_ref"]); T0=np.array([P["foreign_ref"][c] for c in cells]).T
pf=pd.read_parquet(os.path.join(W,"stage1_betas_GSE87571_FULL.parquet")); pf.index=pf.index.map(str)
def fit(beta):
    mapped,_=C.stage_1s_scale_map(beta.to_dict(),"stage1_noob_450K"); v=np.array([mapped.get(m,np.nan) for m in M]); ok=~np.isnan(v)
    X=np.column_stack([Ab0[ok],T0[ok]]); coef,_=nnls(X,v[ok]); return pd.Series(coef[Ab0.shape[1]:]/max(coef.sum(),1e-9),index=cells)
# ---- null on all 732 (shard per array)
rows={}
for k,g in enumerate(pf.columns):
    if os.path.exists(os.path.join(OUT,"STOP")): print("STOP",flush=True); break
    sh=os.path.join(OUT,f"null_{g}.json")
    if os.path.exists(sh): rows[g]=pd.Series(json.load(open(sh))); continue
    s=fit(pf[g].dropna()); json.dump(s.to_dict(),open(sh+".tmp","w")); os.replace(sh+".tmp",sh); rows[g]=s
    if k%100==99: print(f"null {k+1}/{len(pf.columns)}",flush=True)
N=pd.DataFrame(rows).T; N.to_csv(os.path.join(OUT,"null_732.csv"))
tare=pd.read_parquet(os.path.join(W,"results/tare01/PROC_TARE_01_per_array.parquet")); tare=tare[tare.lab=="GSE87571"].set_index("gsm"); chip=tare.reindex(N.index)["chip"]
fp={c:0 for c in cells}; unsp=0; n=0
for ch in chip.dropna().unique():
    te=chip.index[chip==ch]; tr=chip.index[(chip!=ch)&chip.notna()]; L=N.loc[tr].quantile(.99); h=(N.loc[te]>L)
    for c in cells: fp[c]+=int(h[c].sum())
    unsp+=int((h.sum(axis=1)>=3).sum()); n+=len(te)
b1=max(fp.values())/n<=0.015 and unsp/n<=0.01
print("B1 leave-one-chip-out FP %:",{c:round(fp[c]/n*100,2) for c in cells},f"| unspecific {unsp}/{n} ->","MET" if b1 else "FAILED",flush=True)
floors=N.quantile(.99); bias={c:float(N[c].median()) for c in cells}
# ---- B3/B4 spikes: 12 quiet hosts not among the exploration 6
per=pd.read_csv(os.path.join(W,"handoff/heldout2d_per_array.csv")).set_index("gsm"); quiet=[g for g in per[per.ndet==0].index if g in pf.columns]
hosts=quiet[6:18]; mu=SPG._cell_means(os.path.join(W,"atlas_work/IAMAtlasREBUILD.csv")); mu.index=mu.index.map(str)
FULL=["Cortical_neurons","Glia"]; spk=[]
for c in cells:
    if os.path.exists(os.path.join(OUT,"STOP")): break
    members=c[7:].split("+") if c.startswith("family:") else [c]; col=mu[[m for m in members if m in mu.columns]].mean(axis=1).dropna()
    for f in (0.02,0.05,0.10):
        for h in hosts:
            b=pf[h].dropna(); loci=b.index.intersection(col.index); s=b.copy(); s.loc[loci]=(1-f)*b.loc[loci]+f*col.loc[loci].values
            r=fit(s); fired=[x for x in cells if r[x]>floors[x]]
            spk.append(dict(cell=c,f=f,host=h,amp=float(r[c]),detected=bool(r[c]>floors[c]),named=(r.idxmax()==c),n_fired=len(fired)))
    d=pd.DataFrame([x for x in spk if x["cell"]==c]); print(f"  {c:<48} det@.02 {d[d.f==.02].detected.mean():.2f} @.05 {d[d.f==.05].detected.mean():.2f} @.10 {d[d.f==.10].detected.mean():.2f} | named@.05 {d[d.f==.05].named.mean():.2f} @.10 {d[d.f==.10].named.mean():.2f}",flush=True)
S=pd.DataFrame(spk); S.to_csv(os.path.join(OUT,"spikes.csv"),index=False)
full=S[S.cell.isin(FULL)]; thin=S[~S.cell.isin(FULL)]
b3=(full[full.f==.05].detected.mean()>=.9) and (thin[thin.f==.05].detected.mean()>=.75) and (S[S.f==.10].detected.mean()>=.9)
b4=(S[S.f==.10].named.mean()>=.8) and (S[S.f==.05].named.mean()>=.6)
print(f"B3 detection: full@.05 {full[full.f==.05].detected.mean():.2f} thin@.05 {thin[thin.f==.05].detected.mean():.2f} all@.10 {S[S.f==.10].detected.mean():.2f} ->","MET" if b3 else "FAILED")
print(f"B4 naming: @.10 {S[S.f==.10].named.mean():.2f} @.05 {S[S.f==.05].named.mean():.2f} ->","MET" if b4 else "FAILED")
json.dump({"B1":{"met":b1,"fp":{c:fp[c]/n for c in cells},"unspecific":unsp,"n":n},"B3":{"met":bool(b3)},"B4":{"met":bool(b4)},"floors":floors.to_dict(),"bias":bias,"hosts":hosts},open(os.path.join(W,"handoff/stage2d03_results.json"),"w"),indent=1)
print("RESULTS WRITTEN",flush=True)
