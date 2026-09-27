#!/usr/bin/env python3
# INSTRUMENT-TEST: PROC-FOREIGNSCORE-01. Own output dir; polls STOP. Bar results only.
import os, sys, json, numpy as np, pandas as pd
W=os.path.dirname(os.path.abspath(__file__)); CH=os.path.join(W,"iamrepo/Biological_Physics/MethylPhys/chain"); sys.path.insert(0,CH); sys.path.insert(0,CH+"/Synthetic_Patient_Generator")
import cpg_conductor as C, synthetic_patient_generator as SPG
OUT=os.path.join(W,"results/foreignscore01"); os.makedirs(OUT,exist_ok=True); open(os.path.join(OUT,"PID"),"w").write(str(os.getpid()))
pf=pd.read_parquet(os.path.join(W,"stage1_betas_GSE87571_FULL.parquet")); pf.index=pf.index.map(str)
mu=SPG._cell_means(os.path.join(W,"atlas_work/IAMAtlasREBUILD.csv")); mu.index=mu.index.map(str)
pci=json.load(open(C._find("iamatlas_percell_identity_loci_v1_1.json")))["cells"]
c2c=json.load(open(C._find("IAMAtlasREBUILD_celltype_to_class.json"))); c2c=c2c.get("celltype_to_class",c2c)
BLOOD={c for c,k in c2c.items() if k in ("immune","progenitor","stem_adult")}
def Hb(m): m=min(max(float(m),1e-9),1-1e-9); return -(m*np.log2(m)+(1-m)*np.log2(1-m))
def A_of(beta, cell):
    e=pci[cell]; loci=[l for l in e["loci"] if l in beta.index]; v=beta.reindex(loci).dropna(); return Hb(v.mean())/float(e["H_min"]), len(v)
per=pd.read_csv(os.path.join(W,"handoff/heldout2d_per_array.csv")).set_index("gsm"); quiet=[g for g in per[per.ndet==0].index if g in pf.columns]
used=set(json.load(open(os.path.join(W,"handoff/stage2d03_results.json")))["hosts"]); hosts=[g for g in quiet[18:] if g not in used][:12]
CELLS=["Cortical_neurons","Glia","Colon_epithelial_cells","Kidney","Hepatocytes","Pancreatic_beta_cells"]; FR=[0.02,0.05,0.10,0.20,0.50]
rows=[]
for h in hosts:
    if os.path.exists(os.path.join(OUT,"STOP")): break
    b=pf[h].dropna(); mapped0,_=C.stage_1s_scale_map(b.to_dict(),"stage1_noob_450K"); m0=pd.Series(mapped0,dtype=float); m0.index=m0.index.map(str)
    o0=C.run_full(b.to_dict(),os.path.join(W,"atlas_work/IAMAtlasREBUILD.csv"),cfg={"age":60,"pipeline":"stage1_noob_450K","lab":"GSE87571"})
    blood0={c:v["A"] for c,v in o0["cells_all"].items() if v.get("present") and c in BLOOD and v.get("A") is not None}
    for cell in CELLS:
        col=mu[cell].dropna(); loci=b.index.intersection(col.index)
        for f in FR:
            sh=os.path.join(OUT,f"{h}_{cell}_{f}.json")
            if os.path.exists(sh): rows.append(json.load(open(sh))); continue
            s=b.copy(); s.loc[loci]=(1-f)*b.loc[loci]+f*col.loc[loci].values
            o=C.run_full(s.to_dict(),os.path.join(W,"atlas_work/IAMAtlasREBUILD.csv"),cfg={"age":60,"pipeline":"stage1_noob_450K","lab":"GSE87571"})
            mapped,_=C.stage_1s_scale_map(s.to_dict(),"stage1_noob_450K"); ms=pd.Series(mapped,dtype=float); ms.index=ms.index.map(str)
            A_raw,n=A_of(ms,cell)
            fd=o["foreign_detection"]; fh=(fd.get("cells",{}).get(cell) or {}).get("f_hat"); det=cell in (fd.get("detected") or [])
            # inverted: blood reconstruction at the identity loci from the solver's blood-cell fractions
            cf={c:v["fraction"] for c,v in o["cells_all"].items() if c in BLOOD and (v.get("fraction") or 0)>0 and c in mu.columns}
            tot=sum(cf.values()); inv=None
            if fh and fh>0.005 and tot>0:
                idl=[l for l in pci[cell]["loci"] if l in ms.index and l in mu.index]
                bb=sum(mu.loc[idl,c].fillna(mu.loc[idl].mean(axis=1))*w for c,w in cf.items())/tot
                bc=((ms.reindex(idl)-(1-fh)*bb)/fh).clip(0.001,0.999); inv=Hb(bc.mean())/float(pci[cell]["H_min"])
            bloodA={c:v["A"] for c,v in o["cells_all"].items() if v.get("present") and c in BLOOD and v.get("A") is not None}
            dblood=max((abs(bloodA[c]-blood0[c]) for c in bloodA if c in blood0),default=0.0)
            rec=dict(host=h,cell=cell,f=f,A_raw=A_raw,n_loci=n,f_hat=fh,detected=det,A_inv=inv,max_dA_blood=dblood,chain_A=(o["cells_all"].get(cell) or {}).get("A"),chain_present=(o["cells_all"].get(cell) or {}).get("present"))
            json.dump(rec,open(sh+".tmp","w"),default=float); os.replace(sh+".tmp",sh); rows.append(rec)
    print(f"host {h} done ({len(rows)} spikes)",flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(OUT,"spikes.csv"),index=False)
D["dA_raw"]=(D.A_raw-1).abs(); D["dA_inv"]=(D.A_inv-1).abs()
floors={}
print("\ncell | f | median |dA| raw (inside .05) | median |dA| inverted (inside .05) | detected | f_hat/f")
for cell in CELLS:
    fr=None; fi=None
    for f in FR:
        d=D[(D.cell==cell)&(D.f==f)]
        r_med=d.dA_raw.median(); r_in=(d.dA_raw<=0.05).mean(); i_med=d.dA_inv.median(); i_in=(d.dA_inv<=0.05).mean()
        if fr is None and r_med<=0.02 and r_in>=0.9: fr=f
        if fi is None and i_med<=0.02 and i_in>=0.9: fi=f
        print(f"  {cell:<24} {f:.2f}  raw {r_med:.4f} ({r_in:.2f})  inv {i_med:.4f} ({i_in:.2f})  det {d.detected.mean():.2f}  f_hat/f {(d.f_hat/d.f).median():.2f}")
    floors[cell]={"raw_floor":fr,"inverted_floor":fi}
b4=all(D[(D.cell==c)&(D.f==0.5)].dA_raw.median()<=0.02 and D[(D.cell==c)&(D.f==0.5)].dA_inv.median()<=0.02 for c in CELLS)
b5=D[D.f<=0.10].max_dA_blood.max()<=0.005
print("\nfloors:",floors); print("B4 both readings inside .02 at f=.50 ->","MET" if b4 else "FAILED"); print(f"B5 blood A unmoved (max {D[D.f<=0.10].max_dA_blood.max():.4f}) ->","MET" if b5 else "FAILED")
json.dump({"floors":floors,"B4":bool(b4),"B5":{"met":bool(b5),"max":float(D[D.f<=0.10].max_dA_blood.max())},"hosts":hosts},open(os.path.join(W,"handoff/foreignscore01_results.json"),"w"),indent=1)
print("RESULTS WRITTEN",flush=True)
