#!/usr/bin/env python3
"""PROC-CMB-04 (supersedes CMB-01, which failed C2/C4 as sealed) — CHAIN_COMMISSIONING row 4.6: the patient's sky, commissioned against held-out healthy arrays (PREREG sealed 2026-09-21).
Inputs (CPG_KIT_DATA): stage2_fractions.json (selection + Stage 2 class fractions, 80 arrays x 4 labs, seed 2028) and the four
laboratory beta matrices (Stage 1 noob): betas_GSE87571.pkl, betas_GSE42861_controls.pkl, betas_GSE111629_controls.pkl, betas_GSE125105_controls.pkl.
Outputs: results/proc_cmb_01.json, per-lab residual scales (Runtime Matrices/Patient_CMB/residual_scale_<lab>.npz), plates for C6.
"""
import os, sys, json, pickle, hashlib, math, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..","..")); ENG=os.path.join(BP,"MethylPhys/chain")
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data")); OUT=os.path.join(HERE,"results"); os.makedirs(OUT,exist_ok=True)
sys.path.insert(0,ENG); import stage_4_6_patient_cmb as S
RT=os.path.join(ENG,"Runtime Matrices"); ATLAS=os.path.join(BP,"MethylPhys/atlas/IAMAtlasREBUILD.csv")
LABS={"GSE87571":"betas_GSE87571.pkl","GSE42861":"betas_GSE42861_controls.pkl","GSE111629":"betas_GSE111629_controls.pkl","GSE125105":"betas_GSE125105_controls.pkl"}
fr=json.load(open(os.path.join(DATA,"stage2_fractions.json"))); sel=fr["selection"]; F={g:v["class_fractions"] for g,v in fr["fractions"].items()}
mp=json.load(open(os.path.join(RT,"A_Scoring_Module/beta_scale_maps_v1.json")))["maps"]["stage1_noob_450K"]; slope,icpt=mp["slope"],mp["intercept"]
ident=json.load(open(os.path.join(RT,"A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json"))); LOCI={c:v["loci"] for c,v in ident.items() if isinstance(v,dict) and "loci" in v}; HMIN={c:v["H_min"] for c,v in ident.items() if isinstance(v,dict) and "H_min" in v}
print("loading atlas ...",flush=True); AT=pd.read_csv(ATLAS,index_col="cpg_id"); MEANS=AT[[f"{c}_mean" for c in S.CLASSES]].copy(); MEANS.columns=S.CLASSES
mapping=S.load_mapping(); res={"maps":{"stage1_noob_450K":mp}}
PF=S.presence_floors({g:F[g] for lab in LABS for g in sel[lab]["panel"]}); json.dump({"_meta":{"proc":"PROC-CMB-04","rule":"non-blood classes: max(0.02, p99 of Stage 2 fraction over the 160 healthy PANEL arrays); blood lineage (immune, progenitor, stem_adult): 0.02","seed":2028},"floors":PF},open(os.path.join(RT,"Patient_CMB","presence_floors_v1.json"),"w"),indent=1); res["presence_floors"]=PF; print("presence floors:",{k:round(v,3) for k,v in PF.items()})
def mapped(B,g): b=B[g].dropna(); return ((b-icpt)/slope).clip(1e-6,1-1e-6)
# C1 retired formula on the 11 cached test arrays (pure-class immune mean / posterior SD)
cache=os.path.join(DATA,"betas_cache.pkl")
if os.path.exists(cache):
    c=pickle.load(open(cache,"rb")); sd=AT["immune_sd"]; mu=AT["immune_mean"]; fr1=[]
    for g,b in c.items():
        b=pd.Series(b) if not isinstance(b,pd.Series) else b; bm=((b-icpt)/slope); j=pd.concat([bm.rename("b"),mu,sd],axis=1,join="inner").dropna(); j=j[j.immune_sd>=1e-4]
        fr1.append(float(((j.b-j.immune_mean).abs()/j.immune_sd>2).mean()))
    res["C1"]={"n_arrays":len(fr1),"frac_abs_z_gt2_retired_formula":fr1,"median":float(np.median(fr1))}; print(f"C1 retired formula: median frac |z|>2 = {np.median(fr1):.3f} over {len(fr1)} healthy arrays (healthy expectation 0.05)")
else: res["C1"]={"note":"betas_cache.pkl absent"}
# C5 mapping determinism
import subprocess; m1=os.path.join(RT,"Patient_CMB/iamatlas_cpg_to_healpix_nside128.npz"); tmp=os.path.join(OUT,"map_rebuild.npz")
r=subprocess.run([sys.executable,os.path.join(RT,"Patient_CMB/build_healpix_mapping.py"),ATLAS,os.path.join(BP,"MethylPhys/atlas/external_manifests/EPIC_plus_HM450_combined_manifest_normalized.csv"),tmp],capture_output=True,text=True)
h1=hashlib.sha256(open(m1,"rb").read()).hexdigest(); h2=hashlib.sha256(open(tmp,"rb").read()).hexdigest(); nm=int(np.load(tmp)["n_missing"]); os.remove(tmp)
res["C5"]={"sha_runtime":h1[:16],"sha_rebuild":h2[:16],"n_missing":nm,"pass":h1==h2}; print(f"C5 mapping: rebuild sha {'==' if h1==h2 else '!='} runtime sha; unmapped atlas CpGs {nm} -> {'PASS' if h1==h2 else 'FAIL'}")
# C2/C3/C4
scales={}; held={}; REND={}; QUIET={}; res["C2"]={}; res["C4"]={"arrays":0,"conforming":0,"violations":[]}
for lab,fn in LABS.items():
    print(f"{lab}: loading ...",flush=True); B=pickle.load(open(os.path.join(DATA,fn),"rb")); panel=sel[lab]["panel"]; test=sel[lab]["test"]
    P=pd.DataFrame({g:mapped(B,g) for g in panel}); sc=S.build_residual_scale(P,{g:F[g] for g in panel},MEANS)
    S.save_scale(sc,os.path.join(RT,"Patient_CMB",f"residual_scale_{lab}.npz"),lab,{"panel":panel,"seed":2028,"pipeline":"stage1_noob_450K","proc":"PROC-CMB-04","zero":"m = panel mean residual, subtracted at patient time"}); scales[lab]=sc
    held[lab]={g:mapped(B,g) for g in test}; del B,P
    fz=[];mz=[];skies={}
    for g,b in held[lab].items():
        sky=S.patient_sky(b,F[g],MEANS,sc,LOCI,mapping,presence_floors_by_class=PF); fz.append(sky["all"]["frac_abs_z_gt2"]); mz.append(sky["all"]["median_z"])
        res["C4"]["arrays"]+=1; f=F[g]; rule_ok=all((sky["classes"][c]["assessable"]==(f.get(c,0)>=PF[c])) for c in S.CLASSES)
        nonblood=any(sky["classes"][c]["assessable"] for c in ("stromal","cycling","secretory","terminal","stem_pluri")); imm=sky["classes"]["immune"]["assessable"]
        for c in S.CLASSES:
            if sky["classes"][c]["assessable"]: REND[c]=REND.get(c,0)+1; QUIET.setdefault(c,[]).append(sky["classes"][c]["frac_abs_z_gt2"])
        res["C4"]["conforming"]+=int(rule_ok); res["C4"].setdefault("nonblood_rendered",0); res["C4"]["nonblood_rendered"]+=int(nonblood); res["C4"].setdefault("immune_rendered",0); res["C4"]["immune_rendered"]+=int(imm)
        if not rule_ok or nonblood: res["C4"]["violations"].append({"gsm":g,"rule_ok":rule_ok,"nonblood":nonblood,"fractions":{c:round(f.get(c,0),3) for c in S.CLASSES if f.get(c,0)>=0.01}})
        skies[g]=sky
    med_f,med_z=float(np.median(fz)),float(np.median(mz)); ok=0.03<=med_f<=0.08 and abs(med_z)<=0.15
    res["C2"][lab]={"n_test":len(fz),"median_frac_abs_z_gt2":med_f,"median_z":med_z,"frac_range":[float(min(fz)),float(max(fz))],"pass":ok,"immune_panel_median_frac":float(np.median([s["classes"]["immune"].get("frac_abs_z_gt2",np.nan) for s in skies.values()]))}
    print(f"C2 {lab}: held-out n={len(fz)} median frac|z|>2 {med_f:.3f} [{min(fz):.3f}-{max(fz):.3f}] median z {med_z:+.3f} | immune-loci frac {res['C2'][lab]['immune_panel_median_frac']:.3f} -> {'PASS' if ok else 'FAIL'}")
    if lab=="GSE87571":   # C6 on the first held-out array
        g=test[0]; a1=S.render_plate(skies[g],os.path.join(OUT,f"plate_{g}_run1.png"),f"{g} · {lab} · healthy held-out"); a2=S.render_plate(S.patient_sky(held[lab][g],F[g],MEANS,sc,LOCI,mapping,presence_floors_by_class=PF),os.path.join(OUT,f"plate_{g}_run2.png"),f"{g} · {lab} · healthy held-out")
        res["C6"]={"gsm":g,"identical_pixel_arrays":bool(np.array_equal(a1,a2)),"pass":bool(np.array_equal(a1,a2))}; print(f"C6 regeneration {g}: identical pixel arrays {res['C6']['pass']}")
res["C2"]["labs_passing"]=sum(1 for l in LABS if res["C2"][l]["pass"]); res["C2"]["pass"]=res["C2"]["labs_passing"]>=3
res["C4"]["pass"]=res["C4"]["conforming"]==res["C4"]["arrays"]; res["C4"]["rendered_per_class"]=REND; res["C4"]["rendered_panel_quietness"]={c:(float(np.median(v)) if v else None) for c,v in QUIET.items()}
print(f"C4'' gating on TEST arrays: rule {res['C4']['conforming']}/{res['C4']['arrays']} -> {'PASS' if res['C4']['pass'] else 'FAIL'} | rendered per class: {REND} | median frac|z|>2 on rendered panels: { {c:round(q,3) for c,q in res['C4']['rendered_panel_quietness'].items() if q is not None} }")
# C3 cross-lab: each lab's scale applied to every other lab's held-out
res["C3"]={}
for la in LABS:
    for lb in LABS:
        if la==lb: continue
        fz=[S.patient_sky(b,F[g],MEANS,scales[la],None,mapping,presence_floors_by_class=PF)["all"]["frac_abs_z_gt2"] for g,b in list(held[lb].items())[:20]]
        mz=[S.patient_sky(b,F[g],MEANS,scales[la],None,mapping,presence_floors_by_class=PF)["all"]["median_z"] for g,b in list(held[lb].items())[:20]]
        res["C3"][f"{la}->{lb}"]={"median_frac_abs_z_gt2":float(np.median(fz)),"median_z":float(np.median(mz))}
print("C3 cross-lab (scale of A on held-out of B): "+"; ".join(f"{k} {v['median_frac_abs_z_gt2']:.3f}/{v['median_z']:+.2f}" for k,v in res["C3"].items()))
res["row_4_6_commissioned"]=bool(res["C2"]["pass"] and res["C4"]["pass"] and res["C5"]["pass"] and res.get("C6",{}).get("pass"))
json.dump(res,open(os.path.join(OUT,"proc_cmb_04.json"),"w"),indent=1,default=str); print("row 4.6 commissioned:",res["row_4_6_commissioned"])
