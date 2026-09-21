#!/usr/bin/env python3
"""PROC-SWITCH-01 conformance: the conductor REPORTS the identity-loci gauge with the three-layer reference.
Runs on the kit's cached whole-blood betas; no download. Exit non-zero on any failure.
  S1  gauge_surface == identity_loci, scale MAPPED, reportable with a lab zero, A_abs present
  S2  without a lab zero: lab_zero == UNSET, reportable False, A_abs None
  S3  healthy arrays: >= 4/5 IN_BAND (identity_band_v3 p10-p90 admits 80%)
"""
import os, sys, json, pickle
HERE=os.path.dirname(os.path.abspath(__file__)); ENG=os.path.abspath(os.path.join(HERE,"..","..","CPG_Engine"))
sys.path.insert(0,ENG); import cpg_conductor as C
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data"))
ATLAS=os.path.abspath(os.path.join(HERE,"..","..","IAM_Atlas","IAMAtlasREBUILD.csv"))
cache=pickle.load(open(os.path.join(DATA,"betas_cache.pkl"),"rb"))
# lab zeros as the four-cohort full-cohort residual medians (identity_band_v3.json _meta)
Z=json.load(open(os.path.join(ENG,"Runtime Matrices","A_Scoring_Module","identity_band_v3.json")))["_meta"]["cohorts"]
zU=Z["GSE87571_Uppsala"]["z_lab_full_cohort"]; zK=Z["GSE42861_Karolinska"]["z_lab_full_cohort"]
WB={"GSM2333901":(58,"healthy",zU),"GSM2333905":(67,"healthy",zU),"GSM2333950":(43,"healthy",zU),
    "GSM1051533":(60,"healthy",zK),"GSM1051534":(60,"healthy",zK),"GSM1051525":(60,"RA",zK),"GSM1051526":(60,"RA",zK)}
fails=[]; inband=0; nh=0
for k,(age,arm,lab) in WB.items():
    if k not in cache: continue
    b=cache[k]; b=b.to_dict() if hasattr(b,"to_dict") else dict(b)
    im=C.run_full(b,ATLAS,cfg={"age":age,"pipeline":"stage1_noob_450K","lab_zero":lab})["classes"]["immune"]
    im0=C.run_full(b,ATLAS,cfg={"age":age,"pipeline":"stage1_noob_450K"})["classes"]["immune"]
    if not (im["gauge_surface"]=="identity_loci" and im["scale"].startswith("MAPPED") and im["reportable"] and im["A_abs"] is not None): fails.append(("S1",k,im))
    if not (im0["lab_zero"]=="UNSET" and im0["reportable"] is False and im0["A_abs"] is None): fails.append(("S2",k,im0))
    if arm=="healthy": nh+=1; inband+= im["placement"]=="IN_BAND"
    print(f"{k} {arm:<8} A_mapped {im['A_mapped']:.4f} A_abs {im['A_abs']:.4f} {im['placement']:<10} | UNSET -> reportable {im0['reportable']}")
if nh and inband < 4: fails.append(("S3",f"{inband}/{nh} healthy in band"))
print(f"S3 healthy in band {inband}/{nh}")
print("GAUGE SWITCH CONFORMANCE:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
