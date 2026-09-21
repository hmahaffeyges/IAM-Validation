#!/usr/bin/env python3
"""Kit test for Stage 4.6 (row 4.6): (1) without a lab scale the sky is NOT AVAILABLE; (2) with GSE87571's scale, a cached healthy
GSE87571 array renders immune, masks the five non-blood classes unless Stage 2 puts them above the measured floor, and reads a quiet sky
(frac |z|>2 < 0.10); (3) the plate regenerates with an identical pixel array. Needs the cached test betas (CPG_KIT_DATA/betas_cache.pkl)."""
import os, sys, json, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..","..")); ENG=os.path.join(BP,"CPG_Engine")
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data")); sys.path.insert(0,ENG); import cpg_conductor as C, stage_4_6_patient_cmb as S
ATLAS=os.path.join(BP,"IAM_Atlas/IAMAtlasREBUILD.csv"); c=pickle.load(open(os.path.join(DATA,"betas_cache.pkl"),"rb")); g="GSM2333901"; b=c[g].dropna()
a=C.stage_a_cells(b.to_dict(),ATLAS); beta_rm,_=C.stage_1s_scale_map(b.to_dict(),"stage1_noob_450K")
o0=C.stage_4_6_patient_sky(beta_rm,a,cfg={"lab":None},atlas_csv=ATLAS); assert o0["available"] is False and "NOT AVAILABLE" in o0["status"], o0["status"]; print("1 no scale -> NOT AVAILABLE: ok")
o=C.stage_4_6_patient_sky(beta_rm,a,cfg={"lab":"GSE87571"},atlas_csv=ATLAS); assert o["available"]
fl=o["presence_floors"]; f=a["class_fractions"]
assert o["classes"]["immune"]["assessable"]; assert all(o["classes"][k]["assessable"]==(f.get(k,0)>=fl[k]) for k in S.CLASSES); assert o["all"]["frac_abs_z_gt2"]<0.10, o["all"]
print(f"2 {g}: immune rendered; gate follows floors; frac|z|>2 {o['all']['frac_abs_z_gt2']:.3f}, median z {o['all']['median_z']:+.3f}; rendered: {[k for k in S.CLASSES if o['classes'][k]['assessable']]}")
p1=S.render_plate(o["_sky"],os.path.join(HERE,"results","test_sky_run1.png"),f"{g} kit test"); p2=S.render_plate(o["_sky"],os.path.join(HERE,"results","test_sky_run2.png"),f"{g} kit test")
assert np.array_equal(p1,p2); print("3 plate regenerates identically: ok"); print("test_patient_sky: PASS")
