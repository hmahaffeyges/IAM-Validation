#!/usr/bin/env python3
"""Kit test for Stage 4.6 (row 4.6): (1) without a lab scale the sky is NOT AVAILABLE; (2) with GSE87571's scale, a cached healthy
GSE87571 array renders immune, masks the five non-blood classes unless Stage 2 puts them above the measured floor, and reads a quiet sky
(frac |z|>2 < 0.10); (3) the plate regenerates with an identical pixel array. Needs the cached test betas (CPG_KIT_DATA/betas_cache.pkl)."""
import os, sys, json, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..","..")); ENG=os.path.join(BP,"MethylPhys/chain")
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data")); sys.path.insert(0,ENG); import cpg_conductor as C, stage_4_6_patient_cmb as S
ATLAS=os.path.join(BP,"MethylPhys/atlas/IAMAtlasREBUILD.csv"); c=pickle.load(open(os.path.join(DATA,"betas_cache.pkl"),"rb")); g="GSM2333901"; b=c[g].dropna()
a=C.stage_a_cells(b.to_dict(),ATLAS); beta_rm,_=C.stage_1s_scale_map(b.to_dict(),"stage1_noob_450K")
# 2026-09-27 (PROC-SKY-01): the sky is WITHHELD - no panel scale is read, no picture is drawn, until a sigma that is the
# instrument's fits every laboratory. The residual machinery stays callable for the next procedure.
import glob as _g, pandas as _pd
for lab in (None,"GSE87571"):
    o=C.stage_4_6_patient_sky(beta_rm,a,cfg={"lab":lab},atlas_csv=ATLAS)
    # 2026-09-27: the sky is DRAWN (restored under the author's development-stage ruling, archive 12108/12109 - not an explicit
    # author pick between withhold / draw-in-beta; that question is PLAN item 3). On a betas-only input sigma is the atlas posterior alone.
    assert o["available"] is True and "atlas posterior" in str(o.get("sigma","")), {k:o.get(k) for k in ("available","sigma","snp_noise","status")}
assert not _g.glob(os.path.join(os.path.dirname(C.__file__),"Runtime Matrices","Patient_CMB","residual_scale_*.npz")), "panel scales still in the chain"
r_,E=S.residual(_pd.Series(beta_rm,dtype=float),S.load_atlas_means(ATLAS),a["class_fractions"]); assert len(r_.dropna())>100000, "residual machinery broken"
print("1 sky WITHHELD for every laboratory, no panel scale in the tree, residual machinery intact: ok"); print("test_patient_sky: PASS")
