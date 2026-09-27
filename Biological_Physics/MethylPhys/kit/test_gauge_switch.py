# 2026-09-25: S1 expected a tier on every healthy blood array. PROC-FOREIGN-01 commissioned the
# composition guard, which withholds the tier word when more than 2.07 % of a specimen is assigned
# outside the blood lineage - a rate the pre-registration fixed at no more than 5 % of healthy
# arrays. GSM2333950 is one such array (foreign 0.0282, immune fraction 0.9663): its tier is now
# withheld BY DESIGN, not by defect. This test therefore accepts a withheld tier when
# composition_verified is False, and still requires one whenever it is True.
#!/usr/bin/env python3
"""PROC-SWITCH-01 conformance: the conductor runs the identity-loci class gauge as the internal blood-like gate.
Runs on the kit's cached whole-blood betas; no download. Exit non-zero on any failure.
  S1  gauge_surface == identity_loci, scale MAPPED, internal_gate, A present, no tier word, no population key
  S2  the composition check runs on every array
  S3  class A within 0.10 of the fixed point on whole blood
"""
import os, sys, json, pickle
HERE=os.path.dirname(os.path.abspath(__file__)); ENG=os.path.abspath(os.path.join(HERE,"..","..","MethylPhys/chain"))
sys.path.insert(0,ENG); import cpg_conductor as C
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data"))
ATLAS=os.path.abspath(os.path.join(HERE,"..","..","MethylPhys/atlas","IAMAtlasREBUILD.csv"))
cache=pickle.load(open(os.path.join(DATA,"betas_cache.pkl"),"rb"))
# 2026-09-27 (MEASURE, DON'T COMPARE): the class gauge is an internal gate. No lab zero, no age term, no band.
#   S1  gauge_surface == identity_loci, scale MAPPED, internal_gate True, A present, tier None (the class carries no tier word)
#   S2  composition check runs on every whole-blood array: composition_verified is a bool and foreign_fraction is a number
#   S3  the reading is on the fixed point's scale: every whole-blood array's class A is within 0.10 of 1.00
WB={"GSM2333901":58,"GSM2333905":67,"GSM2333950":43,"GSM1051533":60,"GSM1051534":60,"GSM1051525":60,"GSM1051526":60}
fails=[]; n=0
for k,age in WB.items():
    if k not in cache: continue
    b=cache[k]; b=b.to_dict() if hasattr(b,"to_dict") else dict(b)
    im=C.run_full(b,ATLAS,cfg={"age":age,"pipeline":"stage1_noob_450K"})["classes"]["immune"]; n+=1
    if not (im.get("gauge_surface")=="identity_loci" and str(im.get("scale","")).startswith("MAPPED") and im.get("internal_gate") is True and im.get("A") is not None and im.get("tier") is None): fails.append(("S1",k,im))
    if not (isinstance(im.get("composition_verified"),bool) and isinstance(im.get("foreign_fraction"),(int,float))): fails.append(("S2",k,im))
    if abs(im["A"]-1.0)>0.10: fails.append(("S3",k,im["A"]))
    for banned in ("A_abs","placement","lab_zero","age_reference_c","band_status"):
        if banned in im: fails.append(("S1-banned-key",k,banned))
    print(f"{k} age {age}  class A {im['A']:.4f}  foreign {im.get('foreign_fraction')}  verified {im.get('composition_verified')}")
print(f"{n} arrays; gauge = internal gate, no population layer on the record")
print("GAUGE SWITCH CONFORMANCE:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
