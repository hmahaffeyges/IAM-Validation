#!/usr/bin/env python3
"""Kit test, row 7 (PROC-TIER-01): (T1) every tier boundary in tier_breakpoints.json, both sides +/-1e-6, through cpg_tiers.tier_of and
the report builder's _tier - same answer; no literal breakpoints remain in tier code. (T3) not reportable -> tier None; on the cached
whole-blood arrays only reportable identity components carry a tier word. (T4) A at/above 1/H_min -> AT_CEILING."""
import os, sys, json, re, pickle, warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..","..")); ENG=os.path.join(BP,"CPG_Engine"); sys.path.insert(0,ENG)
import cpg_tiers as T, cpg_conductor as C, cpg_report_builder as R
s=T.scheme(); bounds=sorted({b[1] for b in s["bands"]}|{b[2] for b in s["bands"] if b[2] and b[2]<900})
for x in bounds:
    for a in (x-1e-6, x+1e-6):
        t1,_=T.tier_of(a); t2=R._tier(a); assert t1==t2, (a,t1,t2)
lo=T.tier_of(0.95-1e-6)[0]; hi=T.tier_of(0.95)[0]; assert (lo,hi)==("SUPPRESSED","NORMAL")
nb={b[0]:(b[1],b[2]) for b in s["bands"]}; on=nb["ELEVATED"][0]   # onset read from the JSON, never typed (PROC-TIER-02 caught a 1.01 literal here)
assert T.tier_of(on-1e-6)[0]=="NORMAL" and T.tier_of(on)[0]=="ELEVATED" and T.tier_of(s["warburg"])[0]=="SIGNIFICANTLY_ELEVATED" and T.tier_of(s["breach"])[0]=="BREACH"
src=open(os.path.join(ENG,"cpg_report_builder.py")).read()+open(os.path.join(ENG,"cpg_conductor.py")).read()
lits=[m.group(0) for m in re.finditer(r"(?:A|a)\s*[<>]=?\s*1\.0[147]\b|(?:A|a)\s*[<>]=?\s*1\.10?\b|(?:A|a)\s*[<>]=?\s*0\.95\b",src)]
assert not lits, lits; print(f"T1 one definition: {len(bounds)} boundaries x 2 sides agree; 0 literal breakpoints in tier code: ok")
assert T.tier_of(1.2, reportable=False)[0] is None and T.tier_of(None)[0] is None
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data")); c=pickle.load(open(os.path.join(DATA,"betas_cache.pkl"),"rb")); ATLAS=os.path.join(BP,"IAM_Atlas/IAMAtlasREBUILD.csv")
b=c["GSM2333901"].dropna().to_dict(); out=C.run_full(b,ATLAS,cfg={"age":72,"pipeline":"stage1_noob_450K","lab_zero":-0.0117,"lab":"GSE87571"})
cl=out["classes"]; bad=[(k,v.get("tier")) for k,v in cl.items() if isinstance(v,dict) and not v.get("reportable") and v.get("tier") is not None]; assert not bad, bad
rep=[(k,v["tier"],v["A_abs"]) for k,v in cl.items() if isinstance(v,dict) and v.get("reportable")]; assert rep and all(t for _,t,_ in rep), rep
print(f"T3 s108: reportable -> tier {rep}; {sum(1 for k,v in cl.items() if isinstance(v,dict) and not v.get('reportable'))} non-reportable components carry tier None: ok")
out2=C.run_full(b,ATLAS,cfg={"age":72,"pipeline":"stage1_noob_450K","lab_zero":None,"lab":"GSE87571"}); assert all(v.get("tier") is None for v in out2["classes"].values() if isinstance(v,dict)); print("T3 lab_zero UNSET -> every tier None: ok")
hm=json.load(open(os.path.join(ENG,"Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json")))["immune"]["H_min"]
t,n=T.tier_of(1.0/hm+0.01,h_min=hm); assert t=="AT_CEILING" and f"{1.0/hm:.4f}" in n; t2,_=T.tier_of(1.0/hm-0.01,h_min=hm); assert t2=="BREACH"
print(f"T4 ceiling 1/H_min(immune)={1.0/hm:.4f}: above -> AT_CEILING, just below -> BREACH: ok"); print("test_tiers: PASS")
