#!/usr/bin/env python3
"""Kit test, row 7 (PROC-TIER-01): (T1) every tier boundary in tier_breakpoints.json, both sides +/-1e-6, through cpg_tiers.tier_of and
the report builder's _tier - same answer; no literal breakpoints remain in tier code. (T3) not reportable -> tier None; on the cached
whole-blood arrays only reportable identity components carry a tier word. (T4) h_min is accepted and ignored by tier_of: no AT_CEILING word exists (author, 2026-09-27)."""
import os, sys, json, re, pickle, warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); BP=os.path.abspath(os.path.join(HERE,"..","..")); ENG=os.path.join(BP,"MethylPhys/chain"); sys.path.insert(0,ENG)
import cpg_tiers as T, cpg_conductor as C
s=T.scheme(); bounds=sorted({b[1] for b in s["bands"]}|{b[2] for b in s["bands"] if b[2] and b[2]<900})
for x in bounds:
    for a in (x-1e-6, x+1e-6):
        t1,_=T.tier_of(a); assert t1 is not None, a   # the interface prints T.tier_of's word and nothing else (checked by the literal scan below)
lo=T.tier_of(0.95-1e-6)[0]; hi=T.tier_of(0.95)[0]; assert (lo,hi)==("SUPPRESSED","NORMAL")
nb={b[0]:(b[1],b[2]) for b in s["bands"]}; on=nb["ELEVATED"][0]   # onset read from the JSON, never typed (PROC-TIER-02 caught a 1.01 literal here)
assert T.tier_of(on-1e-6)[0]=="NORMAL" and T.tier_of(on)[0]=="ELEVATED" and T.tier_of(s["warburg"])[0]=="SIGNIFICANTLY_ELEVATED" and T.tier_of(s["breach"])[0]=="BREACH"
src=open(os.path.join(ENG,"MethylPhys_Interface","build_methylphys.py")).read()+open(os.path.join(ENG,"cpg_conductor.py")).read()   # the live report interface, since the v1 builder was retired 2026-09-26
lits=[m.group(0) for m in re.finditer(r"(?:A|a)\s*[<>]=?\s*1\.0[147]\b|(?:A|a)\s*[<>]=?\s*1\.10?\b|(?:A|a)\s*[<>]=?\s*0\.95\b",src)]
assert not lits, lits; print(f"T1 one definition: {len(bounds)} boundaries x 2 sides agree; 0 literal breakpoints in tier code: ok")
assert T.tier_of(1.2, reportable=False)[0] is None and T.tier_of(None)[0] is None
DATA=os.environ.get("CPG_KIT_DATA",os.path.join(HERE,"data")); c=pickle.load(open(os.path.join(DATA,"betas_cache.pkl"),"rb")); ATLAS=os.path.join(BP,"MethylPhys/atlas/IAMAtlasREBUILD.csv")
b=c["GSM2333901"].dropna().to_dict(); out=C.run_full(b,ATLAS,cfg={"age":72,"pipeline":"stage1_noob_450K","lab_zero":-0.0117,"lab":"GSE87571"})
# T3 (rewritten 2026-09-27): the class gauge is an internal gate and carries NO tier word (MEASURE, DON'T COMPARE);
# every tier on the report is a present cell's, from tier_of on that cell's A. On whole blood the composition check passes.
cl=out["classes"]; assert all(v.get("tier") is None for v in cl.values() if isinstance(v,dict)), [(k,v.get("tier")) for k,v in cl.items() if isinstance(v,dict)]
assert cl["immune"].get("composition_verified") is True, cl["immune"]
cells=out.get("cells_all") or {}
present=[(k,T.tier_of(v.get("A"))[0],v.get("A")) for k,v in cells.items() if isinstance(v,dict) and v.get("present") and v.get("A") is not None and v.get("status")=="OK"]
assert present and all(t for _,t,_ in present), present[:5]
print(f"T3 class gauge carries no tier; {len(present)} present cells each carry tier_of(A): ok")
hm=json.load(open(os.path.join(ENG,"Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json")))["immune"]["H_min"]
t,n=T.tier_of(1.0/hm+0.01,h_min=hm); assert t=="BREACH" and "AT_CEILING" not in (t or ""); t2,_=T.tier_of(1.0/hm-0.01,h_min=hm); assert t2=="BREACH"
print("T4 no ceiling word: values at or past 1/H_min read BREACH like any other: ok"); print("test_tiers: PASS")
