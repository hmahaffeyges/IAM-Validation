#!/usr/bin/env python3
"""PROC-SMALL-01: the false-positive rate on healthy donors that did NOT set the threshold.

5.3 % on the 38 threshold donors is definitional - the threshold is their 95th percentile. This runs the
frozen panel on 60 Uppsala donors never used for anything, which is the only number worth quoting.
"""
import glob, json, os, re, sys, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"stage1"); CH="iamrepo/Biological_Physics/MethylPhys/chain"; sys.path.insert(0,CH)
import numpy as np
from stage_1_idat_calibration import calibrate_idat_to_beta
import stage_2c_trace_detection as T

panel=T.load_panel()
used=set(json.load(open("handoff/floor_scan.json")).keys()) | {"GSM1051533","GSM2333901","GSM2333905"}
pat=re.compile(r"(GSM\d+)_(\d{9,12})_(R0\dC0\d)_(Grn|Red)\.idat\.gz$")
pairs={}
for p in glob.glob("idats_gse87571/*.idat.gz"):
    m=pat.search(os.path.basename(p))
    if m: pairs.setdefault(m.group(1),{})[m.group(4)]=p
cand=sorted(g for g,v in pairs.items() if len(v)==2 and g not in used)
sel=cand[:60]
print(f"{len(cand)} unused Uppsala donors available; testing {len(sel)}", flush=True)
out={}
if os.path.exists("handoff/small01_heldout.json"): out=json.load(open("handoff/small01_heldout.json"))
for n,g in enumerate(sel,1):
    if g in out: continue
    t0=time.time()
    try:
        beta,_=calibrate_idat_to_beta(pairs[g]["Grn"], pairs[g]["Red"])
        s=(beta.iloc[:,0] if hasattr(beta,"columns") else beta).dropna()
        r=T.detect(s.to_dict(), panel)
    except Exception as e:
        out[g]={"error":f"{type(e).__name__}: {e}"}; print(f"  {g}: {out[g]['error'][:80]}", flush=True); continue
    out[g]={c:{"t":r[c]["t"],"detected":r[c]["detected"]} for c in ("secretory","cycling")}
    json.dump(out, open("handoff/small01_heldout.json","w"), indent=1)
    if n%10==0: print(f"  [{n}/{len(sel)}] {g} ({time.time()-t0:.0f}s)", flush=True)
ok={g:v for g,v in out.items() if "error" not in v}
print(f"\nheld-out healthy donors: {len(ok)}", flush=True)
for c in ("secretory","cycling"):
    ts=[ok[g][c]["t"] for g in ok if ok[g][c]["t"] is not None]
    det=sum(1 for g in ok if ok[g][c]["detected"])
    print(f"  {c:<10} detected {det}/{len(ok)} = {100*det/max(len(ok),1):.1f}% "
          f"| t median {np.median(ts):.2f} max {max(ts):.2f} (threshold {panel['thresholds'][c]['t95']:.2f})",
          flush=True)
