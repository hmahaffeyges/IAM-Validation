#!/usr/bin/env python3
"""CLASS GUARD (2026-09-28). The rule: a cell's class is the name of the floor it is divided by. Nothing else reads it.

Fails (exit 1) on:
  CODE  - any live chain module other than the ALLOWED H_min readers that reads the cell->class map, groups/averages by class,
          or names a class A / class score. Existing uses scheduled for the v2 switch-over are listed in PENDING_V2 with the
          inventory line that schedules them; the guard reports them as PENDING, and fails on any NEW one.
  TEXT  - "class A", "class A-score", "class score", "class gauge", "class reading", "per-class A" in a live document or report tab.
          Record documents are exempt when they carry the banner line RECORD_BANNER in their first 40 lines.
Writes kit/results/class_guard.json."""
import os, re, sys, json, glob
HERE=os.path.dirname(os.path.abspath(__file__)); MP=os.path.dirname(HERE)
RECORD_BANNER="In the commissioned instrument a class is only the floor a cell is divided by"
ALLOWED={"chain/Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py","hmin_calibration/mphys_mcmc_g002.py","kit/class_guard.py"}
PENDING_V2={  # CLASS_USE_INVENTORY.md, "Removed in the live chain (at the v2 switch-over)"
 "chain/cpg_conductor.py":"Stage B class gauge; groups table",
 "chain/legacy_iam_deconvolver/legacy_iam_deconvolver.py":"twin merge 'same class required'; per-class presence gate",
 "chain/disease_matching.py":"per-class identity CpGs",
 "chain/MethylPhys_Interface/build_methylphys.py":"reads the class map for the internal gate and the H_min label",
 "chain/nilc_celltype_deconvolver.py":"NILC cross-check reads the class map from its metadata",
 "kit/HELDOUT_2D.py":"hands the class map to the deconvolver (follows the deconvolver's twin merge)",
 "kit/cpg_kit.py":"marker-panel helper returns the class map with the class-keyed panels",
 "kit/build_percell_identity.py":"per-cell identity loci target the class H_min_beta; replaced by the v2 identity-loci build",
}
LABEL_ONLY={  # read the map only to print the floor's name beside the cell - the allowed use; route through the H_min lookup at v2
 "kit/build_cell_descriptions.py":"prints each cell's floor (class name + H_min)",
}
# a LOAD or USE of the class map, not a filename in a list: open/_find/load of it, attribute or key access, class_A(), groupby class
CODE=re.compile(r"(open|_find|find|load)\([^)\n]*celltype_to_class|\[[\"']celltype_to_class[\"']\]|\.celltype_to_class\b|get\([\"']celltype_to_class|celltype_class_map\s*=|\bclass_A\(|groupby\([^)\n]*class|\bclass_gauge\b",re.I)
TEXT=re.compile(r"\bclass(?:-| )(?:A|A-score|A-scores|score|scores|gauge|reading|readings)\b|\bper-class A\b",re.I)
def live(f): return not re.search(r"RETIRED|/results/|GENERATED|\.git/|/Record/",f)
hits=[]
for f in sorted(glob.glob(os.path.join(MP,"**/*.py"),recursive=True)):
    r=os.path.relpath(f,MP)
    if not live(r) or r in ALLOWED or not r.startswith(("chain/","kit/")) or os.path.basename(r).startswith(("test_","PROC_")): continue
    src=open(f,encoding="utf-8",errors="ignore").read()
    if RECORD_BANNER in "\n".join(src.split("\n")[:40]): continue
    n=len(CODE.findall(src))
    if n:
        st="PENDING_V2" if r in PENDING_V2 else ("LABEL" if r in LABEL_ONLY else "FAIL")
        hits.append({"file":r,"kind":"code","n":n,"status":st,"why":PENDING_V2.get(r) or LABEL_ONLY.get(r) or "new class use outside the H_min lookup"})
DOCS=["sop/MethylPhys_CPG_SOP.md","README.md","doors/RUNBOOK.md","doors/START_HERE.md","doors/HANDOFF.md"]+glob.glob(os.path.join(MP,"papers/*.tex"))
# SOP sections that describe the class gauge the chain still runs until the v2 switch-over (CLASS_USE_INVENTORY.md)
SOP_PENDING_V2=("§43.","Part II — The step-by-step","Part II-A","What this version is","Tiers, stated once")
ALLOW_LINE=re.compile(r"^\*Record|no class A|No class has an A|no class gauge|No class gauge|nothing pools cells by class|until 2026-09-28|class_guard",re.I)
def sop_status(lines):
    st="LIVE"; sec=""; out=[]
    for i,l in enumerate(lines):
        if re.match(r"^#{1,3} ",l):
            sec=l; look=lines[i+1:i+7]; m=[re.match(r"^> \*\*STATUS: ([A-Z ]+)\*\*",x) for x in look]; m=[x for x in m if x]
            if m: st=m[0].group(1).strip()
            elif l.startswith(("## ","# ")): st="LIVE"
        out.append((st,sec))
    return out
for f in DOCS:
    p=f if os.path.isabs(f) else os.path.join(MP,f)
    if not os.path.exists(p): continue
    s=open(p,encoding="utf-8",errors="ignore").read(); L=s.split("\n")
    if RECORD_BANNER in "\n".join(L[:40]): continue
    ss=sop_status(L) if p.endswith("MethylPhys_CPG_SOP.md") else [("LIVE","")]*len(L)
    fail=pend=0; where=[]
    for k,((st,sec),l) in enumerate(zip(ss,L),1):
        if st!="LIVE" or ALLOW_LINE.search(l): continue
        n=len(TEXT.findall(l))
        if not n: continue
        if any(x in sec for x in SOP_PENDING_V2) or "`stage_b_" in l: pend+=n
        else: fail+=n; where.append(k)
    r=os.path.relpath(p,MP)
    if fail: hits.append({"file":r,"kind":"text","n":fail,"status":"FAIL","why":"class-reading vocabulary in a live document, lines "+",".join(map(str,where[:12]))})
    if pend: hits.append({"file":r,"kind":"text","n":pend,"status":"PENDING_V2","why":"describes the class gauge the chain runs until the v2 switch-over"})
os.makedirs(os.path.join(HERE,"results"),exist_ok=True)
json.dump({"rule":"a class is only the floor a cell is divided by","hits":hits},open(os.path.join(HERE,"results","class_guard.json"),"w"),indent=1)
fails=[h for h in hits if h["status"]=="FAIL"]; pend=[h for h in hits if h["status"]=="PENDING_V2"]
for h in hits: print(f"  {h['status']:<10} {h['kind']:<4} {h['n']:>4}  {h['file']}  - {h['why']}")
print(f"class guard: {len(fails)} FAIL, {len(pend)} PENDING_V2, {sum(h['status']=='LABEL' for h in hits)} LABEL")
sys.exit(1 if fails else 0)
