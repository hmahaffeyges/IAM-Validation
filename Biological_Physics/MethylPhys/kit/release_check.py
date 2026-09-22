#!/usr/bin/env python3
"""release_check.py - CHAIN_COMMISSIONING row N, as ONE command.

Runs every safeguard the chain has and writes a machine-readable verdict to results/release_check.json,
which the patient report reads and prints on its Safeguards tab. A guard that cannot run for want of data
is recorded SKIPPED with what it needs - never PASS.

    python3 release_check.py                     # everything it can run here
    python3 release_check.py --only anchor,tiers  # a subset
    CPG_KIT_DATA=/path/to/data python3 release_check.py

Exit code 0 only if no guard FAILED (skips are allowed and are reported as skips).
"""
import os, sys, json, time, subprocess, argparse, hashlib
HERE=os.path.dirname(os.path.abspath(__file__)); BIO=os.path.dirname(os.path.dirname(HERE)); REPO=os.path.dirname(BIO)
OUT=os.path.join(HERE,"results"); os.makedirs(OUT,exist_ok=True)
ENG=os.path.join(BIO,"MethylPhys/chain")

# (key, human name, what it guards, argv, needs)
GUARDS=[
 ("formula","A-score formula self-test","the per-cell surface is mean_i H(beta_i)/H_min and the module refuses H(beta_mean) - the N7 defect cannot come back",
  [sys.executable,"PROC_FORMULA_01.py"],None),
 ("anchor","Sealed cohort anchors","the sealed 648-sample foundation-cohort per-cell scores still reproduce from raw public data (r = 1.00000)",
  [sys.executable,"PROC_ANCHOR_01.py"],"GSE51032/GSE51057 series matrices"),
 ("decon","Deconvolver conformance","the composition solver reproduces the answer key shipped with the test package",
  [sys.executable,"PROC_DECON_01.py"],"10_TEST_DATA"),
 ("sep","Atlas separability","the measured separability of the blood classes - the finding NILC reported and the atlas confirmed",
  [sys.executable,"PROC_SEP_03.py"],None),
 ("gauge","Gauge switch","run_full reports the identity-loci gauge, not the retired marker-union statistic",
  [sys.executable,"test_gauge_switch.py"],"10_TEST_DATA"),
 ("tiers","One tier definition","every tier word in the chain comes from tier_breakpoints.json; no literal breakpoint survives in engine code; AT_CEILING at 1/H_min",
  [sys.executable,"test_tiers.py"],None),
 ("sky","Patient sky","the sky refuses a laboratory with no measured residual scale and masks classes below their presence floor",
  [sys.executable,"test_patient_sky.py"],"10_TEST_DATA"),
 ("labzero","Laboratory zero","panels under 40 arrays are refused and a reading with no zero is marked NOT REPORTABLE",
  [sys.executable,"test_lab_zero.py"],None),
 ("cmb","Sky commissioning","PROC-CMB-05: a healthy sky is quiet at the measured level in every commissioned laboratory",
  [sys.executable,"PROC_CMB_05.py"],"four laboratories' Stage 1 betas"),
 ("bidir","Bidirectional detector","PROC-BIDIR-01: the sealed directional panel reproduces, including re-extraction from raw GEO",
  [sys.executable,"PROC_BIDIR_01.py"],"GSE153712 supplementary matrix"),
 ("detection","Detection-vocabulary scan","no document claims the chain cannot detect something it has not been run on",
  [sys.executable,"finding_check.py","--detection-scan"],None),
]

ENV=dict(os.environ)
ENV["CPG_KIT_ENGINE"]=ENV.get("CPG_KIT_ENGINE") or ENG
ENV["CPG_KIT_RUNTIME"]=ENV.get("CPG_KIT_RUNTIME") or os.path.join(ENG,"Runtime Matrices","A_Scoring_Module")
# prepend, never setdefault: an inherited PYTHONPATH would otherwise hide the engine from the kit scripts
# every directory under the engine that holds a module, so a kit script's bare import resolves wherever the
# file actually lives in the tree (2026-09-22: PROC_SEP_03 needed lineage_splitter, PROC_DECON_01 the solver)
_dirs=[ENG]+[dp for dp,_,fs in os.walk(ENG) if any(f.endswith(".py") for f in fs) and "__pycache__" not in dp]
ENV["PYTHONPATH"]=os.pathsep.join(_dirs+[ENV.get("PYTHONPATH","")])
def run(argv, cwd, timeout=3600):
    t=time.time()
    try:
        p=subprocess.run(argv,cwd=cwd,capture_output=True,text=True,timeout=timeout,env=ENV)
        return p.returncode, (p.stdout or "")+(p.stderr or ""), time.time()-t
    except subprocess.TimeoutExpired:
        return 124,"TIMEOUT",time.time()-t
    except FileNotFoundError as e:
        return 127,repr(e),time.time()-t

# A guard is PASS only if it SAYS so. Exit code 0 is not evidence: 2026-09-22 the bidirectional guard exited 0
# while its own last line read "row 4.5 commissioned: False" (its raw-data bar had been skipped), and an
# exit-code rule reported that as PASS. Never again - an explicit marker or it is not a pass.
PASS_MARK=("-> pass","=> pass",": pass","conformance: pass","commissioned: true","all bars pass","0 unqualified")
FAIL_MARK=("-> fail","=> fail",": fail","commissioned: false","assertionerror","traceback (most recent call last)")
SKIP_MARK=("no such file","not found","filenotfound","modulenotfound","importerror","needs ","timeout")
def verdict(rc, out):
    low=out.lower()
    if any(m in low for m in FAIL_MARK) and not any(m in low for m in SKIP_MARK): return "FAIL"
    if any(m in low for m in SKIP_MARK): return "SKIPPED"
    if rc==0 and any(m in low for m in PASS_MARK): return "PASS"
    if rc!=0: return "FAIL"
    return "INCONCLUSIVE"   # ran, exited 0, said nothing a machine can read - not a pass

def check_links():
    """Every relative path in the live documentation set must resolve (link_check.py, 2026-09-22)."""
    import subprocess, sys, os
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "link_check.py")],
                       capture_output=True, text=True)
    print(r.stdout.strip().split("\n")[-1])
    return r.returncode == 0


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--only",default=""); a=ap.parse_args()
    want=set(x.strip() for x in a.only.split(",") if x.strip())
    sha=subprocess.run(["git","-C",REPO,"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip()
    rows=[]
    for key,name,guards,argv,needs in GUARDS:
        if want and key not in want: continue
        rc,out,dt=run(argv,HERE)
        v=verdict(rc,out)
        tail=[l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
        rows.append({"key":key,"name":name,"guards":guards,"status":v,"seconds":round(dt,1),
                     "detail":tail[0][:300],"needs":needs if v=="SKIPPED" else None,"argv":" ".join(os.path.basename(x) for x in argv)})
        print(f"{v:<8} {name:<32} {round(dt,1):>6}s  {tail[0][:90]}",flush=True)
    rec={"commit":sha,"run_at":time.strftime("%Y-%m-%d %H:%M"),"host_note":"run wherever the chain is run; the report prints this verdict and its commit",
         "n_pass":sum(1 for r in rows if r["status"]=="PASS"),"n_fail":sum(1 for r in rows if r["status"]=="FAIL"),
         "n_skipped":sum(1 for r in rows if r["status"]=="SKIPPED"),"n_inconclusive":sum(1 for r in rows if r["status"]=="INCONCLUSIVE"),"guards":rows}
    p=os.path.join(OUT,"release_check.json"); json.dump(rec,open(p,"w"),indent=1)
    print(f"\n{rec['n_pass']} pass, {rec['n_fail']} fail, {rec['n_skipped']} skipped, {rec['n_inconclusive']} inconclusive -> {p}  (commit {sha})")
    sys.exit(1 if (rec["n_fail"] or rec["n_inconclusive"]) else 0)

if __name__=="__main__": main()
