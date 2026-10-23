"""CANON/workflow_check.py (2026-10-23; author: "You MUST make sure this does not continue to happen", after a 4-hour alignment ended on a
missing package). Every GitHub workflow that runs anything from the chain or a test folder (Biological_Physics/MethylPhys/chain, doors/data or reproduce)
must (1) install the chain's own requirements.txt and (2) run chain/preflight.py right after, before its long part. Exit 1 naming each workflow
and what it lacks.  Usage: python3 CANON/workflow_check.py [--check]"""
import os, re, sys, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); bad = []; n = 0
for p in sorted(glob.glob(os.path.join(ROOT, ".github", "workflows", "*.yml"))):
    s = open(p, encoding="utf-8").read(); f = os.path.basename(p)
    if not re.search(r"MethylPhys/(chain|doors/data|reproduce)/", s): continue
    viasetup = "reproduce/setup.sh" in s   # setup.sh installs the requirements and runs preflight
    n += 1
    if not viasetup and "Biological_Physics/MethylPhys/chain/requirements.txt" not in s: bad.append(f"{f}: does not install chain/requirements.txt")
    if not viasetup and "chain/preflight.py" not in s: bad.append(f"{f}: does not run chain/preflight.py before its long part")
for b in bad: print("  " + b)
print(f"workflow_check: {'REFUSED (%d)' % len(bad) if bad else 'ok'} ({n} workflows run the chain)"); sys.exit(1 if bad else 0)
