"""The reference door (chain/reference_door.py, lock 1): synthetic, written and committed before it runs. Every array through the one command
(run_sample.py --betas). Runs only on a disposable checkout (GitHub; it alters and restores one atlas file of that checkout).
Arrays: one constructed healthy whole blood, the IAM-Atlas v2.1 levels of six blood cells in typical shares with array noise at the real arrays'
median (design I 0.0066, design II 0.0098).
Bars: 1. the clean run's bundle carries reference.checked = true, names IAM-Atlas v2.1 as its healthy reference and records no tare;
2. with one byte appended to one IAM Sequencing Atlas file the run stops before any reading (exit 7, REFERENCE_REFUSED, the file named);
3. with a file added to Seq_Atlas that is not in the reference manifest, the run stops the same way.
Usage: python3 reference_door_synthetic.py OUTDIR OUT.txt"""
import os, sys, json, subprocess, shutil
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
SEQ = os.path.join(CH, "Runtime Matrices", "Seq_Atlas"); out, OUT = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
sys.path.insert(0, CH); import stage_metaa_atlas as SA
S = {"neutrophils": .60, "monocytes": .07, "cd4 t cells": .15, "cd8 t cells": .08, "b cells": .05, "nk cells": .05}
des, sets, _ = SA._ruler(); idx = des.index; F = pd.Series(S)
MU = pd.concat([SA._col("mu", k).reindex(idx).rename(k) for k in S], axis=1); c = MU.notna().mul(F, axis=1).sum(1)
e = (MU.fillna(0).mul(F, axis=1).sum(1) / c).where(c >= 0.99).dropna(); d = des.reindex(e.index).to_numpy(); g = np.random.default_rng(20261107)
x = e.to_numpy() + np.where(d == "I", 0.0066, 0.0098) * g.standard_normal(len(e))
f = os.path.join(out, "ref_betas.csv"); pd.DataFrame({"cpg_id": e.index, "beta": np.round(np.clip(x, 0, 1), 4)}).to_csv(f, index=False)
def run(tag):
    b = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", b],
                       capture_output=True, text=True)
    return r.returncode, (json.load(open(b)) if os.path.exists(b) else {}), r.stdout[-600:] + r.stderr[-300:]
L = ["The reference door: synthetic, every array through run_sample.py"]
rc, B, t = run("clean"); R = B.get("reference") or {}
b1 = rc == 0 and R.get("checked") is True and "IAM-Atlas v2.1" in (R.get("healthy_reference") or []) and R.get("tares") == []
L.append(f"clean: exit {rc} | reference {json.dumps({k: R.get(k) for k in ('healthy_reference', 'checked', 'files_checked', 'tares')})}")
p = os.path.join(SEQ, "floor_sites_v1.json"); keep = open(p, "rb").read()
try:
    open(p, "ab").write(b" "); rc2, B2, t2 = run("altered")
finally: open(p, "wb").write(keep)
b2 = rc2 == 7 and B2.get("status") == "REFERENCE_REFUSED" and any("floor_sites_v1.json" in x for x in (B2.get("reference") or {}).get("refused", [])) and "met_a_whole" not in B2
L.append(f"one byte added to Seq_Atlas/floor_sites_v1.json: exit {rc2} | status {B2.get('status')} | refused {(B2.get('reference') or {}).get('refused')}")
q = os.path.join(SEQ, "unlisted_reference.parquet"); shutil.copy(os.path.join(SEQ, "floor_sites_v1.parquet"), q)
try: rc3, B3, t3 = run("unlisted")
finally: os.remove(q)
b3 = rc3 == 7 and B3.get("status") == "REFERENCE_REFUSED" and any("unlisted_reference.parquet" in x for x in (B3.get("reference") or {}).get("refused", []))
L.append(f"an unlisted file in Seq_Atlas: exit {rc3} | status {B3.get('status')} | refused {(B3.get('reference') or {}).get('refused')}")
L += [f"bar 1 (clean: checked, atlas named, no tare): {'met' if b1 else 'not met'}", f"bar 2 (altered file refused before any reading): {'met' if b2 else 'not met'}",
      f"bar 3 (unlisted file refused): {'met' if b3 else 'not met'}", "synthetic: " + ("met" if b1 and b2 and b3 else "not met")]
if not b1: L.append("clean run's output tail: " + t.replace("\n", " | ")[-500:])
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
