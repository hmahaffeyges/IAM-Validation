"""DEV-CSCORE-LIMIT-01 synthetic through the one command (bar 1; written 2026-10-25 before any healthy C-Score is scored against the limit).
Constructed whole bloods: IAM-Atlas v2.1's cells mixed in fixed shares (MIX), each site plus an independent scatter (sd 0.02, seed 20261025),
read by run_sample.py --specimen "whole blood". 12 such arrays: their C-Scores (each cell at a share of at least 1 %, and the whole genome)
must sit at or below the derived limit (limit.py) in at least 90 % of values. One more array with a planted run: 200 neighboring neutrophil
identity sites (in genome order) moved 10 % of the way toward 0.5: its neutrophil C must sit above the limit.
Usage: python3 limit_synthetic_chain.py WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); W, OUT = sys.argv[1], sys.argv[2]; os.makedirs(W, exist_ok=True)
RS = os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface"); sys.path.insert(0, os.path.join(RS, "..")); sys.path.insert(0, H)
import stage_metaa_atlas as SA, limit as LM
MIX = {"neutrophils": 0.60, "monocytes": 0.10, "naive cd4 t cells": 0.12, "naive cd8 t cells": 0.08, "naive b cells": 0.05, "nk cells": 0.05}
MU = pd.concat([SA._col("mu", c).rename(c) for c in MIX], axis=1).dropna(); e = MU.mul(pd.Series(MIX), axis=1).sum(1)
g = np.random.default_rng(20261025); L = [f"constructed whole bloods: {len(e)} sites, mix {MIX}"]; vals = []
NI = SA._genome_order(pd.Index(SA.identity_sites("neutrophils")).intersection(e.index)); run = NI[2000:2200]
def read(nm, b):
    p = os.path.join(W, nm + "_betas.csv"); pd.DataFrame({"beta": np.clip(b, 0.001, 0.999)}, index=pd.Index(e.index, name="probe")).to_csv(p)
    q = subprocess.run([sys.executable, "run_sample.py", "--betas", p, "--specimen", "whole blood", "--id", nm, "--out", os.path.join(W, nm + ".html"),
                        "--bundle", os.path.join(W, nm + ".json")], cwd=RS, capture_output=True, text=True)
    if q.returncode: L.append(f"{nm} | exit {q.returncode} | {q.stdout[-300:]} {q.stderr[-300:]}"); return {}
    return json.load(open(os.path.join(W, nm + ".json"))).get("cscore_each_cell") or {}
for k in range(12):
    E = read(f"IND{k:02d}", e.to_numpy() + g.normal(0, 0.02, len(e)))
    for c in (E.get("cells") or []):
        if c.get("C") is not None and (c.get("share") or 0) >= 0.01: vals.append((f"IND{k:02d} {c['cell']}", c["C"], LM.limit(c["n_sites"])))
    w = E.get("whole_genome") or {}
    if w.get("C") is not None: vals.append((f"IND{k:02d} whole genome", w["C"], LM.limit(w["n_sites"])))
b = e.copy(); b[run] = b[run] + 0.10 * (0.5 - b[run]); E = read("PLANTED", b.to_numpy() + g.normal(0, 0.02, len(e)))
pn = next((c for c in (E.get("cells") or []) if c["cell"] == "neutrophils"), {})
L += ["reading | C | limit | at or below"] + [f"{n} | {c} | {l:.4f} | {c <= l}" for n, c, l in vals]
s = np.mean([c <= l for _, c, l in vals]) if vals else 0.0
L.append(f"independent arrays: {len(vals)} values, share at or below the limit {s:.4f}")
L.append(f"planted run: neutrophil C {pn.get('C')} | limit {LM.limit(pn['n_sites']) if pn.get('n_sites') else None}")
ok = len(vals) > 0 and s >= 0.9 and pn.get("C") is not None and pn["C"] > LM.limit(pn["n_sites"])
L.append(f"bar 1 (at least 90 % at or below the limit; the planted run above it): {'met' if ok else 'not met'}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-6:]))
