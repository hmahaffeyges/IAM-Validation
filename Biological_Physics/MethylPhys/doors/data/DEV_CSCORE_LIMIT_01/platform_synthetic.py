"""DEV-CSCORE-LIMIT-01, 450K platform term, synthetic through the one command (written 2026-10-26 before any GSE67393 array is read).
Constructed 450K whole bloods on the 450K manifest's probes that IAM-Atlas v2.1 holds: the atlas's cells in fixed shares (MIX), plus a scatter
whose neighbor correlation (in genome order) is the frozen 450K term's (an AR filter giving lag-1 to lag-9 correlations near the term's), sd
0.02, seed 20261026. 12 such arrays: their whole-genome C must sit at or below limit_450k in at least 90 %, and at least 2 of the 12 must sit
above the plain limit (showing the term is needed). One more array with a planted run of 2,000 neighboring sites moved 10 % toward 0.5 must
sit above limit_450k. Read with the healthy 450K bloods' own flags (--cell-panel atlas --dev-metaa-allsites). Usage: python3 platform_synthetic.py MANIFEST450K WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); MF, W, OUT = sys.argv[1:4]; os.makedirs(W, exist_ok=True)
RS = os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface"); sys.path.insert(0, os.path.join(RS, "..")); sys.path.insert(0, H)
import stage_metaa_atlas as SA, limit as LM
T = json.load(open(os.path.join(H, "limit_04", "platform_term_450k.json"))); r = np.array(T["neighbor_corr_median"])
p450 = pd.read_csv(MF, usecols=["IlmnID"], low_memory=False).IlmnID.dropna(); p450 = p450[p450.str.startswith("cg")]
MIX = {"neutrophils": 0.60, "monocytes": 0.10, "naive cd4 t cells": 0.12, "naive cd8 t cells": 0.08, "naive b cells": 0.05, "nk cells": 0.05}
MU = pd.concat([SA._col("mu", c).rename(c) for c in MIX], axis=1).dropna(); MU = MU.loc[MU.index.intersection(p450)]
e = MU.mul(pd.Series(MIX), axis=1).sum(1); e = e.reindex(SA._genome_order(e.index)); n = len(e)
g = np.random.default_rng(20261026)
k = np.array([1.0] + [0.0] * 9); a = np.sqrt(np.clip(r, 0, None)); k[1:] = a / max(1.0, a.sum()) * 1.0   # moving-average weights: corr at lag j ~ sum k_i k_{i+j} / sum k^2
def scatter():
    z = np.convolve(g.standard_normal(n + 9), k, mode="valid")[:n]; return 0.02 * z / z.std()
def read(nm, b):
    p = os.path.join(W, nm + "_betas.csv"); pd.DataFrame({"beta": np.clip(b, 0.001, 0.999)}, index=pd.Index(e.index, name="probe")).to_csv(p)
    q = subprocess.run([sys.executable, "run_sample.py", "--betas", p, "--specimen", "whole blood", "--cell-panel", "atlas", "--dev-metaa-allsites", "--id", nm, "--out", os.path.join(W, nm + ".html"),
                        "--bundle", os.path.join(W, nm + ".json")], cwd=RS, capture_output=True, text=True)
    if q.returncode: return None, f"exit {q.returncode} {q.stdout[-200:]}"
    B = json.load(open(os.path.join(W, nm + ".json"))); return ((B.get("cscore_each_cell") or {}).get("whole_genome") or {}), B.get("array_type")
L = [f"constructed 450K whole bloods: {n} sites; frozen term {T['factor']}; MA weights {np.round(k, 4).tolist()}", "array | array type | C | plain limit | limit_450k | lag-1 neighbor corr"]
under = over_plain = 0; m = 0
for j in range(12):
    w, at = read(f"P{j:02d}", e.to_numpy() + scatter())
    if not w or w.get("C") is None: L.append(f"P{j:02d} | {at} | not read"); continue
    m += 1; lp, l4 = LM.limit(w["n_sites"]), LM.limit_450k(w["n_sites"]); under += w["C"] <= l4; over_plain += w["C"] > lp
    L.append(f"P{j:02d} | {at} | {w['C']} | {lp:.4f} | {l4:.4f} | {(w.get('neighbor_corr') or [None])[0]}")
b = e.to_numpy() + scatter(); s0 = n // 2; b[s0:s0 + 2000] += 0.10 * (0.5 - b[s0:s0 + 2000]); w, at = read("PLANTED", b)
pl = w.get("C") if w else None; l4p = LM.limit_450k(w["n_sites"]) if w else None
L.append(f"PLANTED | {at} | {pl} | limit_450k {l4p}")
ok = m == 12 and under / m >= 0.9 and over_plain >= 2 and pl is not None and pl > l4p
L.append(f"at or below limit_450k {under} of {m}; above the plain limit {over_plain}; planted above limit_450k: {pl is not None and pl > l4p}")
L.append(f"synthetic (>= 90 % under limit_450k; >= 2 above the plain limit; planted above): {'met' if ok else 'not met'}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-16:]))
