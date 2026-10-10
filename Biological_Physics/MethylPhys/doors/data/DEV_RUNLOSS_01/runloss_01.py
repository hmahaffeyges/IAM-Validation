"""DEV-RUNLOSS-01: a molecule-level reading of run-type methylation loss (written 2026-10-10; simulation first).
Territory: molecules with >= 6 CpG calls whose CpGs average vehicle beta >= 0.8 (methylated in the healthy reference, pooled vehicle .pat files).
Per sample: K = share of territory molecules >= 80 % methylated, L = share <= 20 % methylated, d_eff = 1 - beta_T(sample)/beta_T(vehicle)
(beta_T = methylated calls / calls in territory). L_scat(d_eff) = L expected if the same total loss were scattered: each methylated call lost
independently with probability d_eff on the vehicle's own territory molecules (computed, seeded). Excess run loss: L - L_scat.
Usage: runloss_01.py sim VEH1.pat.gz VEH2.pat.gz      (planted scattered and run loss on vehicle 1, vehicle beta from both)
       runloss_01.py read VEH1 VEH2 SAMPLE.pat.gz ...   (readings)"""
import sys, numpy as np, pandas as pd
NCPG = 30_000_000
def load(p):
    d = pd.read_csv(p, sep="\t", header=None, names=["chr", "start", "pat", "n"], usecols=[1, 2, 3], dtype={"start": np.int64, "pat": str, "n": np.int64})
    L = d.pat.str.len().values; b = np.frombuffer("".join(d.pat.values).encode(), np.uint8)
    off = np.r_[0, np.cumsum(L)[:-1]]; mol = np.repeat(np.arange(len(d)), L); pos = np.repeat(d.start.values, L) + (np.arange(len(b)) - np.repeat(off, L))
    return dict(mol=mol, pos=pos, C=(b == 67), T=(b == 84), n=d.n.values, N=len(d))
def beta_v(files):
    c = np.zeros(NCPG); t = np.zeros(NCPG)
    for f in files:
        w = f["n"][f["mol"]]; c += np.bincount(f["pos"], weights=f["C"] * w, minlength=NCPG)[:NCPG]; t += np.bincount(f["pos"], weights=(f["C"] | f["T"]) * w, minlength=NCPG)[:NCPG]
    with np.errstate(invalid="ignore", divide="ignore"): return c / t
def read(f, bv, CX=None):
    C, T = (f["C"], f["T"]) if CX is None else (CX[0], f["T"] | CX[1]); called = C | T   # a planted loss reads as T
    calls = np.bincount(f["mol"], weights=called, minlength=f["N"]); mc = np.bincount(f["mol"], weights=C, minlength=f["N"])
    bvp = np.nan_to_num(bv[f["pos"]]) * called; tb = np.bincount(f["mol"], weights=bvp, minlength=f["N"])
    with np.errstate(invalid="ignore", divide="ignore"): terr = (calls >= 6) & (tb / calls >= 0.8)
    w = f["n"] * terr; frac = np.where(calls > 0, mc / np.maximum(calls, 1), 0)
    return dict(molecules=int(w.sum()), K=float((w * (frac >= 0.8)).sum() / w.sum()), L=float((w * (frac <= 0.2)).sum() / w.sum()),
                beta_T=float((w * mc).sum() / (w * calls).sum()))
def scat(f, d, seed):
    r = np.random.default_rng(seed); x = f["C"] & (r.random(len(f["C"])) < d); return f["C"] & ~x, x
def runl(f, rr, seed):
    r = np.random.default_rng(seed); lost = (r.random(f["N"]) < rr)[f["mol"]]; return f["C"] & ~lost, f["C"] & lost
def _both(f, rr, d):
    c1, x1 = runl(f, rr, 6); c2, x2 = scat(dict(f, C=c1), d, 8); return c2, x1 | x2
if __name__ == "__main__":
    mode = sys.argv[1]; V = [load(p) for p in sys.argv[2:4]]; bv = beta_v(V); v0 = read(V[0], bv)
    def Lscat(d): return read(V[0], bv, scat(V[0], d, 7))["L"]
    print("vehicle 1:", {k: round(v, 4) for k, v in v0.items()}, flush=True)
    if mode == "sim":
        print("planted            | d_eff  | K      | L      | L_scat(d_eff) | excess L")
        for lab, CX in [("scattered 0.10", scat(V[0], 0.10, 1)), ("scattered 0.22", scat(V[0], 0.22, 2)), ("scattered 0.40", scat(V[0], 0.40, 3)),
                             ("run 0.10", runl(V[0], 0.10, 4)), ("run 0.22", runl(V[0], 0.22, 5)), ("run 0.10 + scat 0.10", _both(V[0], 0.10, 0.10))]:
            x = read(V[0], bv, CX); de = 1 - x["beta_T"] / v0["beta_T"]; ls = Lscat(de)
            print(f"{lab:19s}| {de:.4f} | {x['K']:.4f} | {x['L']:.4f} | {ls:.4f}        | {x['L'] - ls:+.4f}", flush=True)
    else:
        for p in sys.argv[4:]:
            f = load(p); x = read(f, bv); de = 1 - x["beta_T"] / v0["beta_T"]; ls = Lscat(de)
            print(p.split("/")[-1][:11], {k: round(v, 4) for k, v in x.items()}, f"d_eff {de:.4f} L_scat {ls:.4f} excess {x['L'] - ls:+.4f}", flush=True)
