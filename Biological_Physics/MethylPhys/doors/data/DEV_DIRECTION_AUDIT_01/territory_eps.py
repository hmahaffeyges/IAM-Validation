"""DEV-DIRECTION-AUDIT-01 condition 2: IAM-A copy error per territory (development, 2026-10-11). Stage Q's rule unchanged (stage_q_iam_a.pat_site_table):
a molecule qualifies with >= 6 calls and >= 80 % C; at each interior call (in call order, '.' skipped) one opportunity; an error is a T whose two
neighbouring calls are C. Qualifying molecules are split by their mean HEALTHY beta (reference beta from healthy files): held (>= 0.8), gained (<= 0.2),
other. Built-in check: errors / opportunities over ALL qualifying molecules equals Stage Q's eps for the same file.
Usage: from territory_eps import load, beta_ref, read  |  python3 territory_eps.py CHAIN PAT [REF_PAT ...]  (check against Stage Q)"""
import sys, os, numpy as np, pandas as pd
def load(p):
    d = pd.read_csv(p, sep="\t", header=None, names=["chr", "start", "pat", "n"], dtype={"chr": str, "start": np.int64, "pat": str, "n": np.int64})
    L = d.pat.str.len().values; b = np.frombuffer("".join(d.pat).encode(), np.uint8).copy()
    mol = np.repeat(np.arange(len(d)), L); pos = np.repeat(d.start.values, L) + (np.arange(len(b)) - np.repeat(np.r_[0, np.cumsum(L)[:-1]], L))
    return dict(d=d, L=L, b=b, mol=mol, pos=pos)
def beta_ref(files):
    mx = max(int(f["pos"].max()) for f in files) + 1; c = np.zeros(mx); t = np.zeros(mx)
    for f in files:
        w = f["d"].n.values[f["mol"]]; C = f["b"] == 67; T = f["b"] == 84
        c += np.bincount(f["pos"], C * w, mx); t += np.bincount(f["pos"], (C | T) * w, mx)
    with np.errstate(invalid="ignore", divide="ignore"): return c / t
def read(f, bv):
    b, mol, n = f["b"], f["mol"], f["d"].n.values; N = len(n); C = b == 67; called = C | (b == 84)
    calls = np.bincount(mol, called, N); nc = np.bincount(mol, C, N); q = (calls >= 6) & (nc >= 0.8 * calls)
    bvx = np.zeros(len(b)); ok = called & (f["pos"] < len(bv)); bvx[ok] = np.nan_to_num(bv[f["pos"][ok]])
    mb = np.bincount(mol, bvx, N) / np.maximum(np.bincount(mol, ok, N), 1)
    ci = np.flatnonzero(called); m = mol[ci]; v = C[ci]
    first = np.r_[True, m[1:] != m[:-1]]; last = np.r_[m[1:] != m[:-1], True]; inner = ~first & ~last
    prv = np.r_[False, v[:-1]]; nxt = np.r_[v[1:], False]; err = inner & ~v & prv & nxt
    opp_m = np.bincount(m, inner, N); err_m = np.bincount(m, err, N); out = {}
    for name, sel in (("all", q), ("held", q & (mb >= 0.8)), ("gained", q & (mb <= 0.2)), ("other", q & (mb > 0.2) & (mb < 0.8))):
        o = float((opp_m * n)[sel].sum()); e = float((err_m * n)[sel].sum())
        out[name] = dict(molecules=int(n[sel].sum()), opportunities=o, errors=e, eps=e / o if o else None)
    return out
if __name__ == "__main__":
    sys.path.insert(0, sys.argv[1]); import stage_q_iam_a as Q
    f = load(sys.argv[2]); refs = [load(p) for p in sys.argv[3:]] or [f]; r = read(f, beta_ref(refs)); T = Q.pat_site_table(sys.argv[2])
    eq = float((T.err_A.sum() + T.err_B.sum()) / (T.opp_A.sum() + T.opp_B.sum()))
    print({k: (v["molecules"], v["opportunities"], round(v["eps"], 6) if v["eps"] else None) for k, v in r.items()}); print("Stage Q eps", eq, "| reader all", r["all"]["eps"], "| identical:", abs(eq - r["all"]["eps"]) < 1e-15)
