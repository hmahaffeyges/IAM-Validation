"""DEV-DIRECTION-AUDIT-01, IAM-A and its C-score under planted GAIN (healthy molecules only; 2026-10-11). Question: Stage Q reads only molecules
>= 80 % methylated; gain makes normally unmethylated molecules fully methylated, so they ENTER the reading. Does IAM-A or its C-score see that?
Plant on one healthy .pat: per-CpG beta from the file itself; normally unmethylated molecules = >= 6 calls, mean beta <= 0.2. A share r of them is
made fully methylated, then each call is turned to T with probability eps_g (the gained territory's copy error). Read with Stage Q's own code
(pat_site_table, cscore). Grid: r 0 / 0.01 / 0.05 x eps_g = 0.5, 1, 2 x the file's own eps. One pattern line is planted as a whole (its n molecules
share the planted pattern). Usage: python3 plant_gain_iama.py CHAIN_DIR PAT OUT.csv"""
import sys, os, gzip, math, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import stage_q_iam_a as Q
PAT, OUT = sys.argv[2], sys.argv[3]; W = os.path.dirname(os.path.abspath(OUT))
d = pd.read_csv(PAT, sep="\t", header=None, names=["chr", "start", "pat", "n"], dtype={"chr": str, "start": np.int64, "pat": str, "n": np.int64})
L = d.pat.str.len().values; b = np.frombuffer("".join(d.pat).encode(), np.uint8); mol = np.repeat(np.arange(len(d)), L)
pos = np.repeat(d.start.values, L) + (np.arange(len(b)) - np.repeat(np.r_[0, np.cumsum(L)[:-1]], L)); C = b == 67; T_ = b == 84; w = d.n.values[mol]
mx = int(pos.max()) + 1; beta = np.bincount(pos, C * w, mx) / np.maximum(np.bincount(pos, (C | T_) * w, mx), 1)
calls = np.bincount(mol, C | T_, len(d)); mb = np.bincount(mol, beta[pos] * (C | T_), len(d)) / np.maximum(calls, 1)
U = (calls >= 6) & (mb <= 0.2)
H = lambda e: -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
def run(r, eg, seed):
    g = np.random.default_rng(seed); pick = U & (g.random(len(d)) < r); nb = b.copy(); m = pick[mol] & (C | T_)
    nb[m] = 67; nb[m & (g.random(len(nb)) < eg)] = 84
    p = os.path.join(W, f"_plant_{seed}.pat.gz"); pats = np.split(nb, np.cumsum(L)[:-1])
    with gzip.open(p, "wt") as f:
        for c_, s_, pt, n_ in zip(d.chr.values, d.start.values, (x.tobytes().decode() for x in pats), d.n.values): f.write(f"{c_}\t{s_}\t{pt}\t{n_}\n")
    Tb = Q.pat_site_table(p); os.remove(p); cs = Q.cscore(Tb)
    e = float((Tb.err_A.sum() + Tb.err_B.sum()) / (Tb.opp_A.sum() + Tb.opp_B.sum()))
    return dict(opportunities=int(Tb.opp_A.sum() + Tb.opp_B.sum()), qualifying_lines=Tb.attrs["n_qualifying_lines"], eps=e, C=cs["C"], C_se_null=cs.get("se_null"), n_blocks=cs["n_blocks"])
base = run(0.0, 0.0, 1); e0 = base["eps"]; rows = [dict(r=0.0, eps_g_over_eps0=None, **base, IAMA_rel=1.0)]
print("baseline", base, flush=True)
for r in (0.01, 0.05):
    for k in (0.5, 1.0, 2.0):
        x = run(r, k * e0, int(100 * r) * 10 + int(k * 2)); rows.append(dict(r=r, eps_g_over_eps0=k, **x, IAMA_rel=H(x["eps"]) / H(e0)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False); print("normally unmethylated molecules (lines):", int(U.sum()))
