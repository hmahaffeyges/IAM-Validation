"""DEV-LINK-IAMA-METAA-02: Stage Q's measured response to a known loss δ (the note's 'dε/dδ measured on the vehicle molecules, in silico').
Every methylated call (C) on every molecule is turned to T with probability δ (seeded); the file is then read by Stage Q's own
pat_site_table (≥ 6 calls, ≥ 80 % C; isolated errors only). Reported per δ: ε measured, ε by the simple form ε_v + δ(1-ε_v), IAM-A_rel
measured and simple, and the share of molecules Stage Q still reads. Run: python3 insilico_loss_02.py CHAIN PAT.gz OUT.csv [max_lines]"""
import sys, os, gzip, math, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import stage_q_iam_a as Q
PAT, OUT = sys.argv[2], sys.argv[3]; MAXL = int(sys.argv[4]) if len(sys.argv) > 4 else None
DELTAS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]; SEED = 20261010
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
lines = []
with gzip.open(PAT, "rt") as f:
    for i, l in enumerate(f):
        if MAXL and i >= MAXL: break
        lines.append(l.rstrip("\n").split("\t"))
def plant(d, path):
    r = np.random.default_rng(SEED); out = []
    for q in lines:
        n = int(q[3])
        for _ in range(n):                                            # one line per molecule, so each gets its own losses
            p = q[2]
            if d > 0:
                a = np.frombuffer(p.encode(), dtype="S1").copy(); c = a == b"C"; a[c & (r.random(len(a)) < d)] = b"T"; p = a.tobytes().decode()
            out.append(f"{q[0]}\t{q[1]}\t{p}\t1\n")
    with gzip.open(path, "wt", compresslevel=1) as g: g.writelines(out)
rows = []; e0 = None
for d in DELTAS:
    tmp = OUT + f".d{d}.pat.gz"; plant(d, tmp); T = Q.pat_site_table(tmp)
    e = float(T.err_A.sum() + T.err_B.sum()) / float(T.opp_A.sum() + T.opp_B.sum()); nq = T.attrs["n_qualifying_lines"]
    if e0 is None: e0, n0 = e, nq
    es = e0 + d * (1 - e0)
    rows.append(dict(delta=d, eps=e, eps_simple=es, IAMA_rel=H(e) / H(e0), IAMA_rel_simple=H(es) / H(e0), share_read=nq / n0))
    os.remove(tmp); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
