"""DEV-IAMA-COMMISSION-01 synthetic test 1: the sampling floor of the repeatability test C2. One healthy Loyfer granulocyte whole file
(GSM5652313, hg19, loyfer_pat_v1, GEO GSE186458). Every molecule is sent at random to half A or half B (seeded); each half is then thinned to a
share f of its molecules (f = 1, 0.25, 0.05, 0.01) to stand in for shallower libraries. Each half is read by Stage Q's own pat_site_table
(>= 6 calls, >= 80 % C, isolated errors only): eps = errors / opportunities; IAM-A ratio = H(eps_A) / H(eps_B). Three seeds per depth.
The spread of that ratio is the smallest repeat difference any pair of libraries can show at that depth: C2's pass value is set above it.
Writes split_floor_01_output.csv. Usage: python3 split_floor_01.py CHAIN_DIR PAT TMPDIR"""
import sys, os, gzip, math, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); import stage_q_iam_a as Q
PAT, TMP = sys.argv[2], sys.argv[3]; os.makedirs(TMP, exist_ok=True); HERE = os.path.dirname(os.path.abspath(__file__))
H = lambda e: -(e * math.log2(e) + (1 - e) * math.log2(1 - e)); F = (1.0, 0.25, 0.05, 0.01); SEEDS = (1, 2, 3)
outs = {(s, f, h): gzip.open(os.path.join(TMP, f"s{s}_f{f}_{h}.pat.gz"), "wt", compresslevel=1) for s in SEEDS for f in F for h in "AB"}
rg = {s: np.random.default_rng(20261011 + s) for s in SEEDS}; mol = 0
with gzip.open(PAT, "rt") as fh:
    for l in fh:
        q = l.rstrip("\n").split("\t"); n = int(q[3]); mol += n
        for s in SEEDS:
            a = rg[s].binomial(n, 0.5); u = rg[s].random(n)
            for f in F:
                na = int((u[:a] < f).sum()); nb = int((u[a:] < f).sum())
                if na: outs[(s, f, "A")].write(f"{q[0]}\t{q[1]}\t{q[2]}\t{na}\n")
                if nb: outs[(s, f, "B")].write(f"{q[0]}\t{q[1]}\t{q[2]}\t{nb}\n")
for g in outs.values(): g.close()
rows = []
for s in SEEDS:
    for f in F:
        e = {}
        for h in "AB":
            p = os.path.join(TMP, f"s{s}_f{f}_{h}.pat.gz"); T = Q.pat_site_table(p)
            e[h] = (float(T.err_A.sum() + T.err_B.sum()) / float(T.opp_A.sum() + T.opp_B.sum()), int(T.opp_A.sum() + T.opp_B.sum())); os.remove(p)
        rows.append(dict(seed=s, share=f, molecules_per_half=round(mol * f / 2), eps_A=e["A"][0], eps_B=e["B"][0], opp_A=e["A"][1], ratio=H(e["A"][0]) / H(e["B"][0])))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "split_floor_01_output.csv"), index=False)
print(R.groupby("share").ratio.agg(lambda x: float(np.max(np.abs(x - 1)))).rename("max |ratio-1|").to_string())
