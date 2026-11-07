"""DEV-ONE-DRAW-01, the copies correction through the one command: synthetic, written and committed before it runs (development).
The instrument's job: a healthy plasma whose calls are copies of fewer molecules reads as the same plasma read from its molecules.
Construction (seed 20261106): a healthy plasma by construction at the IAM-Atlas v2.1 sites with a sequencing CpG: at each site the truth is the
committed array-site columns (Runtime Matrices/Seq_Atlas/array_sites, mu, each cell's donor median) mixed in one GSE186007 healthy plasma's
own Deconvolver shares (copies_synthetic_shares.csv, cells at 0.01 or more, renormalized); molecules Poisson(6) a site, each methylated with
the truth's probability; then the same molecules copied 1, 4 and 12 times (each molecule 1 + Poisson(c - 1) copies). Each constructed plasma
is written as a uint16 .lbeta and read by run_sample.py with the plasma readings' flags, once without and once with --dev-seq-copies.
Bars: 1. copies 1: the flag changes whole-sample Met-A by at most 0.01. 2. copies 4 and 12: with the flag, whole-sample Met-A within 0.03 of
the copies-1 reading without it. Recorded: the readings without the flag, the copies the chain measures, and the domains.
Usage: python3 copies_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
AS = os.path.join(MP, "chain", "Runtime Matrices", "Seq_Atlas", "array_sites"); V21 = os.path.join(MP, "atlas", "v2.1"); IF = os.path.join(MP, "chain", "MethylPhys_Interface")
HG19 = 28217448; rng = np.random.default_rng(20261106)
F = pd.read_csv(os.path.join(HERE, "copies_synthetic_shares.csv")).set_index("cell")["share"]
mu = {}
for part in (1, 2, 3):
    f = os.path.join(AS, f"mu_part{part}.parquet"); cols = [c for c in pq.read_schema(f).names if c in F.index]
    if cols:
        T = pq.read_table(f, columns=cols).to_pandas()
        for c in cols: mu[c] = np.where(T[c].to_numpy() == 65535, np.nan, T[c].to_numpy() / 1e4)
assert set(mu) == set(F.index), sorted(set(F.index) - set(mu))
truth = sum(F[c] * mu[c] for c in F.index)
ids = pq.read_table(os.path.join(V21, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(V21, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
cpg = pd.Series(ids).map(K).to_numpy(); ok = ~pd.isna(cpg) & np.isfinite(truth)
cpg = cpg[ok].astype(np.int64); p = np.clip(truth[ok], 0, 1); _, first = np.unique(cpg, return_index=True); cpg, p = cpg[first], p[first]
n1 = rng.poisson(6, len(p)); k1 = rng.binomial(n1, p)
FL = ["--full-chain", "--dev-seq-ref", "auto", "--dev-whole-sites", "--dev-seq-level", "donors", "--dev-terr-counted"]
res = {}; L = [f"constructed healthy plasma: {len(p):,} sites, molecules Poisson(6), shares of {len(F)} cells from copies_synthetic_shares.csv"]
for c in (1, 4, 12):
    if c == 1: k, n = k1, n1
    else:
        cm = 1 + rng.poisson(c - 1, int(k1.sum())); cu = 1 + rng.poisson(c - 1, int((n1 - k1).sum()))
        site_m = np.repeat(np.arange(len(p)), k1); site_u = np.repeat(np.arange(len(p)), n1 - k1)
        k = np.bincount(site_m, cm, len(p)).astype(np.int64); n = k + np.bincount(site_u, cu, len(p)).astype(np.int64)
    a = np.zeros((HG19, 2), np.uint16); a[cpg - 1, 0] = np.minimum(k, 65534); a[cpg - 1, 1] = np.minimum(n, 65534); fb = os.path.join(W, f"c{c}.lbeta"); a.tofile(fb)
    for flag in (False, True):
        J = os.path.join(W, f"c{c}_{int(flag)}.json")
        r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", fb, "--dev-seq-values", "1", "--specimen", "plasma", "--id", f"c{c}_{int(flag)}"] + FL +
                           (["--dev-seq-copies"] if flag else []) + ["--out", J.replace(".json", ".html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
        B = json.load(open(J)) if os.path.exists(J) else {}; w = B.get("met_a_whole") or {}
        res[(c, flag)] = dict(A=w.get("A"), domains=((w.get("by_place") or {}).get("domains") or {}).get("A"), copies=((B.get("development") or {}).get("copies") or {}).get("mean_copies"), rc=r.returncode)
        L.append(f"copies {c}, flag {'on' if flag else 'off'}: whole-sample Met-A {w.get('A')}, domains {res[(c, flag)]['domains']}, copies measured {res[(c, flag)]['copies']}, exit {r.returncode}"
                 + ("" if r.returncode == 0 else " | " + r.stdout[-200:].replace("\n", " ")))
        print(L[-1], flush=True)
A = lambda c, f: res[(c, f)]["A"]
b1 = A(1, False) is not None and A(1, True) is not None and abs(A(1, True) - A(1, False)) <= 0.01
b2 = all(A(c, True) is not None and A(1, False) is not None and abs(A(c, True) - A(1, False)) <= 0.03 for c in (4, 12))
L += [f"bar 1 (copies 1, the flag changes whole-sample Met-A by at most 0.01): {'met' if b1 else 'not met'}",
      f"bar 2 (copies 4 and 12 with the flag, within 0.03 of copies 1 without it): {'met' if b2 else 'not met'}", "synthetic: " + ("met" if b1 and b2 else "not met")]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L[-3:]))
