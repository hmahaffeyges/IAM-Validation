"""DEV-ONE-DRAW-01, the copies correction over the plasmas' own range: sweep synthetic, written and committed before it runs (development).
The instrument's job: a healthy plasma reads the same at any depth and any number of copies per molecule.
Why: the first copies synthetic (copies_synthetic.py) met its bars at 6 molecules a site and 4 and 12 copies; the GSE186007 plasmas hold about
0.07 to 4.3 molecules a covered site and 1.2 to 270 copies (reading_2/plasma_lab3_r2_copies.txt), a range it never tried, and the third reading
was not met. Construction as copies_synthetic.py (the same truth; seed 20261107 + row): at grid row ROW of copies_sweep_grid.csv, molecules
Poisson(lam) a site; each molecule copied c times, c = 1 + floor(exp(N(m, 1))) with m chosen so the dispersion E[c^2]/E[c] equals the row's D
(D = 1: no copies). Each constructed plasma read by run_sample.py with the plasma readings' flags, once without and once with --dev-seq-copies.
Usage: python3 copies_sweep.py ROW WORKDIR OUT.json (one grid point; copies_sweep_score.py scores the grid)."""
import os, sys, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
ROW, W, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
AS = os.path.join(MP, "chain", "Runtime Matrices", "Seq_Atlas", "array_sites"); V21 = os.path.join(MP, "atlas", "v2.1"); IF = os.path.join(MP, "chain", "MethylPhys_Interface")
HG19 = 28217448; G = pd.read_csv(os.path.join(HERE, "copies_sweep_grid.csv")).set_index("row").loc[ROW]; lam, Dt = float(G.lam), float(G.D)
rng = np.random.default_rng(20261107 + ROW)
F = pd.read_csv(os.path.join(HERE, "copies_synthetic_shares.csv")).set_index("cell")["share"]; mu = {}
for part in (1, 2, 3):
    f = os.path.join(AS, f"mu_part{part}.parquet"); cols = [c for c in pq.read_schema(f).names if c in F.index]
    if cols:
        T = pq.read_table(f, columns=cols).to_pandas()
        for c in cols: mu[c] = np.where(T[c].to_numpy() == 65535, np.nan, T[c].to_numpy() / 1e4)
truth = sum(F[c] * mu[c] for c in F.index)
ids = pq.read_table(os.path.join(V21, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(V21, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
cpg = pd.Series(ids).map(K).to_numpy(); ok = ~pd.isna(cpg) & np.isfinite(truth)
cpg = cpg[ok].astype(np.int64); p = np.clip(truth[ok], 0, 1); _, first = np.unique(cpg, return_index=True); cpg, p = cpg[first], p[first]

def copies(n, m): return 1 + np.floor(np.exp(rng.normal(m, 1.0, n))).astype(np.int64)

def m_for(D):
    """Bisection on m: the dispersion E[c^2]/E[c] of the copy law equals D."""
    lo, hi = -6.0, 8.0; z = np.random.default_rng(1).normal(0, 1, 2_000_000)
    for _ in range(60):
        mid = (lo + hi) / 2; c = 1 + np.floor(np.exp(mid + z)); d = (c * c).mean() / c.mean(); lo, hi = (mid, hi) if d < D else (lo, mid)
    return (lo + hi) / 2

n1 = rng.poisson(lam, len(p)); k1 = rng.binomial(n1, p)
if Dt == 1: k, n, m = k1, n1, None
else:
    m = m_for(Dt); cm = copies(int(k1.sum()), m); cu = copies(int((n1 - k1).sum()), m)
    k = np.bincount(np.repeat(np.arange(len(p)), k1), cm, len(p)).astype(np.int64)
    n = k + np.bincount(np.repeat(np.arange(len(p)), n1 - k1), cu, len(p)).astype(np.int64)
a = np.zeros((HG19, 2), np.uint16); a[cpg - 1, 0] = np.minimum(k, 65534); a[cpg - 1, 1] = np.minimum(n, 65534); fb = os.path.join(W, f"r{ROW}.lbeta"); a.tofile(fb)
cov = n > 0
res = dict(row=ROW, lam=lam, D=Dt, m=m, sites=int(len(p)), covered=int(cov.sum()),
           molecules_per_covered=round(float(n1[cov].mean()), 4) if cov.any() else None, calls_per_covered=round(float(n[cov].mean()), 4) if cov.any() else None)
FL = ["--full-chain", "--dev-seq-ref", "auto", "--dev-whole-sites", "--dev-seq-level", "donors", "--dev-terr-counted"]
for flag in (False, True):
    J = os.path.join(W, f"r{ROW}_{int(flag)}.json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", fb, "--dev-seq-values", "1", "--specimen", "plasma", "--id", f"r{ROW}_{int(flag)}"] + FL +
                       (["--dev-seq-copies"] if flag else []) + ["--out", J.replace(".json", ".html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
    B = json.load(open(J)) if os.path.exists(J) else {}; w = B.get("met_a_whole") or {}
    res["on" if flag else "off"] = dict(A=w.get("A"), domains=((w.get("by_place") or {}).get("domains") or {}).get("A"),
                                        reason=str(w.get("reason"))[:120] if w.get("A") is None else None,
                                        copies=((B.get("development") or {}).get("copies") or {}).get("mean_copies"), rc=r.returncode,
                                        tail=None if r.returncode == 0 else r.stdout[-300:])
json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(res)[:600])
