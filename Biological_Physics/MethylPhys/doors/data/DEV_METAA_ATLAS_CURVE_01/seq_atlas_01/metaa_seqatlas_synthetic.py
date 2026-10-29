"""Met-A against the IAM Sequencing Atlas (plan 35.30): synthetic, written and committed before it runs.
The instrument's job: a healthy sequenced cell reads Met-A 1 against its own cell's sequencing reference, at any depth.

Known answer: naive CD4 T cells' 6,000 Met-A identity sites (Met-A's site rule on IAM-Atlas v2.1); at each, a planted healthy level on the
sequencing scale (the marked side 0.80 to 0.99, the unmarked side 0.00 to 0.06, uniform), written as a constructed Met-A sequencing reference
in the committed format. Healthy samples: k ~ Binomial(n, level) at every site, n ~ Poisson(depth), depths 5, 8, 12, 20, 30, 45, 60, three
samples each, seed 20261104, as uint16 .lbeta files on the hg19 numbering. Each read by the one command (run_sample.py --dev-seq-beta FILE
--dev-seq-values 1 --dev-metaa-atlas 'naive cd4 t cells' --dev-seq-ref REFERENCE).
Bars: 1. the median Met-A over the 21 samples within 0.05 of 1; 2. |Spearman rho| between Met-A and depth at most 0.2.
Usage: python3 metaa_seqatlas_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from scipy.stats import spearmanr
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, pyarrow.parquet as pq
CELL = "naive cd4 t cells"; HG19 = 28217448; rng = np.random.default_rng(20261104)
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
S = [s for s in SA.identity_sites(CELL) if s in K.index]; mu = SA._col("mu", CELL).reindex(S)
lv = np.where(mu.to_numpy() > 0.5, rng.uniform(0.80, 0.99, len(S)), rng.uniform(0.00, 0.06, len(S)))
R = pd.DataFrame(dict(cell=CELL, probe=S, cpg=K.reindex(S).to_numpy(np.int64)))
for c in ("donors", "reads", "meth", "spread", "disorder", "halfA", "halfB", "readA", "readB"): R[c] = np.uint16(65535)
R["level"] = np.round(lv * 1e4).astype(np.uint16); REF = os.path.join(W, "metaa_seq_reference_synthetic.parquet"); R.to_parquet(REF, index=False)
IF = os.path.join(CH, "MethylPhys_Interface"); rows = []
for depth in (5, 8, 12, 20, 30, 45, 60):
    for j in range(3):
        n = rng.poisson(depth, len(S)); k = rng.binomial(n, lv); a = np.zeros((HG19, 2), np.uint16); idx = R.cpg.to_numpy() - 1
        a[idx, 0] = k; a[idx, 1] = n; f = os.path.join(W, "s.lbeta"); a.tofile(f); J = os.path.join(W, "s.json")
        r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted " + CELL, "--id", f"syn{depth}_{j}",
                            "--dev-metaa-atlas", CELL, "--dev-seq-ref", REF, "--out", os.path.join(W, "s.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
        m = (json.load(open(J)).get("development") or {}).get("metaa_atlas") or {}
        rows.append(dict(depth=depth, rep=j, A=m.get("A"), reading=m.get("reading"), n_sites=m.get("n_sites"), reason=m.get("reason"))); os.remove(f)
D = pd.DataFrame(rows); D.to_csv(os.path.join(W, "readings.csv"), index=False); ok = D.A.notna()
med = float(D.A[ok].median()) if ok.any() else float("nan"); rho = spearmanr(D.depth[ok], D.A[ok]).correlation if ok.sum() > 2 else float("nan")
L = ["Met-A against the IAM Sequencing Atlas: synthetic, read by run_sample.py --dev-seq-ref", f"seed 20261104, {CELL}, {len(S)} identity sites, 7 depths x 3 samples",
     f"read {int(ok.sum())} of {len(D)} | reading: {D.reading.dropna().iloc[0] if ok.any() else D.reason.iloc[0]}",
     "Met-A by depth: " + ", ".join(f"{d} {g.A.median():.4f}" for d, g in D[ok].groupby("depth")),
     f"bar 1 median Met-A {med:.4f} (within 0.05 of 1): {'met' if abs(med - 1) <= 0.05 else 'not met'}",
     f"bar 2 Spearman rho with depth {rho:+.3f} (|rho| at most 0.2): {'met' if abs(rho) <= 0.2 else 'not met'}",
     f"synthetic: {'met' if ok.all() and abs(med - 1) <= 0.05 and abs(rho) <= 0.2 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
