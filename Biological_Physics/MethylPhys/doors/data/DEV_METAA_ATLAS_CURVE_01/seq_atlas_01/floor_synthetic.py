"""The library floor (plan 35.30): synthetic, written and committed before it runs. The instrument's job: a library's own opposite-call rate
at the floor sites is recovered as planted. Known answer: at the committed floor sites (chain/Runtime Matrices/Seq_Atlas/floor_sites_v1.parquet),
calls n ~ Poisson(depth) and opposite calls ~ Binomial(n, rate): rates 0.002, 0.005, 0.01, 0.02 at depths 5 and 20, seed 20261105, as uint16
.lbeta files on the hg19 numbering, each read by the one command (run_sample.py --dev-seq-beta FILE --dev-seq-values 1). Bar: every
recovered rate within 3 counting standard deviations (sqrt(rate (1 - rate) / calls)) of the planted rate.
Usage: python3 floor_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); IF = os.path.join(CH, "MethylPhys_Interface")
D = pd.read_parquet(os.path.join(CH, "Runtime Matrices", "Seq_Atlas", "floor_sites_v1.parquet")); HG19 = 28217448; rng = np.random.default_rng(20261105)
L = ["The library floor: synthetic, read by the one command", f"seed 20261105, {len(D):,} floor sites ({(D.side == 'unmarked').sum():,} unmarked, {(D.side == 'marked').sum():,} marked)",
     "rate | depth | recovered | calls | within 3 sd"]; ok = []
for rate in (0.002, 0.005, 0.01, 0.02):
    for depth in (5, 20):
        n = rng.poisson(depth, len(D)); o = rng.binomial(n, rate); k = np.where(D.side == "unmarked", o, n - o)
        a = np.zeros((HG19, 2), np.uint16); a[D.cpg.to_numpy() - 1, 0] = k; a[D.cpg.to_numpy() - 1, 1] = n; f = os.path.join(W, "s.lbeta"); a.tofile(f); J = os.path.join(W, "s.json")
        r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", "floorsyn",
                            "--out", os.path.join(W, "s.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
        F = (json.load(open(J)).get("development") or {}).get("library_floor") or {}; got = F.get("rate"); nn = (F.get("unmarked") or {}).get("calls", 0) + (F.get("marked") or {}).get("calls", 0)
        sd = (rate * (1 - rate) / nn) ** 0.5 if nn else float("nan"); w = got is not None and abs(got - rate) <= 3 * sd; ok.append(w)
        L.append(f"{rate} | {depth} | {got} | {nn:,} | {'met' if w else 'not met'}"); os.remove(f)
L.append(f"synthetic: {'met' if all(ok) else 'not met'}"); open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
