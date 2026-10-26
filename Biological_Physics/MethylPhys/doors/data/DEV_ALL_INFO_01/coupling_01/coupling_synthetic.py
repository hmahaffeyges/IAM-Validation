"""DEV-ALL-INFO-01, nearest-neighbor coupling (plan 49.5; written 2026-10-25 before any real sample's coupling is read). Constructed molecule files
(.pat): 200,000 held molecules of 10 calls on every autosome, each call unmethylated with chance 0.02, or with chance R right after an unmethylated
call (R = 0.02: independent; 0.1 and 0.3: clustered). Each file read by run_sample.py --dev-count-only as sorted B cells; the coupling is copied
from the bundle. Bars: independent reads 0.9 to 1.1; R 0.1 reads at least 2; the three in order. Usage: python3 coupling_synthetic.py WORKDIR OUT.txt
Run 2 (run 1, coupling_synthetic_run1.txt, disclosed: the coupling was pairs over the error share squared, which the held rule pulls to the
exact value 0.8728 under independence; it read 0.875. The coupling now uses each molecule's own expectation, 1 under independence)."""
import sys, os, gzip, json, subprocess, numpy as np
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True); g = np.random.default_rng(20261025)
RS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain", "MethylPhys_Interface")
R0 = json.load(open(os.path.join(RS, "..", "Runtime Matrices", "IAM_A_Positions", "hg19_cpg_chrom_ranges.json")))["ranges"]
L = ["after an error | coupling | error share | held error"]; got = []
for r in (0.02, 0.1, 0.3):
    f = os.path.join(W, f"c{r}.pat.gz"); rows = []
    for k in range(200000):
        c = f"chr{1 + k % 22}"; a0, b0 = R0[c]; i = a0 + 50 + (k // 22) * 40
        s, prev = [], False
        for j in range(10):
            e = g.random() < (r if prev else 0.02); s.append("T" if e else "C"); prev = e
        rows.append((c, i, "".join(s)))
    rows.sort(key=lambda x: (int(x[0][3:]), x[1]))
    with gzip.open(f, "wt") as h:
        for c, i, s in rows: h.write(f"{c}\t{i}\t{s}\t1\n")
    b = os.path.join(W, f"c{r}.json")
    subprocess.run([sys.executable, "run_sample.py", "--pat", os.path.abspath(f), "--specimen", "sorted b cells", "--id", f"cpl{r}", "--dev-count-only",
                    "--out", os.path.join(W, f"c{r}.html"), "--bundle", b], cwd=RS, capture_output=True, text=True, check=True)
    q = json.load(open(b))["iam_a_count_only"]["count"]; got.append(q["coupling"])
    L.append(f"{r} | {q['coupling']:.3f} | {q['coupling_err_share']:.4f} | {q['held_err']:.4f}")
ok = 0.9 <= got[0] <= 1.1 and got[1] >= 2 and got[0] < got[1] < got[2]
L.append(f"bars (independent 0.9-1.1; R 0.1 at least 2; in order): {'met' if ok else 'not met'}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
