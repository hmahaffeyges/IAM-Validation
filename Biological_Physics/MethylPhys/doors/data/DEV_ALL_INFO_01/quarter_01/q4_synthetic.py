"""DEV-ALL-INFO-01 quarter-depth test, synthetic (written 2026-10-25 before any quarter reading is made). A spent molecule file (a .pat from the committed
pipeline) and the same file keeping each molecule with chance 1/4 (binomial on each line's count,
seed 20261025), both read by run_sample.py under the same specimen. The quarter must read IAM-A and Met-A whole sample within 0.02 of the whole file.
Usage: python3 q4_synthetic.py PAT.gz WORKDIR OUT.txt [SPECIMEN]
Run 2 (run 1, disclosed, q4_synthetic_run1.txt: on GSM2387333, a shallow file, the quarter was withheld by the sample-shape check, so the
synthetic moved to a file at this study's full depth: GSM4914785's 20 chunks merged, spent)."""
import sys, os, gzip, json, subprocess, numpy as np
P, W, OUT = sys.argv[1:4]; SP = sys.argv[4] if len(sys.argv) > 4 else "sorted b cells"; os.makedirs(W, exist_ok=True); g = np.random.default_rng(20261025)
RS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain", "MethylPhys_Interface")
q = os.path.join(W, "quarter.pat.gz"); kept = tot = 0
with gzip.open(P, "rt") as fi, gzip.open(q, "wt") as fo:
    for l in fi:
        f = l.rstrip("\n").split("\t"); n = int(f[3]); k = int(g.binomial(n, 0.25)); tot += n; kept += k
        if k: f[3] = str(k); fo.write("\t".join(f) + "\n")
R = {}
for nm, f in (("whole", P), ("quarter", q)):
    b = os.path.join(W, nm + ".json")
    subprocess.run([sys.executable, "run_sample.py", "--pat", os.path.abspath(f), "--specimen", SP, "--id", "q4" + nm, "--dev-seq-values", "1",
                    "--out", os.path.join(W, nm + ".html"), "--bundle", b], cwd=RS, capture_output=True, text=True, check=True)
    B = json.load(open(b)); R[nm] = ((B.get("iam_a") or {}).get("A"), (B.get("met_a_whole") or {}).get("A"))
L = [f"molecules: whole {tot} | quarter {kept} ({kept / tot:.4f})", "reading | whole | quarter | diff"]
ok = True
for i, nm in enumerate(("IAM-A", "Met-A whole sample")):
    a, b = R["whole"][i], R["quarter"][i]
    if a is None or b is None: L.append(f"{nm} | {a} | {b} | missing"); ok = False; continue
    L.append(f"{nm} | {a:.4f} | {b:.4f} | {b - a:+.4f}"); ok &= abs(b - a) <= 0.02
L.append(f"synthetic (both within 0.02): {'met' if ok else 'not met'}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
