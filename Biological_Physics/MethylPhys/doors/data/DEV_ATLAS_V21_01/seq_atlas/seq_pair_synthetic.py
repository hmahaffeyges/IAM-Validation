"""IAM Sequencing Atlas (plan 35.30): synthetic for the pair comparison (run_sample.py --build seq-pair), written and committed before it
runs. The instrument's job: two reads of the same cell score about 1; two different cells score far above. Known answer: 400,000 CpGs, a
cell's level at each (a third near unmarked, half near marked, the rest anywhere) and a second cell differing at a random 10 %; three files at
Poisson(3) reads (sparse, as single-cell pools are): two of the first cell, one of the second, seed 20261106, uint8 .beta files.
Bars: 1. same cell weighted score within 0.9 to 1.1; 2. the two different cells score at least 1.5. Usage: python3 seq_pair_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); IF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain", "MethylPhys_Interface"))
HG19 = 28217448; NB = 400000; rng = np.random.default_rng(20261106)
u = rng.random(NB); a = np.where(u < 1 / 3, rng.beta(1, 30, NB), np.where(u < 5 / 6, rng.beta(30, 3.5, NB), rng.random(NB))); b = a.copy(); k = rng.random(NB) < 0.10; b[k] = 1 - b[k]
def write(path, lv):
    n = rng.poisson(3, NB); m = rng.binomial(n, lv); x = np.zeros((NB, 2), np.uint8); x[:, 0] = np.minimum(m, 255); x[:, 1] = np.minimum(n, 255)
    with open(path, "wb") as h: h.truncate(2 * HG19); h.seek(0); h.write(x.tobytes())
F = {nm: os.path.join(W, nm + ".beta") for nm in ("a1", "a2", "b1")}; write(F["a1"], a); write(F["a2"], a); write(F["b1"], b)
def pair(x, y):
    J = os.path.join(W, "p.json"); r = subprocess.run([sys.executable, "run_sample.py", "--build", "seq-pair", "--build-input", F[x], "--build-samples", F[y], "--bundle", J], cwd=IF, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
    return json.load(open(J))
s, d = pair("a1", "a2"), pair("a1", "b1"); b1, b2 = 0.9 <= s["weighted"] <= 1.1, d["weighted"] >= 1.5
L = ["IAM Sequencing Atlas: pair comparison synthetic, read by run_sample.py --build seq-pair", "seed 20261106, 400,000 CpGs, reads Poisson(3)",
     f"same cell: weighted {s['weighted']} (0.9 to 1.1: {'met' if b1 else 'not met'}), distance {s['distance']}, sites {s['sites']:,}",
     f"different cells: weighted {d['weighted']} (at least 1.5: {'met' if b2 else 'not met'}), distance {d['distance']}, sites {d['sites']:,}",
     f"synthetic: {'met' if b1 and b2 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
