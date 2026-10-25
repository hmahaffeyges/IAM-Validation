"""DEV-ERASE-01, version 2 for bar 3 at 50,000-CpG tiles (written 2026-10-25 before the deep pair is read): 2,500 tiles of 2,000 CpGs, the planted E
constant over each run of 25 tiles (100 summed tiles), 400 molecules per tile. Was: DEV-ERASE-01, the graded maintenance test, bar 1 (synthetic; written 2026-10-25 before any count): constructed molecules through the one
command. 600 tiles of 2,000 CpGs on one chromosome, each with a planted held-mark E drawn evenly from 2.6 to 4.5 (seed 20261025); 1,500
molecules of 10 CpGs per tile, every call methylated except where an error is planted. Each call is unmethylated with probability p, chosen so that the lone-error rate the counter reads on the molecules it counts as held (its own rule,
enumerated over all 1,024 patterns) is exactly e = 1/(1 + exp(E)).
Three samples: the control; E shifted by -0.7 in every tile (R: slope 1, intercept -0.7); E scaled by 0.8 (Phi: slope 0.8, intercept 0).
Each read by run_sample.py --dev-count-only --dev-count-tiles; then graded_score.py on both pairs.  Usage: python3 graded_synthetic.py WORKDIR OUT.txt"""
import sys, os, subprocess, gzip, numpy as np
from scipy.optimize import brentq
W, OUT = sys.argv[1], sys.argv[2]; os.makedirs(W, exist_ok=True); H = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface")
sys.path.insert(0, RS + "/.."); import stage_q2_selfref as Q2, itertools
PAT = np.array(list(itertools.product((0, 1), repeat=10)))   # 1 = unmethylated call
KEEP = (10 - PAT.sum(1)) >= Q2.QUAL_MIN_METH * 10   # the counter's held-molecule rule (QUAL_MIN_CALLS <= 10)
ISO = np.array([sum(1 for q in range(1, 9) if r[q] and not r[q - 1] and not r[q + 1]) for r in PAT])
def counted(x):   # the lone-error rate the counter reads when each call is unmethylated with probability x (second run, disclosed: the first planted
    w = x ** PAT.sum(1) * (1 - x) ** (10 - PAT.sum(1)); return (w * ISO)[KEEP].sum() / (8 * w[KEEP].sum())   # p(1-p)^2, ignoring that heavily hit molecules are not held)
g = np.random.default_rng(20261025); E0 = np.repeat(g.uniform(2.6, 4.5, 100), 25)   # from 2.6 (first run from 2.0 stopped: a shift of -0.7 below about 1.75 asks for a lone-error rate above the 4/27 a molecule can show)
def write(name, E):
    with gzip.open(os.path.join(W, name + ".pat.gz"), "wt") as f:
        for t, e_ in enumerate(E):
            e = 1 / (1 + np.exp(e_)); p = brentq(lambda x: counted(x) - e, 1e-6, 0.99)   # counted() rises to 0.155 (E 1.69); every planted E here is above it
            st = g.integers(0, 2000 - 10, 400); m = g.random((400, 10)) < p
            for s0, row in sorted(zip(st, m), key=lambda z: z[0]): f.write(f"chr1\t{1 + t * 2000 + s0}\t{''.join('T' if x else 'C' for x in row)}\t1\n")
for name, E in (("SYN_CTRL", E0), ("SYN_SHIFT", E0 - 0.7), ("SYN_SCALE", E0 * 0.8)):
    write(name, E); P = os.path.join(W, name + ".pat.gz")
    r = subprocess.run([sys.executable, "run_sample.py", "--pat", P, "--specimen", "cultured cells", "--seq-cell", name, "--id", name, "--dev-count-only",
                        "--dev-count-genome", "mm10", "--dev-count-tiles", os.path.join(W, name + "_tiles.csv.gz"), "--out", os.path.join(W, name + ".html"),
                        "--bundle", os.path.join(W, name + ".json")], cwd=RS, capture_output=True, text=True); print(name, "exit", r.returncode, r.stdout[-200:] if r.returncode else "", flush=True)
r = subprocess.run([sys.executable, os.path.join(H, "graded_score.py"), W, OUT, "SYN_CTRL", "SYN_SHIFT", "SYN_CTRL", "SYN_SCALE"], capture_output=True, text=True); print(r.stdout, r.stderr[-500:])
open(OUT, "a").write("planted: SYN_SHIFT slope 1, intercept -0.7 (R); SYN_SCALE slope 0.8, intercept 0 (Phi); bar 1 met if each is recovered within 0.05 and called correctly\n")
