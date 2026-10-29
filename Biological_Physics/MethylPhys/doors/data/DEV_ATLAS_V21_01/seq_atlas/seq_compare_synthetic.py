"""IAM Sequencing Atlas (plan 35.30): synthetic for the identity comparison (run_sample.py --build seq-compare), written and committed before it
runs. The instrument's job: a new healthy donor of a cell reads nearest that cell.

Planted: three cells, each with its own level at every one of the first 400,000 CpGs (a third near unmarked, half near marked, the rest
anywhere; each cell differs from the others at a random 10 % of sites), four donors each (two in each sealed half), donor term sd 0.02, reads
Poisson(20), seed 20261031. The mini-atlas is built by the one command (--build seq-atlas, two pieces); then, for each cell, a fifth donor
(not an input, sparse: Poisson(8) reads) is compared by the one command (--build seq-compare).
Bars: 1. each new donor's nearest cell is its own cell; 2. the comparison counts exactly the CpGs where the donor has at least 5 reads and
every cell has a level.
Usage: python3 seq_compare_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); IF = os.path.join(CH, "MethylPhys_Interface")
HG19 = 28217448; NB = 400000; rng = np.random.default_rng(20261031)
u = rng.random(NB); base = np.where(u < 1 / 3, rng.beta(1, 30, NB), np.where(u < 5 / 6, rng.beta(30, 3.5, NB), rng.random(NB)))
CELLS = ("cell one", "cell two", "cell three"); L = {}
for c in CELLS:
    x = base.copy(); k = rng.random(NB) < 0.10; x[k] = 1 - x[k]; L[c] = x
def write(path, lv, lam):
    p = np.clip(lv + rng.normal(0, 0.02, NB), 0, 1); n = rng.poisson(lam, NB); m = rng.binomial(n, p)
    a = np.zeros((NB, 2), np.uint8); a[:, 0] = np.minimum(m, 255); a[:, 1] = np.minimum(n, 255)
    with open(path, "wb") as h: h.truncate(2 * HG19); h.seek(0); h.write(a.tobytes())
    return n
rows = []
for c in CELLS:
    for d in range(4):
        f = os.path.join(W, f"{c.replace(' ', '_')}_{d}.beta"); write(f, L[c], 20); rows.append(dict(gsm=f"S{len(rows)}", title=f"d{d}", cell=c, half="A" if d < 2 else "B", file=f))
S = os.path.join(W, "samples.csv"); pd.DataFrame(rows).to_csv(S, index=False); PD = os.path.join(W, "pieces"); os.makedirs(PD, exist_ok=True)
def one(args):
    r = subprocess.run([sys.executable, "run_sample.py"] + args, cwd=IF, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
for s0 in (0, NB // 2): one(["--build", "seq-atlas", "--build-samples", S, "--build-block", f"{s0}:{s0 + NB // 2}", "--build-out", os.path.join(PD, f"piece_{s0:08d}_{s0 + NB // 2:08d}.parquet")])
T = pd.concat([pd.read_parquet(os.path.join(PD, f)) for f in sorted(os.listdir(PD))]); allc = np.ones(NB, bool)
for c in CELLS: allc &= T[f"{c}|level"].to_numpy() != 65535
out = ["IAM Sequencing Atlas: identity comparison synthetic, built and read by run_sample.py --build seq-atlas and --build seq-compare", f"seed 20261031, {NB:,} CpGs, 3 cells x 4 donors"]; ok = []
for c in CELLS:
    f = os.path.join(W, f"new_{c.replace(' ', '_')}.beta"); n = write(f, L[c], 8); J = os.path.join(W, f"cmp_{c.replace(' ', '_')}.json")
    one(["--build", "seq-compare", "--build-input", f, "--build-samples", PD, "--bundle", J]); R = json.load(open(J))
    want = int(((n >= 5) & allc).sum()); b1, b2 = R["ranked"][0] == c, R["sites"] == want; ok += [b1, b2]
    out.append(f"{c}: nearest {R['ranked'][0]} ({'met' if b1 else 'not met'}) | distances " + ", ".join(f"{k} {R['cells'][k]['distance']:.4f}" for k in R["ranked"])
               + f" | halves apart {R['cells'][c].get('halves_apart')} | sites {R['sites']:,} vs {want:,} ({'met' if b2 else 'not met'})")
out.append(f"synthetic (both bars, three cells): {'met' if all(ok) else 'not met'}")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
