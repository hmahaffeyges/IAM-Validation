"""IAM Sequencing Atlas (plan 35.30): synthetic 2 for the identity comparison, columns of unequal depth (written and committed before it runs;
after donor 3's test showed a column pooled from sparse reads sits far from its own cell on the plain distance).
The instrument's job: a new healthy donor of a cell reads nearest that cell, however deep or shallow that cell's column is.

Planted: four cells, each with its own level at every one of the first 400,000 CpGs (a third near unmarked, half near marked, the rest
anywhere; each cell differs from the shared base at a random 10 % of sites). Three deep cells: four donors each (two per sealed half),
reads Poisson(20). "cell shallow": two donors (one per half), reads Poisson(1.5), as a column pooled from sparse single-cell reads. Donor
term sd 0.02, seed 20261103. The mini-atlas is built by the one command (--build seq-atlas); a new donor of each cell (Poisson(8) reads) is
compared by the one command (--build seq-compare).
Bars: 1. by the depth-weighted score, each new donor's nearest cell is its own, the shallow cell's included; 2. the comparison counts exactly
the CpGs where the donor has at least 5 reads and every cell has a level. The plain distance is shown beside it, not scored.
Usage: python3 seq_compare_synthetic2.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); IF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain", "MethylPhys_Interface"))
HG19 = 28217448; NB = 400000; rng = np.random.default_rng(20261103)
u = rng.random(NB); base = np.where(u < 1 / 3, rng.beta(1, 30, NB), np.where(u < 5 / 6, rng.beta(30, 3.5, NB), rng.random(NB)))
CELLS = ("cell one", "cell two", "cell three", "cell shallow"); L = {}
for c in CELLS:
    x = base.copy(); k = rng.random(NB) < 0.10; x[k] = 1 - x[k]; L[c] = x
def write(path, lv, lam):
    p = np.clip(lv + rng.normal(0, 0.02, NB), 0, 1); n = rng.poisson(lam, NB); m = rng.binomial(n, p)
    a = np.zeros((NB, 2), np.uint8); a[:, 0] = np.minimum(m, 255); a[:, 1] = np.minimum(n, 255)
    with open(path, "wb") as h: h.truncate(2 * HG19); h.seek(0); h.write(a.tobytes())
    return n
rows = []
for c in CELLS:
    nd = 2 if c == "cell shallow" else 4
    for d in range(nd):
        f = os.path.join(W, f"{c.replace(' ', '_')}_{d}.beta"); write(f, L[c], 1.5 if c == "cell shallow" else 20)
        rows.append(dict(gsm=f"S{len(rows)}", title=f"d{d}", cell=c, half="A" if d < nd // 2 else "B", file=f))
S = os.path.join(W, "samples.csv"); pd.DataFrame(rows).to_csv(S, index=False); PD = os.path.join(W, "pieces"); os.makedirs(PD, exist_ok=True)
def one(args):
    r = subprocess.run([sys.executable, "run_sample.py"] + args, cwd=IF, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
for s0 in (0, NB // 2): one(["--build", "seq-atlas", "--build-samples", S, "--build-block", f"{s0}:{s0 + NB // 2}", "--build-out", os.path.join(PD, f"piece_{s0:08d}_{s0 + NB // 2:08d}.parquet")])
T = pd.concat([pd.read_parquet(os.path.join(PD, f)) for f in sorted(os.listdir(PD))]); allc = np.ones(NB, bool)
for c in CELLS: allc &= T[f"{c}|level"].to_numpy() != 65535
out = ["IAM Sequencing Atlas: identity comparison synthetic 2 (unequal depth), built and read by run_sample.py --build seq-atlas and --build seq-compare",
       f"seed 20261103, {NB:,} CpGs, 3 cells x 4 donors (reads Poisson 20) and a shallow cell of 2 donors (Poisson 1.5)"]; ok = []
for c in CELLS:
    f = os.path.join(W, f"new_{c.replace(' ', '_')}.beta"); n = write(f, L[c], 8); J = os.path.join(W, f"cmp_{c.replace(' ', '_')}.json")
    one(["--build", "seq-compare", "--build-input", f, "--build-samples", PD, "--bundle", J]); R = json.load(open(J))
    want = int(((n >= 5) & allc).sum()); b1, b2 = R["weighted_ranked"][0] == c, R["sites"] == want; ok += [b1, b2]
    out.append(f"{c}: weighted nearest {R['weighted_ranked'][0]} ({'met' if b1 else 'not met'}) | weighted " + ", ".join(f"{k} {R['cells'][k]['weighted']:.3f}" for k in R["weighted_ranked"])
               + f" | plain nearest {R['ranked'][0]} | plain " + ", ".join(f"{k} {R['cells'][k]['distance']:.4f}" for k in R["ranked"])
               + f" | sites {R['sites']:,} vs {want:,} ({'met' if b2 else 'not met'})")
out.append(f"synthetic 2 (both bars, four cells): {'met' if all(ok) else 'not met'}")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
