"""IAM Sequencing Atlas (plan 35.30), step 1 synthetic: written and committed before it runs. The instrument's job: the atlas reports each
cell's healthy level where donors were measured, leaves blank where they were not, and its two noise maps carry noise only.

Planted: two cells (b cells, cd8 t cells), four donors each (two in each sealed half), over the first 200,000 CpGs of the hg19 numbering.
Each site's planted level: 40 % near unmarked (0.03), 50 % near marked (0.90), 10 % anywhere; each donor's level is the site's level moved by a
donor term (normal, sd 0.02), kept in [0, 1]; reads per donor per site Poisson(20); methylated reads binomial; 5 % of sites have no reads in a
cell. Seed 20261029. Files are written as full-length hg19 .beta files (sparse, only the block holds data) and built by the one command
(run_sample.py --build seq-atlas).

Bars (all three must hold, in both cells):
 1. Levels: median |built level - mean planted donor level| at sites with reads, at most 0.02.
 2. Noise maps: halfA - halfB and readA - readB each average within 0.005 of 0; readA - readB's spread within 20 % of the spread counting
    predicts (sqrt(p(1 - p)(1/nA + 1/nB)) at each site, from the planted level and the reads in each read half).
 3. Blanks: every site with no reads in a cell is blank in all of that cell's level columns, and no site with reads is blank.
Usage: python3 seq_atlas_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain"))
HG19 = 28217448; NB = 200000; rng = np.random.default_rng(20261029)
u = rng.random(NB); L = np.where(u < 0.4, rng.beta(1, 30, NB), np.where(u < 0.9, rng.beta(30, 3.5, NB), rng.random(NB)))
rows, truth = [], {}
for cell in ("b cells", "cd8 t cells"):
    empty = rng.random(NB) < 0.05; P = []; Nn = []
    for d in range(4):
        p = np.clip(L + rng.normal(0, 0.02, NB), 0, 1); n = rng.poisson(20, NB); n[empty] = 0; m = rng.binomial(n, p)
        a = np.zeros((NB, 2), np.uint8); a[:, 0] = np.minimum(m, 255); a[:, 1] = np.minimum(n, 255)
        f = os.path.join(W, f"SYN_{cell.replace(' ', '_')}_{d}.beta")
        with open(f, "wb") as h: h.truncate(2 * HG19); h.seek(0); h.write(a.tobytes())
        rows.append(dict(gsm=f"SYN{len(rows)}", title=f"syn{d}", cell=cell, half="A" if d < 2 else "B", file=f)); P.append(p); Nn.append(n)
    truth[cell] = dict(p=np.mean(P, 0), n=np.array(Nn), empty=empty)
S = os.path.join(W, "samples.csv"); pd.DataFrame(rows).to_csv(S, index=False); O = os.path.join(W, "block.parquet")
r = subprocess.run([sys.executable, "run_sample.py", "--build", "seq-atlas", "--build-samples", S, "--build-block", f"0:{NB}", "--build-out", O, "--bundle", os.path.join(W, "build.json")],
                   cwd=os.path.join(CH, "MethylPhys_Interface"), capture_output=True, text=True)
if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
T = pd.read_parquet(O); B = 65535; L_ = ["IAM Sequencing Atlas step 1 synthetic (plan 35.30), built by run_sample.py --build seq-atlas", f"block 0-{NB:,}, 2 cells x 4 donors, seed 20261029"]
ok = []
for cell, t in truth.items():
    lv = T[f"{cell}|level"].to_numpy(); has = ~t["empty"]
    b1 = float(np.median(np.abs(lv[has] / 1e4 - t["p"][has])))
    hA, hB, rA, rB = (T[f"{cell}|{k}"].to_numpy().astype(float) / 1e4 for k in ("halfA", "halfB", "readA", "readB"))
    d1, d2 = (hA - hB)[has], (rA - rB)[has]
    nA = (t["n"] // 2).sum(0)[has]; nB = (t["n"] - t["n"] // 2).sum(0)[has]; pp = t["p"][has]
    pred = float(np.sqrt(np.mean(pp * (1 - pp) * (1 / nA + 1 / nB)))); obs = float(np.std(d2))
    b2 = abs(d1.mean()) <= 0.005 and abs(d2.mean()) <= 0.005 and abs(obs / pred - 1) <= 0.2
    cols = [f"{cell}|{k}" for k in ("level", "lo", "hi", "halfA", "halfB", "readA", "readB", "disorder")]
    b3 = all((T[c].to_numpy()[t["empty"]] == B).all() for c in cols) and not (lv[has] == B).any()
    ok += [b1 <= 0.02, b2, b3]
    L_ += [f"{cell}: bar 1 median |level - planted| {b1:.4f} ({'met' if b1 <= 0.02 else 'not met'})",
           f"{cell}: bar 2 halfA-halfB mean {d1.mean():+.5f}, readA-readB mean {d2.mean():+.5f}, spread {obs:.4f} vs counting {pred:.4f} ({'met' if b2 else 'not met'})",
           f"{cell}: bar 3 blanks where no reads, none where reads ({'met' if b3 else 'not met'})"]
L_.append(f"synthetic (all bars, both cells): {'met' if all(ok) else 'not met'}")
open(OUT, "w").write("\n".join(L_) + "\n"); print("\n".join(L_))
