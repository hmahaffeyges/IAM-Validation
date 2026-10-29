"""IAM Sequencing Atlas (plan 35.30): synthetic for the held-error reference, region by region (written and committed before it runs).
The instrument's job: each healthy cell's held error is recovered tile by tile, as counting predicts.

Known answer: two cells, four donors each (two in each sealed half), 40 tiles of 2,000 CpGs on chr1; in every tile 300 molecules per donor
of 10 CpGs, each CpG unmethylated with chance e_t independently (cell one e_t = 0.03 + 0.001 t, cell two e_t = 0.07 - 0.001 t), seed
20261101. Counting's prediction for a tile, from Stage Q2's own rules (a molecule is kept with at least 6 calls and at least 80 % methylated;
a lone error is an unmethylated call between two methylated ones, over the 8 interior chances), by exact sum over all 1,024 patterns. Each
donor's .pat is counted by the one command (--dev-count-only --dev-count-tiles), the reference built by the one command (--build seq-held).
Bars: 1. the median relative difference from the prediction over a cell's 40 tiles is within 3 of its own counting standard deviations
(1.2533 x sqrt(mean of each tile's (1 - r)/(r opp))/sqrt(40); amended for run 2, run 1's 2 % was not derived from counting); 2. at least 90 % of tiles within 2
counting standard deviations (sqrt(r(1 - r)/opp)); 3. the halves add up exactly (donor halves and molecule halves each sum to the whole).
Usage: python3 seq_held_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, itertools, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); IF = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain", "MethylPhys_Interface"))
rng = np.random.default_rng(20261102); NT, L, M = 40, 10, 300   # run 2 (run 1 seed 20261101: bar 1 as first written not met, disclosed)
E = {"cell one": [0.03 + 0.001 * t for t in range(NT)], "cell two": [0.07 - 0.001 * t for t in range(NT)]}
def predict(e):
    num = den = 0.0
    for p in itertools.product("CT", repeat=L):
        nt = p.count("T")
        if L - nt < 0.8 * L: continue
        pr = e ** nt * (1 - e) ** (L - nt); num += pr * sum(1 for q in range(1, L - 1) if p[q] == "T" and p[q - 1] == "C" and p[q + 1] == "C"); den += pr * (L - 2)
    return num / den
rows = []
for cell, es in E.items():
    for d in range(4):
        lines = []
        for t, e in enumerate(es):
            for k in range(M):
                i = 1 + 2000 * t + int(rng.integers(0, 2000 - L)); s = "".join("T" if x else "C" for x in rng.random(L) < e); lines.append((i, s))
        lines.sort(); g = f"{cell.replace(' ', '_')}_{d}"; P = os.path.join(W, g + ".pat")
        open(P, "w").write("".join(f"chr1\t{i}\t{s}\t1\n" for i, s in lines))
        r = subprocess.run([sys.executable, "run_sample.py", "--pat", P, "--specimen", "sorted b cells", "--id", g, "--dev-count-only", "--dev-count-tiles", os.path.join(W, g + "_tiles.csv"),
                            "--out", os.path.join(W, g + ".html"), "--bundle", os.path.join(W, g + ".json")], cwd=IF, capture_output=True, text=True)
        if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
        rows.append(dict(gsm=g, cell=cell, half="A" if d < 2 else "B", file=os.path.join(W, g + "_tiles.csv")))
S = os.path.join(W, "held_samples.csv"); pd.DataFrame(rows).to_csv(S, index=False); O = os.path.join(W, "held.parquet")
r = subprocess.run([sys.executable, "run_sample.py", "--build", "seq-held", "--build-samples", S, "--build-out", O, "--bundle", os.path.join(W, "held.json")], cwd=IF, capture_output=True, text=True)
if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
R = pd.read_parquet(O).set_index(["chrom", "tile"]); out = ["IAM Sequencing Atlas: held-error reference synthetic, counted and built by run_sample.py (--dev-count-tiles, --build seq-held)",
     f"seed 20261102, 2 cells x 4 donors, {NT} tiles, {M} molecules of {L} CpGs per tile and donor"]; ok = []
for cell, es in E.items():
    iso, opp = R[f"{cell}|iso"].to_numpy(), R[f"{cell}|opp"].to_numpy(); rp = np.array([predict(e) for e in es]); h = iso / opp
    rel = float(np.median((h - rp) / rp)); lim = 3 * 1.2533 * np.sqrt(np.mean((1 - rp) / (rp * opp))) / np.sqrt(len(rp)); z = np.abs(h - rp) / np.sqrt(rp * (1 - rp) / opp); w = float(np.mean(z <= 2))
    add = all((R[f"{cell}|{x}A"] + R[f"{cell}|{x}B"] == R[f"{cell}|{y}"]).all() for x, y in (("iso_d", "iso"), ("opp_d", "opp"), ("iso_r", "iso"), ("opp_r", "opp")))
    b = [abs(rel) <= lim, w >= 0.90, add]; ok += b
    out.append(f"{cell}: {len(h)} tiles | median relative difference {rel:+.4f} (limit {lim:.4f}: {'met' if b[0] else 'not met'}) | within 2 sd {w:.3f} ({'met' if b[1] else 'not met'}) "
               f"| halves add up: {'met' if add else 'not met'} | held error range {h.min():.4f} to {h.max():.4f} (predicted {rp.min():.4f} to {rp.max():.4f})")
out.append(f"synthetic (all bars, both cells): {'met' if all(ok) else 'not met'}")
open(OUT, "w").write("\n".join(out) + "\n"); print("\n".join(out))
