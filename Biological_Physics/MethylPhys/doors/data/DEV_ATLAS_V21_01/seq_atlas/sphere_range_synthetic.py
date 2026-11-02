"""The IAM Sequencing Atlas's 95 % range of each cell's average (run_sample.py --build seq-sphere-cols; stage_seq_atlas.range_of_average;
checklist D4): synthetic, read by the one command (rewritten 2026-11-01 so the reading comes from run_sample.py; the first version called
the chain function directly). The instrument's job: the range a cell's average is given matches how far that average really scatters.
Known answer (seed 20261112): 20,000 sites, each with a true cell level (Beta(0.5, 0.5)) and a person-to-person spread around it
(logit-normal, sd 0.3, stated); for k = 1, 2 and 3 donors, each counted at Poisson(33) reads, built 200 times, each build one constructed
cell in the committed layout (meth, reads, donor_sd; the inputs list giving each cell its k donors). The truth is the persons' average
level. Bars, for k = 2 and 3, at sites whose truth lies between 0.1 and 0.9: mean stated variance (mu_se squared) over the true variance
of the level about its truth within 0.9 to 1.1; coverage of the truth by the range within 0.90 to 0.99. k = 1 recorded, not scored.
Usage: python3 sphere_range_synthetic.py WORKDIR OUT.txt"""
import os, sys, subprocess, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from scipy.special import expit, logit
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(H, "..", "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py")
W = sys.argv[1]; C = os.path.join(W, "cols"); O = os.path.join(W, "out"); os.makedirs(C, exist_ok=True)
rng = np.random.default_rng(20261112); S = 20000; T = 200; BL = 65535
mu = np.clip(rng.beta(0.5, 0.5, S), 0.002, 0.998); truth = np.mean(expit(logit(mu)[None, :] + rng.normal(0, 0.3, (4000, S))), 0)
ME, RD, SD, inp = {}, {}, {}, []
for k in (1, 2, 3):
    for t in range(T):
        c = f"k{k} build {t:03d}"; lv = expit(logit(mu)[None, :] + rng.normal(0, 0.3, (k, S))); n = rng.poisson(33, (k, S)).clip(1); m = rng.binomial(n, lv)
        ME[c] = m.sum(0).astype(np.uint16); RD[c] = n.sum(0).astype(np.uint16)
        SD[c] = np.rint(np.std(m / n, axis=0, ddof=1) * 1e4).astype(np.uint16) if k >= 2 else np.full(S, BL, np.uint16); inp += [dict(cell=c, gsm=f"{c} donor {j}") for j in range(k)]
names = list(ME); thirds = np.array_split(np.arange(len(names)), 3)
for stem, A in (("meth", ME), ("reads", RD), ("donor_sd", SD)):
    for p, js in enumerate(thirds, 1): pq.write_table(pa.table({names[j]: A[names[j]] for j in js}), os.path.join(C, f"{stem}_part{p}.parquet"))
pd.DataFrame(inp).to_csv(os.path.join(W, "inputs.csv"), index=False)
r = subprocess.run([sys.executable, RS, "--build", "seq-sphere-cols", "--build-input", C, "--build-samples", os.path.join(W, "inputs.csv"), "--build-out", O], capture_output=True, text=True)
if r.returncode: sys.exit(r.stdout[-800:] + r.stderr[-1500:])
dec = lambda a: np.where(a == BL, np.nan, a.astype("float64") / 1e4); G = {}
for p in (1, 2, 3):
    for stem in ("mu", "mu_lo", "mu_hi", "mu_se"):
        t_ = pq.read_table(os.path.join(O, f"{stem}_part{p}.parquet"))
        for c in t_.column_names: G[(stem, c)] = dec(t_.column(c).to_numpy())
inner = (truth > 0.1) & (truth < 0.9); res = {}; L = ["read by run_sample.py --build seq-sphere-cols on constructed cells (seed 20261112, 20,000 sites, 200 builds, Poisson(33) reads a donor, person sd 0.3 logit)"]
for k in (1, 2, 3):
    cs = [c for c in names if c.startswith(f"k{k} ")]
    est = np.stack([G[("mu", c)] for c in cs]); se = np.stack([G[("mu_se", c)] for c in cs]); lo = np.stack([G[("mu_lo", c)] for c in cs]); hi = np.stack([G[("mu_hi", c)] for c in cs])
    ratio = float((se[:, inner] ** 2).mean() / ((est[:, inner] - truth[inner]) ** 2).mean()); cov = float(((lo <= truth) & (truth <= hi))[:, inner].mean()); res[k] = (ratio, cov)
    L.append(f"k = {k} donors: stated over true variance {ratio:.3f}; coverage of the truth {cov:.3f}")
ok = all(0.9 <= res[k][0] <= 1.1 and 0.90 <= res[k][1] <= 0.99 for k in (2, 3))
L.append(f"bars (k = 2 and 3): variance ratio within 0.9 to 1.1 and coverage within 0.90 to 0.99: {'met' if ok else 'not met'}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
