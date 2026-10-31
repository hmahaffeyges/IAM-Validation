"""Met-A against the IAM Sequencing Atlas with the pooled level as the healthy expectation (stage_metaa_atlas.SEQ_LEVEL = "pooled"):
synthetic, written and committed before it runs. The instrument's job: a healthy sequenced cell reads Met-A Normal against its own cell's
sequencing reference, at any depth.
Why: the atlas stores each cell's level as the median of its donors' own levels; with 1 to 3 donors of a few dozen reads, the median sits at
exactly 0 or 1 where a donor has no read the other way (86 % of the B cells' unmarked identity sites, against 49 % for the pooled count), so the
expected disorder there is 0 and a healthy sample reads high. The pooled count (methylated over all reads) estimates the cell's methylation
probability, of which a healthy sample's reads are draws.
Known answer: naive CD4 T cells' identity sites; truth: marked side uniform 0.80 to 0.99, unmarked side Beta(0.3, 60) (mean 0.005); three
donors, each k ~ Binomial(n, truth), n ~ Poisson(30) (pooled near 90 reads, the atlas's median at identity sites is 88); the constructed
reference in the committed format holds level = the donors' median and meth, reads = their pooled counts. Samples: k ~ Binomial(n, truth),
n ~ Poisson(depth), depths 5, 8, 12, 20, 30, 45, 60, three each, seed 20261107, uint16 .lbeta. Each read by the one command (run_sample.py
--dev-seq-beta FILE --dev-seq-values 1 --dev-metaa-atlas CELL --dev-seq-ref REFERENCE) with the pooled level (scored) and with the median
(--dev-seq-level median; recorded, not scored). Bars: 1. median Met-A with the pooled level within 0.05 of 1 (Normal); 2. |Spearman rho| with
depth at most 0.2. Usage: python3 metaa_pooled_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from scipy.stats import spearmanr
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, pyarrow.parquet as pq
CELL = "naive cd4 t cells"; HG19 = 28217448; rng = np.random.default_rng(20261107)
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
S = [s for s in SA.identity_sites(CELL) if s in K.index]; mu = SA._col("mu", CELL).reindex(S)
tru = np.where(mu.to_numpy() > 0.5, rng.uniform(0.80, 0.99, len(S)), rng.beta(0.3, 60, len(S)))
nd = np.maximum(rng.poisson(30, (len(S), 3)), 1); kd = rng.binomial(nd, tru[:, None])
R = pd.DataFrame(dict(cell=CELL, probe=S, cpg=K.reindex(S).to_numpy(np.int64)))
for c in ("spread", "disorder", "halfA", "halfB", "readA", "readB"): R[c] = np.uint16(65535)
R["donors"] = np.uint16(3); R["level"] = np.round(np.median(kd / nd, 1) * 1e4).astype(np.uint16)
R["meth"] = kd.sum(1).astype(np.uint16); R["reads"] = nd.sum(1).astype(np.uint16)
REF = os.path.join(W, "ref.parquet"); R.to_parquet(REF, index=False)
IF = os.path.join(CH, "MethylPhys_Interface"); rows = []
def one(f, tag, level):
    J = os.path.join(W, f"{tag}.json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted " + CELL, "--id", tag, "--dev-metaa-atlas", CELL,
                        "--dev-seq-ref", REF, "--dev-seq-level", level, "--out", os.path.join(W, "s.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
    m = (json.load(open(J)).get("development") or {}).get("metaa_atlas") or {} if os.path.exists(J) else {}
    if m.get("A") is None: print(tag, m.get("reason"), r.stdout[-600:], r.stderr[-600:])
    return m
for depth in (5, 8, 12, 20, 30, 45, 60):
    for j in range(3):
        n = rng.poisson(depth, len(S)); k = rng.binomial(n, tru); a = np.zeros((HG19, 2), np.uint16); idx = R.cpg.to_numpy() - 1
        a[idx, 0] = k; a[idx, 1] = n; f = os.path.join(W, "s.lbeta"); a.tofile(f)
        p = one(f, f"p{depth}_{j}", "pooled"); md = one(f, f"m{depth}_{j}", "median")
        rows.append(dict(depth=depth, rep=j, A=p.get("A"), ref_pooled=p.get("reference"), A_median=md.get("A"), ref_median=md.get("reference"))); os.remove(f)
D = pd.DataFrame(rows); D.to_csv(os.path.join(W, "readings.csv"), index=False); ok = D.A.notna() & D.ref_pooled.astype(str).str.contains("pooled")
med = float(D.A[ok].median()) if ok.any() else float("nan"); rho = spearmanr(D.depth[ok], D.A[ok]).correlation if ok.sum() > 2 else float("nan")
L = ["Met-A against the IAM Sequencing Atlas with the pooled level: synthetic, read by run_sample.py --dev-seq-ref FILE (--dev-seq-level)",
     f"seed 20261107, {CELL}, {len(S)} identity sites, three donors near 30 reads each; read with the pooled level {int(ok.sum())} of {len(D)}",
     f"the reference: level at exactly 0 or 1 on {float(((R.level == 0) | (R.level == 10000)).mean()):.3f} of sites (median of donors), pooled at exactly 0 or 1 on {float(((R.meth == 0) | (R.meth == R.reads)).mean()):.3f}",
     "Met-A by depth, the pooled level: " + ", ".join(f"{d} {g.A.median():.4f}" for d, g in D[ok].groupby("depth")),
     "Met-A by depth, the median of the donors (recorded, not scored): " + ", ".join(f"{d} {g.A_median.median():.4f}" for d, g in D.groupby("depth")),
     f"bar 1 median Met-A {med:.4f} (within 0.05 of 1): {'met' if abs(med - 1) <= 0.05 else 'not met'}",
     f"bar 2 Spearman rho with depth {rho:+.3f} (|rho| at most 0.2): {'met' if abs(rho) <= 0.2 else 'not met'}",
     f"synthetic: {'met' if ok.all() and abs(med - 1) <= 0.05 and abs(rho) <= 0.2 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
