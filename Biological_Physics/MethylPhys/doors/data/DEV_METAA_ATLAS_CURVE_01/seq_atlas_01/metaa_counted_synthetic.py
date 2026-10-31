"""Met-A against the IAM Sequencing Atlas with the reference counted (stage_metaa_atlas.expected_H_counted): synthetic, written and committed
before it runs. The instrument's job: a healthy sequenced cell reads Met-A 1 against its own cell's sequencing reference, at any depth.

Why a new synthetic: the first (metaa_seqatlas_synthetic.py) drew samples from the reference's own levels, so the reference was exact by
construction; the site-by-site diagnostic found 42 % of real unmarked identity sites at a reference level of exactly 0, counted from a few
hundred reads. Here the healthy level is a hidden truth and the reference is counted from it.
Known answer: naive CD4 T cells' identity sites; truth: the marked side uniform 0.80 to 0.99, the unmarked side Beta(0.3, 60) (mean 0.005,
most sites near 0); the reference's donor half B counted from the truth, mB ~ Binomial(nB, truth), nB ~ Poisson(150), its level mB / nB,
written as a constructed reference in the committed format with its counts. Samples: k ~ Binomial(n, truth), n ~ Poisson(depth), depths
5, 8, 12, 20, 30, 45, 60, three samples each, seed 20261105, uint16 .lbeta. Each read twice by the one command (run_sample.py --dev-seq-beta
FILE --dev-seq-values 1 --dev-metaa-atlas CELL --dev-seq-ref REFERENCE --dev-seq-ref-half B): with the counts (scored) and with the counts
removed, the reference level taken as exact (recorded, not scored). Bars: 1. median Met-A with the counts within 0.02 of 1; 2. |Spearman
rho| with depth at most 0.2. Usage: python3 metaa_counted_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from scipy.stats import spearmanr
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, pyarrow.parquet as pq
CELL = "naive cd4 t cells"; HG19 = 28217448; rng = np.random.default_rng(20261105)
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
S = [s for s in SA.identity_sites(CELL) if s in K.index]; mu = SA._col("mu", CELL).reindex(S)
tru = np.where(mu.to_numpy() > 0.5, rng.uniform(0.80, 0.99, len(S)), rng.beta(0.3, 60, len(S)))
nB = np.maximum(rng.poisson(150, len(S)), 1); mB = rng.binomial(nB, tru)
R = pd.DataFrame(dict(cell=CELL, probe=S, cpg=K.reindex(S).to_numpy(np.int64)))
for c in ("donors", "reads", "meth", "spread", "disorder", "halfA", "readA", "readB"): R[c] = np.uint16(65535)
R["level"] = np.round(mB / nB * 1e4).astype(np.uint16); R["halfB"] = R["level"]
REF0 = os.path.join(W, "ref_exact.parquet"); R.to_parquet(REF0, index=False)
R["mA"] = 0; R["nA"] = 0; R["mB"] = mB.astype(np.int64); R["nB"] = nB.astype(np.int64); REF1 = os.path.join(W, "ref_counted.parquet"); R.to_parquet(REF1, index=False)
IF = os.path.join(CH, "MethylPhys_Interface"); rows = []
def one(f, ref, tag):
    J = os.path.join(W, f"{tag}.json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted " + CELL, "--id", tag, "--dev-metaa-atlas", CELL,
                        "--dev-seq-ref", ref, "--dev-seq-ref-half", "B", "--out", os.path.join(W, "s.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-1500:], r.stderr[-1500:]); raise SystemExit(1)
    return (json.load(open(J)).get("development") or {}).get("metaa_atlas") or {}
for depth in (5, 8, 12, 20, 30, 45, 60):
    for j in range(3):
        n = rng.poisson(depth, len(S)); k = rng.binomial(n, tru); a = np.zeros((HG19, 2), np.uint16); idx = R.cpg.to_numpy() - 1
        a[idx, 0] = k; a[idx, 1] = n; f = os.path.join(W, "s.lbeta"); a.tofile(f)
        c1 = one(f, REF1, f"c{depth}_{j}"); c0 = one(f, REF0, f"e{depth}_{j}")
        rows.append(dict(depth=depth, rep=j, A=c1.get("A"), counted=c1.get("expectation_counted"), A_exact=c0.get("A"), reason=c1.get("reason"))); os.remove(f)
D = pd.DataFrame(rows); D.to_csv(os.path.join(W, "readings.csv"), index=False); ok = D.A.notna() & (D.counted == True)
med = float(D.A[ok].median()) if ok.any() else float("nan"); rho = spearmanr(D.depth[ok], D.A[ok]).correlation if ok.sum() > 2 else float("nan")
L = ["Met-A against the IAM Sequencing Atlas, the reference counted: synthetic, read by run_sample.py --dev-seq-ref --dev-seq-ref-half B",
     f"seed 20261105, {CELL}, {len(S)} identity sites, reference half B ~150 reads per site, 7 depths x 3 samples; read with the counts {int(ok.sum())} of {len(D)}",
     "Met-A by depth, with the counts: " + ", ".join(f"{d} {g.A.median():.4f}" for d, g in D[ok].groupby("depth")),
     "Met-A by depth, the reference level taken as exact (recorded, not scored): " + ", ".join(f"{d} {g.A_exact.median():.4f}" for d, g in D.groupby("depth")),
     f"bar 1 median Met-A {med:.4f} (within 0.02 of 1): {'met' if abs(med - 1) <= 0.02 else 'not met'}",
     f"bar 2 Spearman rho with depth {rho:+.3f} (|rho| at most 0.2): {'met' if abs(rho) <= 0.2 else 'not met'}",
     f"synthetic: {'met' if ok.all() and abs(med - 1) <= 0.02 and abs(rho) <= 0.2 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
