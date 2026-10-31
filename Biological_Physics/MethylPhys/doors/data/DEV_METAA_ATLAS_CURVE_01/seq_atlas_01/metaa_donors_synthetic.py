"""Met-A against the IAM Sequencing Atlas with the expectation taken donor by donor (stage_metaa_atlas.expected_H_donors, SEQ_LEVEL "donors"):
synthetic, written and committed before it runs. The instrument's job: a healthy sequenced person reads Met-A Normal against their cell's
healthy donors, at any depth.
Why: healthy people differ at a site. The pooled level of several donors has higher entropy than any one of them, so a person outside the
atlas read low against it (the outside neurons, 0.9191); the median of a few sparse donors sits at 0 or 1 and reads high.
Known answer: naive CD4 T cells' identity sites. At each site a population level (marked side uniform 0.80 to 0.99, unmarked Beta(0.3, 60));
each healthy person's own level drawn around it with real person-to-person spread (logit-normal, sd 0.6 on the logit scale). Three donors,
each counted at Poisson(40) reads; the constructed reference (committed format) holds their median (level) and pooled counts (meth, reads),
and the constructed per-donor file their own counts. Samples: new people from the same population, k ~ Binomial(n, own level),
n ~ Poisson(depth), depths 5, 8, 12, 20, 30, 45, 60, three each, seed 20261108, uint16 .lbeta. Each read by the one command
(run_sample.py --dev-seq-beta FILE --dev-seq-values 1 --dev-metaa-atlas CELL --dev-seq-ref REFERENCE --dev-seq-donors DONORS) donor by
donor (scored), with the pooled level and with the median (recorded). Bars: 1. median Met-A donor by donor within 0.05 of 1; 2. |Spearman
rho| with depth at most 0.2. Usage: python3 metaa_donors_synthetic.py WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.special import expit, logit
W, OUT = sys.argv[1:3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, pyarrow.parquet as pq
CELL = "naive cd4 t cells"; HG19 = 28217448; rng = np.random.default_rng(20261108)
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
S = [s for s in SA.identity_sites(CELL) if s in K.index]; mu = SA._col("mu", CELL).reindex(S)
pop = np.clip(np.where(mu.to_numpy() > 0.5, rng.uniform(0.80, 0.99, len(S)), rng.beta(0.3, 60, len(S))), 1e-4, 1 - 1e-4)
person = lambda: expit(logit(pop) + rng.normal(0, 0.6, len(S)))
pd_ = np.stack([person() for _ in range(3)], 1); nd = np.maximum(rng.poisson(40, (len(S), 3)), 1); kd = rng.binomial(nd, pd_)
cpg = K.reindex(S).to_numpy(np.int64)
R = pd.DataFrame(dict(cell=CELL, probe=S, cpg=cpg))
for c in ("spread", "disorder", "halfA", "halfB", "readA", "readB"): R[c] = np.uint16(65535)
R["donors"] = np.uint16(3); R["level"] = np.round(np.median(kd / nd, 1) * 1e4).astype(np.uint16); R["meth"] = kd.sum(1).astype(np.uint16); R["reads"] = nd.sum(1).astype(np.uint16)
REF = os.path.join(W, "ref.parquet"); R.to_parquet(REF, index=False)
DON = os.path.join(W, "donors.parquet")
pd.concat([pd.DataFrame(dict(cell=CELL, probe=S, cpg=cpg, gsm=f"D{j}", half="A", m=kd[:, j].astype(np.uint16), n=nd[:, j].astype(np.uint16))) for j in range(3)]).to_parquet(DON, index=False)
IF = os.path.join(CH, "MethylPhys_Interface"); rows = []
def one(f, tag, level):
    J = os.path.join(W, f"{tag}.json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted " + CELL, "--id", tag, "--dev-metaa-atlas", CELL,
                        "--dev-seq-ref", REF, "--dev-seq-donors", DON, "--dev-seq-level", level, "--out", os.path.join(W, "s.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
    m = (json.load(open(J)).get("development") or {}).get("metaa_atlas") or {} if os.path.exists(J) else {}
    if m.get("A") is None: print(tag, m.get("reason"), r.stdout[-400:], r.stderr[-400:])
    return m
for depth in (5, 8, 12, 20, 30, 45, 60):
    for j in range(3):
        own = person(); n = rng.poisson(depth, len(S)); k = rng.binomial(n, own); a = np.zeros((HG19, 2), np.uint16)
        a[cpg - 1, 0] = k; a[cpg - 1, 1] = n; f = os.path.join(W, "s.lbeta"); a.tofile(f)
        d = one(f, f"d{depth}_{j}", "donors"); p = one(f, f"p{depth}_{j}", "pooled"); md = one(f, f"m{depth}_{j}", "median")
        rows.append(dict(depth=depth, rep=j, A=d.get("A"), how=d.get("expectation"), A_pooled=p.get("A"), A_median=md.get("A"))); os.remove(f)
D = pd.DataFrame(rows); D.to_csv(os.path.join(W, "readings.csv"), index=False); ok = D.A.notna() & D.how.astype(str).str.contains("donor by donor")
med = float(D.A[ok].median()) if ok.any() else float("nan"); rho = spearmanr(D.depth[ok], D.A[ok]).correlation if ok.sum() > 2 else float("nan")
L = ["Met-A against the IAM Sequencing Atlas, the expectation donor by donor: synthetic, read by run_sample.py --dev-seq-ref FILE --dev-seq-donors FILE (--dev-seq-level)",
     f"seed 20261108, {CELL}, {len(S)} identity sites, three donors near 40 reads each, person-to-person spread sd 0.6 (logit); read donor by donor {int(ok.sum())} of {len(D)}",
     "Met-A by depth, donor by donor: " + ", ".join(f"{d} {g.A.median():.4f}" for d, g in D[ok].groupby("depth")),
     "Met-A by depth, the pooled level (recorded, not scored): " + ", ".join(f"{d} {g.A_pooled.median():.4f}" for d, g in D.groupby("depth")),
     "Met-A by depth, the median of the donors (recorded, not scored): " + ", ".join(f"{d} {g.A_median.median():.4f}" for d, g in D.groupby("depth")),
     f"bar 1 median Met-A {med:.4f} (within 0.05 of 1): {'met' if abs(med - 1) <= 0.05 else 'not met'}",
     f"bar 2 Spearman rho with depth {rho:+.3f} (|rho| at most 0.2): {'met' if abs(rho) <= 0.2 else 'not met'}",
     f"synthetic: {'met' if ok.all() and abs(med - 1) <= 0.05 and abs(rho) <= 0.2 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
