"""Can the IAM-Deconvolver tell naive from memory CD4 T cells on sequencing? Synthetic, written and committed before it runs, through the one
command (development; plan 60.3). Why: the domains and the whole-sample reading of lymphocytes follow the naive or memory make-up of a sample
(BLUEPRINT, inside one laboratory), but the Deconvolver's measured memory share tracked it only in part (DEV_ATLAS_V21_01.md, 11-07); its
naive and memory CD4 columns rest on one donor each. The instrument's job: the Deconvolver reports the naive and memory CD4 shares a sample holds.
One grid row per run (subset_grid.csv): a constructed sequenced sample whose truth at IAM-Atlas v2.1's sites is a mix of the IAM Sequencing
Atlas's own columns (array_sites): CD4 at share 0.3 or 0.7, split between naive CD4 and memory CD4 (central and effector memory, half each)
at memory fraction 0, 0.25, 0.5, 0.75 or 1; the rest monocytes 0.4 of it, NK cells 0.3, B cells 0.3. Each site's level varied as a person
(logit-normal, sd 0.3, stated), reads Poisson(lam) per site at lam 3, 15 or 40. Read by run_sample.py --dev-seq-beta --dev-seq-values 1
--dev-seq-ref auto --cell-panel atlas --specimen "sorted cd4 t cells". Bars (subset_synthetic_score.py): 1. at lam 15 and 40, every row's
measured memory fraction (memory over memory plus naive CD4 columns) within 0.15 of the planted; 2. at lam 15 and 40, the measured total CD4
(naive, memory and plain CD4 columns) within 0.1 of the planted. lam 3 is recorded, where it breaks.
Range of the data: real: outside libraries 3 to 71 calls a site, the labeled cell 0.24 to 0.80 of the sample (crosslab_chain_table.csv);
synthetic: 3, 15 and 40 reads a site, CD4 at 0.3 and 0.7, memory fraction 0 to 1.
Usage: python3 subset_synthetic.py ROW WORKDIR OUT.json"""
import os, sys, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.special import expit, logit
ROW, W, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
AS = os.path.join(MP, "chain", "Runtime Matrices", "Seq_Atlas", "array_sites"); V21 = os.path.join(MP, "atlas", "v2.1"); IF = os.path.join(MP, "chain", "MethylPhys_Interface")
HG19 = 28217448; G = pd.read_csv(os.path.join(HERE, "subset_grid.csv")).set_index("row").loc[ROW]; mf, lam, c4 = float(G.memfrac), float(G.lam), float(G.cd4)
rng = np.random.default_rng(20261107 + 100 * ROW); rest = 1 - c4
F = {"naive cd4 t cells": c4 * (1 - mf), "t central memory cd4": c4 * mf / 2, "t effector memory cd4": c4 * mf / 2,
     "monocytes": rest * 0.4, "nk cells": rest * 0.3, "b cells": rest * 0.3}
mu = {}
for part in (1, 2, 3):
    f = os.path.join(AS, f"mu_part{part}.parquet"); cols = [c for c in pq.read_schema(f).names if c in F]
    if cols:
        T = pq.read_table(f, columns=cols).to_pandas()
        for c in cols: mu[c] = np.where(T[c].to_numpy() == 65535, np.nan, T[c].to_numpy() / 1e4)
assert set(mu) == set(F), set(F) - set(mu)
truth = sum(F[c] * mu[c] for c in F if F[c] > 0)
ids = pq.read_table(os.path.join(V21, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(V21, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"]
cpg = pd.Series(ids).map(K).to_numpy(); ok = ~pd.isna(cpg) & np.isfinite(truth)
cpg = cpg[ok].astype(np.int64); p = np.clip(truth[ok], 0, 1); _, first = np.unique(cpg, return_index=True); cpg, p = cpg[first], p[first]
p = expit(logit(np.clip(p, 1e-4, 1 - 1e-4)) + rng.normal(0, 0.3, len(p)))
n = rng.poisson(lam, len(p)); k = rng.binomial(n, p)
a = np.zeros((HG19, 2), np.uint16); a[cpg - 1, 0] = np.minimum(k, 65534); a[cpg - 1, 1] = np.minimum(n, 65534); fb = os.path.join(W, f"s{ROW}.lbeta"); a.tofile(fb)
J = os.path.join(W, f"s{ROW}.json")
r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", fb, "--dev-seq-values", "1", "--dev-seq-ref", "auto", "--cell-panel", "atlas",
                    "--specimen", "sorted cd4 t cells", "--id", f"s{ROW}", "--out", J.replace(".json", ".html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
B = json.load(open(J)) if os.path.exists(J) else {}
sh = {x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells") or []}
json.dump(dict(row=ROW, memfrac=mf, lam=lam, cd4=c4, planted=F, found=sh, label_check=B.get("label_check"), rc=r.returncode,
               tail=None if r.returncode == 0 else (r.stdout + r.stderr)[-500:]), open(OUT, "w"), indent=1)
os.remove(fb); print(ROW, r.returncode, sorted(sh.items(), key=lambda x: -x[1])[:6])
