"""The islands on sequencing with the expectation from the reference's own reads (stage_metaa_atlas.expected_H_mix_counts, run_sample.py
--dev-terr-counted): synthetic, written and committed before it runs. The instrument's job: a healthy sequenced sample reads Normal at the
islands.
Why: the islands are sites every cell holds unmethylated, so a healthy sample's disorder there is rare off-pattern reads; the plug-in
expectation treats each cell's pooled level, counted from about 100 reads, as exact, which reads about 3 % high on a sample healthy by
construction, and the sample's own reads sit in its own pool.
Known answer: base, the spent healthy B-cell input GSM5652316 (its .beta; every site outside the islands unchanged, so the Deconvolver
reads it as it is). At each island site and for each of the atlas's 56 cells, a site level (the cell's pooled level in the committed array
columns, clipped to 0.0001..0.3); each person's own level around it (logit-normal, sd 0.3, stated); per cell two donors in each sealed half,
each counted at Poisson(33) reads, written as a constructed per-half island counts file in the committed format. Six new B-cell persons
(seed 20261111): at every island site k ~ Binomial(n, own level), n the base sample's own reads there; written as uint16 .lbeta. Each read by
the one command (--full-chain --dev-seq-ref auto --dev-seq-ref-half B --dev-terr-counted --dev-terr-counts FILE), the islands from half B's
reads. Bar (one): the six samples' median islands reading within 0.02 of 1. Recorded: the plug-in reading beside it.
Usage: python3 islands_counted_synthetic.py BASE.beta WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd
from scipy.special import expit, logit
BASE, W, OUT = sys.argv[1:4]; os.makedirs(W, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_seq_atlas as Q, pyarrow.parquet as pq
SA.SEQ_ATLAS = Q.ARRAY_COLS   # the sequencing atlas's folder, as the one command sets it
SA.SEQ_LEVEL = "pooled"; rng = np.random.default_rng(20261111); HG19 = 28217448
P = pd.Index(SA.territory_sets()["islands"])
K = pq.read_table(os.path.join(SA.ATLAS, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe")["loyfer_cpg_index"].reindex(P).dropna().astype(np.int64)
probes = K.index; cpg = K.to_numpy()
cells = sorted(pd.read_csv(os.path.join(CH, "..", "doors", "data", "DEV_ATLAS_V21_01", "seq_atlas", "seq_atlas_inputs.csv")).cell.str.strip().unique())
lev = {c: np.clip(SA._col("mu", c).reindex(probes).fillna(0.001).to_numpy(), 1e-4, 0.3) for c in cells}
person = lambda c: expit(logit(lev[c]) + rng.normal(0, 0.3, len(probes)))
rows = []
for c in cells:
    d = {}
    for h in ("A", "B"):
        m_ = np.zeros(len(probes), np.int64); n_ = np.zeros(len(probes), np.int64)
        for _ in range(2): n1 = rng.poisson(33, len(probes)); m_ += rng.binomial(n1, person(c)); n_ += n1
        d["m" + h] = m_.astype(np.uint16); d["n" + h] = n_.astype(np.uint16)
    rows.append(pd.DataFrame(dict(cell=c, probe=probes, **d)))
CF = os.path.join(W, "terr_counts.parquet"); pd.concat(rows, ignore_index=True).to_parquet(CF, index=False)
b = np.fromfile(BASE, np.uint8).reshape(-1, 2).astype(np.uint16); assert len(b) == HG19, len(b)
IF = os.path.join(CH, "MethylPhys_Interface"); res = []
for t in range(6):
    a = b.copy(); n = a[cpg - 1, 1].astype(np.int64); a[cpg - 1, 0] = rng.binomial(n, person("b cells")); f = os.path.join(W, f"s{t}.lbeta"); a.tofile(f); J = os.path.join(W, f"s{t}.json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", f"s{t}", "--full-chain", "--dev-seq-ref", "auto",
                        "--dev-seq-ref-half", "B", "--dev-terr-counted", "--dev-terr-counts", CF, "--out", os.path.join(W, "s.html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
    I_ = ((json.load(open(J)).get("territories") or {}).get("sets") or {}).get("islands", {}) if os.path.exists(J) else {}
    c_ = I_.get("counted") or {}; res.append(dict(t=t, A=c_.get("A"), plugin=I_.get("A_plugin", I_.get("A")), sites=c_.get("sites"), left_out=c_.get("left_out"), cells=c_.get("cells"), reason=c_.get("reason")))
    if c_.get("A") is None: print(t, c_, r.stdout[-300:], r.stderr[-300:])
    os.remove(f); print(res[-1], flush=True)
D = pd.DataFrame(res); D.to_csv(os.path.join(W, "readings.csv"), index=False); x = pd.to_numeric(D.A, errors="coerce").dropna(); med = float(x.median()) if len(x) else float("nan")
L = ["The islands on sequencing, the expectation from the reference's own reads: synthetic, read by run_sample.py --dev-terr-counted --dev-terr-counts FILE (half B)",
     f"seed 20261111, base GSM5652316 (spent), {len(probes)} island sites, 56 cells, two donors per half near 33 reads, person spread sd 0.3 (logit); read {len(x)} of {len(D)}",
     "islands, counted: " + ", ".join(f"{v:.4f}" for v in x), "islands, plug-in (recorded, not scored): " + ", ".join(f"{v:.4f}" for v in pd.to_numeric(D.plugin, errors="coerce").dropna()),
     f"bar 1 median {med:.4f} (within 0.02 of 1): {'met' if len(x) == len(D) and abs(med - 1) <= 0.02 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
