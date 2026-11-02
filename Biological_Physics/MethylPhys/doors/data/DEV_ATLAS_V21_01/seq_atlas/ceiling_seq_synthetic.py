"""The distance to the entropy ceiling on sequencing (checklist B8; stage_metaa_atlas.ceiling_counted, read by run_sample.py): synthetic,
written and committed before it runs. The instrument's job: a healthy sample sits at the healthy end of the gauge, a sample whose every
site has gone to one half sits at the ceiling, and one half way between reads between.
Known answer (seed 20261119): base, the spent healthy B-cell input GSM5652316's .beta, for its read count at every CpG. At every CpG an
IAM-Atlas v2.1 site names, a person's level: the IAM Sequencing Atlas's B-cell level plus N(0, donor_sd), held inside 0 and 1 ("healthy");
the same moved half way to one half ("half way"); one half at every site ("ceiling"). Methylated reads drawn Binomial(n, level) at the
base's own reads; written as .lbeta and read by the one command, --specimen "sorted b cells", the healthy passes' settings. Bars, on the
sorted-cell reading's distance_to_ceiling: healthy at least 0.9; ceiling at most 0.1; half way strictly between them. Recorded: the
islands' and the in-mix B-cell reading's distance.
Usage: python3 ceiling_seq_synthetic.py BASE.beta WORKDIR OUT.txt"""
import os, sys, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
BASE, W, OUT = sys.argv[1:4]; os.makedirs(W, exist_ok=True)
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain")); IF = os.path.join(CH, "MethylPhys_Interface")
AT = os.path.join(CH, "..", "atlas", "v2.1"); SC = os.path.join(CH, "Runtime Matrices", "Seq_Atlas", "sphere_cols"); HG19 = 28217448
ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(AT, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe").reindex(ids)
def col(stem, c="b cells"):
    for k in (1, 2, 3):
        f = os.path.join(SC, f"{stem}_part{k}.parquet")
        if c in pq.read_schema(f).names:
            u = pq.read_table(f, columns=[c]).column(c).to_numpy(); v = u.astype("float64") / 1e4; v[u == 65535] = np.nan; return v
mu, sd = col("mu"), col("donor_sd"); cpg = K.loyfer_cpg_index.to_numpy(); ok = ~pd.isna(cpg) & np.isfinite(mu)
cpg = cpg[ok].astype(np.int64); mu = mu[ok]; sd = np.nan_to_num(sd[ok]); ch = K.chrom_hg19.to_numpy()[ok]; ps = K.pos_hg19.to_numpy()[ok].astype(float)
cpg, first = np.unique(cpg, return_index=True); mu, sd, ch, ps = mu[first], sd[first], ch[first], ps[first]
b = np.fromfile(BASE, np.uint8).reshape(-1, 2).astype(np.uint16); assert len(b) == HG19, len(b); n = b[cpg - 1, 1].astype(np.int64)
res = {}; rng = np.random.default_rng(20261119)
FL = ["--full-chain", "--dev-seq-ref", "auto", "--dev-seq-level", "donors", "--dev-terr-counted"]
dev = rng.normal(0, 1, len(mu)) * sd; L0 = np.clip(mu + dev, 0, 1)
for name, lv in (("healthy", L0), ("half way", L0 + 0.5 * (0.5 - L0)), ("ceiling", np.full(len(mu), 0.5))):
    a = b.copy(); a[cpg - 1, 0] = rng.binomial(n, lv); tag = name.replace(" ", ""); f = os.path.join(W, tag + ".lbeta"); a.tofile(f); J = os.path.join(W, tag + ".json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", "ceil_" + tag, *FL,
                        "--out", os.path.join(W, tag + ".html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
    B = json.load(open(J)) if os.path.exists(J) else {}; os.remove(f)
    srt = ((B.get("development") or {}).get("metaa_atlas") or {}); isl = (((B.get("territories") or {}).get("sets") or {}).get("islands") or {})
    res[name] = (srt.get("ceiling") or {}).get("distance_to_ceiling")
    print(name, "| sorted A", srt.get("A"), "ceiling", srt.get("ceiling"), "| islands A", isl.get("A"), "distance", (isl.get("ceiling") or {}).get("distance_to_ceiling"), "| rc", r.returncode, flush=True)
    if res[name] is None: print(r.stdout[-400:], r.stderr[-800:], flush=True)
h, m_, c = res["healthy"], res["half way"], res["ceiling"]; ok = None not in (h, m_, c)
b1 = ok and h >= 0.9; b2 = ok and c <= 0.1; b3 = ok and c < m_ < h
L = [f"read by run_sample.py (seed 20261119): distance to the ceiling, sorted-cell reading: healthy {h}, half way {m_}, ceiling {c}",
     f"bar 1 healthy at least 0.9: {'met' if b1 else 'not met'}", f"bar 2 ceiling at most 0.1: {'met' if b2 else 'not met'}",
     f"bar 3 half way between: {'met' if b3 else 'not met'}", f"synthetic: {'met' if b1 and b2 and b3 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
