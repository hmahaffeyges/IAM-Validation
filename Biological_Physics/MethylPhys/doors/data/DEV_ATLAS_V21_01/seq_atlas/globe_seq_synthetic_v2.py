"""VERSION 2 (2026-11-02, written before it runs; version 1 not met as written, see the record): seed 20261118; the planted person's sites
on chromosome 8 (100 to 115 Mb) each moved 0.3 toward the opposite end (down where the level is above 0.5, up where at or below), so no
site is cut off at 0 or 1; the globe counts blocks of 10 sites (its "sites_changed" is the number of changed blocks), so the bars are in
blocks: the healthy three's median at most 50 changed blocks; the planted one at least that median plus half the planted blocks (planted
sites measured / 10).
The report globe on sequencing (report_globe_v4.build with the sample's own counts; run_sample.py, the sequencing atlas): synthetic,
written and committed before it runs. The instrument's job: a healthy sequenced sample shows little change on its globe; a real
regional change shows.
Known answer (seed 20261117): base, the spent healthy B-cell input GSM5652316's .beta, for its read count at every CpG. At every CpG an
IAM-Atlas v2.1 site names (sites_sequencing_key), a new person's level: the IAM Sequencing Atlas's B-cell level there (sphere columns, mu)
plus a person deviation N(0, donor_sd) (0 where donor_sd is blank), held inside 0 and 1; the methylated reads drawn Binomial(n, level),
n the base's own reads. Three such people ("healthy 1" to "healthy 3"), and a fourth with every site on chromosome 8 from 100 to 115 Mb
raised by 0.3 ("planted"). Each written as .lbeta and read by the one command, --specimen "sorted b cells", the healthy passes' settings
(--full-chain --dev-seq-ref auto --dev-seq-level donors --dev-terr-counted). Bars: the healthy three's median number of changed sites at
most 50; the planted one's changed sites at least that median plus half the planted sites it measured.
Usage: python3 globe_seq_synthetic.py BASE.beta WORKDIR OUT.txt"""
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
plant = (ch == "chr8") & (ps >= 100e6) & (ps < 115e6); rng = np.random.default_rng(20261118); res = {}
FL = ["--full-chain", "--dev-seq-ref", "auto", "--dev-seq-level", "donors", "--dev-terr-counted"]
for name in ("healthy 1", "healthy 2", "healthy 3", "planted"):
    lv = np.clip(mu + rng.normal(0, 1, len(mu)) * sd, 0, 1)
    if name == "planted": lv = np.where(plant, np.where(lv > 0.5, lv - 0.3, lv + 0.3), lv)
    a = b.copy(); a[cpg - 1, 0] = rng.binomial(n, lv); tag = name.replace(" ", ""); f = os.path.join(W, tag + ".lbeta"); a.tofile(f); J = os.path.join(W, tag + ".json")
    r = subprocess.run([sys.executable, "run_sample.py", "--dev-seq-beta", f, "--dev-seq-values", "1", "--specimen", "sorted b cells", "--id", "syn_" + tag, *FL,
                        "--out", os.path.join(W, tag + ".html"), "--bundle", J], cwd=IF, capture_output=True, text=True)
    B = json.load(open(J)) if os.path.exists(J) else {}; g = B.get("globe") or {}; s = B.get("info_sphere") or {}
    res[name] = g.get("sites_changed"); os.remove(f)
    print(name, "changed", g.get("sites_changed"), "of", g.get("sites_compared"), "| territories", s.get("territories_changed"), "| Fisher-Rao", s.get("fisher_rao_rms_rad"), "| rc", r.returncode, flush=True)
    if g.get("sites_changed") is None: print(r.stdout[-400:], r.stderr[-800:], flush=True)
hm = float(np.median([res[k] for k in ("healthy 1", "healthy 2", "healthy 3") if res[k] is not None])) if all(res[k] is not None for k in res) else None
npl = int((plant & (n > 0)).sum()) // 10; b1 = hm is not None and hm <= 50; b2 = hm is not None and res["planted"] >= hm + 0.5 * npl
L = [f"read by run_sample.py (version 2, seed 20261118); planted blocks (planted sites measured / 10): {npl}"] + [f"{k}: changed blocks {v}" for k, v in res.items()] + [
    f"bar 1 healthy median changed blocks at most 50: {hm}: {'met' if b1 else 'not met'}", f"bar 2 planted at least the healthy median plus half the planted blocks ({None if hm is None else hm + 0.5 * npl}): {'met' if b2 else 'not met'}",
     f"synthetic: {'met' if b1 and b2 else 'not met'}"]
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
