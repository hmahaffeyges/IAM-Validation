"""build_atlas_ruler.py - builds Runtime Matrices/Stage_T/atlas_ruler_v1.json, the atlas ruler (development, written 2026-10-23; plan 0.2 and
35.14; author: 'we need a standard, a ruler, something to calibrate each sample to'). Run once; the matrix is a frozen input like the atlas.

The ruler keeps the self-tare's method (a straight line for each probe design from the array's own low and high anchor means) and takes its
targets from IAM-Atlas v2.1 instead of reference arrays, so every array lands on the atlas's scale. Anchors, by fixed rules, nothing chosen by
hand: sites measured in all 79 cell types, on both the 450K and EPIC v1 chips, on autosomes; LOW = every cell type below 0.10 and outside every
CpG island and its 2 kb shores (UCSC hg19 cpgIslandExt), so the islands where early cancer gains are never anchors; HIGH = every cell type
above 0.90. Targets: the atlas's mean over each set. Inputs (fetched by the caller into MANIFESTS): HumanMethylation450k_15017482_v3.csv.gz and
HumanMethylationEPIC_manifest_v2.csv.gz (array-manifest-files.s3.amazonaws.com), cpgIslandExt.txt.gz (hgdownload.soe.ucsc.edu, hg19).
Usage: python3 build_atlas_ruler.py MANIFESTS_DIR"""
import os, sys, json, hashlib, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); AT = os.path.join(H, "..", "..", "atlas", "v2.1"); MD = sys.argv[1]
OUT = os.path.join(H, "..", "Runtime Matrices", "Stage_T", "atlas_ruler_v1.json")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist(); N = len(ids); idx = pd.Index(ids)
S = np.zeros(N); C = np.zeros(N); MN = np.full(N, np.inf, np.float32); MX = np.full(N, -np.inf, np.float32); NM = np.zeros(N, int); ncell = 0
for k in (1, 2, 3):
    t = pq.read_table(os.path.join(AT, f"mu_part{k}.parquet"))
    for c in t.column_names:
        v = t.column(c).to_numpy().astype(np.float32); ok = v != 65535; v = v / 10000
        S[ok] += v[ok]; C[ok] += 1; MN[ok] = np.minimum(MN[ok], v[ok]); MX[ok] = np.maximum(MX[ok], v[ok]); ncell += 1
    del t
full = C == ncell; mu = np.where(C > 0, S / np.maximum(C, 1), np.nan)
def man(f, cols):
    d = pd.read_csv(os.path.join(MD, f), usecols=lambda c: c in cols, low_memory=False, dtype=str).dropna(subset=["Name"]).set_index("Name")
    return d[~d.index.duplicated()]
m450 = man("HumanMethylation450k_15017482_v3.csv.gz", ("Name", "Infinium_Design_Type", "OLD_CHR", "OLD_MAPINFO"))
mE = man("HumanMethylationEPIC_manifest_v2.csv.gz", ("Name", "Infinium_Design_Type"))
both = idx.isin(m450.index) & idx.isin(mE.index); des = m450["Infinium_Design_Type"].reindex(idx).to_numpy()
ch = m450["OLD_CHR"].reindex(idx).astype(str).to_numpy(); p19 = pd.to_numeric(m450["OLD_MAPINFO"].reindex(idx), errors="coerce").to_numpy()
auto = ~np.isin(ch, ["X", "Y", "nan", "None"])
cgi = pd.read_csv(os.path.join(MD, "cpgIslandExt.txt.gz"), sep="\t", header=None, usecols=[1, 2, 3], names=["chr", "s", "e"]); near = np.zeros(N, bool)
for c, g in cgi.groupby("chr"):
    m = np.where((ch == c.replace("chr", "")) & np.isfinite(p19))[0]
    if not len(m): continue
    o = np.argsort(g.s.values); s = g.s.values[o] - 2000; e = np.maximum.accumulate(g.e.values[o] + 2000)
    j = np.searchsorted(s, p19[m], side="right") - 1; near[m] = (j >= 0) & (p19[m] <= e[np.clip(j, 0, None)])
low = full & (MX < 0.10) & both & auto & ~near; high = full & (MN > 0.90) & both & auto
sets, ref = {}, {}
for d in ("I", "II"):
    for nm, m in (("low", low), ("high", high)):
        k = m & (des == d); sets[f"{d}_{nm}"] = list(idx[k]); ref[f"{d}_{nm}"] = round(float(np.mean(mu[k])), 6)
dE = mE["Infinium_Design_Type"].reindex(idx).to_numpy()   # every site's probe design: EPIC v1's where the site is on it, else 450K's
design = {i: (a if a in ("I", "II") else b) for i, a, b in zip(idx, dE, des) if a in ("I", "II") or b in ("I", "II")}
J = dict(version="atlas_ruler_v1", label="DEVELOPMENT - not commissioned", rule=__doc__.split("\n\n")[1].replace("\n", " "),
         n={k: len(v) for k, v in sets.items()}, ref_anchors=ref, sets=sets, design=design,
         inputs={"atlas_sites_sha256": sha(os.path.join(AT, "sites.parquet")), **{f: sha(os.path.join(MD, f)) for f in ("HumanMethylation450k_15017482_v3.csv.gz", "HumanMethylationEPIC_manifest_v2.csv.gz", "cpgIslandExt.txt.gz")}})
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(J, open(OUT, "w")); print(J["n"], J["ref_anchors"], len(design), "designs")
