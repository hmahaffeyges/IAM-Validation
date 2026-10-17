"""DEV-ATLAS-USE-02: Loyfer's own reference, projected onto array probes exactly as his paper does (wgbstools beta_to_450k, then the
mean of each of his 39 groups). Written 2026-10-16 before any value is read.
 - inputs: his 205 samples' .beta files from GEO GSE186458 (loyfer_files_groups.csv: GEO file, his group from his Supplementary Table S1;
   the two GEO files not in S1 are left out, as he left them out); probe -> CpG index from atlas/v2/inputs/probe_map_summary.csv.gz
   (1-based wgbstools index; all 931 of his mapped S12 markers fall inside his own printed marker ranges, checked 2026-10-16)
 - per sample: beta = meth / cov at the probe's CpG where cov >= 3 (wgbstools beta_to_450k default cov_thresh 3), else missing
 - per group: mean over his samples with a value (nanmean), as averaging his per-sample beta_to_450k output
Output: loyfer_array_reference.parquet (IlmnID x 39 groups) and its SHA-256.
Usage: python build_loyfer_array_ref.py <probe_map_summary.csv.gz> <out dir> [workers]"""
import os, sys, json, hashlib, urllib.request, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
H = os.path.dirname(os.path.abspath(__file__))
def beta_at(path, idx, cov_thresh=3):
    a = np.fromfile(path, dtype=np.uint8).reshape(-1, 2)[idx - 1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"): v = a[:, 0] / a[:, 1]
    v[a[:, 1] < cov_thresh] = np.nan; return v
def fetch(url, dst):
    for t in range(5):
        try:
            urllib.request.urlretrieve(url, dst + ".part"); os.replace(dst + ".part", dst); return dst
        except Exception as e: err = e
    raise err
if __name__ == "__main__":
    pm, out = sys.argv[1], sys.argv[2]; W = int(sys.argv[3]) if len(sys.argv) > 3 else 8; os.makedirs(out, exist_ok=True)
    P = pd.read_csv(pm, usecols=["IlmnID", "cpg_index"]).dropna().drop_duplicates("IlmnID"); idx = P.cpg_index.astype(np.int64).values
    F = pd.read_csv(os.path.join(H, "loyfer_files_groups.csv")).dropna(subset=["Group"]); assert len(F) == 205 and F.Group.nunique() == 39
    tmp = os.path.join(out, "beta"); os.makedirs(tmp, exist_ok=True); V = {}; done = 0
    def one(r):
        p = fetch(r.url, os.path.join(tmp, r.file)); v = beta_at(p, idx); os.remove(p); return r.file, r.Group, v
    with ThreadPoolExecutor(W) as X:
        for f, g, v in X.map(one, [r for r in F.itertuples()]):
            V.setdefault(g, []).append(v); done += 1
            if done % 20 == 0: print("read", done, "of 205", flush=True)
    R = pd.DataFrame({g: np.nanmean(np.vstack(L), axis=0) for g, L in sorted(V.items())}, index=P.IlmnID.values).astype("float32"); R.index.name = "IlmnID"
    p = os.path.join(out, "loyfer_array_reference.parquet"); R.to_parquet(p, compression="zstd")
    s = hashlib.sha256(open(p, "rb").read()).hexdigest()
    json.dump(dict(groups=list(R.columns), probes=len(R), samples=done, missing_share=float(R.isna().mean().mean()), sha256=s), open(os.path.join(out, "loyfer_array_reference.json"), "w"), indent=1)
    print("Loyfer array reference:", R.shape, "sha256", s)
