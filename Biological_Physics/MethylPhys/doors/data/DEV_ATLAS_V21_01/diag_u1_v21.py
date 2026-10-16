"""DIAGNOSTIC ONLY (2026-10-16, after U1 was scored once; nothing here re-scores U1 or changes its bar). Why do 23 of 27 cell-platform pairs miss?
For every U1-scored array, against its cell's 90 % predictive interval (chain Stage A21, exactly as scored):
  offset        median of (value - interval midpoint) over its sites: one overall shift of the array
  below, above  share of sites under the interval and over it (a shift puts the misses on one side; extra spread puts them on both)
  cov_own_shift coverage after removing the array's own offset (diagnostic: what one per-array number would recover)
Per laboratory: medians of the above. Usage: python diag_u1_v21.py <blocks> <shards> <out> [workers]"""
import os, sys, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import score_u1_v21 as U
def one(iv, sh):
    y = pd.read_parquet(sh).iloc[:, 0]; j = iv.index.intersection(y.index); lo = iv.loc[j, "lo"].values; hi = iv.loc[j, "hi"].values; v = y.loc[j].values.astype(float)
    k = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(v); lo, hi, v = lo[k], hi[k], v[k]; mid = (lo + hi) / 2; off = float(np.median(v - mid)); w = v - off
    return dict(sites=int(k.sum()), offset=off, below=float((v < lo).mean()), above=float((v > hi).mean()), coverage=float(((v >= lo) & (v <= hi)).mean()), cov_own_shift=float(((w >= lo) & (w <= hi)).mean()))
if __name__ == "__main__":
    blocks, shards, out = sys.argv[1:4]; wk = int(sys.argv[4]) if len(sys.argv) > 4 else 8; os.makedirs(out, exist_ok=True)
    T = pd.read_csv(os.path.join(H, "u1_samples_v21.csv"), comment="#"); T = T[T.status == "scored"]
    I = pd.read_csv(os.path.join(H, "identity_v21_arrays.csv")); I = I[(I.set != "build") & (I.status == "ok")][["sample", "series", "base"]]
    T = T.merge(I, on=["sample", "series"], how="left"); IV, lab = U.intervals(blocks, sorted(set(T.cell)), wk)
    R = pd.DataFrame([dict(sample=r["sample"], series=r.series, cell=r.cell, family=r.family, **one(IV[r.cell], os.path.join(shards, r.base + ".parquet"))) for _, r in T.iterrows()])
    R.to_csv(os.path.join(out, "diag_u1_arrays.csv"), index=False)
    L = R.groupby("series").agg(n=("sample", "size"), offset=("offset", "median"), below=("below", "median"), above=("above", "median"), coverage=("coverage", "median"), cov_own_shift=("cov_own_shift", "median"),
                                 inband=("coverage", lambda x: float(x.between(0.80, 0.97).mean())), inband_own_shift=("cov_own_shift", lambda x: float(x.between(0.80, 0.97).mean()))).round(4)
    L.to_csv(os.path.join(out, "diag_u1_labs.csv")); print(L.sort_values("inband").to_string())
    print(json.dumps(dict(arrays=len(R), median_abs_offset=float(R.offset.abs().median()), share_inband=float(R.coverage.between(0.80, 0.97).mean()), share_inband_own_shift=float(R.cov_own_shift.between(0.80, 0.97).mean()),
                          misses_one_sided=float((np.maximum(R.below, R.above) > 3 * np.minimum(R.below, R.above)).mean()))))
