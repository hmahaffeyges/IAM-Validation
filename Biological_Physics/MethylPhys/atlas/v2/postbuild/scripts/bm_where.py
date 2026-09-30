import glob, json, numpy as np, pandas as pd, os
from deconv_v2 import DeconvV2
exec(open("bm_test.py").read().split("specs = ")[0])
out = []
for p in sorted(glob.glob("/home/ubuntu/data/v2run01/shards/*.parquet")):
    b = pd.read_parquet(p).iloc[:, 0]; b.index = b.index.astype(str)
    f1 = D1.deconvolve(b, n_boot=0)["fractions"]; f0 = D0.deconvolve(b, n_boot=0)["fractions"]
    out.append({c: (f1.get(c, 0), f0.get(c, 0)) for c in set(f1) | set(f0)})
cells = sorted(out[0]); W = pd.DataFrame({c: [np.median([o[c][0] for o in out]), np.median([o[c][1] for o in out])] for c in cells}, index=["with_BM", "without_BM"]).T
W["change"] = W.without_BM - W.with_BM; W = W[(W.with_BM > 0.005) | (W.without_BM > 0.005)].sort_values("change")
print(W.round(4).to_string())
