"""DEV-SECONDLAB-01 scale diagnosis (2026-10-15): why ENCODE's loose share reads 6x Loyfer's on B cells. Same 2,000-CpG tile rule, loose below
0.70, tiles with >= 200 calls, on (a) ENCODE's own published per-CpG calls for the counted run (ENCFF703XLD, GRCh38) and (b) a Loyfer B cell's
published hg38 beta file (GSM5652317). Usage: python scale_diagnosis.py <ENCFF703XLD.bed.gz> <GSM5652317...hg38.beta>"""
import sys, json, os, numpy as np, pandas as pd
def tiles_stats(M, C):
    n = len(M) // 2000; Mt = M[:n * 2000].reshape(n, 2000).sum(1); Ct = C[:n * 2000].reshape(n, 2000).sum(1); k = Ct >= 200; mm = Mt[k] / Ct[k]
    return dict(tiles=int(k.sum()), loose_share=round(float((mm < 0.70).mean()), 3), tile_median=round(float(np.median(mm)), 3),
                share_by_cut={str(c): round(float((mm < c).mean()), 3) for c in (0.60, 0.65, 0.70, 0.75)}, global_mean=round(float(M.sum() / C.sum()), 4))
out = {}
parts = [ch[ch[0].isin([f"chr{i}" for i in range(1, 23)])][[0, 1, 9, 10]] for ch in pd.read_csv(sys.argv[1], sep="\t", header=None, chunksize=5_000_000, low_memory=False)]
E = pd.concat(parts); E.columns = ["chr", "pos", "cov", "pct"]; E = E[E["cov"] >= 1]; E["m"] = E.pct / 100 * E["cov"]
Ms, Cs = [], []
for c, g in E.groupby("chr", sort=False):
    g = g.sort_values("pos"); n = len(g) // 2000 * 2000; Ms.append(g.m.values[:n]); Cs.append(g["cov"].values[:n])
out["encode_b_ENCFF703XLD"] = tiles_stats(np.concatenate(Ms), np.concatenate(Cs).astype(float))
a = np.fromfile(sys.argv[2], dtype=np.uint8).reshape(-1, 2).astype(float); out["loyfer_b_GSM5652317"] = tiles_stats(a[:, 0], a[:, 1])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale_diagnosis.json"), "w"), indent=1); print(json.dumps(out, indent=1))
