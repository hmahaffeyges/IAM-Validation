"""DEV-METAA-ATLAS-01, exploration 3 (written and committed 2026-10-21 before it runs; not sealed; the arrays are already spent by step 3).
Step 3 (sorted_01/) found the tared spread of sorted cells against IAM-Atlas v2.1 too wide (4.2 to 8.6 %). Question: is the spread the
sorting purity? A sorted sample with a few percent of other cells mixed in has more entropy at the cell's identity sites, so it reads higher,
and purity differs from sample to sample. Through the chain: each IDAT pair by chain Stage 1, Met-A against the atlas
(stage_metaa_atlas.read, unchanged) and the cell shares of the commissioned IAM-Deconvolver (stage_v4.stage_d21, unchanged). Per cell type:
the Spearman correlation of the tared reading (step 3's A_rel, recomputed the same way) with the off-target share (1 minus the share of the
sorted cell's own group), the share Normal in the purest half against the less pure half, and whether a planted 1 % loss is larger than
the purest half's spread. Nothing is fitted or corrected: this only tells whether purity explains the spread.
Usage: python3 sorted_purity_03.py OUTDIR   (writes sorted_purity_03_rows.csv and sorted_purity_03_summary.txt)"""
import os, sys, re, time, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H_, "..", "..", "..", "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA, stage_v4 as V
from stage_1_idat_calibration import calibrate_idat_to_beta
from scipy.stats import spearmanr
GROUP = {"monocytes": "monocytes", "naive b cells": "b cells", "naive cd4 t cells": "cd4 t cells", "naive cd8 t cells": "cd8 t cells", "nk cells": "nk cells"}
out = sys.argv[1]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "work"); os.makedirs(W, exist_ok=True)
S = pd.read_csv(os.path.join(H_, "gse184269_sorted_samples.csv")); rows = []
def files(g):
    u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"
    for t in range(6):
        try:
            names = sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))); got = {}
            for n in names:
                p = os.path.join(W, n); urllib.request.urlretrieve(u + n, p); got["Grn" if "_Grn" in n else "Red"] = p
            if len(got) == 2: return got
        except Exception: pass
        time.sleep(20 * (t + 1))
    raise RuntimeError(f"{g}: IDATs not fetched")
for k, r in S.iterrows():
    try:
        f = files(r.gsm); o = calibrate_idat_to_beta(f["Grn"], f["Red"], verbose=False); b = pd.Series(o[0] if isinstance(o, tuple) else o, dtype="float64")
        rec = SA.read(b, r.atlas_cell); d = V.stage_d21(b); sh = {c["cell"]: c["share"] for c in (d.get("cells") or [])}
        S_ = SA.identity_sites(r.atlas_cell); mu = SA._col("mu", r.atlas_cell).reindex(S_); bp = b.copy(); x = b.reindex(S_); ok = x.notna() & mu.notna()
        bp.loc[x[ok].index] = (x[ok] + 0.01 * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6); p = SA.read(bp, r.atlas_cell)
        rows.append(dict(gsm=r.gsm, atlas_cell=r.atlas_cell, A=rec.get("A"), A_planted=p.get("A"), own_share=sh.get(GROUP[r.atlas_cell]), **{f"share_{c}": v for c, v in sh.items()}))
    except Exception as e:
        rows.append(dict(gsm=r.gsm, atlas_cell=r.atlas_cell, reason=f"{type(e).__name__}: {e}"[:200]))
    for q in os.listdir(W): os.remove(os.path.join(W, q))
    print(k + 1, "of", len(S), r.gsm, rows[-1].get("A"), rows[-1].get("own_share"), flush=True)
R = pd.DataFrame(rows)
for c, g in R.groupby("atlas_cell"):
    for i in g.index:
        refs = [a for j, a in g.A.items() if j != i and pd.notna(a)]
        R.loc[i, "A_rel"] = SA.tare(R.A[i], refs)["A_rel"]; R.loc[i, "A_rel_planted"] = SA.tare(R.A_planted[i], refs)["A_rel"]
R["off_target"] = 1 - R.own_share; R.to_csv(os.path.join(out, "sorted_purity_03_rows.csv"), index=False)
L = ["DEV-METAA-ATLAS-01 exploration 3: is the spread of sorted cells against the atlas their sorting purity? (not sealed)"]
for c, g in R.dropna(subset=["A_rel", "off_target"]).groupby("atlas_cell"):
    rho, pv = spearmanr(g.off_target, g.A_rel); m = g.off_target.median(); pure, rest = g[g.off_target <= m], g[g.off_target > m]
    nrm = lambda x: x.A_rel.between(0.95, 1.05).mean(); sp = float(np.std(pure.A_rel / pure.A_rel.median(), ddof=1)); ch = float((pure.A_rel_planted - pure.A_rel).mean())
    L.append(f"{c}: n {len(g)} | own-group share median {g.own_share.median():.3f} (range {g.own_share.min():.3f}-{g.own_share.max():.3f}) | Spearman(off-target, A_rel) "
             f"{rho:+.2f} (p {pv:.3f}) | Normal: purest half {nrm(pure):.2f}, less pure half {nrm(rest):.2f} | purest half spread {sp:.4f}, planted 1 % = {ch / sp:.1f} x")
open(os.path.join(out, "sorted_purity_03_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
