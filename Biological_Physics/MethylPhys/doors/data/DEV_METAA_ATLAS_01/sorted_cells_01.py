"""DEV-METAA-ATLAS-01, design step 3, exploration (2026-10-21; not sealed: "we explore without sealing until we are certain").
The sorted blood cells of GSE184269, a laboratory IAM-Atlas v2.1 was not fitted on, never read before (CANON/data_spent.csv): 24 monocytes,
23 naive CD4 T, 23 naive B, 20 NK and 19 naive CD8 T arrays (gse184269_sorted_samples.csv, from the series' GEO descriptions).
Through the chain: each IDAT pair is calibrated by chain Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta) and read by the chain's
development stage, stage_metaa_atlas.read, against its matching atlas cell; then tared leave-one-out against the other arrays of the same
cell type in this series (stage_metaa_atlas.tare). A planted copy of each array (a 1 % loss of the cell's pattern) is tared against the same
references. Reported per cell: the share Normal after the tare, the tared spread, the planted change, the untared A (the atlas offset).
Usage: python3 sorted_cells_01.py OUTDIR   (writes OUTDIR/sorted_cells_01_rows.csv and sorted_cells_01_summary.txt; IDATs deleted after use)"""
import os, sys, re, time, urllib.request, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H_, "..", "..", "..", "chain"); sys.path.insert(0, CH)
import stage_metaa_atlas as SA
from stage_1_idat_calibration import calibrate_idat_to_beta
out = sys.argv[1]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "work"); os.makedirs(W, exist_ok=True)
S = pd.read_csv(os.path.join(H_, "gse184269_sorted_samples.csv")); rows = []
def files(g):
    u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"
    for t in range(6):
        try:
            names = sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))); got = {}
            for n in names:
                p = os.path.join(W, n); urllib.request.urlretrieve(u + n, p)
                assert open(p, "rb").read(2) == b"\x1f\x8b" and os.path.getsize(p) > 1_000_000; got["Grn" if "_Grn" in n else "Red"] = p
            if len(got) == 2: return got
        except Exception as e: err = e
        time.sleep(20 * (t + 1))
    raise RuntimeError(f"{g}: IDATs not fetched")
for k, r in S.iterrows():
    try:
        f = files(r.gsm); o = calibrate_idat_to_beta(f["Grn"], f["Red"], verbose=False); b = pd.Series(o[0] if isinstance(o, tuple) else o, dtype="float64")
        rec = SA.read(b, r.atlas_cell); S_ = SA.identity_sites(r.atlas_cell); mu = SA._col("mu", r.atlas_cell).reindex(S_)
        bp = b.copy(); x = b.reindex(S_); ok = x.notna() & mu.notna(); bp.loc[x[ok].index] = (x[ok] + 0.01 * (0.5 - mu[ok])).clip(1e-6, 1 - 1e-6)
        p = SA.read(bp, r.atlas_cell)
        rows.append(dict(gsm=r.gsm, src=r.src, atlas_cell=r.atlas_cell, A=rec.get("A"), A_planted=p.get("A"), shift=rec.get("shift_per_1pct_loss"),
                         n_sites=rec.get("n_sites"), reason=rec.get("reason", "")))
    except Exception as e:
        rows.append(dict(gsm=r.gsm, src=r.src, atlas_cell=r.atlas_cell, reason=f"{type(e).__name__}: {e}"[:200]))
    for q in os.listdir(W): os.remove(os.path.join(W, q))
    print(k + 1, "of", len(S), r.gsm, r.src, rows[-1].get("A"), rows[-1].get("reason", ""), flush=True)
R = pd.DataFrame(rows)
for c, g in R.groupby("atlas_cell"):
    for i in g.index:
        refs = [a for j, a in g.A.items() if j != i]
        R.loc[i, "A_rel"] = SA.tare(R.A[i], refs)["A_rel"]; R.loc[i, "A_rel_planted"] = SA.tare(R.A_planted[i], refs)["A_rel"]
R.to_csv(os.path.join(out, "sorted_cells_01_rows.csv"), index=False); L = ["DEV-METAA-ATLAS-01 design step 3: GSE184269 sorted cells against IAM-Atlas v2.1 (exploration, not sealed)"]
for c, g in R.groupby("atlas_cell"):
    a = g.A_rel.dropna(); ch = (g.A_rel_planted - g.A_rel).dropna(); sp = float(a.std())
    L.append(f"{c}: read {len(a)} of {len(g)} | untared A median {g.A.median():.4f} | after the tare Normal {a.between(0.95, 1.05).mean():.2f}, spread {sp:.4f} | "
             f"planted 1 % loss: change {ch.mean():+.4f} = {ch.mean() / sp if sp else float('nan'):.1f} x the tared spread")
open(os.path.join(out, "sorted_cells_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
