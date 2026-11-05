"""The same-laboratory tare on sequencing, measured on committed readings (development; spent data only; nothing commissioned).
A laboratory's offset on sorted-cell Met-A is removed by a tare (A_rel = A / median of the same laboratory's healthy references of the same cell,
stage_metaa_atlas.tare) only if the offset is one factor for the laboratory and cell, the same for every donor. Then (1) the spread among
one laboratory's healthy donors of a cell is no wider than inside the atlas's own laboratory, and (2) a tared healthy sample reads like the atlas's
own healthy samples, with extra spread from the tare's references that shrinks with their number. This measures both:
  (1) within-cell spread (coefficient of variation) of own-cell Met-A: the atlas's laboratory (healthy pass 7), BLUEPRINT, the second laboratory;
  (2) for each laboratory and cell with at least n + 1 healthy donors, every sample tared on n others drawn from the rest (up to 2,000 draws per
      sample, seed fixed), for n = 3, 5, 8: the tared readings' median, spread, and share inside 0.95 to 1.05; beside the untared readings.
Usage: python3 same_lab_tare.py OUT.csv OUT.txt   (run from this folder)"""
import sys, numpy as np, pandas as pd
rng = np.random.default_rng(20261105)
A7 = pd.read_csv("../../../DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/healthy_fullchain7_readings.csv"); A7 = A7[A7.reading == "Met-A sorted cell"].dropna(subset=["A"])
BP = pd.read_csv("blueprint_healthy_test.csv").dropna(subset=["Met-A own cell"]); L2 = pd.read_csv("lab2_cd8_test.csv").dropna(subset=["Met-A own cell"])
D = pd.concat([pd.DataFrame(dict(lab="atlas laboratory (pass 7)", cell=A7.cell, A=A7.A)),
               pd.DataFrame(dict(lab="BLUEPRINT", cell=BP.cell, A=BP["Met-A own cell"])),
               pd.DataFrame(dict(lab="second laboratory (GSE245832)", cell=L2.cell, A=L2["Met-A own cell"]))], ignore_index=True)
cv = lambda x: float(np.std(x, ddof=1) / np.mean(x))
rows = []; L = ["(1) within-cell spread of own-cell Met-A (cells with at least 3 healthy donors):"]
for lab, g in D.groupby("lab", sort=False):
    c = g.groupby("cell").A.agg(count="count", median="median", cv=cv); c = c[c["count"] >= 3]
    L.append(f"  {lab}: {len(c)} cells, {int(c['count'].sum())} samples; untared median {g[g.cell.isin(c.index)].A.median():.4f}; within-cell spread median {c.cv.median():.4f}")
L.append("(2) tared on n healthy references of the same laboratory and cell (each sample left out of its own references):")
for n in (3, 5, 8):
    for lab, g in D.groupby("lab", sort=False):
        T = []
        for cell, h in g.groupby("cell"):
            a = h.A.to_numpy()
            if len(a) < n + 1: continue
            for i in range(len(a)):
                o = np.delete(a, i); k = min(2000, 1 if len(o) == n else 2000)
                T += [a[i] / np.median(rng.choice(o, n, replace=False)) for _ in range(k)]
        if not T: L.append(f"  n {n}, {lab}: no cell has {n + 1} donors"); continue
        T = np.array(T); ins = float(((T >= 0.95) & (T <= 1.05)).mean())
        rows.append(dict(n=n, lab=lab, draws=len(T), median=np.median(T), spread=np.std(T) / np.mean(T), inside=ins))
        L.append(f"  n {n}, {lab}: tared median {np.median(T):.4f}, spread {np.std(T) / np.mean(T):.4f}, inside 0.95 to 1.05 {ins:.3f} ({len(T)} draws)")
pd.DataFrame(rows).to_csv(sys.argv[1], index=False); open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
