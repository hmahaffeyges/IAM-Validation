"""DEV-SELFTARE-03, the readings summarized (exploration, not sealed). Each variant's Met-A followed by the median tare (Stage T step 2):
each array against the median of the same-slide healthy arrays (at least 3, itself excluded), else the same series. Held to the bars that
adopted the current self-tare (DEV-SELFTARE-02): replicates within-person SD <= 0.020 and >= 95 % in Normal (0.95-1.05); other-laboratory
neutrophils every array in Normal. And the new question: a planted 1 % neutrophil loss at every site must be found (mean change over 2 SE).
Usage: python3 selftare03_score.py ROWS_DIR   Writes selftare03_summary.txt here."""
import os, sys, glob, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
T = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(sys.argv[1], "selftare03_*_rows.csv"))], ignore_index=True)
err = T[T.case == "ERROR"]; T = T[T.case != "ERROR"].copy(); T["A"] = T.A.astype(float); T["slide"] = T.slide.astype(str)
def tare(d):
    h = d[d.case == "healthy"]; out = []
    for _, x in d.iterrows():
        ref = h[(h.gsm != x.gsm) & (h.slide == x.slide)]
        if len(ref) < 3: ref = h[(h.gsm != x.gsm) & (h.series == x.series)]
        out.append(x.A / ref.A.median() if len(ref) >= 3 and pd.notna(x.A) else np.nan)
    d = d.copy(); d["A_rel"] = out; return d
L = [f"rows {len(T)}; arrays not read: {err.gsm.nunique()}" + (f" ({'; '.join(sorted(set(err.variant.astype(str))))[:200]})" if len(err) else "")]
for v in ("absolute", "relative", "none"):
    L.append(f"== self-tare {v}, then the median tare")
    R = tare(T[(T.set == "replicates") & (T.variant == v)])
    if len(R):
        sd = R.groupby("person").A_rel.std().mean(); inn = R.A_rel.between(0.95, 1.05).sum()
        L.append(f"  replicates: within-person SD {sd:.4f} (bar <= 0.020) | in Normal {inn}/{R.A_rel.notna().sum()} (bar >= 95 %)")
    O = tare(T[(T.set == "otherlab") & (T.variant == v)])
    if len(O): L.append(f"  other-laboratory neutrophils: in Normal {O.A_rel.between(0.95, 1.05).sum()}/{O.A_rel.notna().sum()} (bar: all), range {O.A_rel.min():.3f}-{O.A_rel.max():.3f}")
    G = T[(T.set == "groupa") & (T.variant == v)]
    if len(G):
        G = pd.concat([tare(G[G.case.isin(["healthy", c])]).assign(cc=c) for c in ("loss_every_site", "loss_identity_sites")])
        for c in ("loss_every_site", "loss_identity_sites"):
            g = G[G.cc == c]; p = g[g.case == c].set_index("gsm").A_rel; h = g[g.case == "healthy"].set_index("gsm").A_rel; d = (p - h).dropna()
            L.append(f"  Group-A healthy, 1 % neutrophil loss {c.replace('loss_', 'at ').replace('_', ' ')}: tared change {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f}, n {len(d)}) "
                     f"-> {'found' if d.mean() > 2 * d.std(ddof=1) / np.sqrt(len(d)) else 'not found'}; healthy spread after the tare {h.std():.4f}")
open(os.path.join(H, "selftare03_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
