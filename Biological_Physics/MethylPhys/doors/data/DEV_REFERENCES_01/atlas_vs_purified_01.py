"""DEV-REFERENCES-01, exploration (2026-10-21, not sealed): the two healthy references the chain holds, compared site by site.
(1) Met-A's purified-array reference: the eight blood groups' healthy means at Met-A's 6,000 neutrophil identity sites
    (Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json, 91 purified arrays of GSE110554 and GSE167998, the chain's Stage 1).
(2) IAM-Atlas v2.1's posterior means for the six panel cells (atlas/v2.1 compact copy, 21 sources, arrays and sequencing).
Per cell: the methylation gap, its correlation, and the entropy H(beta) Met-A reads; for neutrophils, the gap by methylation level and Met-A
read on the atlas neutrophil as if it were a sample (mean H over the sites / the healthy reference 0.330263 bits from metA_floors_v1_3.json).
Note: (1)'s two studies are among (2)'s sources, so agreement is not independent. Writes atlas_vs_purified_01_summary.txt here."""
import os, sys, json, numpy as np, pandas as pd
H_ = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H_, "..", "..", "..", "chain"); sys.path.insert(0, CH)
import stage_deconv_v21 as D
RM = os.path.join(CH, "Runtime Matrices", "Met_A_Floors")
B = json.load(open(os.path.join(RM, "blood_composition_EPIC_v1.json"))); idx = pd.Index(B["neutrophil_sites"]); P = B["profiles_at_neutrophil_sites"]
HREF = json.load(open(os.path.join(RM, "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]["floor"]
R = D.reference(list(D.BLOOD6)); mu = R["mu"]
G = {"neutrophils": "NEU", "monocytes": "MONO", "b cells": "B", "cd4 t cells": "CD4T", "cd8 t cells": "CD8T", "nk cells": "NK"}
def H(b): b = np.clip(b, 1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
L = [f"Met-A's healthy reference (metA_floors_v1_3.json): {HREF:.6f} bits per site"]
for c, g in G.items():
    a = mu[c].reindex(idx); m = pd.Series(P[g], index=idx); k = a.notna() & m.notna(); d = a[k] - m[k]
    L.append(f"{c}: {int(k.sum())} sites; mean |gap| {d.abs().mean():.4f}, 95th percentile {d.abs().quantile(.95):.4f}, mean gap {d.mean():+.4f}, "
             f"r {np.corrcoef(a[k], m[k])[0, 1]:.4f}; mean entropy atlas {H(a[k]).mean():.4f}, purified arrays {H(m[k]).mean():.4f}")
a = mu["neutrophils"].reindex(idx); m = pd.Series(P["NEU"], index=idx); k = a.notna() & m.notna(); a, m = a[k], m[k]; dH = H(a) - H(m)
bins = pd.cut(m, [0, .05, .1, .3, .7, .9, .95, 1.0], include_lowest=True)
for b, g in m.groupby(bins, observed=True):
    i = g.index; L.append(f"neutrophils, purified level {b}: {len(i)} sites, gap {(a[i] - m[i]).mean():+.4f}, entropy {H(m[i]).mean():.4f} -> {H(a[i]).mean():.4f}, share of the entropy gap {dH[i].sum() / dH.sum():.2f}")
L.append(f"Met-A read on the atlas neutrophil as a sample: {H(a).mean() / HREF:.4f} (Normal 0.95 to 1.05); the purified profile on itself: {H(m).mean() / HREF:.4f}")
open(os.path.join(H_, "atlas_vs_purified_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
