"""DEV-METAA-ATLAS-01, design additions 1 and 2, on the atlas alone (2026-10-21; no array read, not sealed). For each panel cell: the healthy
width (chain/stage_metaa_atlas.width) at its 6,000 identity sites, the effective number of sites if each were weighted by one over its width
squared (Kish), and what a 1 % loss of the pattern moves Met-A by, equal and weighted (the weighted sum written here, the chain unchanged).
Usage: python3 width_weight_04.py   (writes width_weight_04_summary.txt beside itself)"""
import os, sys, numpy as np
H_ = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H_, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
L = ["DEV-METAA-ATLAS-01 width weighting, on the atlas alone"]
for c in ["neutrophils", "monocytes", "b cells", "cd4 t cells", "cd8 t cells", "nk cells"]:
    S = SA.identity_sites(c); w = SA.width(c).reindex(S).to_numpy(); mu = SA._col("mu", c).reindex(S).to_numpy(); x = mu + 0.01 * (0.5 - mu)
    wt = 1 / w ** 2; ne = wt.sum() ** 2 / (wt ** 2).sum()
    eq = SA._H(x).mean() / SA._H(mu).mean() - 1; we = (wt * SA._H(x)).sum() / (wt * SA._H(mu)).sum() - 1
    L.append(f"{c}: width {w.min():.4f}-{w.max():.4f} (median {np.median(w):.4f}) | effective sites {ne:.0f} of {len(S)} | 1 % loss moves A: equal {eq:+.5f}, weighted {we:+.5f}")
open(os.path.join(H_, "width_weight_04_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
