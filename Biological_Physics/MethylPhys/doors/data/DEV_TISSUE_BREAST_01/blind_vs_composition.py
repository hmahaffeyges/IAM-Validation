"""DEV-TISSUE-BREAST-01 synthetic test 6 (2026-10-11; real normals, no plant). Real normal tissue varies in epithelial share. Is the plate-tared
Met-A at the composition-blind sites (blind_sites_test.py) independent of it? Spearman of each tared normal's reading with its fitted epithelial
share (f_luminal + f_basal, null_normals_output.csv) and adipocyte share; and the same for the composition-aware reading, for contrast.
PASS: |Spearman| < 0.2 with p > 0.05 for the blind reading. Writes blind_vs_composition_output.txt. Usage: python3 blind_vs_composition.py POSTDIR BETADIR"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.argv = sys.argv[:3]
src = open(os.path.join(HERE, "blind_sites_test.py")).read(); exec(src[:src.index("tared = np.array")])   # blind sites, base, ref
T = pd.read_csv(os.path.join(HERE, "null_normals_output.csv")).set_index("sample").loc[keep]
T["blind_tared"] = [base[s] / ref[s] for s in keep]
T["comp_tared"] = [T.loc[s, "comp"] / np.median([T.loc[o, "comp"] for o in keep if plates[o] == plates[s] and o != s]) for s in keep]
T["epi"] = T.f_luminal + T.f_basal; out = [f"normals {len(T)} | epithelial share {T.epi.min():.3f}-{T.epi.max():.3f}"]
for r in ("blind_tared", "comp_tared"):
    for c in ("epi", "f_adipocytes"):
        rho, p = spearmanr(T[r], T[c]); out.append(f"{r} vs {c}: Spearman {rho:+.3f} (p {p:.1e})")
rho, p = spearmanr(T.blind_tared, T.epi); out.append("SYNTHETIC TEST 6 " + ("PASSED" if abs(rho) < 0.2 and p > 0.05 else "FAILED"))
open(os.path.join(HERE, "blind_vs_composition_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
