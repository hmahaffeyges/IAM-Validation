"""DEV-LOWFRAC-01 basis of MIN_READ_FRACTION (chain/conductor_v3.py): the shift a known 1 % loss of the neutrophil pattern causes in whole-blood
Met-A, by neutrophil fraction, computed with the chain's own reader (stage_m_blood) and matrices (blood_composition_EPIC_v1 profiles).
A noiseless whole blood is built as e = sum_g f_g mu_g with the other groups in their median healthy proportions; the 1 % loss is the chain's own
definition (x + f*0.01*(0.5 - mu_NEU)). Run: python3 shift_vs_fraction_01.py"""
import os, sys, numpy as np, pandas as pd
CH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../chain")); sys.path.insert(0, CH); os.chdir(CH)
import conductor_v3 as C3
B = C3._bc(); S = pd.Index(B["neutrophil_sites"]); P = {g: pd.Series(v, index=S, dtype="float64") for g, v in B["profiles_at_neutrophil_sites"].items()}
OTHER = {"EOS": 0.03, "BASO": 0.005, "MONO": 0.08, "B": 0.06, "NK": 0.05, "CD4T": 0.16, "CD8T": 0.08}; tot = sum(OTHER.values())
def H(b): b = b.clip(1e-6, 1 - 1e-6); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
print("fraction | shift per 1 % loss | 2 % | 4 %")
for f in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
    fr = {"NEU": f, **{g: (1 - f) * v / tot for g, v in OTHER.items()}}; e = sum(v * P[g] for g, v in fr.items() if g in P)
    base = float(H(e).mean() / H(e).mean()); sh = {}
    for loss in (0.01, 0.02, 0.04):
        x = e + f * loss * (0.5 - P["NEU"]); sh[loss] = float(H(x).mean() / H(e).mean()) - base
    print(f"{f:.2f}     | {sh[0.01]:.4f} | {sh[0.02]:.4f} | {sh[0.04]:.4f}")
print("MIN_READ_FRACTION in the chain:", C3.MIN_READ_FRACTION)
