"""DEV-LINK-IAMA-METAA-02: the IAM-A window each dose must fall in, from the measured Met-A alone (2026-10-10, before any EM-seq reading).
Derivation (DEV-LINK-IAMA-METAA-01, nothing fitted): a loss δ on methylated molecules moves a methylated identity site to μ(1-δ).
Curve A (loss only): unmethylated sites unchanged. Curve B (symmetric bound): unmethylated sites to μ + (1-μ)δ.
Met-A_pred(δ) = mean H(β') / mean H(μ) over the 6,000 identity sites, μ = the two vehicles' mean (identity_sites_vehicle_mu.csv).
Invert each curve at the measured Met-A_rel -> δ_A (largest δ allowed) and δ_B (smallest). Copy error: ε = ε_v + δ(1-ε_v), ε_v = the vehicle
EM-seq copy error (Stage Q). Prediction: IAM-A_rel = H(ε)/H(ε_v) lies between H(ε_v+δ_B(1-ε_v))/H(ε_v) and H(ε_v+δ_A(1-ε_v))/H(ε_v).
Run: python3 predict_iama_window_02.py [eps_v ...]   (default: a grid, the window for any vehicle copy error)"""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(os.path.join(HERE, "identity_sites_vehicle_mu.csv")); R = pd.read_csv(os.path.join(HERE, "metaa_dose_02_rows.csv"))
hi = S[S.set == "hi"].mu_vehicle.values; lo = S[S.set == "lo"].mu_vehicle.values
def H(b): b = np.clip(b, 1e-9, 1 - 1e-9); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
den = np.concatenate([H(hi), H(lo)]).mean()
def curve(d, sym): return np.concatenate([H(hi * (1 - d)), H(lo + (1 - lo) * d) if sym else H(lo)]).mean() / den
def invert(m, sym):
    dd = np.linspace(0, 0.6, 6001); c = np.array([curve(d, sym) for d in dd]); i = int(np.argmax(c))   # curve rises to a maximum, then falls
    if m > c[i]: return None
    return float(brentq(lambda d: curve(d, sym) - m, 0, dd[i]))
eps_list = [float(x) for x in sys.argv[1:]] or [0.02, 0.03, 0.04, 0.05, 0.06]
print("dose        Met-A_rel  delta_B  delta_A  | IAM-A_rel window [B, A] for vehicle copy error eps_v")
for t, g in R.groupby("treatment", sort=False):
    if t == "vehicle": continue
    for lab, m in [(f"{t} mean", g.MetA_rel.mean())] + [(f"  {a}", v) for a, v in zip(g.array, g.MetA_rel)]:
        dA, dB = invert(m, False), invert(m, True)
        win = "  ".join(f"{e:.2f}: [{H(e + dB * (1 - e)) / H(e):.3f}, {H(e + dA * (1 - e)) / H(e):.3f}]" if dA is not None and dB is not None else f"{e:.2f}: beyond curve A" for e in eps_list)
        print(f"{lab:22s} {m:.4f}  {dB if dB is not None else float('nan'):.4f}  {dA if dA is not None else float('nan'):.4f}  | {win}")
print("curve maxima: A", round(max(curve(d, False) for d in np.linspace(0, 0.6, 601)), 4), "B", round(max(curve(d, True) for d in np.linspace(0, 0.6, 601)), 4))
