"""DEV-LINK-IAMA-METAA-02: the IAM-A window each dose must fall in, from the measured Met-A alone (2026-10-10, before any EM-seq reading).
Derivation (DEV-LINK-IAMA-METAA-01, nothing fitted): a loss δ on methylated molecules moves a methylated identity site to μ(1-δ).
Curve A (loss only): unmethylated sites unchanged. Curve B (symmetric bound): unmethylated sites to μ + (1-μ)δ.
Met-A_pred(δ) = mean H(β') / mean H(μ) over the 6,000 identity sites, μ = the two vehicles' mean (identity_sites_vehicle_mu.csv).
Invert each curve at the measured Met-A_rel -> δ_A (largest δ allowed) and δ_B (smallest). IAM-A_rel at each δ from Stage Q's measured
response to that loss on the vehicle molecules. Prediction: the measured IAM-A_rel lies between the two.
Run: python3 predict_iama_window_02.py [RESPONSE.csv from insilico_loss_02.py on the vehicle molecules]"""
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
# The step δ -> IAM-A_rel is Stage Q's measured response (insilico_loss_02.py on the vehicle molecules), not the simple form
# ε_v + δ(1-ε_v): Stage Q reads only molecules >= 80 % methylated and only isolated errors, so its reading rises about half as fast.
# Argument: the in-silico response CSV. A δ beyond the range where >= 70 % of molecules are read (the note's rule) gives no limit.
RESP = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else pd.read_csv(os.path.join(HERE, "insilico_standin_neutrophil_SRR9888330.csv"))
dmax = float(RESP[RESP.share_read >= 0.70].delta.max())
def iama(d): return None if d is None or d > dmax else float(np.interp(d, RESP.delta, RESP.IAMA_rel))
print(f"response: {os.path.basename(sys.argv[1]) if len(sys.argv) > 1 else 'neutrophil stand-in SRR9888330'} | eps_v {RESP.eps.iloc[0]:.4f} | readable (>= 70 %) to delta {dmax:.2f}")
print("dose                   Met-A_rel  delta_B  delta_A | IAM-A_rel window [curve B, curve A]")
for t, g in R.groupby("treatment", sort=False):
    if t == "vehicle": continue
    for lab, m in [(f"{t} mean", g.MetA_rel.mean())] + [(f"  {a}", v) for a, v in zip(g.array, g.MetA_rel)]:
        dA, dB = invert(m, False), invert(m, True); wB, wA = iama(dB), iama(dA)
        fmt = lambda x: f"{x:.3f}" if x is not None else "none"
        print(f"{lab:22s} {m:.4f}  {dB if dB is not None else float('nan'):.4f}  {dA if dA is not None else float('nan'):.4f} | [{fmt(wB)}, {fmt(wA)}]"
              + ("" if wA is not None else "  (upper limit outside the readable range: lower limit only)"))
print("curve maxima: A", round(max(curve(d, False) for d in np.linspace(0, 0.6, 601)), 4), "B", round(max(curve(d, True) for d in np.linspace(0, 0.6, 601)), 4))
