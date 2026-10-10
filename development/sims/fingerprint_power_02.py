"""DEV-FINGERPRINT-01 power through Stage Q's measured response (2026-10-10; replaces fingerprint_power_01.py's curve, which was written against
the simple-form IAM-A). Curve B (DEV-LINK-IAMA-METAA-01 table: Met-A_rel against simple-form IAM-A_rel) is re-expressed against Stage Q's IAM-A_rel
with the response measured on healthy prostate molecules (IAMA_rel_simple -> IAMA_rel). Precisions unchanged (Met-A 0.020 per array; IAM-A
repeat 0.009 per run, a Stage Q quantity). Same detection rule. Run: python3 fingerprint_power_02.py RESPONSE.csv"""
import sys, numpy as np, pandas as pd
R = pd.read_csv(sys.argv[1])
XS = np.array([1.00, 1.02, 1.05, 1.10, 1.16]); YB = np.array([1.000, 1.023, 1.056, 1.115, 1.188])     # curve B against simple-form IAM-A
XQ = np.interp(XS, R.IAMA_rel_simple, R.IAMA_rel)                                                    # the same points in Stage Q's IAM-A
SLOPE = np.polyfit(XQ, YB, 1)[0]; SD_MET, SD_IAM = 0.020, 0.009
def curveB(i): return np.interp(i, XQ, YB, right=YB[-1] + SLOPE * (i - XQ[-1]))
def detect(excess, nA, nW, I_true, N=4000, seed=11):
    r = np.random.default_rng(seed); vI = SD_IAM ** 2 * (1 / nW + 1 / max(nW, 4)); hit = 0
    se = np.sqrt(SD_MET ** 2 * (2 / nA) + SLOPE ** 2 * vI)
    for _ in range(N):
        I = I_true + r.normal(0, np.sqrt(vI)); M = curveB(I_true) + excess + r.normal(0, SD_MET * np.sqrt(2 / nA))
        hit += (M - curveB(I)) / se > 1.645
    return hit / N
I_true = float(np.interp(1.15, R.IAMA_rel_simple, R.IAMA_rel))
print("curve B points in Stage Q IAM-A:", XQ.round(4).tolist(), f"| slope {SLOPE:.3f} (simple form 1.17) | design point 1.15 -> {I_true:.4f}")
print("excess  2arr/4runs(design)  2arr/2runs  3arr/5runs")
for ex in (0.0, 0.02, 0.04, 0.06, 0.10): print(f"{ex:5.2f} {detect(ex, 2, 4, I_true):12.2f} {detect(ex, 2, 2, I_true):11.2f} {detect(ex, 3, 5, I_true):11.2f}")
