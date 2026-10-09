"""DEV-SYNTH-LEVERS-01 cancer both-ways test (GSE86833 design): power to detect Met-A above curve B of DEV-LINK-IAMA-METAA-01.
Precisions: Met-A held-out spread 0.020 per array (commissioning, neutrophils); IAM-A repeat 0.009 per run (DEV-IAMA-XCELL-01).
Run: python3 fingerprint_power_01.py"""
import numpy as np
XS = np.array([1.00, 1.02, 1.05, 1.10, 1.16]); YB = np.array([1.000, 1.023, 1.056, 1.115, 1.188])   # curve B, DEV-LINK-IAMA-METAA-01 table
SLOPE = np.polyfit(XS, YB, 1)[0]; SD_MET, SD_IAM = 0.020, 0.009
def curveB(i): return np.interp(i, XS, YB, right=YB[-1] + SLOPE * (i - XS[-1]))
def detect(excess, nA, nW, I_true=1.15, N=4000, seed=11):
    r = np.random.default_rng(seed); vI = SD_IAM ** 2 * (1 / nW + 1 / max(nW, 4)); hit = 0
    se = np.sqrt(SD_MET ** 2 * (2 / nA) + SLOPE ** 2 * vI)
    for _ in range(N):
        I = I_true + r.normal(0, np.sqrt(vI)); M = curveB(I_true) + excess + r.normal(0, SD_MET * np.sqrt(2 / nA))
        hit += (M - curveB(I)) / se > 1.645
    return hit / N
if __name__ == "__main__":
    print(f"slope of curve B {SLOPE:.3f}"); print("excess  2arr/2runs 2arr/4runs 3arr/5runs")
    for ex in (0.0, 0.02, 0.04, 0.06, 0.10):
        print(f"{ex:5.2f} " + " ".join(f"{detect(ex, a, w):10.2f}" for a, w in ((2, 2), (2, 4), (3, 5))))
