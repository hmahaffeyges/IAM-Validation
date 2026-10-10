"""DEV-SAM-LEVER-01, power through Stage Q's measured response (2026-10-10; replaces the power columns of sam_lever_01.py).
sam_lever_01 compared the effect in true copy error against a spread converted by the simple form, and its cross-lab column used the withdrawn
reference-free cross-species table. Here both are in Stage Q's own units: the calculated fold rise f = ε/ε0 (common.restore_ratio_A, unchanged)
is read through Stage Q's measured response to a planted loss (true fold eps_simple/eps_v -> IAMA_rel), and compared with an IAM-A spread between
animals of SD 0.01 / 0.02 / 0.04 (0.02 = the healthy donor spread on Stage Q). Two stand-in responses until the mouse wild-type molecules exist:
HCT116 vehicle (EM-seq) and Loyfer prostate epithelium (WGBS). Test: one-sided Mann-Whitney, 6 knockout vs 8 wild type, alpha 0.05.
Run: python3 sam_lever_02.py RESPONSE.csv [...]"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import mannwhitneyu
from common import restore_ratio_A, EPS0
from sam_lever_01 import FOLD
def power(eff, sd, n1=6, n2=8, sims=4000, seed=1):
    r = np.random.default_rng(seed)
    return float(np.mean([mannwhitneyu(r.normal(eff, sd, n1), r.normal(0, sd, n2), alternative="greater").pvalue < 0.05 for _ in range(sims)]))
for f in sys.argv[1:]:
    R = pd.read_csv(f); fold = R.eps_simple / R.eps.iloc[0]
    print(f"== response {os.path.basename(f)} (eps_v {R.eps.iloc[0]:.4f})")
    print("SAM_uM renewed  true_IAMA  StageQ_IAMA  power sd0.01  sd0.02  sd0.04")
    for S0 in (30, 60, 90):
        for ren in (0.3, 0.6, 1.0):
            A, e = restore_ratio_A(S0, FOLD, ren); q = float(np.interp(e / EPS0, fold, R.IAMA_rel)); eff = q - 1
            print(f"{S0:6d} {ren:7.1f} {A:10.4f} {q:12.4f} {power(eff, 0.01):12.2f} {power(eff, 0.02):7.2f} {power(eff, 0.04):7.2f}", flush=True)
