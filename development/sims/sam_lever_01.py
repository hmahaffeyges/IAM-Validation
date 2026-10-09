"""DEV-SAM-LEVER-01: predicted IAM-A for the Mat1a knockout (SAM -74 %) and power, 6 knockout vs 8 wild type. Run: python3 sam_lever_01.py"""
import math, numpy as np
from scipy.stats import mannwhitneyu
from common import restore_ratio_A, within_species_sd, H2, EPS0
FOLD = 1 / (1 - 0.74)                     # Lu et al. 2001: hepatic AdoMet -74 %
SD_XSP = within_species_sd()
SD_LAB = 0.02 / ((H2(EPS0 * 1.01) / H2(EPS0) - 1) / math.log(1.01))   # IAM-A donor SD 0.02 in SD of ln eps
def power(eff, sd, n1=6, n2=8, sims=4000, seed=1):
    r = np.random.default_rng(seed)
    return float(np.mean([mannwhitneyu(r.normal(eff, sd, n1), r.normal(0, sd, n2), alternative="greater").pvalue < 0.05 for _ in range(sims)]))
if __name__ == "__main__":
    print(f"fold {FOLD:.2f} | within-species SD ln eps {SD_XSP:.3f} | same-lab SD ln eps {SD_LAB:.4f}")
    print("SAM_uM renewed IAM-A  power(same-lab) power(2x) power(cross-lab)")
    for S0 in (30, 60, 90):
        for ren in (0.3, 0.6, 1.0):
            A, e = restore_ratio_A(S0, FOLD, ren); eff = math.log(e / EPS0)
            print(f"{S0:6d} {ren:7.1f} {A:6.3f} {power(eff, SD_LAB):14.2f} {power(eff, 2 * SD_LAB):9.2f} {power(eff, SD_XSP):16.2f}")
