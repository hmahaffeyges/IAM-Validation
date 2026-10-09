"""DEV-SYNTH-LEVERS-01 hamster torpor (GSE199815, 3 v 3): power for the passive-bit fall (30 K colder, eps x 0.70) and the driven-bit
small residual (20 % or none renewed). Run: python3 hibernation_01.py"""
import math, numpy as np
from scipy.stats import ttest_ind
from common import within_species_sd
SD = within_species_sd()
def power(eff, n=3, sims=6000, seed=2):
    r = np.random.default_rng(seed)
    return float(np.mean([ttest_ind(r.normal(eff, SD, n), r.normal(0, SD, n)).pvalue < 0.05 for _ in range(sims)]))
if __name__ == "__main__":
    for lab, eff in (("passive bit, eps falls 30 %", math.log(0.70)), ("driven bit, 20 % renewed", math.log(0.94)), ("driven bit, none renewed", 0.0)):
        print(f"{lab}: power 3 v 3 = {power(eff):.2f}")
