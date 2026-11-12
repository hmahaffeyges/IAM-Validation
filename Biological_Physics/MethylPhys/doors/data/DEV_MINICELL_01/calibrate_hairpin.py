"""DEV-MINICELL-01 first calibration from the hairpin genotypes (GSE116482, mouse stem cells; restores per loss at the most fully held sites,
from the spent genotype counts). Writes calibrate_hairpin.txt. Development; a simulation of the model, not data."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "minicell"))
import numpy as np, minicell as mc
from scipy.optimize import brentq
LAG = 0.10                              # copier lag, assumed (cycles)
SKO = (17.23 + 15.52) / 2               # no writers, no erasers (Tet1-3 + Dnmt3a/b/c knockout), two replicates
WT = (12.36 + 11.86) / 2                # wild type, two replicates
LEVEL = 1 - 0.04376                     # held level at held-marked sites: the human somatic canon value stands in (limit)
def rpl_alone(m, phases=20):
    p0 = np.array([0.0, 0.0, 1.0]); s = [p0 @ mc._cycle_matrix(m, 0.0, 0.0, LAG, t_obs=(j + 0.5) / phases)[1] for j in range(phases)]
    U, H, M = np.mean(s, axis=0); return M / H
m = brentq(lambda x: rpl_alone(x) - SKO, 0.5, 0.99999)
def wt(e):
    d = brentq(lambda x: (lambda u: u[2] + u[1] / 2)(mc.class_state(m, x, e, LAG)) - LEVEL, 1e-6, 0.999)
    U, H, M = mc.class_state(m, d, e, LAG); return M / H, d
L = [f"maintenance alone: restores per loss {SKO:.2f} -> m = {m:.4f} per cycle (lag {LAG})", f"wild type target: {WT:.2f}; held level {LEVEL:.3f}"]
for e in (0.0, 0.01, 0.03, 0.06, 0.10): r, d = wt(e); L.append(f"eraser e = {e:.2f}: wild-type restores per loss {r:.2f} (writer d = {d:.3f})")
open(os.path.join(HERE, "calibrate_hairpin.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
