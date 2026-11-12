"""The hold as a proportional regulator (candidate derivation of the healthy middle; development; nothing sealed as a verdict).
Where IAM's Law is: the loop gain K equals the coin in thermal kicks at the encoding surface, K = M = dG_ATP/(R T_cell) (canon M_cell).
The held error is the balance between writing (on SAM) and erasing (on oxygen and 2-oxoglutarate) at the held sites, independent of
division (canon rule 25). Borrowed (textbook control): a proportional loop with gain K leaves a steady error 1/(1+K). Borrowed (statistical
mechanics, equipartition / fluctuation-dissipation): a deviation held by a proportional restoring stiffness against thermal noise keeps a
mean energy of kT/2, so its size relative to the restoring energy is kT / (restoring energy), whatever the rates. This script computes the
candidate's numbers from the canon, then checks by simulation that the steady error of a proportional hold does not depend on how fast the
writer and eraser work (only on the gain), which is why such a hold is independent of division."""
import os, json, math, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); C = json.load(open(os.path.join(H, "..", "..", "..", "CANON", "iam_canon.json")))["constants"]
M = float(C["M_cell"]["value"]); Eh = float(C["E_hold_meth"]["value"]); phi = float(C["phi"]["value"]) if "phi" in C else Eh / M
L = [f"M (canon) = {M}", f"held error 1/(1+M) = {1/(1+M):.5f}   (healthy lone-loss intercept, pooled 197 samples: 0.04526, 95 % 0.04375-0.04657; meter-corrected 0.046-0.048)",
     f"holding energy ln M = {math.log(M):.4f} kT   (canon E_hold, raw counter at each sample's own loose share: {Eh})",
     f"phi = ln M / M = {math.log(M)/M:.4f}   (canon phi: {phi:.4f})",
     f"the floor (a wall M kicks high, Boltzmann): 1/(1+e^M) = {1/(1+math.exp(M)):.3e}; the middle (a proportional hold of gain M): 1/(1+M) = {1/(1+M):.5f}"]
# Langevin check: dx = -(k/g) x dt + sqrt(2 kT / g) dW, kT = 1, stiffness k = K; the rate scale g (how fast writer and eraser act) varied.
g0 = np.random.default_rng(20261112); L.append("\nProportional hold against thermal noise (kT = 1): steady <x^2> x K, by gain K and rate scale")
for K in (5.0, 20.94, 80.0):
    row = []
    for g in (0.2, 1.0, 5.0):
        dt = 0.02 * g / K; n = 200000; x = 0.0; s = 0.0; a = K / g * dt; b = math.sqrt(2 * dt / g); z = g0.standard_normal(n)
        for i in range(n):
            x += -a * x + b * z[i]
            if i > n // 10: s += x * x
        row.append(f"rate x{1/g:g}: {s / (n - n // 10 - 1) * K:.3f}")
    L.append(f"  K = {K}: " + "; ".join(row) + "   (equipartition: 1.000 at every rate)")
t = "\n".join(L) + "\n"; open(os.path.join(H, "regulator_summary.txt"), "w").write(t); print(t)

# The open step, second route (fluctuation-dissipation through the Einstein relation): a restoring drive of work W across one mark
# against thermal diffusion; the odds are the diffusive passage time over the driven one. Mean first-passage time from 0 (reflecting)
# to l (absorbing) with drift v and diffusion D: T(v) = int_0^l dy int_0^y dz exp(-(v/D)(y - z)) / D.
from scipy.integrate import quad
def _mfpt(v, D=1.0, l=1.0): return quad(lambda y: quad(lambda z: math.exp(-(v / D) * (y - z)), 0, y)[0] / D, 0, l)[0]
_L2 = ["\nSecond route (Einstein relation): odds = diffusive / driven passage across one mark, by W/kT (= Pe)"]
for _pe in (5.0, M, 80.0): _L2.append(f"  W/kT = {_pe:.2f}: odds {_mfpt(0.0) / _mfpt(_pe):.3f} (W/2kT = {_pe/2:.3f})")
_L2.append("Both routes give odds in proportion to the coin. The first (equipartition with the virial half) gives exactly M but rests on mapping a\n"
           "continuous restoring coordinate onto a two-state site; the second gives about W/2kT, so odds of M need the drive's work over one mark\n"
           "to be two coins' worth or a different geometry. The open step is narrowed to this factor of order one.")
_t2 = "\n".join(_L2) + "\n"; open(os.path.join(H, "regulator_summary.txt"), "a").write(_t2); print(_t2)
