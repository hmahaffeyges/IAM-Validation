"""DEV-ERASE-01 prediction, computed before any data set is searched (2026-10-14).
Author's statement: the half and half in DNA methylation is one half writing, one half erasing (the virial pair of IAM's Law: the
kinetic kept and the heat written, K = Q = |V|/2). In the copy channel eps = f/(f+g): if the loss f is the writer's own misses f_w plus
erasure f_e in equal halves, removing the eraser leaves f_w, and the copy error falls to f_w/(f_w+f_e+... ) of its value, i.e. the RATIO
eps(eraser removed) / eps(same-lab control) = (1 - eps_c)/(1 - eps_ko) * f_w/(f_w+f_e) ~ 1/(1 + f_e/f_w). Inputs, all already committed:
healthy eps 0.032 (PROC-CHANNEL-01), DNMT1 discrimination D 47.4 / 63.7 / 80 measured outside the cell (DEV-WRITER-01)."""
import math, os
HERE = os.path.dirname(os.path.abspath(__file__))
eps_c = 0.032
L = ["DEV-ERASE-01 prediction (prediction_erase_01.py), computed before any data set is searched"]
def ratio(r, eps=eps_c):                      # r = f_e / f_w ; eps_ko from removing f_e at fixed g
    gf = (1 - eps) / eps; fw_over_ftot = 1 / (1 + r); eps_ko = 1 / (1 + gf / fw_over_ftot); return eps_ko / eps, eps_ko
for lab, r in (("equal halves (author)", 1.0), ("virial 2:1 (eraser = 2 x writer)", 2.0), ("virial 1:2 (eraser = writer / 2)", 0.5), ("no eraser share", 0.0)):
    q, e = ratio(r); L.append(f"{lab}: ratio {q:.3f}, eps with the eraser removed {e:.4f}")
for Dv in (47.4, 63.7, 80.0):
    L.append(f"writer alone at D {Dv}: eps {1/(1+Dv):.4f} (ratio to 0.032: {1/(1+Dv)/eps_c:.3f})")
_h = [math.log(Dv) - math.log(2) for Dv in (47.4, 80.0)]
L.append(f"holding with equal halves over the writer's range (ln D - ln 2): {_h[0]:.2f}-{_h[1]:.2f} kT; measured {math.log((1-eps_c)/eps_c):.2f} kT")
L += [f"RULE (fixed now): MET if the eraser-removed / same-lab-control copy-error ratio is within 0.40-0.60 (excludes the virial 2:1 at {ratio(2.0)[0]:.3f}",
      f"and 1:2 at {ratio(0.5)[0]:.3f}); NOT MET if 0.85 or above (no change: the extra loss is not erasure); anything else INCONCLUSIVE."]
open(os.path.join(HERE, "prediction_erase_01.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
