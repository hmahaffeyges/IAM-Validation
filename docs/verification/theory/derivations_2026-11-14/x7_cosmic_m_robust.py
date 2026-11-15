#!/usr/bin/env python3
"""EXPLORATION (register X7/R8; GitHub explore.yml), 2026-11-14 night. Follow-up to x7_cosmic_m.py: the typical collapsing halo's cosmic M number
grew close to E(a)^2 from z = 4 to today. Here, integrated rather than per e-fold: (1/2) ln[M(a)/M(1)] against ln E(a) = 1 - 1/a = -z, for
definitions a cosmologist would vary: overdensity 200 x mean, 200 x critical, virial (Bryan-Norman); peak height nu = 1 and nu = 2 (rarer halos);
sigma_8 0.77 and 0.81. M = (1/2) mu m_p v^2 / (k T_H), v^2 = G M_h / R_h. Writes xout/x7_cosmic_m_robust_summary.txt."""
import subprocess, sys
try:
    __import__("colossus")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "colossus"], check=True)
import os, numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
G, Mpc, kpc, Msun = 6.674e-11, 3.0857e22, 3.0857e19, 1.989e30
zs = np.array([4, 3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0])
L = ["X7 robustness: (1/2) ln[M(z)/M(0)] for the typical collapsing halo, against ln E = -z (exploration)",
     "  " + " " * 34 + "  ".join(f"z={z:<4}" for z in zs) + "   max |diff| z<=2   max |diff| z<=4"]
L.append("  " + f"{'target ln E = -z':34s}" + "  ".join(f"{-z:6.2f}" for z in zs))
for s8 in (0.8111, 0.77):
    cosmology.setCosmology("p18x", params={"flat": True, "H0": 67.36, "Om0": 0.3153, "Ob0": 0.0493, "sigma8": s8, "ns": 0.9649, "relspecies": True}, persistence="")
    c = cosmology.getCurrent(); h = c.h
    for mdef in ("200m", "200c", "vir"):
        for nu in (1.0, 2.0):
            Mn = []
            for z in zs:
                Mh = peaks.massFromPeakHeight(nu, z)                       # Msun/h, the halo just collapsing at peak height nu
                R = mass_so.M_to_R(Mh, z, mdef) * kpc / h                    # physical kpc/h -> m
                v2 = G * (Mh / h * Msun) / R
                Mn.append(v2 / (c.Hz(z)))                                    # M up to constants: v^2 / H
            y = 0.5 * np.log(np.array(Mn) / Mn[-1])
            d2 = np.max(np.abs(y[zs <= 2] + zs[zs <= 2])); d4 = np.max(np.abs(y + zs))
            L.append("  " + f"{'s8 ' + str(s8) + ' ' + mdef + ' nu=' + str(nu):34s}" + "  ".join(f"{v:6.2f}" for v in y) + f"   {d2:8.3f}         {d4:8.3f}")
L += ["Reading guide (exploration): if each row tracks -z closely, sqrt(M) of the typical collapsing halo carries E(a)'s shape whatever reasonable",
      "definition is used; if the agreement depends on the choice, the lead is a coincidence of one definition."]
open(os.path.join(OUT, "x7_cosmic_m_robust_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
