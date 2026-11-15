#!/usr/bin/env python3
"""EXPLORATION (register X7/R8; GitHub explore.yml), 2026-11-14 night. Why does the typical collapsing halo's M grow close to 2/a per e-fold?
Analytic split (200 x mean density): M_number ~ v^2 / H, v^2 = G M_h / R ~ M_h^(2/3) rho_m^(1/3), so
   d ln M / d ln a = (2/3) d ln M_* / d ln a  - 1  - d ln H / d ln a,
with d ln M_* / d ln a = f / s, f = d ln D / d ln a (growth rate), s = -d ln sigma / d ln M at M_* (= (n_eff + 3)/6, the local CDM spectral slope).
Columns: f, s, n_eff, the three terms, their sum, the direct numerical slope (check), and 2/a. Writes xout/x7_cosmic_m_why_summary.txt."""
import subprocess, sys
try:
    __import__("colossus")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "colossus"], check=True)
import os, numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
c = cosmology.setCosmology("p18x", params={"flat": True, "H0": 67.36, "Om0": 0.3153, "Ob0": 0.0493, "sigma8": 0.8111, "ns": 0.9649, "relspecies": True}, persistence="")
zs = np.array([9, 5, 4, 3, 2, 1.5, 1, 0.5, 0.2, 0.0])
L = ["   z     f      s     n_eff   (2/3)f/s   -1    -dlnH/dlna   sum     direct   2/a"]
for z in zs:
    a = 1 / (1 + z); e = 1e-3
    lnD = lambda zz: np.log(c.growthFactor(zz))
    f = -(lnD(z + e * (1 + z)) - lnD(z - e * (1 + z))) / (2 * e)          # d ln D / d ln a, with d ln a = -dz/(1+z)
    Ms = peaks.massFromPeakHeight(1.0, z); R = peaks.lagrangianR(Ms)
    s = -(np.log(c.sigma(peaks.lagrangianR(Ms * 1.01), 0)) - np.log(c.sigma(peaks.lagrangianR(Ms / 1.01), 0))) / (2 * np.log(1.01))
    dlnH = (np.log(c.Hz(z - e * (1 + z))) - np.log(c.Hz(z + e * (1 + z)))) / (2 * e)
    t1 = (2 / 3) * f / s; tot = t1 - 1 - dlnH
    Mn = lambda zz: (peaks.massFromPeakHeight(1.0, zz)) ** (2 / 3) * (1 + zz) / c.Hz(zz)
    direct = (np.log(Mn(z - e * (1 + z))) - np.log(Mn(z + e * (1 + z)))) / (2 * e)
    L.append(f"  {z:4.1f}  {f:5.3f}  {s:5.3f}  {6 * s - 3:6.2f}   {t1:7.3f}   -1   {-dlnH:8.3f}   {tot:6.3f}   {direct:6.3f}   {2 / a:5.2f}")
L += ["Reading guide (exploration): which term carries the 1/a: the CDM spectrum's running slope at the collapsing mass (s falls as M_* grows), the",
      "growth rate f (Lambda slows it late), or the Hubble term."]
open(os.path.join(OUT, "x7_cosmic_m_why_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
