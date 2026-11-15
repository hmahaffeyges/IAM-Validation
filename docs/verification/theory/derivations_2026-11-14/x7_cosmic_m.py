#!/usr/bin/env python3
"""EXPLORATION (register X7, R8; GitHub explore.yml), 2026-11-14 night. The book (p1_02, the M number, cosmological bullet): 'M integrated
self-consistently over cosmic time with gravitational feedback is read as the activation function E(a) = exp(1 - 1/a) ... Carrying out that integration
is the next step.' First step: the cosmic M number itself, M(a) = E_drive / (k T_H(a)), with E_drive the virial kinetic energy per particle of the
typical collapsing halo (mass M_* where sigma(M_*) D(a) = 1.686; v^2 = G M_*/R_vir at 200 x mean density; mu = 0.6 m_p), T_H = hbar H / 2 pi k.
Reported: log10 M, its growth per e-fold d ln M / d ln a, beside the record's required growth d ln E / d ln a = 1/a = 1 + z, and the same for the
mass-weighted mean v^2 over halos above 1e8 Msun. Planck 2018, colossus. Writes xout/x7_cosmic_m_summary.txt."""
import subprocess, sys
try:
    __import__("colossus")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "colossus"], check=True)
import os, numpy as np
from colossus.cosmology import cosmology
from colossus.lss import peaks, mass_function
_trap = getattr(np, "trapezoid", None) or np.trapz
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
c = cosmology.setCosmology("p18x", params={"flat": True, "H0": 67.36, "Om0": 0.3153, "Ob0": 0.0493, "sigma8": 0.8111, "ns": 0.9649, "relspecies": True}, persistence="")
G, mp, k, hbar, Msun, Mpc = 6.674e-11, 1.6726e-27, 1.380649e-23, 1.0546e-34, 1.989e30, 3.0857e22
h = 0.6736
zs = np.array([9, 7, 5, 4, 3, 2, 1.5, 1, 0.5, 0.2, 0.0]); a = 1 / (1 + zs)
def v2_of(Mh, z):                                   # Mh in Msun/h; R_vir at 200 x mean matter density (physical)
    rho = 200 * c.rho_m(z) * 1e9 * Msun / Mpc**3 * h**2   # colossus rho in Msun h^2 / kpc^3
    M = Mh / h * Msun; R = (3 * M / (4 * np.pi * rho)) ** (1 / 3)
    return G * M / R
H = c.Hz(zs) * 1e3 / Mpc
Mstar = np.array([peaks.massFromPeakHeight(1.0, z) for z in zs])
TH = hbar * H / (2 * np.pi * k)
Mn_star = 0.5 * 0.6 * mp * np.array([v2_of(m, z) for m, z in zip(Mstar, zs)]) / (k * TH)
lnM = np.linspace(np.log(1e8), np.log(1e16), 300); Mg = np.exp(lnM)
vbar = []
for z in zs:
    dn = mass_function.massFunction(Mg, z, q_in="M", q_out="dndlnM", mdef="200m", model="tinker08")
    w = dn * Mg; vbar.append(_trap(w * v2_of(Mg, z), lnM) / _trap(w, lnM))
Mn_bar = 0.5 * 0.6 * mp * np.array(vbar) / (k * TH)
lna = np.log(a)
g_star, g_bar = np.gradient(np.log(Mn_star), lna), np.gradient(np.log(Mn_bar), lna)
L = ["X7 cosmic M number (exploration): M = virial kinetic energy per particle / (k T_H).",
     "   z    M_* (Msun/h)   log10 M(M_*)   dlnM/dlna   log10 M(mean)  dlnM/dlna   required dlnE/dlna = 1+z"]
for i, z in enumerate(zs):
    L.append(f"  {z:4.1f}  {Mstar[i]:11.3e}    {np.log10(Mn_star[i]):8.3f}     {g_star[i]:7.3f}     {np.log10(Mn_bar[i]):8.3f}    {g_bar[i]:7.3f}      {1 + z:5.2f}")
L += ["Reading guide (exploration): M is enormous (the drive per particle against the horizon's tiny kT_H), so any 'Boltzmann factor in M' reading needs",
      "a per-record normalisation; what the row asks first is whether M's growth per e-fold has the shape of 1 + z (falling toward 1 today)."]
open(os.path.join(OUT, "x7_cosmic_m_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
