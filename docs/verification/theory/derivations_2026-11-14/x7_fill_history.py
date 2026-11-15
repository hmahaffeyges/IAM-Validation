#!/usr/bin/env python3
"""EXPLORATION (register X7; GitHub explore.yml), 2026-11-15 afternoon. Follow-up to x7_tiles.py, whose lead was that the record keeps a ~constant
share (~Omega_m/2) of the horizon's capacity between today and maturity. Check the lead across the whole history: the fill fraction
f(a) = S_info(a)/S_geo(a), S_geo(a) = S_geo(1)/(H/H0)^2, S_info from the book's bottom-up rate priced at T_H (same code and canon amplitude as
x7_tiles.py), at a = 0.1 ... 12, for PS, ST and Tinker08. A constant f across the history would be a property of the ledger; f constant only
after today would be a late-time coincidence of the 1/Omega_L growth. Writes xout/x7_fill_history_summary.txt."""
import os, sys, subprocess, numpy as np
try:
    import colossus
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "colossus"], check=True)
from colossus.cosmology import cosmology
from colossus.lss import mass_function
trap = getattr(np, "trapezoid", None) or np.trapz
cumtrap = lambda y, x: np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
P18 = {'flat': True, 'H0': 67.36, 'Om0': 0.3153, 'Ob0': 0.0493, 'sigma8': 0.8111, 'ns': 0.9649, 'relspecies': True}
c = cosmology.setCosmology('p18chains', params=P18, persistence='')
Om0 = 0.3153
lnM = np.linspace(np.log(1e4), np.log(1e17), 320); M = np.exp(lnM)
lna = np.linspace(np.log(0.02), np.log(12.0), 340); a = np.exp(lna); z = 1 / a - 1; i1 = np.argmin(abs(lna))
show = (0.1, 0.2, 0.33, 0.5, 0.67, 1.0, 1.5, 2, 4, 12)
L = ["X7 fill-fraction history (exploration): f(a) = S_info(a)/S_geo(a), canon amplitude f(1) = Omega_m/2 = %.4f" % (Om0 / 2),
     "  model     " + " ".join(f"a={x:<5}" for x in show)]
for model, mdef in (("press74", "fof"), ("sheth99", "fof"), ("tinker08", "200m")):
    try:
        E = c.Ez(z); Om = c.Om(z); Dc = 18 * np.pi**2 + 82 * (Om - 1) - 39 * (Om - 1)**2
        dn = np.array([mass_function.massFunction(M, zz, q_in='M', q_out='dndlnM', mdef=mdef, model=model) for zz in z])
        U = trap(dn * M**(5 / 3) * ((Dc * E**2)[:, None])**(1 / 3), lnM, axis=1)
        S = cumtrap(a**-3 * np.gradient(U, lna) / E**4, lna)
        f = (S / S[i1]) * (Om0 / 2) * E**2          # S_geo(a)/S_geo(1) = 1/E^2
        L.append(f"  {model:9s} " + " ".join(f"{np.interp(np.log(x), lna, f):7.4f}" for x in show))
    except Exception as e:
        L.append(f"  {model:9s} failed: {type(e).__name__}: {str(e)[:120]}")
L.append("Reading guide: flat across a = 0.1-12 means the ledger keeps a fixed share of the horizon's capacity (a property to derive); rising to ~Omega_m/2 "
         "and flat only after a ~ 1 means a late-time coincidence of the capacity's 1/Omega_L growth.")
open(os.path.join(OUT, "x7_fill_history_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
