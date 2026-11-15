#!/usr/bin/env python3
"""EXPLORATION (register X7; GitHub explore.yml), 2026-11-14 night. Stock versus flow. The book's bottom-up rate (verify_bottom_up_exponent.py) counts
the binding energy NEWLY written per e-fold, w = dU/dln a, and its exponent runs (5.3-5.8 at z = 9 to 2.0-2.4 at z = 1). The cell holds its record by
paying continuously in proportion to what it holds (restores per held mark). Cosmic analogue (re-virialisation, the book's own open route): each
halo re-records its state once per dynamical time, t_dyn ~ 1/(H sqrt(Delta_c Omega_m(z)/2)) at fixed overdensity, so the writing per e-fold is
U x (1/(H t_dyn)) ~ U sqrt(Delta_c Omega_m(z)). Rates tried, each through the book's n_eff = d ln[(rate)/f]/d ln D:
  flow (book) dU/dln a;  stock U;  stock re-recorded per dynamical time U sqrt(Delta_c Om(z));  flow + stock re-recorded.
Same halo model and parameters as the book's script. Writes xout/x7_stock_summary.txt."""
import subprocess, sys
try:
    __import__("colossus")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "colossus"], check=True)

import os, numpy as np
_trap = getattr(np, 'trapezoid', None) or np.trapz
from colossus.cosmology import cosmology
from colossus.lss import mass_function
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
P18 = {'flat': True, 'H0': 67.36, 'Om0': 0.3153, 'Ob0': 0.0493, 'sigma8': 0.8111, 'ns': 0.9649, 'relspecies': True}
c = cosmology.setCosmology('p18chains', params=P18, persistence='')
lna = np.linspace(np.log(0.01), 0, 600); a = np.exp(lna); z = 1 / a - 1
lnM = np.linspace(np.log(1e4), np.log(1e17), 320); M = np.exp(lnM)
E = c.Ez(z); D = c.growthFactor(z); f = np.gradient(np.log(D), lna); Om = c.Om(z); Dc = 18 * np.pi**2 + 82 * (Om - 1) - 39 * (Om - 1)**2
def neff(rate): return np.gradient(np.log(np.clip(rate, 1e-300, None) / f), np.log(D))
def migrate(w, k):
    L = np.zeros_like(w); dx = np.diff(lna)
    for i in range(1, len(w)):                      # implicit step: stable for large k
        L[i] = (L[i - 1] + dx[i - 1] * w[i]) / (1 + dx[i - 1] * k[i])
    return k * L
zs = (9, 5, 4, 3, 2, 1, 0.05)
at = lambda arr, q: np.interp(-q, -z, arr)
win = (z >= 1) & (z <= 9); win_md = (z >= 2.3) & (z <= 9)
L = ["X7 exploration: stock versus flow of halo writing. n_eff, the book's definition.",
     "  rate                       " + "  ".join(f"z={q:<4}" for q in zs) + "  | mean z2.3-9 | spread z1-9 (max-min)"]
for model, mdef in (("press74", "fof"), ("sheth99", "fof"), ("tinker08", "200m")):
    dn = np.array([mass_function.massFunction(M, zz, q_in='M', q_out='dndlnM', mdef=mdef, model=model) for zz in z])
    U = _trap(dn * M**(5 / 3) * ((Dc * E**2)[:, None])**(1 / 3), lnM, axis=1)
    w = np.clip(np.gradient(U, lna), 0, None)
    L.append(f" {model}")
    rr = U * np.sqrt(Dc * Om)
    rows = [("flow (book)", w), ("stock U", U), ("stock re-recorded", rr), ("flow + stock re-recorded", w + rr / rr[-1] * w[-1])]
    for name, j_ in rows:
        ne = neff(j_)
        L.append(f"  {name:26s} " + "  ".join(f"{at(ne, q):6.2f}" for q in zs) + f"  |   {ne[win_md].mean():5.2f}     |   {ne[win].max() - ne[win].min():5.2f}")
L += ["", "Reading guide: the book's target is n_eff = 7/2 at every z (spread 0). Which rate, if any, flattens the running near 3.5. Exploration."]
open(os.path.join(OUT, "x7_stock_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
