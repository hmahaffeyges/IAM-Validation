#!/usr/bin/env python3
"""EXPLORATION (not a test), register X7, 2026-11-14 evening. Runs on GitHub (explore.yml), never on the laptop.
Read first (brief X7): book p1_02 sec:law_exponent, p4_14 sec:qr_exponent, app_C3 app:der:exponent, docs/verification/theory/BOTTOM_UP_EXPONENT.md:
the halos' own writing rate (virial kinetic half over the measured mass functions) has a RUNNING exponent n_eff (5.3-5.8 at z = 9, 7/2 at z = 3.2-4.0,
2.0-2.4 at z = 1); the top-down E(a) needs a constant 7/2. Open routes named there: mergers/re-virialisation, and the rho_crit(z) scaling of K.
NEW HERE (not in the book or register): the book's two-entry ledger (p3_07: 'information migrates from the first [local] to the second [cosmic] at a
rate set by their temperature difference'). Local records L are written at the book's bottom-up rate w = dU/dln a and migrate to the cosmic ledger:
    dL/dln a = w - k(a) L,     cosmic writing rate per e-fold  j = k(a) L.
Two clocks are tried: k per e-fold constant (migration time a fixed fraction of the Hubble time) and k per unit time constant (k(a) = k_t / E(a)).
For each, n_eff of the cosmic rate, the book's definition d ln[(j/f)]/d ln D, and its spread over z = 1-9. Same code and parameters as
docs/verification/scripts/verify_bottom_up_exponent.py. Writes xout/x7_migration_summary.txt."""
import os, numpy as np
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
L = ["X7 exploration: two-entry ledger (local records migrate to the cosmic horizon). n_eff of the cosmic writing rate, book's definition.",
     "  clock           k      " + "  ".join(f"z={q:<4}" for q in zs) + "  | mean z2.3-9 | spread z1-9 (max-min)"]
for model, mdef in (("press74", "fof"), ("sheth99", "fof"), ("tinker08", "200m")):
    dn = np.array([mass_function.massFunction(M, zz, q_in='M', q_out='dndlnM', mdef=mdef, model=model) for zz in z])
    U = np.trapezoid(dn * M**(5 / 3) * ((Dc * E**2)[:, None])**(1 / 3), lnM, axis=1)
    w = np.clip(np.gradient(U, lna), 0, None)
    L.append(f" {model}")
    rows = [("none (book)", None, w)]
    for k in (0.5, 1.0, 2.0, 4.0): rows.append(("per e-fold", k, migrate(w, np.full_like(w, k))))
    for kt in (0.5, 1.0, 2.0): rows.append(("per unit time", kt, migrate(w, kt / E)))
    for name, k, j in rows:
        ne = neff(j)
        L.append(f"  {name:14s} {('' if k is None else f'{k:4.1f}'):5s}  " + "  ".join(f"{at(ne, q):6.2f}" for q in zs)
                 + f"  |   {ne[win_md].mean():5.2f}     |   {ne[win].max() - ne[win].min():5.2f}")
L += ["", "Reading guide: the book's target is n_eff = 7/2 at every z (spread 0). A migration clock that pulls the spread down while holding the",
      "mean near 3.5 would be the two-ledger route to the constant 7/2; one that only shifts the curve is not. Exploration; nothing sealed."]
open(os.path.join(OUT, "x7_migration_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
