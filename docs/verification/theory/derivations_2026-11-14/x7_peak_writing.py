#!/usr/bin/env python3
"""EXPLORATION (register X7; GitHub explore.yml), 2026-11-15. Principle written in the register before this ran: book p1_02 (properties of E(a)):
'the record written per e-fold, dE/d ln a = E/a, is largest at a = 1'; for E = exp(c(1 - 1/a)) the peak is at a = c. So c is the scale factor at
which the universe writes the most record per e-fold. Bottom-up, nothing fitted: the book's own writing rate (docs/verification/scripts/
verify_bottom_up_exponent.py, same code and parameters): U(a) = sum over halos of the virial kinetic half, K ~ M^(5/3) (Delta_c rho_crit)^(1/3),
three measured mass functions. If the record's energy ledger follows U (rho_info ~ E), two independent readings of c come from the same U:
  (1) the scale factor where dU/dln a peaks (c = a_peak);
  (2) U(far future)/U(today) = e^c.
Also reported: the record per e-fold priced at T_H per horizon area (the top-down measure), a^-3 H dU/dln a, and where it peaks.
Writes xout/x7_peak_writing_summary.txt."""
import os, sys, subprocess, numpy as np
try:
    import colossus
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "colossus"], check=True)
from colossus.cosmology import cosmology
from colossus.lss import mass_function
trap = getattr(np, "trapezoid", None) or np.trapz
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
P18 = {'flat': True, 'H0': 67.36, 'Om0': 0.3153, 'Ob0': 0.0493, 'sigma8': 0.8111, 'ns': 0.9649, 'relspecies': True}
c = cosmology.setCosmology('p18chains', params=P18, persistence='')
L = ["X7 peak writing (exploration): book's bottom-up U(a), Planck 2018, three mass functions"]
lnM = np.linspace(np.log(1e4), np.log(1e17), 320); M = np.exp(lnM)
def U_of(a, model, mdef):
    z = 1 / a - 1; E = c.Ez(z); Om = c.Om(z); Dc = 18 * np.pi**2 + 82 * (Om - 1) - 39 * (Om - 1)**2
    dn = np.array([mass_function.massFunction(M, zz, q_in='M', q_out='dndlnM', mdef=mdef, model=model) for zz in z])
    return trap(dn * M**(5 / 3) * ((Dc * E**2)[:, None])**(1 / 3), lnM, axis=1), E
for model, mdef in (("press74", "fof"), ("sheth99", "fof"), ("tinker08", "200m")):
    lna = np.linspace(np.log(0.05), np.log(1.0), 200); a = np.exp(lna)
    U, E = U_of(a, model, mdef); W = np.gradient(U, lna); R = a**-3 * E * W
    line = f"  {model:9s} dU/dlna peaks at a = {a[np.argmax(W)]:.3f}" + (" (still rising at a = 1)" if np.argmax(W) == len(a) - 1 else "")
    line += f"; record per e-fold per horizon area peaks at a = {a[np.argmax(R)]:.3f}"
    try:
        af = np.exp(np.linspace(0, np.log(12.0), 80)); Uf, _ = U_of(af, model, mdef)
        Wf = np.gradient(Uf, np.log(af))
        aw = np.concatenate([a, af[1:]]); Ww = np.concatenate([W, Wf[1:]])
        line += f" | extended to a = 12: dU/dlna peaks at a = {aw[np.argmax(Ww)]:.3f}; U(12)/U(1) = {Uf[-1] / Uf[0]:.3f} (ln = {np.log(Uf[-1] / Uf[0]):.3f})"
    except Exception as e:
        line += f" | future not computed ({type(e).__name__}: {str(e)[:80]})"
    L.append(line)
L += ["Canon E = exp(1 - 1/a): dE/dlna peaks at a = 1 and E(inf)/E(1) = e (ln = 1).",
      "Reading guide: if U's per-e-fold peak and ln[U(future)/U(today)] both sit near 1, structure formation's own energy ledger fixes c = 1 two ways;",
      "a peak far from 1 with ln near 1 (or the reverse) means the energy ledger is not E, and the feedback (book step 9) or the pricing must enter."]
open(os.path.join(OUT, "x7_peak_writing_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
