#!/usr/bin/env python3
"""EXPLORATION (register X7; GitHub explore.yml), 2026-11-15 afternoon. Two items, principles written in the register first.
(A) Seam check: the book's bottom-up energy ledger U(a) (same code as x7_peak_writing.py) on ONE continuous grid a = 0.05..12, so the per-e-fold
    peak and ln U(12)/U(1) carry no join between separate past and future grids; all three mass functions.
(B) The author's tile route for S_0 (app_C3: rho_info ~ exp(S_info/S_0), S_0 sets the coefficient of 1/a). The record entropy written per e-fold,
    priced at the horizon temperature T_H = hbar H/(2 pi k), is dS_info/dln a = (V_H dU_phys/dln a)/(k T_H), V_H the Hubble volume, U_phys = a^-3 U
    (book eq:Idot, eq:dSdt). Its NORMALIZATION is fixed by the virial relation the canon uses (rho_info(1) = beta_m rho_crit, beta_m = Omega_m/2), so
    only the SHAPE comes from U. Candidates for S_0, each reported as the c it implies (c = Delta S_info(after today)/S_0):
      (i)  S_0 = today's horizon capacity S_geo(1) = pi c^3/(G hbar H0^2) in nats (tiles);
      (ii) S_0 = today's record S_info(1) (then c = ratio still to come / written so far);
      (iii) S_0 = the horizon capacity at maturity, S_geo(inf) = S_geo(1)/Omega_L.
Writes xout/x7_tiles_summary.txt."""
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
Om0, OL0 = 0.3153, 1 - 0.3153
# physical constants (SI) and today's horizon capacity in nats
G, hbar, cl, kB = 6.674e-11, 1.0546e-34, 2.998e8, 1.3807e-23; H0 = 67.36e3 / 3.0857e22
Sgeo1 = np.pi * cl**5 / (G * hbar * H0**2)
lnM = np.linspace(np.log(1e4), np.log(1e17), 320); M = np.exp(lnM)
lna = np.linspace(np.log(0.05), np.log(12.0), 300); a = np.exp(lna); z = 1 / a - 1
i1 = np.argmin(abs(lna))
L = [f"X7 tiles and seam check (exploration). Horizon capacity today S_geo(1) = {Sgeo1:.3e} nats; at maturity S_geo(1)/Omega_L = {Sgeo1 / OL0:.3e}"]
for model, mdef in (("press74", "fof"), ("sheth99", "fof"), ("tinker08", "200m")):
    try:
        E = c.Ez(z); Om = c.Om(z); Dc = 18 * np.pi**2 + 82 * (Om - 1) - 39 * (Om - 1)**2
        dn = np.array([mass_function.massFunction(M, zz, q_in='M', q_out='dndlnM', mdef=mdef, model=model) for zz in z])
        U = trap(dn * M**(5 / 3) * ((Dc * E**2)[:, None])**(1 / 3), lnM, axis=1)
        W = np.gradient(U, lna)
        # (B) record entropy per e-fold, shape only: V_H a^-3 W / T_H ~ E^-3 a^-3 W / E = a^-3 W / E^4 (comoving U -> physical, Hubble volume, T_H ~ H)
        s = a**-3 * W / E**4; S = cumtrap(s, lna)
        # normalize so that the record's ENERGY today equals beta_m rho_crit V_H (canon): k T_H S_info(1) = (Om0/2) * (3 H0^2/(8 pi G)) c^2 * (4 pi/3)(c/H0)^3
        Erec1 = (Om0 / 2) * (3 * H0**2 / (8 * np.pi * G)) * cl**2 * (4 * np.pi / 3) * (cl / H0)**3
        TH1 = hbar * H0 / (2 * np.pi * kB); Sinfo1 = Erec1 / (kB * TH1)
        S_abs = S / S[i1] * Sinfo1; dS_after = S_abs[-1] - S_abs[i1]
        L.append(f"  {model:9s} (A) one grid: dU/dlna peaks at a = {a[np.argmax(W)]:.3f}; ln U(12)/U(1) = {np.log(U[-1] / U[i1]):.3f}")
        L.append(f"  {model:9s} (B) S_info(1) = {Sinfo1:.3e} nats ({Sinfo1 / Sgeo1:.3f} of today's capacity); written after today to a = 12: {dS_after:.3e} nats; "
                 f"c from S_0 = capacity today {dS_after / Sgeo1:.3f}; = record today {dS_after / Sinfo1:.3f}; = capacity at maturity {dS_after / (Sgeo1 / OL0):.3f}; "
                 f"fill at a = 12: {S_abs[-1] / (Sgeo1 / OL0):.3f}")
    except Exception as e:
        L.append(f"  {model:9s} failed: {type(e).__name__}: {str(e)[:120]}")
L += ["Canon: c = 1. Reading guide: a candidate S_0 that gives c near 1 for PS and ST together is the lead; the fill fractions say how full the horizon",
      "gets (1 = black-hole full; 1/2 = the settled half of what was paid)."]
open(os.path.join(OUT, "x7_tiles_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
