#!/usr/bin/env python3
"""DESI DR1 growth from the ShapeFit ratios, against LCDM and IAM in the MGCAMB form. numpy, scipy.

Data (public, read from the paper source of DESI 2024 V, arXiv:2411.12021, JCAP 2025 09 008):
  * Table "Results from the ShapeFit baseline fit", ShapeFit+BAO rows: f sigma_s8 / (f sigma_s8)_fid for the six DR1 bins;
  * Table "Fiducial values" (Appendix): (f sigma_s8)_fid at each bin's effective redshift.
  f sigma8 = ratio x (f sigma_s8)_fid, with DESI's own assumption that the fiducial sound horizon is the true one (r_d = r_d,fid,
  so f sigma_s8 becomes f sigma8). Errors: the mean of the upper and lower 68 % errors; bins treated as independent (diagonal).
  SDSS DR16: the six points of Alam et al. 2021 (PRD 103 083533) Table III (MGS, BOSS z=0.38 and 0.51, eBOSS LRG, ELG, QSO).
Models (no parameter adjusted): LCDM background, Planck 2018 TT,TE,EE+lowE+lensing (Omega_m = 0.3153, Omega_L = 0.6847,
  sigma8 = 0.8111); IAM in the MGCAMB tracking form mu(a) = 1 + mu0 Omega_DE(a)/Omega_DE,0 with mu0 = -0.13495, Sigma = 1;
  linear growth from the same early amplitude (a_i = 1e-3), so sigma8(IAM) = 0.8111 x D_IAM(1)/D_LCDM(1).
The book once quoted chi2 = 4.51 (LCDM) and 5.24 (IAM) on the six DESI bins and 6.19 and 6.95 on SDSS DR16 with no script.
This script is the reproduction attempt; it prints what the public data give.
Run: python docs/verification/scripts/verify_shapefit_chi2.py
"""
import numpy as np
from scipy.integrate import solve_ivp

Om, OL, S8_PLANCK, MU0 = 0.3153, 0.6847, 0.8111, -0.13495
H2 = lambda a: Om * a**-3 + OL
Oma = lambda a: Om * a**-3 / H2(a)
mu_mg = lambda a: 1 + MU0 * (OL / H2(a)) / OL


def grow(m):
    def r(l, y):
        a = np.exp(l)
        return [y[1], -(2 - 1.5 * Om * a**-3 / H2(a)) * y[1] + 1.5 * Oma(a) * m(a) * y[0]]
    return solve_ivp(r, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)


LC, MG = grow(lambda a: 1.0), grow(mu_mg)
D = lambda S, a: S.sol(np.log(a))[0]
f = lambda S, a: S.sol(np.log(a))[1] / S.sol(np.log(a))[0]
s8 = {"LCDM": S8_PLANCK, "IAM": S8_PLANCK * D(MG, 1.0) / D(LC, 1.0)}
model = {"LCDM": LC, "IAM": MG}


def fs8(name, z):
    S, a = model[name], 1 / (1 + z)
    return f(S, a) * D(S, a) / D(S, 1.0) * s8[name]


# DESI 2024 V: bin, z_eff, (f sigma_s8)_fid, ShapeFit+BAO ratio, upper error, lower error
DESI = [("BGS", 0.295, 0.4723, 0.84, 0.19, 0.19), ("LRG1", 0.510, 0.4733, 1.16, 0.13, 0.13),
        ("LRG2", 0.706, 0.4608, 1.04, 0.11, 0.092), ("LRG3", 0.919, 0.4398, 0.997, 0.10, 0.084),
        ("ELG2", 1.317, 0.3944, 0.945, 0.097, 0.077), ("QSO", 1.491, 0.3750, 1.16, 0.12, 0.12)]
# Alam et al. 2021 Table III: name, z, f sigma8, error
SDSS = [("MGS", 0.15, 0.53, 0.16), ("BOSS", 0.38, 0.497, 0.045), ("BOSS", 0.51, 0.459, 0.038),
        ("eBOSS LRG", 0.70, 0.473, 0.041), ("eBOSS ELG", 0.85, 0.315, 0.095), ("eBOSS QSO", 1.48, 0.462, 0.045)]

print(f"sigma8: LCDM {s8['LCDM']:.4f} (Planck 2018), IAM {s8['IAM']:.4f} (same early amplitude, MGCAMB form mu0 = {MU0})")
print("DESI DR1, ShapeFit+BAO ratios x fiducial f sigma_s8 (diagonal):")
c = {"LCDM": 0.0, "IAM": 0.0}
for nm, z, fid, r, up, lo in DESI:
    v, e = r * fid, 0.5 * (up + lo) * fid
    for k in c:
        c[k] += ((v - fs8(k, z)) / e) ** 2
    print(f"   {nm:5s} z {z:.3f}: f sigma8 {v:.3f} +- {e:.3f}   LCDM {fs8('LCDM', z):.3f}  IAM {fs8('IAM', z):.3f}")
print(f"   chi2 (6 bins): LCDM {c['LCDM']:.2f}  IAM {c['IAM']:.2f}   [book had 4.51, 5.24]")
c = {"LCDM": 0.0, "IAM": 0.0}
for nm, z, v, e in SDSS:
    for k in c:
        c[k] += ((v - fs8(k, z)) / e) ** 2
print(f"SDSS DR16 (6 points, diagonal): chi2 LCDM {c['LCDM']:.2f}  IAM {c['IAM']:.2f}   [book had 6.19, 6.95]")
