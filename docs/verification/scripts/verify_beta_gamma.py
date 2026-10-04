#!/usr/bin/env python3
"""beta_gamma bound and the photon-sector theta_s shift (DUAL_SECTOR_VALIDATION_CHECK.md items 8a-8c). numpy, scipy.
Same model as tests/mcmc_final_iam.py:
H_photon = H0 sqrt(Om a^-3 + Or a^-4 + OL + beta_g e^(1-1/a)), r_* fixed, theta_* compared with Planck's.
Every input is taken from ONE Planck fit: Planck 2018 VI (arXiv:1807.06209) Table 2, base LCDM, TT,TE,EE+lowE+lensing:
H0 = 67.36, Omega_m = 0.3153, r_* = 144.43 Mpc, 100 theta_* = 1.04110 +/- 0.00031, z_* = 1089.92.
Omega_r from T_CMB = 2.7255 K and N_eff = 3.046 at h = 0.6736 (massless neutrinos; simplified)."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
c, H0, Om, zs = 299792.458, 67.36, 0.3153, 1089.92          # Planck 2018 TT,TE,EE+lowE+lensing
Or = 2.4728e-5*(1+0.2271*3.046)/(H0/100)**2; OL = 1-Om-Or; obs, err, rs = 0.0104110, 3.1e-6, 144.43
H = lambda a, b: H0*np.sqrt(Om*a**-3 + Or*a**-4 + OL + b*np.exp(1-1/a))
def theta(b):   # adaptive quadrature: a 20,000-point grid shifts the 95 % bound by 4 %
    return rs/(quad(lambda z: 1/H(1/(1+z), b), 0, zs, limit=500, epsabs=0, epsrel=1e-12)[0]*c)
# 1. the emcee script's integral as written: both arrays reversed -> negative distance
z = np.linspace(0, zs, 1000); I = 1/H(1/(1+z), 0.0)
bug = rs/(np.trapezoid(I[::-1], z[::-1])*c)
print(f"1. mcmc_final_iam.py theta_s(beta_g=0) as written = {bug:.8f}  ({(bug-obs)/err:+.0f} sigma) -> the 1.4e-6 bound is an artefact")
# 2. corrected: profile Delta chi2 over beta_g >= 0
t0 = theta(0.0); print(f"2. corrected theta_s(beta_g=0) = {t0:.8f} ({(t0-obs)/err:+.1f} sigma: fixed r_s, simplified)")
c0 = ((t0-obs)/err)**2; lim = {}
for lev, nm in ((1, "68 %"), (4, "95 %"), (9, "99.7 %")):
    lim[lev] = brentq(lambda b: ((theta(b)-obs)/err)**2 - c0 - lev, 1e-6, 0.05); print(f"   beta_g < {lim[lev]:.4f} ({nm}, Delta chi2 = {lev})")
print(f"   beta_g / beta_m < {lim[4]/(Om/2):.3f} (95 %)")
# 3. the matter coupling applied to photon paths, everything else fixed
for b in (0.18, Om/2):
    t = theta(b); print(f"3. beta = {b:.4f} on photon paths: theta_s shift {100*(t/t0-1):+.3f} % = {(t-t0)/err:.1f} sigma at Planck precision")
