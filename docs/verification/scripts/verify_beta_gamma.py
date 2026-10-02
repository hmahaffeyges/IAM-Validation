#!/usr/bin/env python3
"""beta_gamma bound and the photon-sector theta_s shift (DUAL_SECTOR_VALIDATION_CHECK.md items 8a-8c). numpy, scipy.
Same model and data as tests/mcmc_final_iam.py and development/archive/tests_27-29/test_29_beta_gamma_constraint.py:
H_photon = 67.4 sqrt(Om a^-3 + Or a^-4 + OL + beta_g e^(1-1/a)), r_s = 144.43 Mpc fixed, theta_s = 0.0104110 +/- 0.0000031."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
c, H0, Om, Or = 299792.458, 67.4, 0.315, 9.24e-5; OL = 1-Om-Or; obs, err, rs = 0.0104110, 3.1e-6, 144.43
H = lambda a, b: H0*np.sqrt(Om*a**-3 + Or*a**-4 + OL + b*np.exp(1-1/a))
def theta(b):   # adaptive quadrature: a 20,000-point grid shifts the 95 % bound by 4 %
    return rs/(quad(lambda z: 1/H(1/(1+z), b), 0, 1090, limit=500, epsabs=0, epsrel=1e-12)[0]*c)
# 1. the emcee script's integral as written: both arrays reversed -> negative distance
z = np.linspace(0, 1090, 1000); I = 1/H(1/(1+z), 0.0)
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
