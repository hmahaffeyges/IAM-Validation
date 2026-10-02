#!/usr/bin/env python3
"""What Euclid's published mu-Sigma forecast means for IAM's mu(z) (2026-10-02).

Euclid's forecasts (Euclid Collaboration: Albuquerque et al. 2025, arXiv:2506.03008, Table 5 and Sect. 6.1; summarised in Frusciante et al. 2025,
arXiv:2512.09748, Sect. 4.2.1) use mu(z) = 1 + mu0 * Omega_DE(z)/Omega_DE(0). Published 68 % errors on mu_bar0 = 1 + mu0 (LCDM fiducial):
  conservative cuts (k_max 3x2pt 0.25/Mpc, GCsp 0.1/Mpc), GCsp + 3x2pt ............ 23.3 % (US), 23.7 % (SS)   -> sigma(mu0) ~ 0.23
  3x2pt alone with k_max ~ 4/Mpc (ReACT within 1 % of N-body) ...................... 4 %                      -> sigma(mu0) ~ 0.04
  full 3x2pt + GCsp with optimistic k_max (review) .................................. 'of the order of 1 %'      -> sigma(mu0) ~ 0.01
IAM's mu(a) = H2_LCDM/(H2_LCDM + beta_m E(a)), beta_m = Omega_m/2, E(a) = exp(1 - 1/a), has the same mu(z=0) - 1 = -0.136 but falls off much faster
with redshift, so its significance is NOT |mu0| / sigma. This script finds the template mu0 whose linear-growth f sigma8 deficit best matches IAM's
over a redshift range (least squares, unweighted). It is an estimate, not a forecast: a Fisher forecast with IAM's own mu(z) is OPEN.
Checks itself against the chain values: IAM f sigma8 deficit 4.25 % (z=0), 2.17 % (0.3), 1.35 % (0.5), 0.41 % (1).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
Om = 0.3153; OL = 1 - Om; bm = Om / 2                     # canon beta_m = 0.15765 (Planck 2018 Omega_m)
E = lambda a: np.exp(1 - 1 / a)
H2 = lambda a: Om / a**3 + OL
mu_iam = lambda a: H2(a) / (H2(a) + bm * E(a))
mu_tmp = lambda a, m0: 1 + m0 * (OL / H2(a)) / OL
def fD(mu, z):
    """f*D at redshift z with D -> a at early times (same primordial amplitude), linear growth, LCDM background."""
    def rhs(lna, y):
        a = np.exp(lna); h2 = H2(a); dlnh = -1.5 * Om / a**3 / h2
        return [y[1], -(2 + dlnh) * y[1] + 1.5 * Om / a**3 / h2 * mu(a) * y[0]]
    s = solve_ivp(rhs, [np.log(1e-3), 0], [1e-3, 1e-3], dense_output=True, rtol=1e-9, atol=1e-12)
    return s.sol(np.log(1 / (1 + np.asarray(z, float))))[1]
def deficit(mu, z): return 1 - fD(mu, z) / fD(lambda a: 1.0, z)
z = [0, 0.3, 0.5, 1.0]; d = 100 * deficit(mu_iam, z)
print("IAM f sigma8 deficit %:", np.round(d, 2)); assert np.allclose(d, [4.25, 2.17, 1.35, 0.41], atol=0.01)
print("template (mu0 = -0.136) deficit %:", np.round(100 * deficit(lambda a: mu_tmp(a, -0.136), z), 2))
for name, zz in (("spectroscopic bins 0.9-1.7", np.array([0.9, 1.1, 1.3, 1.5, 1.7])), ("photometric range 0.2-2.0", np.linspace(0.2, 2.0, 10)),
                 ("all 0-2", np.linspace(0, 2, 21))):
    dI = deficit(mu_iam, zz); t = deficit(lambda a: mu_tmp(a, -0.1), zz)
    meq = brentq(lambda m: np.sum((deficit(lambda a: mu_tmp(a, m), zz) - dI) * t), -0.5, 0.0)
    print(f"{name:28s} template-equivalent mu0 = {meq:+.3f} | /0.04 = {abs(meq)/0.04:.1f} sigma | /0.23 = {abs(meq)/0.23:.2f} sigma | /0.01 = {abs(meq)/0.01:.0f} sigma")
