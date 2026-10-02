#!/usr/bin/env python3
"""Reproduces DUAL_SECTOR_VALIDATION_CHECK.md (Dual-Sector Validation paper, 23 Feb 2026) on the public Pantheon+ release.
Downloads Pantheon+SH0ES.dat and the STAT+SYS covariance from github.com/PantheonPlusSH0ES/DataRelease. numpy, scipy, pandas. ~10 s."""
import os, urllib.request, numpy as np, pandas as pd
from scipy.optimize import minimize
B = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
for f, u in (("PantheonPlusSH0ES.dat", "Pantheon%2BSH0ES.dat"), ("PantheonPlusSH0ES_STAT+SYS.cov", "Pantheon%2BSH0ES_STAT%2BSYS.cov")):
    if not os.path.exists(f): urllib.request.urlretrieve(B + u, f)
D = pd.read_csv("PantheonPlusSH0ES.dat", sep=r"\s+"); c = 299792.458

# 1. The paper's own setup: zCMB 0.01-2.5, diagonal errors, Nelder-Mead, bounds as printed
s = D[(D.zCMB > 0.01) & (D.zCMB < 2.5)]; z, mb, dm = s.zCMB.values, s.m_b_corr.values, s.m_b_corr_err_DIAG.values
zg = np.linspace(0, z.max(), 3000)
def dl(Om, H0, b, zz, zh=None):
    a = 1/(1+zg); H = H0*np.sqrt(Om*a**-3 + 1-Om + b*np.exp(1-1/a))
    dc = np.concatenate([[0], np.cumsum(0.5*(c/H[1:] + c/H[:-1])*np.diff(zg))]); return (1+(zz if zh is None else zh))*np.interp(zz, zg, dc)
def chi2(p, prior=None):
    Om, H0, b, M = p
    if not (0.2 < Om < 0.4 and 60 < H0 < 75 and -0.3 < b < 0.3 and -20 < M < -18): return 1e10
    r = np.sum(((mb - (M + 5*np.log10(dl(Om, H0, b, z)) + 25))/dm)**2)
    return r + (((H0-prior[0])/prior[1])**2 if prior else 0)
print(f"1. Paper setup, {len(z)} SNe")
for nm, pr, x0 in (("A Planck prior", (67.4, 0.5), [0.315, 67.4, 0, -19.3]), ("B SH0ES prior", (73.04, 1.04), [0.315, 73.04, 0, -19.3]), ("C no prior", None, [0.315, 70, 0, -19.3])):
    r = minimize(chi2, x0, args=(pr,), method="Nelder-Mead", options=dict(maxiter=5000, xatol=1e-6, fatol=1e-6))
    print(f"   {nm:15s} Om {r.x[0]:.4f} H0 {r.x[1]:.2f} beta {r.x[2]:+.4f} M {r.x[3]:.4f} chi2 {r.fun:.2f}")
print("   profile in H0 (minimised over Om, beta, M):")
for H0 in (64, 67.4, 70, 73.04):
    r = minimize(lambda q: chi2([q[0], H0, q[1], q[2]]), [0.315, 0, -19.3+5*np.log10(H0/70)], method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-7, fatol=1e-7))
    print(f"   H0 {H0:5.2f}: chi2 {r.fun:.4f}   M - 5 log10 H0 = {r.x[2]-5*np.log10(H0):.4f}")

# 2. Full STAT+SYS covariance, Hubble flow zHD > 0.01 (Pantheon+ convention); M marginalised analytically (equivalently H0)
raw = np.loadtxt("PantheonPlusSH0ES_STAT+SYS.cov"); N = int(raw[0]); C = raw[1:].reshape(N, N)
m = (D.zHD > 0.01).values; Ci = np.linalg.inv(C[np.ix_(m, m)]); zz, zh, y = D.zHD.values[m], D.zHEL.values[m], D.m_b_corr.values[m]
zg = np.linspace(0, zz.max()+0.01, 4000)
def chiF(Om, b):
    r = y - (5*np.log10(dl(Om, 70.0, b, zz, zh)) + 25); o = np.ones_like(r); d = r - (o@Ci@r)/(o@Ci@o); return d@Ci@d
print(f"2. Full covariance, {m.sum()} SNe")
c0, c1 = chiF(0.315, 0), chiF(0.315, 0.15765)
print(f"   Om 0.315: LCDM {c0:.2f}; beta = 0.15765 on SN distances {c1:.2f}; dchi2 {c1-c0:+.2f}")
bs = np.linspace(-0.3, 0.3, 121); P = np.array([chiF(0.315, b) for b in bs]); i = P.argmin(); ok = bs[P <= P[i]+1]
print(f"   Om 0.315: best beta {bs[i]:+.3f}, 68 % {ok.min():+.3f} to {ok.max():+.3f}")
for b in (0.0, 0.15765):
    g = [(Om, chiF(Om, b)) for Om in np.linspace(0.2, 0.45, 51)]; Om, v = min(g, key=lambda t: t[1]); print(f"   beta {b:.4f} fixed, Om free: Om {Om:.3f} chi2 {v:.2f}")

# 3. S8 implied by the Level 2 IAM chain
print("3. S8 = sigma8 sqrt(Om/0.3), Level 2 Run A (sigma8 0.7998, Om 0.3166): %.3f" % (0.7998*np.sqrt(0.3166/0.3)))
