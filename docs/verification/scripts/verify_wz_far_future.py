#!/usr/bin/env python3
"""Recomputes every number in 'Dark Energy Evolution and the Far Future of an IAM Universe' (Feb 2026). numpy, scipy."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
H0=67.4; Om=0.315; OL=0.685; Gyr=977.792/H0   # 1/H0 in Gyr
E=lambda a: np.exp(1-1/a); H=lambda a: np.sqrt(Om*a**-3+OL)  # in units of H0
print("1. w_info from rho_info proportional to E(a) (continuity: w = -1 - (1/3) dln rho/dln a; dlnE/dlna = 1/a)")
for a in (0.25,0.5,1,2,10): print(f"   a={a:5}: w_info = {-1-1/(3*a):.4f}")
print("   CPL at a=1: w0 = -4/3; wa = -dw/da|1 = -1/3  (NEGATIVE; paper text says 'small positive wa' in 3 places)")
print("2. H_inf = H0 sqrt(OL) =", round(H0*np.sqrt(OL),2), "km/s/Mpc")
print("3. maturity f = E(a)/e, a(f) = -1/ln f; age t(a) = int_0^a da/(a H)")
t=lambda a: quad(lambda x: 1/(x*H(x)),1e-8,a,limit=200)[0]*Gyr; t0=t(1)
for f in (0.01,0.05,0.10,0.20,1/np.e,0.5,0.75,0.90,0.95,0.99):
    a=-1/np.log(f); print(f"   {100*f:5.1f}%  a {a:7.3f}  z {1/a-1:6.2f}  age {t(a):6.1f} Gyr  from now {t(a)-t0:+6.1f}")
print("4. where the rates peak")
r_a=minimize_scalar(lambda a: -E(a)/a**2,bounds=(0.05,5),method="bounded").x
r_ln=minimize_scalar(lambda a: -E(a)/a,bounds=(0.05,5),method="bounded").x
r_t=minimize_scalar(lambda a: -E(a)/a*H(a),bounds=(0.05,5),method="bounded").x
print(f"   dE/da = E/a^2 peaks at a = {r_a:.3f} (inflection of E(a) in a; paper says a = 1)")
print(f"   dE/dln a = E/a peaks at a = {r_ln:.3f} (inflection of E in ln a; this is the 'peaks today' statement)")
print(f"   dE/dt = H E/a peaks at a = {r_t:.3f}, z = {1/r_t-1:.2f}; today dE/dt/e = {H(1)/np.e/Gyr*100:.3f} %/Gyr (paper 2.54); at peak {E(r_t)/r_t*H(r_t)/np.e/Gyr*100:.3f} %/Gyr")
print(f"   paper's formula d(E/e)/da = 1/(e a^2) vs actual E/(e a^2): at a=0.5 {1/(np.e*0.25):.3f} vs {E(0.5)/(np.e*0.25):.3f}")
print("5. DESI DR2 CPL points vs the CPL image of w_info")
for nm,w0,s0,wa,sa in (("DR2+CMB+Pantheon+",-0.838,0.055,-0.62,0.21),("DR2+CMB+Union3",-0.667,0.088,-1.09,0.29),("DR2+CMB+DESY5",-0.752,0.057,-0.86,0.22)):   # arXiv:2503.14738 eqs. 25-27
    print(f"   {nm}: w0 offset {(-4/3-w0)/s0:+.1f} sigma, wa offset {(-1/3-wa)/sa:+.1f} sigma; DESI w(a=1) {w0:.3f} > -1, crossing -1 at a = {1-(-1-w0)/wa:.3f}")
print("6. the informational term lives in the matter-sector rate: H_m^2 = H^2 + beta_m E(a) H0^2 (photon sector = LCDM)")
bm=0.3153/2
for a in (1.0,2.0,10.0,1e6):
    Hm=np.sqrt(H(a)**2+bm*E(a)); print(f"   a={a:g}: H_photon {H0*H(a):6.2f}  H_matter {H0*Hm:6.2f}  ratio {Hm/H(a):.4f}")
print(f"   rho_info today / rho_L = beta_m/OL = {bm/OL:.3f}; at saturation {bm*np.e/OL:.3f}; at z=3 {bm*E(0.25)/OL:.4f}")
