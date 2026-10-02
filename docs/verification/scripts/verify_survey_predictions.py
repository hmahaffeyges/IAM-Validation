#!/usr/bin/env python3
"""Recomputes 'Falsifiable Predictions of the IAM for Euclid, DESI, and Next-Generation Surveys' (25 Feb 2026). numpy, scipy."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
Om,OL=0.3153,0.6847; bm=Om/2
E=lambda a: np.exp(1-1/a); H2=lambda a: Om*a**-3+OL; mu=lambda a: H2(a)/(H2(a)+bm*E(a)); Oma=lambda a: Om*a**-3/H2(a)
def grow(m):
    r=lambda l,y: [y[1],-(2-1.5*Om*np.exp(-3*l)/H2(np.exp(l)))*y[1]+1.5*Oma(np.exp(l))*m(np.exp(l))*y[0]]
    return solve_ivp(r,(np.log(1e-3),0),[1e-3,1e-3],dense_output=True,rtol=1e-10,atol=1e-14)
L=grow(lambda a:1.0); I=grow(mu); D=lambda S,a:S.sol(np.log(a))[0]; f=lambda S,a:S.sol(np.log(a))[1]/S.sol(np.log(a))[0]
print("1. Table 5: paper dD/D, dfs8/fs8 vs growth equation (same early amplitude)")
for z,pd,pf in ((0,-6.8,-10.2),(0.1,-5.7,-8.6),(0.3,-3.9,-5.9),(0.5,-2.6,-3.9),(1.0,-0.9,-1.3),(2.0,-0.1,-0.2)):
    a=1/(1+z); print(f"   z {z}: dmu {100*(mu(a)-1):6.2f}%  dD/D {100*(D(I,a)/D(L,a)-1):6.2f}% (paper {pd})  dfs8 {100*(f(I,a)*D(I,a)/(f(L,a)*D(L,a))-1):6.2f}% (paper {pf})")
print("   paper dPhi/Phi = dmu: with Sigma = 1 the lensing potential follows delta_m, so dPhi/Phi = dD/D")
print("2. Table 4 activation milestones: fraction of 1 - mu(0) reached")
d0=1-mu(1.0); Gyr=977.792/67.36; t=lambda a: quad(lambda x:1/(x*np.sqrt(H2(x))),1e-8,a)[0]*Gyr; t0=t(1)
for q in (0.01,0.05,0.10,0.25,0.5,0.75,0.9):
    a=brentq(lambda a:(1-mu(a))/d0-q,0.05,1); print(f"   {100*q:4.0f}%: a {a:.3f} z {1/a-1:.2f} lookback {t0-t(a):.1f} Gyr")
zz=np.linspace(0,3,30001); dmz=np.gradient(np.array([mu(1/(1+z)) for z in zz]),zz); print(f"   |dmu/dz| peaks at z = {zz[np.argmax(abs(dmz))]:.3f}")
print("   Fig. 4(a) milestones of E(a) itself (10/50/90 %):", [round(1/(1+np.log(q)*-1)**-1-1,2) if False else round(-1/np.log(q)**-1,2) for q in ()] )
for q in (0.1,0.5,0.9): a=1/(1-np.log(q)); print(f"   E(a) = {q}: z = {1/a-1:.2f}")
print("3. ISW source (1-f)D, uniform weight z 0.05-1.5: see verify_s8_trend.py (ratio ~1.03); paper A_ISW = 1.134; Fig. 2 per-sample 1.09-1.17")
print("4. Sirens: 67.161 sqrt(1.1575) =", round(67.161*np.sqrt(1.1575),2), "; GW170817 Abbott 2017: 70.0 +12/-8")
print("5. Table 2 timeline: sigma(mu0) DESI Y5 0.207 > DESI DR2 0.100 (non-monotonic; DESI Y5 alone cannot be weaker than DR2)")
