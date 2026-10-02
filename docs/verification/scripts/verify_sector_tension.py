#!/usr/bin/env python3
"""Recomputes 'Dark Energy or Sector Tension? IAM Confrontation with DESI Full-Shape Growth Rates and Joint Weak Lensing' (Mar 2026).
Exact coupling mu = H^2/(H^2 + beta_m E(a)), beta_m = Om/2, LCDM background, Sigma = 1. numpy, scipy."""
import numpy as np
from scipy.integrate import solve_ivp, quad
Om,OL=0.3153,0.6847; bm=Om/2
E=lambda a: np.exp(1-1/a); H2=lambda a: Om*a**-3+OL; mu=lambda a: H2(a)/(H2(a)+bm*E(a)); Oma=lambda a: Om*a**-3/H2(a)
def grow(m):
    def r(l,y):
        a=np.exp(l); return [y[1],-(2-1.5*Om*a**-3/H2(a))*y[1]+1.5*Oma(a)*m(a)*y[0]]
    return solve_ivp(r,(np.log(1e-3),0),[1e-3,1e-3],dense_output=True,rtol=1e-10,atol=1e-14)
L=grow(lambda a:1.0); I=grow(mu)
D=lambda S,a: S.sol(np.log(a))[0]; f=lambda S,a: S.sol(np.log(a))[1]/S.sol(np.log(a))[0]
print("1. DESI DR2 (arXiv:2503.14738, eqs. 25-27) w0wa and crossing redshift a_x = 1 + (1+w0)/wa")
for nm,w0,wa in (("DESI+CMB",-0.42,-1.75),("DESI+CMB+Pantheon+",-0.838,-0.62),("DESI+CMB+Union3",-0.667,-1.09),("DESI+CMB+DESY5",-0.752,-0.86)):
    ax=1+(1+w0)/wa; print(f"   {nm:20s} w0 {w0:+.3f} wa {wa:+.2f}  z_cross {1/ax-1:.2f}")
print("   paper Table 5 labels: 'BAO+CMB' -0.838/-0.62 is the Pantheon+ row; Union3 -0.752/-0.82 and DESY5 -0.734/-1.05 match no DR2 row")
print("2. at the DESI DR1 tracer redshifts: 1 - mu (paper's 'suppression') vs the actual fsigma8 deficit (same early amplitude)")
for nm,z in (("BGS",0.295),("LRG1",0.510),("LRG2",0.706),("LRG3",0.919),("ELG2",1.317),("QSO",1.491),("today",0.0)):
    a=1/(1+z); r=f(I,a)*D(I,a)/(f(L,a)*D(L,a))
    print(f"   {nm:5s} z {z:5.3f}: 1-mu {100*(1-mu(a)):5.2f} %   fsigma8 deficit {100*(1-r):5.2f} %   sigma8(z) deficit {100*(1-D(I,a)/D(L,a)):4.2f} %   E_G change {100*(f(L,a)/f(I,a)-1):+5.2f} %")
print("3. CMB lensing C_phiphi (Limber, kernel (chi_s-chi)/chi_s, power ~ D^2): exact-form ratio")
zs=1089; c=1.0; chi=lambda z: quad(lambda x: 1/np.sqrt(H2(1/(1+x))),0,z)[0]; cs=chi(zs)
zg=np.linspace(0.02,10,500); ch=np.array([chi(z) for z in zg]); W=((cs-ch)/cs*(1+zg))**2/np.sqrt(H2(1/(1+zg)))
rat=np.array([(D(I,1/(1+z))/D(L,1/(1+z)))**2 for z in zg]); print(f"   C_phiphi IAM/LCDM ~ {np.trapezoid(W*rat,zg)/np.trapezoid(W,zg):.4f}  (paper: 'no change')")
print("4. H_matter = 67.16 sqrt(1 + beta_m) =", round(67.16*np.sqrt(1+0.15765),2), "; SH0ES offset", round((73.04-72.26)/1.04,2), "sigma")
print("5. beta_gamma/beta_m: paper < 8.5e-6 (from the 1.4e-6 bound, reversed-array bug); corrected beta_gamma < 0.0039 gives", round(0.0039/0.15765,4))
