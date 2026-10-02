#!/usr/bin/env python3
"""Recomputes 'The Redshift-Dependent S8 Trend in the Context of IAM' (Mar 2026) from the linear growth equation.
Background LCDM (Planck 2018: Om 0.3153, OL 0.6847); mu(a) = H^2/(H^2 + beta_m E(a)), beta_m = Om/2; Sigma = 1. numpy, scipy."""
import numpy as np
from scipy.integrate import solve_ivp
Om,OL=0.3153,0.6847; bm=Om/2; S8P=0.832
E=lambda a: np.exp(1-1/a); H2=lambda a: Om*a**-3+OL; mu=lambda a: H2(a)/(H2(a)+bm*E(a)); Oma=lambda a: Om*a**-3/H2(a)
def grow(m):
    def r(l,y):
        a=np.exp(l); dlnH=-1.5*Om*a**-3/H2(a)
        return [y[1],-(2+dlnH)*y[1]+1.5*Oma(a)*(m(a))*y[0]]
    return solve_ivp(r,(np.log(1e-3),0),[1e-3,1e-3],dense_output=True,rtol=1e-10,atol=1e-14)
L=grow(lambda a:1.0); I=grow(mu)
D=lambda S,a: S.sol(np.log(a))[0]; f=lambda S,a: S.sol(np.log(a))[1]/S.sol(np.log(a))[0]
print("1. mu0 =", round(mu(1)-1,4), " (paper -0.1349)")
print("2. paper's Eq.5 S8 x mu(z) vs the growth-equation S8 x D_IAM/D_LCDM (what a survey at z extrapolating with LCDM infers)")
for z in (0,0.25,0.5,0.75,1.0,1.5,2.0,3.0):
    a=1/(1+z); r=D(I,a)/D(L,a)
    print(f"   z {z:4}: mu {mu(a):.4f}  paper S8 {S8P*mu(a):.3f}  growth S8 {S8P*r:.4f}  deficit {100*(1-r):.2f} %   f_IAM/f_LCDM {f(I,a)/f(L,a):.4f}")
print("   paper text z=0 0.719 vs its Fig. 2 label 0.702; S8 x mu(0) =",round(S8P*mu(1),3))
print("3. effective growth index gamma = ln f / ln Om(a)")
for z in (0,0.5,1.0):
    a=1/(1+z); print(f"   z {z}: LCDM {np.log(f(L,a))/np.log(Oma(a)):.3f}   IAM {np.log(f(I,a))/np.log(Oma(a)):.3f}   (Nguyen et al. 2023: 0.633 +0.025 -0.024)")
print("4. ISW source (1 - f) D / a, ratio IAM/LCDM (photon potential with Sigma = 1 follows delta_m)")
zz=np.linspace(0.05,1.5,30); num=den=0
for z in zz:
    a=1/(1+z); sI=(1-f(I,a))*D(I,a)/a; sL=(1-f(L,a))*D(L,a)/a; num+=sI*D(I,a); den+=sL*D(L,a)
for z in (0.1,0.5,1.0): a=1/(1+z); print(f"   z {z}: source ratio {((1-f(I,a))*D(I,a))/((1-f(L,a))*D(L,a)):.4f}")
print(f"   ISW-galaxy cross (source x D, uniform in z 0.05-1.5): ratio {num/den:.3f}  (paper: 1.10-1.30)")
print("5. cluster M_lens/M_dyn = 1/mu (linear mu): z 0", round(1/mu(1),3), " z 0.5", round(1/mu(1/1.5),3), " z 1", round(1/mu(0.5),3))
print("6. same growth integration with the MGCAMB form used in the Level 1 chains, mu = 1 + mu0 Omega_DE(a)/Omega_DE0, mu0 = -0.13495")
M=grow(lambda a: 1-0.13495*(OL/H2(a))/OL)
for z in (0,0.5,1.0): a=1/(1+z); print(f"   z {z}: S8 deficit {100*(1-D(M,a)/D(L,a)):.2f} %   fsigma8 deficit {100*(1-f(M,a)*D(M,a)/(f(L,a)*D(L,a))):.2f} %")
print(f"   Level 1 chains (Planck only): sigma8 0.8139 -> 0.8014 = {100*(1-0.8014/0.8139):.2f} %")
print(f"   exact form today: fsigma8 deficit {100*(1-f(I,1)*D(I,1)/(f(L,1)*D(L,1))):.2f} %")
