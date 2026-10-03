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

# ---------------------------------------------------------------------------------------------------------------
# Book carriage (2026-10-03). The trend (MNRAS 528, L20; arXiv:2303.06928) is built from f sigma8(z) (redshift-space distortions)
# with Om held to a Planck+BAO prior (0.3111 +- 0.0056) and data below z_min removed. The informational term's counterpart is
# therefore the S8 a LCDM fit to f sigma8 data at z >= z_min infers, not the lensing amplitude.
import sympy as sp
b=sp.symbols('beta_m',positive=True); Om_s,OL_s=sp.symbols('Omega_m Omega_L',positive=True)
mu1=1/(1+b/(Om_s+OL_s))
print("\n7. sympy: mu0 = mu(1) - 1 =",sp.simplify(mu1-1)," (paper form -beta/(Om+OL+beta)); flat: ",sp.simplify((mu1-1).subs(OL_s,1-Om_s)),
      "=",round(float((-b/(1+b)).subs(b,0.15765)),4))
print("   E(a) < 10 % for z >",round(float(sp.log(10)),3)," ; E at z = 2.2:",round(float(np.exp(1-(1+2.2))),3))
print("8. inferred S8 three ways (S8_Planck 0.832, same early amplitude, Om fixed 0.3153)")
fsr=lambda z: f(I,1/(1+z))*D(I,1/(1+z))/(f(L,1/(1+z))*D(L,1/(1+z)))
for z in (0,0.3,0.5,1.0,1.5,2.0):
    a=1/(1+z); print(f"   z {z}: lensing amplitude {S8P*D(I,a)/D(L,a):.4f}   RSD at z (f sigma8 ratio) {S8P*fsr(z):.4f}   S8 x mu {S8P*mu(a):.3f}")
zs=np.array([0.067,0.15,0.295,0.38,0.51,0.51,0.70,0.706,0.85,0.919,1.317,1.48,1.491]); er=np.array([0.055,0.16,0.094,0.047,0.039,0.064,0.043,0.053,0.095,0.047,0.037,0.045,0.044])
print("   z_min cut on the 13 f sigma8 points of the DESI/SDSS table (errors as published), LCDM template fit of sigma8 at fixed Om:")
for zmin in (0.0,0.2,0.4,0.6,0.8,1.0,1.3):
    k=zs>=zmin; tL=np.array([f(L,1/(1+z))*D(L,1/(1+z)) for z in zs[k]]); w=(tL/er[k])**2; r=np.array([fsr(z) for z in zs[k]])
    S=S8P*(w*r).sum()/w.sum(); t1=tL/D(L,1.0)   # template f D/D(1): f sigma8 per unit sigma8
    sS8=np.sqrt(Om/0.3)/np.sqrt(((t1/er[k])**2).sum())
    print(f"   z_min {zmin}: n {k.sum():2d}  inferred S8 {S:.4f}  (deficit {100*(1-S/S8P):.2f} %)   statistical sigma(S8) {sS8:.3f}  shift/sigma {(S-S8P)/sS8:+.2f}")
print("9. growth index: f sigma8 of gamma = 0.633 (same early amplitude) against the informational term")
from scipy.integrate import quad as _q
def Dg(a,g): return np.exp(-_q(lambda x: Oma(x)**g/x, a, 1)[0])   # D(a)/D(1) with f = Om(a)^g
for z in (0,0.5,1.0,1.5,2.0):
    a=1/(1+z); g633=Oma(a)**0.633*Dg(a,0.633)/Dg(1e-3,0.633)*1e-3; gL=f(L,a)*D(L,a); gI=f(I,a)*D(I,a)
    print(f"   z {z}: f sigma8 ratio to LCDM: gamma 0.633 {g633/gL:.4f}   IAM {gI/gL:.4f}")
