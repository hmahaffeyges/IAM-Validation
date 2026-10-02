#!/usr/bin/env python3
"""Recomputes every number in the G4 papers (CC as Actualized Vacuum Energy; Matter-Antimatter Asymmetry; Baryon Asymmetry
as a Derived Quantity; 18th-chain record) and runs the no-free-factor accumulation test. CODATA via scipy.constants.
Chains: mgcamb_validation/chains/*.1.txt, 30 % burn-in, weighted. numpy, scipy, pandas."""
import numpy as np, pandas as pd, scipy.constants as C, os
from scipy.integrate import quad
hbar,c,G,k=C.hbar,C.c,C.G,C.k; Mpc=3.0857e22; Msun=1.98847e30
lP=np.sqrt(hbar*G/c**3); EP=np.sqrt(hbar*c**5/G)
def chain(stem):
    f=f"mgcamb_validation/chains/{stem}.1.txt"; cols=open(f).readline().lstrip("#").split()
    X=pd.read_csv(f,sep=r"\s+",comment="#",names=cols); return X.iloc[int(0.3*len(X)):]
def ms(v,w): m=np.average(v,weights=w); return m,np.sqrt(np.average((v-m)**2,weights=w))
print("1. Planck 2018 inputs (paper): H0 67.4, Ob 0.0493, Om 0.3153, OL 0.6846")
H0=67.4e3/Mpc; lH=c/H0; Ob,Om,OL=0.0493,0.3153,0.6846
rvac=EP**4/(hbar*c)**3; rc=3*H0**2/(8*np.pi*G)*c**2; obs=OL*rc/rvac
print(f"   rho_vac = {rvac:.3e} J/m3; rho_L = {OL*rc:.3e} J/m3; observed ratio = {obs:.4e}")
print(f"   identity (3 OL/8pi)(lP/lH)^2 = {3*OL/(8*np.pi)*(lP/lH)**2:.4e}  (equal by definition of rho_crit)")
base=2/np.pi*(lP/lH)**2*Ob/Om; corr=base*np.sqrt(OL)
print(f"   (lP/lH)^2 = {(lP/lH)**2:.4e}; baseline (2/pi)(lP/lH)^2 Ob/Om = {base:.4e} (x{base/obs:.3f}); x sqrt(OL) = {corr:.4e} -> {100*(corr/obs-1):+.2f} %")
print(f"   exponent p with OL^p closing the baseline: {np.log(obs/base)/np.log(OL):.3f}")
print(f"   2/pi step: l_P^2/(A_eff/4pi) with A_eff = 2 pi lH^2 = {lP**2/(2*np.pi*lH**2/(4*np.pi))/(lP/lH)**2:.4f} x (lP/lH)^2 (paper: 2/pi = {2/np.pi:.4f})")
print(f"   the relation that remains: Ob/Om = (3/16) sqrt(OL): {Ob/Om:.4f} vs {3/16*np.sqrt(OL):.4f}")
print("2. The 18th chain (iam_baryon_test) and the LCDM chains, eta = 273.9e-10 Ob h^2")
for st in ("iam_baryon_test","lcdm_baseline","planck_rsd_lcdm","planck_bao_lcdm_baseline","planck_pantheon_lcdm_baseline"):
    try: X=chain(st)
    except Exception: continue
    w=X.weight.values; h=X.H0.values/100; ob=X.ombh2.values; om=1-X.omegal.values; r=ob/h**2/om
    m,s=ms(273.9*ob,w); q,qs=ms(r/(3/16*np.sqrt(X.omegal.values)),w)
    print(f"   {st:30s} rows {len(X):6d}  ombh2 {ms(ob,w)[0]:.5f}  eta {m:.3f} +/- {s:.3f}  Ob/Om {ms(r,w)[0]:.4f}  ratio to (3/16)sqrt(OL) {q:.4f} +/- {qs:.4f}")
print("3. The analytic eta (baryon paper eqs. 3, 5): invert for Ob h^2 at Planck Om h^2 = 0.1430")
omh2=0.3153*0.6736**2
for nm,f in (("eq.3",lambda x: 2/np.pi*(lP/(c/(67.36e3/Mpc)))**2*x/omh2),("eq.5",lambda x: 2/np.pi*(lP/(c/(67.36e3/Mpc)))**2*np.sqrt(0.6847)*x/omh2)):
    o=0.6847*3*(67.36e3/Mpc)**2/(8*np.pi*G)*c**2/rvac; x=o/f(1.0)
    print(f"   {nm}: Ob h^2 = {x:.5f}, eta = {273.9*x:.3f}e-10")
print("4. Matter-Antimatter section 7 numbers at the QCD transition (T = 150 MeV, g* = 17.25)")
GeV=1e9*C.e; MP=np.sqrt(hbar*c/G)*c**2/GeV; T=0.150; Hq=1.66*np.sqrt(17.25)*T**2/MP*GeV/hbar; lq=c/Hq
print(f"   a_QCD = {(3.91/17.25)**(1/3)*2.7255*k/GeV/T:.2e}; l_H = {lq:.3e} m = {lq/3.0857e16:.2e} pc (paper 1e-3 pc); horizon bits = {np.pi*lq**2/lP**2:.2e} (paper 1e40)")
print(f"   (Odm + Ode)/Ob = {(0.265+0.685)/0.049:.2f}")
print("5. De Sitter horizon: T_GH S_BH = M_H c^2 (the Hubble-sphere mass-energy), for any H")
TS=hbar*H0/(2*np.pi)*np.pi*c**5/(hbar*G*H0**2); MH=rc/OL*0+3*H0**2/(8*np.pi*G)*4*np.pi/3*(c/H0)**3*c**2
print(f"   T_GH S / (M_H c^2) = {TS/MH:.6f}   so rho_L = OL x (horizon bits x k T_GH ln2 / ln2) / V_H")
print("6. Accumulation test, no free factors: heat released by baryonic virialisation, summed to today")
# Eisenstein-Hu no-wiggle P(k), sigma8 = 0.811; Sheth-Tormen mass function; virial K per unit mass = (3/10) G M / r_vir (uniform sphere)
h=0.6736; Om0=0.3153; Ob0=0.0493; ns=0.9649; s8=0.811; rhom=Om0*3*(100*h*1e3/Mpc)**2/(8*np.pi*G)
def T_EH(kh):
    k_=kh*h; om=Om0*h*h; fb=Ob0/Om0; s=44.5*np.log(9.83/om)/np.sqrt(1+10*(Ob0*h*h)**0.75)
    a=1-0.328*np.log(431*om)*fb+0.38*np.log(22.3*om)*fb**2; g=Om0*h*(a+(1-a)/(1+(0.43*k_*s)**4))
    q=kh*(2.7255/2.7)**2/g; L=np.log(2*np.e+1.8*q); return L/(L+(14.2+731/(1+62.5*q))*q*q)
kk=np.logspace(-4,3,4000); Pk=kk**ns*T_EH(kk)**2
def sig(Rmpch): x=kk*Rmpch; W=3*(np.sin(x)-x*np.cos(x))/x**3; return np.sqrt(np.trapezoid(Pk*W*W*kk**2,kk)/(2*np.pi**2))
A=s8/sig(8.0)
def D(z):
    a=1/(1+z); E=lambda x: np.sqrt(Om0/x**3+1-Om0); g=lambda aa: 2.5*Om0*E(aa)*quad(lambda x:1/(x*E(x))**3,0,aa)[0]
    return g(a)/g(1.0)
lnM=np.linspace(np.log(1e8),np.log(1e16),300); M=np.exp(lnM)*Msun
R=(3*M/(4*np.pi*rhom))**(1/3)/(Mpc/h); sg=np.array([A*sig(r) for r in R]); dls=np.gradient(np.log(sg),lnM)
def stats(z):
    nu=1.686/(sg*D(z)); aST,p,Aq=0.707,0.3,0.3222; f=Aq*np.sqrt(2*aST/np.pi)*nu*(1+(aST*nu*nu)**-p)*np.exp(-aST*nu*nu/2)
    dfdlnM=f*np.abs(dls)                      # mass fraction per ln M
    Hz=100*h*1e3/Mpc*np.sqrt(Om0*(1+z)**3+1-Om0); rvir=(3*M/(4*np.pi*200*3*Hz**2/(8*np.pi*G)))**(1/3)
    K=0.3*G*M/rvir                             # J/kg
    return np.trapezoid(dfdlnM,lnM), np.trapezoid(dfdlnM*K,lnM)
fc,eK=stats(0.0); rhob=Ob0*3*(100*h*1e3/Mpc)**2/(8*np.pi*G)
rhoL=(1-Om0)*3*(100*h*1e3/Mpc)**2/(8*np.pi*G)*c**2
print(f"   z=0: baryon mass fraction in halos > 1e8 Msun = {fc:.3f}; mean virial K per unit collapsed mass = {eK/fc:.3e} J/kg (sigma_eff {np.sqrt(2*eK/fc/3)/1e3:.0f} km/s)")
Q=rhob*eK                                     # J/m3: heat radiated so far = K stored (virial), upper bound if all baryons in halos cooled
print(f"   accumulated virial heat of baryons / rho_L c^2 = {Q/rhoL:.2e}")
Tv=(0.59*C.m_p)*(eK/fc)/(1.5*k); TGH=hbar*100*h*1e3/Mpc/(2*np.pi*k)
print(f"   priced at the horizon (bits = Q/(k T_gas ln2), cost k T_GH ln2 each): x T_GH/T_gas = x {TGH/Tv:.1e} -> {Q/rhoL*TGH/Tv:.1e}")
print(f"   target (3/16) sqrt(OL) structure: rho_L is {1/(Q/rhoL):.1e} times the accumulated virial heat")
print(f"   upper bound, all baryon rest mass converted: Ob/OL = {Ob0/(1-Om0):.3f} of rho_L")
