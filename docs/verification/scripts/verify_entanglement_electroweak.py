#!/usr/bin/env python3
"""Numbers for book chapters p4_21_entanglement_records.tex and p4_22_electroweak.tex.
Sources: 'Entanglement, Decoherence, and the Thermodynamic Cost of Classical Records' (Oct 2026 rev.) and
'Electroweak Symmetry Breaking and the Matter Sector' (Oct 2026 rev.). Constants: scipy.constants (CODATA), PDG 2022, Planck 2018."""
import numpy as np, scipy.constants as C
from scipy.optimize import brentq
from scipy.integrate import quad
hb, kB, G, c = C.hbar, C.k, C.G, C.c
GeV = 1e9*C.e; hbar_GeVs = hb/GeV; hbarc_GeVm = hb*c/GeV
E = lambda a: np.exp(1-1/a)
print("== Entanglement chapter ==")
rho = 2200.0
EG = lambda m: G*m*m/((3*m/(4*np.pi*rho))**(1/3))
tI = lambda m, T: hb*(kB*T)**2*np.log(2)/EG(m)**3
tPD = lambda m: hb/EG(m)
for m, T in [(1e-15, .01), (1e-12, .01), (1e-12, .02)]:
    print(f"E1 m={m:.0e} kg T={T*1e3:.0f} mK: tau_IAM={tI(m,T):.4g} s  tau_PD={tPD(m):.4g} s")
print(f"E2 crossover 10 mK: m={10**brentq(lambda l: np.log(tI(10**l,.01)/tPD(10**l)),-20,-5):.3g} kg")
# CHSH for a Bell state with coherence factor c in the pointer (which-path) basis
sx = np.array([[0,1],[1,0]]); sy = np.array([[0,-1j],[1j,0]]); sz = np.diag([1,-1]); P = [sx, sy, sz]
def rho_deph(cf):
    psi = np.array([1,0,0,1])/np.sqrt(2); r = np.outer(psi, psi.conj())
    r[0,3] *= cf; r[3,0] *= cf; return r
def rho_iso(p):  # Werner: p |Phi+><Phi+| + (1-p) I/4
    psi = np.array([1,0,0,1])/np.sqrt(2); return p*np.outer(psi,psi)+(1-p)*np.eye(4)/4
def Tmat(r): return np.array([[np.trace(r@np.kron(a,b)).real for b in P] for a in P])
def Smax(r):  # Horodecki 1995: S_max = 2 sqrt(sum of two largest eigenvalues of T^T T)
    ev = np.sort(np.linalg.eigvalsh(Tmat(r).T@Tmat(r))); return 2*np.sqrt(ev[-1]+ev[-2])
def S_fixed(r):  # standard settings a=z, a'=x, b=(z+x)/sqrt2, b'=(z-x)/sqrt2
    T = Tmat(r); n = {'z':np.array([0,0,1.]),'x':np.array([1.,0,0])}
    b = (n['z']+n['x'])/np.sqrt(2); bp = (n['z']-n['x'])/np.sqrt(2)
    e = lambda u,v: u@T@v
    return e(n['z'],b)+e(n['z'],bp)+e(n['x'],b)-e(n['x'],bp)
for cf in [1, .75, .5, .25, .1, 0]:
    print(f"E3 dephasing c={cf:.2f}: S_max={Smax(rho_deph(cf)):.4f} (2sqrt(1+c^2)={2*np.sqrt(1+cf**2):.4f}); fixed settings S={S_fixed(rho_deph(cf)):.4f} (sqrt2(1+c)={np.sqrt(2)*(1+cf):.4f}); isotropic p=c: S_max={Smax(rho_iso(cf)):.4f} (2sqrt2 c={2*np.sqrt(2)*cf:.4f})")
print(f"E4 isotropic noise loses violation below p=1/sqrt2={1/np.sqrt(2):.4f} (D>{1-1/np.sqrt(2):.4f}); fixed settings under dephasing lose it below c=sqrt2-1={np.sqrt(2)-1:.4f}")
for eta in [0.5, 1, 2, 3]:
    ce = np.exp(-eta); cr = 1-E(eta)/np.e
    print(f"E5 t/tau={eta}: exponential c={ce:.4f} S_max={2*np.sqrt(1+ce**2):.4f}; assumed ramp c={cr:.4f} S_max={2*np.sqrt(1+cr**2):.4f}")
Gt = 1.42; print(f"E6 top lifetime hbar/Gamma_t (Gamma_t=1.42 GeV, PDG 2022) = {hbar_GeVs/Gt:.3g} s")
print(f"E7 S_info: integrand a^(n-11/2) at n=7/2 is a^-2; integral = -1/a + const; ln E = 1 - 1/a with E(1)=1; E(z=10)={E(1/11):.3g}")
print("== Electroweak chapter ==")
mPl = np.sqrt(hb*c/G)*c**2/GeV  # Planck mass in GeV
def t_of_T(T, gs): return 0.301*mPl/(np.sqrt(gs)*T**2)*hbar_GeVs  # radiation era, t = 1/(2H)
print(f"W1 m_Pl={mPl:.4e} GeV; t(T=159.5 GeV, g*=106.75)={t_of_T(159.5,106.75):.3g} s; t(100 GeV)={t_of_T(100,106.75):.3g} s")
print(f"W2 t(150 MeV, g*=61.75)={t_of_T(.15,61.75):.3g} s; t(150 MeV, g*=17.25)={t_of_T(.15,17.25):.3g} s")
print(f"W3 t(0.1 MeV, g*=3.36)={t_of_T(1e-4,3.36):.3g} s; t(0.07 MeV)={t_of_T(7e-5,3.36):.3g} s")
T0K = 2.7255; T0 = kB*T0K/GeV
a_EW = T0/159.5*(3.938/106.75)**(1/3)
print(f"W4 a_EW={a_EW:.3g}, 1/a_EW={1/a_EW:.3g}, ln E(a_EW)={1-1/a_EW:.4g}")
zs = 1089.9; print(f"W5 T(z*=1089.9)={T0K*(1+zs):.0f} K = {kB*T0K*(1+zs)/C.e:.4f} eV")
h = 0.6736; Om = 0.3153; Ogh2 = 2.4728e-5; Or = Ogh2*(1+0.2271*3.046)/h**2; OL = 1-Om-Or
H0 = 100*h*1e3/C.parsec/1e6
age = lambda z: quad(lambda a: 1/(a*H0*np.sqrt(Or/a**4+Om/a**3+OL)), 0, 1/(1+z), epsabs=0, epsrel=1e-10, limit=200)[0]/(3.15576e7)
print(f"W6 age(z*)={age(zs):.4g} yr; age(z=30)={age(30)/1e6:.4g} Myr; age(z=0)={age(0)/1e9:.4g} Gyr")
print(f"W7 E(z=30)={E(1/31):.3g}; E(z=10)={E(1/11):.3g}; E(z=1)={E(.5):.3g}; mu(z=0)=1/(1+beta_m)={1/(1+0.15765):.4f}, mu0-1={1/(1+0.15765)-1:.4f}")
Ob = 0.0493; Odm = Om-Ob
print(f"W8 Ob/2={Ob/2:.5f}, Odm/2={Odm/2:.5f}, sum={Om/2:.5f}; baryon share={Ob/Om*100:.2f} %, dark share={Odm/Om*100:.2f} %")
MW, MZ = 80.377, 91.1876
print(f"W9 weak range hbar c/(M_W c^2)={hbarc_GeVm/MW:.4g} m; Z: {hbarc_GeVm/MZ:.4g} m")
GF = 1.1663788e-5; print(f"W10 v=(sqrt2 G_F)^(-1/2)={(np.sqrt(2)*GF)**-0.5:.3f} GeV")
sig = 0.18; print(f"W11 string tension 0.18 GeV^2 = {sig/(hbarc_GeVm*1e15):.3f} GeV/fm")
for k in (-1, 1, 2): print(f"W12 virial 2<K> = k<V>: k={k}: <K>/<V> = {k/2}")
