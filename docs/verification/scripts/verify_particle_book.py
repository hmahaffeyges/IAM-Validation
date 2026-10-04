#!/usr/bin/env python3
"""Numbers for book chapters p4_15a_lepton_koide.tex, p4_15b_electron_mass.tex and p4_22b_higgs_record.tex.
Inputs: PDG 2024 (Navas et al., Phys. Rev. D 110, 030001) lepton and boson masses; CODATA via scipy.constants;
Planck 2018 (TT,TE,EE+lowE+lensing) cosmology; book sector values H0 = 67.16 (photon sector), 72.26 (matter sector).
Run: python docs/verification/scripts/verify_particle_book.py > docs/verification/scripts/verify_particle_book_output.txt"""
import numpy as np, scipy, scipy.constants as C
from scipy.optimize import brentq
from scipy.integrate import quad
print("scipy", scipy.__version__, "CODATA m_e =", C.m_e, "alpha =", C.alpha)
# ------------------------------------------------------------------ Koide (K)
me, mm, mt, smt = 0.51099895000, 105.6583755, 1776.93, 0.09          # PDG 2024, MeV
m = np.array([me, mm, mt]); s = np.sqrt(m)
Q = lambda mt_: (me+mm+mt_)/(np.sqrt(me)+np.sqrt(mm)+np.sqrt(mt_))**2
q = Q(mt); dq = (Q(mt+1e-4)-Q(mt-1e-4))/2e-4*smt
print(f"K1 Q(PDG 2024) = {q:.8f} +- {dq:.8f};  2/3 - Q = {2/3-q:.3e} = {(2/3-q)/dq:.2f} sigma;  Q(PDG 2022, m_tau=1776.86) = {Q(1776.86):.8f}")
print(f"K2 sqrt-mass vector (sqrt MeV): ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})")
cth = s.sum()/(np.sqrt(3)*np.linalg.norm(s)); th = np.degrees(np.arccos(cth))
print(f"K3 angle to (1,1,1) = {th:.5f} deg (45 deg exactly <=> Q = 2/3; cos^2 = 1/(3Q) = {1/(3*q):.7f})")
x = s.sum()/3
yx = np.sqrt(6*q-2)
ph = 2*np.pi*np.arange(3)/3            # k=0 tau, k=1 e, k=2 mu
order = np.array([mt, me, mm]); so = np.sqrt(order)
yc = (2/3)*np.sum(so*np.cos(ph)); ys = -(2/3)*np.sum(so*np.sin(ph))
y = np.hypot(yc, ys); d = np.arctan2(ys, yc)
print(f"K4 x = {x:.5f} sqrt(MeV), x^2 = {x**2:.3f} MeV; y/x = {y/x:.6f} (sqrt2 = {np.sqrt(2):.6f}; from Q: {yx:.6f}); delta = {d:.6f} rad; 2/9 = {2/9:.6f}; delta-2/9 = {d-2/9:.2e}")
rec = (x*(1+np.sqrt(2)*np.cos(d+ph)))**2
print(f"K5 masses from (x, delta, y/x = sqrt2), phases k=0,1,2 -> tau, e, mu: {rec[0]:.3f}, {rec[1]:.5f}, {rec[2]:.4f} MeV")
z = (x*(1+np.sqrt(2)*np.cos(ph)))**2
print(f"K6 delta = 0: masses {z[1]:.3f}, {z[2]:.3f}, {z[0]:.2f} MeV")
def dpart(mt_):
    s_ = np.sqrt([mt_, me, mm]); return np.arctan2(-(2/3)*np.sum(s_*np.sin(ph)), (2/3)*np.sum(s_*np.cos(ph)))
sd = abs(dpart(mt+smt)-dpart(mt-smt))/2
print(f"K7 sigma(delta) from m_tau = {sd:.2e} rad; (delta - 2/9)/sigma = {(d-2/9)/sd:.2f}")
mt_pred = brentq(lambda t: Q(t)-2/3, 1700, 1850)
print(f"K8 m_tau from m_e, m_mu at Q = 2/3: {mt_pred:.4f} MeV; PDG 2024 {mt} +- {smt}: pull {(mt-mt_pred)/smt:+.2f} sigma")
# m_tau at delta = 2/9 with x fixed by e and mu? solve x, then tau
def mt_from_delta(dd):
    # e at k=1, mu at k=2: sqrt(me)/sqrt(mm) fixes nothing alone; use x from e and mu with given delta via least squares on e
    xe = np.sqrt(me)/(1+np.sqrt(2)*np.cos(dd+ph[1])); return (xe*(1+np.sqrt(2)*np.cos(dd)))**2, (xe*(1+np.sqrt(2)*np.cos(dd+ph[2])))**2
t29, mu29 = mt_from_delta(2/9)
print(f"K9 delta = 2/9 exactly, x from m_e: m_tau = {t29:.2f} MeV, m_mu = {mu29:.4f} MeV (measured 105.6584)")
dl = np.linspace(0, 2*np.pi, 400001)[:-1]
for n in range(2, 8):
    mn = (1+np.sqrt(2)*np.cos(dl[:, None]+2*np.pi*np.arange(n)/n)).min(1)
    print(f"K10 n={n}: fraction of offsets with all roots positive {np.mean(mn > 0):.4f}  (analytic max(0, 1-n/4) = {max(0, 1-n/4):.4f})")
for n in range(3, 8):
    phn = 2*np.pi*np.arange(n)/n; sq = 1+np.sqrt(2)*np.cos(0.3+phn)
    print(f"K11 n={n}: Q_n with y/x = sqrt2 (signed roots, delta=0.3) = {np.sum(sq**2)/np.sum(sq)**2:.6f}; 2/n = {2/n:.6f}")
# one-loop QED running (MS-bar): mbar(mu) = M [1 - (alpha/pi)(1 + (3/2) ln(mu/M))]; alpha fixed at 1/137.036 (illustrative)
a = C.alpha
for mu in (mt, 91187.6):
    mb = m*(1-(a/np.pi)*(1+1.5*np.log(mu/m)))
    print(f"K12 one-loop QED MSbar at mu = {mu/1e3:.3f} GeV: Q = {np.sum(mb)/np.sum(np.sqrt(mb))**2:.6f}, 2/3 - Q = {2/3-np.sum(mb)/np.sum(np.sqrt(mb))**2:.2e}")
# ------------------------------------------------------------------ Electron (E)
hbar, c, G, kB, al, mE = C.hbar, C.c, C.G, C.k, C.alpha, C.m_e
mP = np.sqrt(hbar*c/G); Mpc = 3.0856775814913673e22
B = lambda H0, p=2.5: (hbar*(H0*1e3/Mpc)*np.log(2)*mP**1.5/(al**p*c**2))**0.4
print(f"E1 bracket B^(2/5) / m_e at H0 = 67.4: {B(67.4)/mE:.6f}")
print(f"E2 fixed point as derived, (2pi)^(-2/5) B: {(2*np.pi)**-0.4*B(67.4)/mE:.6f} m_e; with identified (2pi)^(-1/10): {(2*np.pi)**-0.1*B(67.4)/mE:.7f} m_e")
print(f"E3 (2pi)^(-2/5) = {(2*np.pi)**-0.4:.6f}; (2pi)^(3/10) = {(2*np.pi)**0.3:.6f}; (2pi)^(-1/10) = {(2*np.pi)**-0.1:.6f}; needed {mE/B(67.4):.6f}; equivalent coefficient in N: (2pi)^(3/4) = {(2*np.pi)**0.75:.4f}")
for H in (67.16, 67.36, 67.4, 72.26, 73.04):
    print(f"E4 H0 = {H}: m/m_e - 1 = {1e6*((2*np.pi)**-0.1*B(H)/mE-1):+9.0f} ppm")
print(f"E5 sigma(H0) = 0.54 -> sigma(m)/m = {0.4*0.54/67.36*100:.3f} %; H0 that makes it exact = {67.4*(mE/((2*np.pi)**-0.1*B(67.4)))**2.5:.4f}")
for p in (1.5, 2.0, 2.5, 3.0, 3.5):
    print(f"E6 alpha exponent p = {p}: m/m_e (with (2pi)^(-1/10), H0 = 67.4) = {(2*np.pi)**-0.1*B(67.4, p)/mE:.4f}")
EC = mE*c**2; EH = hbar*(67.4e3/Mpc)
print(f"E7 m_e c^2/(hbar H0) = {EC/EH:.3e} (log10 {np.log10(EC/EH):.2f}); T_C = m_e c^2/(2 pi k_B) = {EC/(2*np.pi*kB):.3e} K; T_GH = {EH/(2*np.pi*kB):.3e} K")
S = np.pi*(mP/mE)**2; Nn = (mP/mE)**1.5
print(f"E8 S_BH(Compton) = pi (m_P/m_e)^2 = {S:.3e}; (m_P/m_e)^(3/2) = {Nn:.3e}; alpha^(5/2) = {al**2.5:.3e}; E_bit = {hbar*(67.4e3/Mpc)*np.log(2)/(2*np.pi):.3e} J")
print(f"E9 Lambda_N := E_bit N / f at m_e (no prefactor) / (m_e c^2) = {hbar*(67.4e3/Mpc)*np.log(2)/(2*np.pi)*Nn/al**2.5/EC:.4f}")
print("E10 black hole: N k T_H ln2 = T_H S = Mc^2/2 (Smarr)")
# ------------------------------------------------------------------ Electroweak / Higgs (H)
GeV = 1e9*C.e; hbar_GeVs = hbar/GeV
mPl = np.sqrt(hbar*c/G)*c**2/GeV
t_of_T = lambda T, gs: 0.301*mPl/(np.sqrt(gs)*T**2)*hbar_GeVs
print(f"H1 t(159.5 GeV, g*=106.75) = {t_of_T(159.5,106.75):.3g} s; t(161 GeV) = {t_of_T(161,106.75):.3g}; t(158 GeV) = {t_of_T(158,106.75):.3g}; t(100 GeV) = {t_of_T(100,106.75):.3g} s")
GF = 1.1663788e-5; v = (np.sqrt(2)*GF)**-0.5
mH, mW, mZ, mtop = 125.20, 80.3692, 91.1880, 172.57
print(f"H2 v = {v:.3f} GeV; lambda = m_H^2/(2 v^2) = {mH**2/(2*v**2):.4f}; y_t = sqrt2 m_t/v = {np.sqrt(2)*mtop/v:.4f}; y_e = {np.sqrt(2)*me*1e-3/v:.3e}; y_tau = {np.sqrt(2)*mt*1e-3/v:.4e}")
for name, M in [("e", me*1e-3), ("mu", mm*1e-3), ("tau", mt*1e-3), ("W", mW), ("Z", mZ), ("H", mH), ("t", mtop)]:
    print(f"H3 {name}: m = {M:.6g} GeV, Compton time hbar/mc^2 = {hbar_GeVs/M:.3e} s, m/T_c = {M/159.5:.3g}")
T0 = C.k*2.7255/GeV; a_EW = T0/159.5*(3.938/106.75)**(1/3)
print(f"H4 a_EW = {a_EW:.3g}; ln E(a_EW) = {1-1/a_EW:.4g}")
h = 0.6736; Om = 0.3153; Ogh2 = 2.4728e-5; Or = Ogh2*(1+0.2271*3.046)/h**2; OL = 1-Om-Or
H0s = 100*h*1e3/Mpc; yr = 3.15576e7
age = lambda aa: quad(lambda u: 1/(u*H0s*np.sqrt(Or/u**4+Om/u**3+OL)), 0, aa, epsabs=0, epsrel=1e-10, limit=200)[0]/yr
for tG in (4.8, 9.0):
    aa = brentq(lambda u: age(u)/1e9-tG, 0.05, 1.0)
    print(f"H5 cosmic age {tG} Gyr: a = {aa:.3f}, z = {1/aa-1:.2f}, E = {np.exp(1-1/aa):.3f}")
a17 = 1/(1-np.log(0.17)); print(f"H6 E = 0.17 at a = {a17:.3f} (z = {1/a17-1:.2f}), cosmic age {age(a17)/1e9:.2f} Gyr")
print(f"H7 R(1) = Omega_m/(beta_m E(1)) with beta_m = Omega_m/2: {Om/((Om/2)*1.0):.1f} (identity)")
print(f"H8 t(150 MeV, g*=61.75) = {t_of_T(.15,61.75):.3g} s; T(z*=1089.9) = {C.k*2.7255*1090.9/C.e:.4f} eV; age(z*) = {age(1/1090.9):.4g} yr")
# ------------------------------------------------------------------ added checks
# K13: delta fixed by e and mu alone with y/x = sqrt2 exactly (x eliminated): ratio sqrt(me/mm) = (1+sqrt2 cos(d+2pi/3))/(1+sqrt2 cos(d+4pi/3))
f = lambda dd: (1+np.sqrt(2)*np.cos(dd+ph[1]))/(1+np.sqrt(2)*np.cos(dd+ph[2])) - np.sqrt(me/mm)
d_emu = brentq(f, 0.15, 0.26); x_emu = np.sqrt(me)/(1+np.sqrt(2)*np.cos(d_emu+ph[1]))
s_emu = 0.0000023/mm/2 * abs(1/((f(d_emu+1e-7)-f(d_emu-1e-7))/2e-7)) * np.sqrt(me/mm)   # sigma(delta) from sigma(m_mu)
print(f"K13 y/x = sqrt2 and (m_e, m_mu): delta = {d_emu:.9f} rad (sigma ~ {s_emu:.1e} from m_mu); 2/9 - delta = {2/9-d_emu:.3e}; m_tau = {(x_emu*(1+np.sqrt(2)*np.cos(d_emu)))**2:.4f} MeV")
print(f"K14 positivity edge delta = pi/12 = {np.pi/12:.6f}; pi/12 - delta = {np.pi/12-d:.6f} rad")
print(f"E11 Bekenstein bound for mass m in radius hbar/mc: S <= 2 pi E R/(hbar c) = 2 pi = {2*np.pi:.4f} (nats); area count pi (m_P/m_e)^2 = {S:.3e}; ratio = {S/(2*np.pi):.3e}")
Hz2 = np.sqrt(Om*27+Or*81+OL); print(f"E12 H(z=2)/H0 = {Hz2:.3f}; (H(z=2)/H0)^(2/5) = {Hz2**0.4:.3f}")
print(f"H9 k_B T ln2 at T_c = 159.5 GeV: {159.5*np.log(2):.2f} GeV per bit")
gs = 106.75; fH = 1.65e-5*(159.5/100)*(gs/100)**(1/6)
print(f"H10 Hubble frequency at T_c redshifted to today (Caprini et al. 2016 normalisation) = {fH:.2e} Hz; x beta/H = 10..1000 -> {10*fH:.1e} .. {1000*fH:.1e} Hz")
# K15: a second harmonic on the orbit, sampled at three phases, aliases onto the first harmonic.
for eps in (1e-5, 1e-4, 1e-3, 3.2e-3):
    qs = []
    for th in np.linspace(0, 2*np.pi, 73):
        rt = 1 + np.sqrt(2)*np.cos(d+ph) + eps*np.sqrt(2)*np.cos(2*(d+ph)+th)
        qs.append(np.sum(rt**2)/np.sum(rt)**2)
    qs = np.array(qs)
    print(f"K15 second-harmonic amplitude a2/y = {eps:.1e} (weight ratio w2/w1 = {eps**2:.1e}): max |Q - 2/3| over its phase = {np.max(abs(qs-2/3)):.2e}")
z3 = np.cos(2*(d+ph)); print(f"K16 on Z3, cos(2 phi_k) = cos(2 pi k/3 - 2 delta) (a first harmonic in k, phase -2 delta): max diff = {np.max(abs(z3-np.cos(ph-2*d))):.1e}")
print(f"K17 Boltzmann bound: w2/w1 = exp(-3 w0^2/(2kT)) <= 1e-5 -> kT <= {3/(2*np.log(1e5)):.4f} w0^2 (1/{1/(3/(2*np.log(1e5))):.2f}); <= 1e-10 (a2/y <= 1e-5) -> kT <= {3/(2*np.log(1e10)):.4f} w0^2 (1/{1/(3/(2*np.log(1e10))):.1f})")
print(f"K18 n = 2 admissible for |cos delta| < 1/sqrt2, i.e. pi/4 < delta < 3pi/4 (mod pi); at delta = {d:.4f}: cos = {np.cos(d):.4f} -> excluded")
