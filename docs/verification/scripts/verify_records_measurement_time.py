#!/usr/bin/env python3
"""Every equation and number carried into Part 2 'Records at the quantum scale' (p2_14_quantum_records.tex),
Part 5 'Duration, the arrow of time ...' (p5_03_time.tex) and Part 5 'Measurement as the writing of a record' (p5_04_measurement.tex).
Sources read in full: Quantum_Darwinism_at_Cosmological_Scales (861 PDF lines), IAM_Measurement_Problem_Quantum (388),
The_Two_Faces_of_Time (249). Corrections: PAPER_ERRATA.md rows QD1-QD9, MP1-MP7, TF1-TF3, GD1-GD4, EN8, V17, EM1, KO2.
Run: python docs/verification/scripts/verify_records_measurement_time.py > ..._output.txt
"""
import numpy as np
import sympy as sp
import scipy.constants as C
from scipy.integrate import solve_ivp, quad
from scipy.special import erfc

hb, G, c, kB, eV = C.hbar, C.G, C.c, C.k, C.e
Msun, kpc, Mpc = 1.98847e30, 3.0857e19, 3.0857e22
Om, OL = 0.3153, 0.6847
bm = Om / 2
H0g, H0m = 67.16, 72.26
tP = np.sqrt(hb * G / c**5)
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))

P("=== A. Records at the quantum scale (Quantum Darwinism) ===")
# A1 Zurek decoherence time ratio tau_D/tau_R = (lambda_th/dx)^2, lambda_th = hbar/sqrt(2 m kB T)
m, T, dx = 1e-3, 300.0, 1e-2
lam = hb / np.sqrt(2 * m * kB * T)
P(f"A1 lambda_th(1 g, 300 K) = {lam:.2e} m; (lambda_th/dx)^2 at dx = 1 cm = {(lam/dx)**2:.1e}")
# A2 halo decoherence time
M, R = 1e12 * Msun, 200 * kpc
tau = hb * R / (G * M**2)
tH = 13.8e9 * 3.15576e7
P(f"A2 tau_D = hbar R/(G M^2), 1e12 Msun, 200 kpc = {tau:.2e} s; 13.8 Gyr = {tH:.3e} s; ratio = 10^{np.log10(tH/tau):.1f}")
# A3 virial theorem for V = -k r^-n on a circular orbit: 2K = n|V|
r, k, n_, mm = sp.symbols('r k n m', positive=True)
V = -k * r**(-n_)
v2 = sp.simplify(r * sp.diff(-V, r) * (-1) / mm)  # m v^2/r = |F| = n k r^(-n-1)
Fmag = sp.diff(k * r**(-n_), r) * (-1)          # n k r^(-n-1)
v2 = sp.simplify(Fmag * r / mm)
K = sp.Rational(1, 2) * mm * v2
P("A3 2K/|V| for V = -k r^-n:", sp.simplify(2 * K / (k * r**(-n_))), "; n = 1 ->", sp.simplify((2 * K / (k * r**(-n_))).subs(n_, 1)))
P(f"A4 beta_m = Omega_m/2 = {bm:.5f}")
# A5 Press-Schechter one-bit-per-particle: n_eff = nu^2 - 1
nu, D = sp.symbols('nu D', positive=True)
nuD = sp.Symbol('nu0', positive=True) / D
F = sp.erfc(nuD / sp.sqrt(2))
dFdlnD = sp.simplify(D * sp.diff(F, D))
neff = sp.simplify(D * sp.diff(sp.log(dFdlnD), D))
P("A5 dF_coll/dlnD =", sp.simplify(dFdlnD.subs(sp.Symbol('nu0', positive=True), nu * D)), "; n_eff = dln(dF/dlnD)/dlnD =",
  sp.simplify(neff.subs(sp.Symbol('nu0', positive=True), nu * D)))
P(f"   n_eff = 7/2 at nu = {np.sqrt(4.5):.4f}; 5/2 at nu = {np.sqrt(3.5):.4f}")
P("   int_0^inf nu^2 exp(-nu^2/2) dnu =", sp.integrate(nu**2 * sp.exp(-nu**2 / 2), (nu, 0, sp.oo)), "(= sqrt(pi/2), no D dependence)")
# A6 top-down exponent, matter domination
a, n = sp.symbols('a n', positive=True)
Hs = a**sp.Rational(-3, 2); rho = a**-3; Ds = a; TH = Hs; AH = Hs**-2
integrand = rho * Ds**n * 1 * Hs / (TH * AH) / (a * Hs)   # dS/da = Idot/(T_H A_H) * dt/da, dt = da/(a H)
expo = sp.simplify(sp.log(sp.simplify(integrand)).expand(force=True) / sp.log(a))
P("A6 dS_info/da exponent:", sp.simplify(sp.powsimp(integrand)), "; S ~ a^(n-9/2); n - 9/2 = -1 -> n =", sp.solve(sp.Eq(n - sp.Rational(9, 2), -1), n)[0])
P("   check S(a) for n = 7/2:", sp.simplify(sp.integrate(integrand.subs(n, sp.Rational(7, 2)), a)))
# A7 numerical slope p of dS/dlna with full LambdaCDM growth (radiation neglected)
H2 = lambda a: Om * a**-3 + OL
def grow():
    def rhs(l, y):
        a = np.exp(l); dlnH = -1.5 * Om * a**-3 / H2(a)
        return [y[1], -(2 + dlnH) * y[1] + 1.5 * Om * a**-3 / H2(a) * y[0]]
    return solve_ivp(rhs, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)
S = grow()
Dn = lambda a: S.sol(np.log(a))[0] / S.sol(0)[0]
fg = lambda a: S.sol(np.log(a))[1] / S.sol(np.log(a))[0]
for nn in (2.5, 3.0, 3.5):
    def slope(a1, a2):
        aa = np.logspace(np.log10(a1), np.log10(a2), 60)
        y = a**0  # placeholder
        H = np.sqrt(H2(aa))
        g = aa**-3 * Dn(aa)**nn * fg(aa) * H      # rho_m D^n f / (T_H A_H), T_H A_H ~ 1/H
        return np.polyfit(np.log(aa), np.log(g), 1)[0]
    P(f"A7 n = {nn}: p(matter era a 0.01-0.1) = {slope(0.01, 0.1):+.2f}; p(late a 0.25-1) = {slope(0.25, 1):+.2f}  (E = exp(1-1/a) needs p = -1)")
# A8 Compton-sphere area count and its Landauer price
mP = np.sqrt(hb * c / G); me = C.m_e
mm_, mP_ = sp.symbols('m m_P', positive=True)
lP2 = sp.Symbol('l_P2', positive=True)
Sexpr = sp.simplify(4 * sp.pi * (sp.Symbol('hbar') / (mm_ * sp.Symbol('c')))**2 / (4 * sp.Symbol('hbar') * sp.Symbol('G') / sp.Symbol('c')**3))
P("A8 S = 4 pi lambdabar_C^2/(4 l_P^2) =", Sexpr, "= pi (m_P/m)^2 with m_P^2 = hbar c/G")
H0s = 67.4e3 / Mpc
Se = np.pi * (mP / me)**2
Ebit = hb * H0s * np.log(2) / (2 * np.pi)
P(f"   electron: S = {Se:.3e}; E_bit = hbar H0 ln2/2pi = {Ebit:.3e} J (H0 = 67.4); S E_bit = {Se*Ebit:.3e} J vs m_e c^2 = {me*c**2:.3e} J (ratio {Se*Ebit/(me*c**2):.2e})")
# A9 black hole bit rate and Smarr
Ms = Msun
Pw = hb * c**6 / (15360 * np.pi * G**2 * Ms**2); TBH = hb * c**3 / (8 * np.pi * G * Ms * kB)
P(f"A9 Gamma = P/(k T ln2) = {Pw/(kB*TBH*np.log(2)):.2f} bits/s; c^3/(1920 G M ln2) = {c**3/(1920*G*Ms*np.log(2)):.2f}; radiation entropy rate (4/3)P/T in bits = {4/3*Pw/TBH/kB/np.log(2):.2f}")
Mm = sp.Symbol('M', positive=True); hbar_, G_, c_, k_ = sp.symbols('hbar G c k_B', positive=True)
T_H = hbar_ * c_**3 / (8 * sp.pi * G_ * Mm * k_); S_BH = k_ * 4 * sp.pi * (2 * G_ * Mm / c_**2)**2 * c_**3 / (4 * G_ * hbar_)
P("   T_H S_BH / (M c^2) =", sp.simplify(T_H * S_BH / (Mm * c_**2)), "(Smarr: E = T_H S_BH = Mc^2/2; with S counted in bits and k T ln2 per bit the same product)")
# A10 mu today, at DESI bins, f sigma8 deficits
mu = lambda a: H2(a) / (H2(a) + bm * np.exp(1 - 1 / a))
P(f"A10 mu(1) = 1/(1+beta_m) = {mu(1.0):.4f}; mu0 = {mu(1.0)-1:+.4f}")
for name, z in [("BGS", 0.295), ("LRG1", 0.510), ("LRG2", 0.706), ("LRG3", 0.934), ("ELG2", 1.321), ("QSO", 1.484), ("z=1", 1.0), ("z=2", 2.0)]:
    P(f"    {name:5s} z = {z:.3f}: mu = {mu(1/(1+z)):.4f}, 1 - mu = {100*(1-mu(1/(1+z))):.2f} %")
def growm(muf):
    def rhs(l, y):
        a = np.exp(l); dlnH = -1.5 * Om * a**-3 / H2(a)
        return [y[1], -(2 + dlnH) * y[1] + 1.5 * Om * a**-3 / H2(a) * muf(a) * y[0]]
    return solve_ivp(rhs, (np.log(1e-3), 0), [1e-3, 1e-3], dense_output=True, rtol=1e-10, atol=1e-14)
SL, SI = growm(lambda a: 1.0), growm(mu)
for z in (0, 0.3, 0.5, 1.0):
    aa = 1 / (1 + z); yl, yi = SL.sol(np.log(aa)), SI.sol(np.log(aa))
    P(f"    f sigma8 deficit at z = {z}: {100*(1 - yi[1]/yl[1]):.2f} %")
P(f"A11 H0(matter) = 67.16 sqrt(1 + beta_m) = {H0g*np.sqrt(1+bm):.2f}")

P("\n=== B. Measurement as the writing of a record ===")
QL = lambda T: kB * T * np.log(2) / eV
for T in (300, 77, 4.2, 4.0, 0.01):
    P(f"B1 Q_L({T} K) = {QL(T):.3e} eV")
for name, Q in [("photon absorbed (2 eV)", 2.0), ("retinal rod (2.5 eV)", 2.5), ("CCD pixel (3 eV)", 3.0)]:
    P(f"B2 {name}: Q/Q_L(300 K) = {Q/QL(300):.0f}")
Qs, QLs = sp.symbols('Q Q_L', positive=True)
P("B3 1 - exp(1 - Q_L/Q)/e - (1 - exp(-Q_L/Q)) =", sp.simplify(1 - sp.exp(1 - QLs / Qs) / sp.E - (1 - sp.exp(-QLs / Qs))))
Ff = lambda Q, T: 1 - np.exp(-QL(T) / Q)
for T in (0.01, 4.0, 300):
    P(f"B4 F(0.1 eV, {T} K) = {Ff(0.1, T):.4f}")
P(f"B5 F at Q/Q_L = 0.04: 1 - {np.exp(-25):.1e}; at Q = Q_L: {1-np.exp(-1):.3f}; at Q/Q_L = 100: {1-np.exp(-0.01):.4f}")
P(f"B6 flight time 5 m: {5/c*1e9:.1f} ns")
# B7 table: gravitational times, 300 K, radii stated
rows = [("electron", 9.109e-31, 1.0e-10), ("C60", 1.2e-24, 5.0e-10), ("virus", 1e-18, 5.0e-8), ("bacterium", 1e-15, 5.0e-7),
        ("dust grain", 1e-12, 5.0e-6), ("sand grain", 1e-6, 5.0e-4), ("cat", 4.0, 0.15), ("human", 70.0, 0.30)]
for name, mass, rad in rows:
    EG = G * mass**2 / rad
    tPD = hb / EG
    tI = hb * (kB * 300)**2 * np.log(2) / EG**3
    tI10 = hb * (kB * 0.01)**2 * np.log(2) / EG**3
    rho = mass / (4 / 3 * np.pi * rad**3)
    P(f"B7 {name:10s} m = {mass:.2e} kg, R = {rad:.1e} m (rho {rho:.1e}): E_G = {EG:.2e} J, tau_PD = {tPD:.2e} s, tau_IAM(300 K) = {tI:.2e} s, (10 mK) = {tI10:.2e} s")
EGcat = G * 16 / 0.15
P(f"B8 cat: tau_IAM(300 K) = {hb*(kB*300)**2*np.log(2)/EGcat**3:.2e} s; t_P = {tP:.3e} s; t_P/tau = {tP/(hb*(kB*300)**2*np.log(2)/EGcat**3):.1e}")
# silica check used elsewhere in the book
rs = (3 * 1e-12 / (4 * np.pi * 2200))**(1 / 3); EGs = G * 1e-24 / rs
P(f"B9 silica 1e-12 kg, 2200 kg/m^3: R = {rs*1e6:.2f} um, E_G = {EGs:.2e} J, tau_IAM(10 mK) = {hb*(kB*0.01)**2*np.log(2)/EGs**3:.0f} s, tau_PD = {hb/EGs*1e6:.2f} us")
# B10 Wigner's friend
Nb = 1000 * 2.0 / QL(300)
P(f"B10 1000 photons x 2 eV / Q_L(300 K) = {Nb:.0f} bits; P(reversal) = 2^-N = 10^-{Nb*np.log10(2):.0f}")
# B11 Bell under pointer-basis dephasing (Horodecki)
cc = sp.Symbol('c', nonnegative=True)
Tm = sp.diag(cc, -cc, 1)  # correlation matrix of (|00><00|+|11><11| + c(|00><11|+h.c.))/2
ev = sorted([e for e in (Tm.T * Tm).diagonal()], key=lambda e: 0)
P("B11 T^T T eigenvalues:", list((Tm.T * Tm).diagonal()), "-> S_max = 2 sqrt(1 + c^2)")
# numerical CHSH maximisation for c = 0.3
def chsh(cv):
    import itertools
    sx = np.array([[0, 1], [1, 0]]); sz = np.diag([1, -1]); sy = np.array([[0, -1j], [1j, 0]])
    rho = 0.5 * np.array([[1, 0, 0, cv], [0, 0, 0, 0], [0, 0, 0, 0], [cv, 0, 0, 1]])
    Tn = np.array([[np.real(np.trace(rho @ np.kron(s1, s2))) for s2 in (sx, sy, sz)] for s1 in (sx, sy, sz)])
    w = np.sort(np.linalg.eigvalsh(Tn.T @ Tn))[::-1]
    return 2 * np.sqrt(w[0] + w[1])
P(f"    numeric S_max(c = 0.3) = {chsh(0.3):.4f}; 2 sqrt(1.09) = {2*np.sqrt(1.09):.4f}; fixed pure-state settings sqrt2(1+c) < 2 for D = 1 - c > {2-np.sqrt(2):.3f}; isotropic 2 sqrt2 (1-D) < 2 for D > {1-1/np.sqrt(2):.3f}")
P(f"B12 photon pairs 1200 km / spins 1.3 km = {1200/1.3:.0f}")

P("\n=== C. Time ===")
aa = sp.Symbol('a', positive=True)
Ea = sp.exp(1 - 1 / aa)
P("C1 E(a): lim a->0 =", sp.limit(Ea, aa, 0, '+'), "; E(1) =", Ea.subs(aa, 1), "; lim a->oo =", sp.limit(Ea, aa, sp.oo), "; dE/da =", sp.simplify(sp.diff(Ea, aa)))
P(f"C2 E(z = 10) = {np.exp(1-11):.2e}; E(z = 2) = {np.exp(1-3):.3f}; E(z = 1) = {np.exp(-1):.3f}")
P(f"C3 E(a_EW = 4.9e-16) = exp({1-1/4.9e-16:.2e})")
P(f"C4 beta_m = {bm:.5f}; H0(matter) = {H0g*np.sqrt(1+bm):.2f}")
print("\n".join(out))
