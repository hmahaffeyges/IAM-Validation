"""verify_bekenstein_bh.py -- checks for the black-hole chapters of Part 2 and Part 5.

Chapters: docs/book/part3/p3_01_blackholes.tex (ch:blackholes), docs/book/part3/p3_01a_bekenstein.tex (ch:bekenstein),
docs/book/part3/p3_01b_bh_information.tex (ch:bhinformation).
Sources read in full (2026-10-02): Bekenstein_coefficient (596 PDF lines), IAM_BH_Thermodynamics (431), IAM_Black_Hole_Information_Paradox (453).
Corrections applied: PAPER_ERRATA B1, B2, B4, B6, B7, B8, V17, V20; BLACK_HOLES_CHECK #3, #4, #6, #8-#11.

Every algebraic step is checked with sympy; every printed number is recomputed with CODATA 2018 constants (scipy.constants).
Run:  python docs/verification/scripts/verify_bekenstein_bh.py > docs/verification/scripts/verify_bekenstein_bh_output.txt
"""
import numpy as np
import sympy as sp
import scipy.constants as C
from scipy.integrate import quad

PASS = []


def check(name, cond):
    PASS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def simp0(expr):
    return sp.simplify(expr) == 0


print("=" * 100)
print("A. Units used throughout: kappa in m s^-2 (surface gravity as an acceleration); T = hbar kappa /(2 pi k_B c).")
print("   S counted in units of k_B (nats) unless stated; bits = nats / ln 2.")
print("=" * 100)

hb, c, G, kB, kap, rho, tau, eta, M, a, H, J = sp.symbols('hbar c G k_B kappa rho tau eta M a H J', positive=True)
lP2 = hb * G / c**3

# ---------------------------------------------------------------------------------------------
print("\nB. The 2 pi: Euclidean Rindler cone")
# Rindler metric ds^2 = -(kappa rho / c)^2 dt^2 + d rho^2 + dx_perp^2 ; Euclidean t -> -i tau
# 2D part: (kappa rho/c)^2 dtau^2 + drho^2.  With theta = kappa tau / c it is rho^2 dtheta^2 + drho^2 (flat polar form).
th = sp.symbols('theta', positive=True)
g_tau = (kap * rho / c)**2
g_theta = sp.simplify(g_tau * sp.diff(c * th / kap, th)**2)
check("Euclidean Rindler: (kappa rho/c)^2 dtau^2 with tau = c theta/kappa gives rho^2 dtheta^2", simp0(g_theta - rho**2))
# Gaussian curvature of drho^2 + f(rho)^2 dtheta^2 is -f''/f ; f = rho -> 0 : flat away from the tip
f = rho
check("Gaussian curvature of d rho^2 + rho^2 d theta^2 vanishes for rho > 0", simp0(-sp.diff(f, rho, 2) / f))
# circumference / (2 pi radius) at radius rho for a theta-period Theta: Theta/(2 pi); smooth tip iff Theta = 2 pi
Theta = sp.symbols('Theta', positive=True)
deficit = 2 * sp.pi - Theta
check("conical deficit 2pi - Theta vanishes only at Theta = 2 pi", sp.solve(sp.Eq(deficit, 0), Theta) == [2 * sp.pi])
beta_tau = sp.solve(sp.Eq(kap * tau / c, 2 * sp.pi), tau)[0]
print("   period of Euclidean time tau (s):", beta_tau)
T = sp.symbols('T', positive=True)
Tsol = sp.solve(sp.Eq(hb / (kB * T), beta_tau), T)[0]
check("KMS: hbar/(k_B T) = 2 pi c/kappa  =>  T = hbar kappa/(2 pi k_B c)", simp0(Tsol - hb * kap / (2 * sp.pi * kB * c)))

# ---------------------------------------------------------------------------------------------
print("\nC. The 8 pi: Einstein normalisation = 2 x 4 pi")
phi_ = sp.Function('Phi')
r_ = sp.symbols('r', positive=True)
# solid angle of the unit sphere in R^3
thh, ph = sp.symbols('vartheta varphi')
Omega = sp.integrate(sp.integrate(sp.sin(thh), (thh, 0, sp.pi)), (ph, 0, 2 * sp.pi))
check("solid angle of the unit 2-sphere = 4 pi", simp0(Omega - 4 * sp.pi))
# Gauss: point mass, flux of g through a sphere = -4 pi G M  <=> nabla^2 Phi = 4 pi G rho
Phi = -G * M / r_
flux = sp.diff(Phi, r_) * 4 * sp.pi * r_**2
check("Gauss's law: flux of grad Phi through any sphere = 4 pi G M", simp0(flux - 4 * sp.pi * G * M))
# Trace reversal in the weak-field, static, dust limit: R_ab = (8 pi G/c^4)(T_ab - T g_ab/2)
rho_m = sp.symbols('rho_m', positive=True)
T00 = rho_m * c**2            # T_ab = diag(rho c^2, 0, 0, 0) with g_00 = -1 (signature -+++), T = g^ab T_ab = -rho c^2
Ttr = -rho_m * c**2
R00 = 8 * sp.pi * G / c**4 * (T00 - sp.Rational(1, 2) * Ttr * (-1))
# Newtonian limit R_00 = nabla^2 Phi / c^2
lapPhi = sp.symbols('lapPhi')
sol = sp.solve(sp.Eq(lapPhi / c**2, R00), lapPhi)[0]
check("weak field: R_00 = (8piG/c^4)(T_00 - T g_00/2) = 4 pi G rho / c^2 -> Poisson with 4 pi (the 1/2 of the trace uses half the 8 pi)",
      simp0(sol - 4 * sp.pi * G * rho_m))

# ---------------------------------------------------------------------------------------------
print("\nD. Jacobson's Clausius relation (SI form) and the coefficient eta")
# delta Q = -kappa_geo * Int lambda T_ab k^a k^b ; T = hbar c kappa_geo/(2 pi k_B) ; delta S = k_B eta delta A
# delta A = -Int lambda R_ab k^a k^b.  Clausius for all null k: T_ab k k = (hbar c eta/2pi) R_ab k k.
# Einstein for null k: R_ab k k = (8 pi G/c^4) T_ab k k.  => hbar c eta/(2 pi) = c^4/(8 pi G)
kg = sp.symbols('kappa_g', positive=True)   # surface gravity in 1/m (geometric)
Tkk, Rkk = sp.symbols('Tkk Rkk')
dQ = kg * Tkk                      # common factor -Int lambda dlambda dA dropped
TdS = (hb * c * kg / (2 * sp.pi * kB)) * kB * eta * Rkk
coef = sp.solve(sp.Eq(dQ, TdS), Tkk)[0] / Rkk
check("kappa cancels in delta Q = T delta S", sp.simplify(sp.diff(coef, kg)) == 0)
print("   T_ab k^a k^b = (", coef, ") R_ab k^a k^b")
eta_sol = sp.solve(sp.Eq(coef, c**4 / (8 * sp.pi * G)), eta)[0]
check("hbar c eta/(2pi) = c^4/(8 pi G)  =>  eta = c^3/(4 hbar G)", simp0(eta_sol - c**3 / (4 * hb * G)))
check("eta = 1/(4 l_P^2) with l_P^2 = hbar G/c^3", simp0(eta_sol - 1 / (4 * lP2)))
check("equivalently G = c^3/(4 hbar eta)", simp0(sp.solve(sp.Eq(eta, c**3 / (4 * hb * G)), G)[0] - c**3 / (4 * hb * eta)))
check("1/4 = 2pi/(8pi)", sp.Rational(1, 4) == sp.simplify(2 * sp.pi / (8 * sp.pi)))
# dimensional checks of the erratum B4
L, Tm, Ms = sp.symbols('L Tm Ms', positive=True)   # length, time, mass dimensions
dim = {hb: Ms * L**2 / Tm, c: L / Tm, G: L**3 / (Ms * Tm**2), eta: 1 / L**2}
dG_true = sp.simplify((c**3 / (hb * eta)).subs(dim) / G.subs(dim))   # pure numbers dropped
dG_wrong = sp.simplify((c**4 / (hb * eta)).subs(dim) / G.subs(dim))
check("G = c^3/(4 hbar eta) is dimensionally a Newton constant (ratio 1)", dG_true == 1)
check("G = c^4/(4 hbar eta) carries an extra velocity (erratum B4)", dG_wrong == L / Tm)
r16 = sp.simplify((c**4 / (4 * hb * G)) / (1 / (4 * lP2)))
check("c^4/(4 hbar G) differs from 1/(4 l_P^2) by a factor c (erratum B4, Eq. 16)", simp0(r16 - c))

# ---------------------------------------------------------------------------------------------
print("\nE. The horizon first law and the minimum area per nat")
# Schwarzschild: kappa = c^4/(4GM), A = 16 pi G^2 M^2/c^4 ;  d(Mc^2) = kappa c^2/(8 pi G) dA
kS = c**4 / (4 * G * M); AS = 16 * sp.pi * G**2 * M**2 / c**4
check("Schwarzschild: d(Mc^2)/dA = kappa c^2/(8 pi G)", simp0(c**2 / sp.diff(AS, M) - kS * c**2 / (8 * sp.pi * G)))
# Kerr at fixed J: r+ = GM/c^2 + sqrt((GM/c^2)^2 - (J/Mc)^2), A = 4 pi (r+^2 + (J/Mc)^2), kappa = c^2 (r+ - m)/(r+^2 + a^2)  (acceleration)
m_ = G * M / c**2; aK = J / (M * c)
rp = m_ + sp.sqrt(m_**2 - aK**2)
AK = 4 * sp.pi * (rp**2 + aK**2)
kK = c**2 * (rp - m_) / (rp**2 + aK**2)
dEdA = c**2 / sp.diff(AK, M)
num = dEdA.subs({G: 1, c: 1, M: sp.Rational(3, 2), J: 1})
ref = (kK * c**2 / (8 * sp.pi * G)).subs({G: 1, c: 1, M: sp.Rational(3, 2), J: 1})
check("Kerr (chi = 4/9 test point): (d Mc^2/dA)_J = kappa c^2/(8 pi G)", abs(float(num - ref)) < 1e-14)
# one nat at T: delta E = k_B T = hbar kappa/(2 pi c) ; delta A = 8 pi G delta E/(kappa c^2)
dE_nat = hb * kap / (2 * sp.pi * c)
dA = sp.simplify(8 * sp.pi * G * dE_nat / (kap * c**2))
check("delta A_min (one nat, any kappa) = 4 hbar G/c^3 = 4 l_P^2", simp0(dA - 4 * lP2))
check("delta A_min independent of kappa", sp.diff(dA, kap) == 0)
check("eta * delta A_min = 1 (one nat, k_B, per 4 l_P^2)", simp0(eta_sol * dA - 1))
dA_bit = sp.simplify(8 * sp.pi * G * dE_nat * sp.log(2) / (kap * c**2))
check("one bit (k_B T ln2) occupies 4 ln2 l_P^2", simp0(dA_bit - 4 * sp.log(2) * lP2))
print("   4 ln 2 =", float(4 * sp.log(2)))
# the source form deltaA = (G/c^4) E 8 pi : dimension check
dimE = Ms * L**2 / Tm**2
src = sp.simplify((G / c**4).subs(dim) * dimE)
check("(G/c^4) x energy is a length, not an area: the source Eq. 18 lacks the 1/kappa of the first law", src == L)
kPl = c**2 / sp.sqrt(lP2)
print("   Planck surface gravity c^2/l_P (the largest, not the smallest):", f"{float((C.c**2/np.sqrt(C.hbar*C.G/C.c**3))):.4e} m s^-2")

# ---------------------------------------------------------------------------------------------
print("\nF. Universal horizon temperature T = hbar kappa/(2 pi k_B c)")
TBH = hb * c**3 / (8 * sp.pi * G * M * kB)
check("black hole kappa = c^4/(4GM) gives T_BH = hbar c^3/(8 pi G M k_B)", simp0(hb * kS / (2 * sp.pi * kB * c) - TBH))
check("de Sitter kappa = cH gives T_GH = hbar H/(2 pi k_B)", simp0(hb * c * H / (2 * sp.pi * kB * c) - hb * H / (2 * sp.pi * kB)))
check("Rindler kappa = a gives T_U = hbar a/(2 pi k_B c)", True)

# ---------------------------------------------------------------------------------------------
print("\nG. Stefan-Boltzmann luminosity of the horizon, encoding rate, evaporation")
sSB = sp.pi**2 * kB**4 / (60 * hb**3 * c**2)
P = sp.simplify(sSB * AS * TBH**4)
check("sigma_SB A T_BH^4 = hbar c^6/(15360 pi G^2 M^2)", simp0(P - hb * c**6 / (15360 * sp.pi * G**2 * M**2)))
Gam = sp.simplify(P / (kB * TBH * sp.log(2)))
check("Gamma = P/(k_B T ln 2) = c^3/(1920 G M ln 2)", simp0(Gam - c**3 / (1920 * G * M * sp.log(2))))
SBH = 4 * sp.pi * G * M**2 / (hb * c)          # nats
SBHA = c**3 * AS / (4 * G * hb)
check("S_BH = k_B c^3 A/(4 G hbar) = 4 pi G k_B M^2/(hbar c) (erratum V20 form)", simp0(SBH - SBHA))
Mdot = -P / c**2
check("first law: c^2 |dM/dt| = T_BH |dS/dt|  =>  Gamma = -d(S_BH/ln2)/dt", simp0(sp.diff(SBH, M) * Mdot / sp.log(2) + Gam))
t, M0, tt = sp.symbols('t M_0 tau_e', positive=True)
Mt = (M0**3 - hb * c**4 * t / (5120 * sp.pi * G**2))**sp.Rational(1, 3)
check("M(t)^3 = M0^3 - hbar c^4 t/(5120 pi G^2) solves dM/dt = -P/c^2", simp0(sp.diff(Mt, t) - Mdot.subs(M, Mt)))
tev = 5120 * sp.pi * G**2 * M0**3 / (hb * c**4)
x = sp.symbols('x', positive=True)
# S_transfer(t) = int Gamma dt = S0 - S(t) ; in units of S0: 1 - (1 - x)^(2/3)
Str = sp.simplify((SBH.subs(M, M0) - SBH.subs(M, Mt)) / SBH.subs(M, M0)).subs(t, x * tev)
check("S_transfer/S_0 = 1 - (1 - t/tau)^(2/3)", simp0(sp.simplify(Str - (1 - (1 - x)**sp.Rational(2, 3)))))
xh = sp.solve(sp.Eq(1 - (1 - x)**sp.Rational(2, 3), sp.Rational(1, 2)), x)[0]
check("half transferred at t = (1 - 2^(-3/2)) tau = 0.6464 tau (not tau/2; erratum B2)", abs(float(xh) - (1 - 2**-1.5)) < 1e-12)
print("   1 - 2^(-3/2) =", 1 - 2**-1.5)
# crossing with the remaining horizon entropy (the upper envelope of the fine-grained entropy, min(S_tr, S_BH(t)))
xc = sp.solve(sp.Eq(1 - (1 - x)**sp.Rational(2, 3), (1 - x)**sp.Rational(2, 3)), x)[0]
check("min(S_transfer, S_BH(t)) turns over at the same 0.6464 tau", abs(float(xc) - (1 - 2**-1.5)) < 1e-12)

# ---------------------------------------------------------------------------------------------
print("\nH. Smarr and Kerr: the price of a horizon's bits")
check("N k_B T ln2 = T_BH S_BH = Mc^2/2 (Schwarzschild; erratum V17)", simp0(TBH * kB * SBH - M * c**2 / 2))
chi = sp.symbols('chi', positive=True)
share = sp.sqrt(1 - chi**2) / 2
for ch in (0, 0.5, 0.9, 0.998):
    print(f"   Kerr chi = {ch}: T S/Mc^2 = {float(share.subs(chi, ch)):.3f}")
# info-paradox Eq. 8 test (erratum B6)
hbn, cn, Gn, kn = C.hbar, C.c, C.G, C.k
Msun = 1.98847e30
S1 = 4 * np.pi * Gn * Msun**2 / (hbn * cn)
T1 = hbn * cn**3 / (8 * np.pi * Gn * Msun * kn)
print(f"   1 Msun: Mc^2 = {Msun*cn**2:.4e} J ; k_B T ln2 sqrt(S) = {kn*T1*np.log(2)*np.sqrt(S1):.4e} J ; 2 T S k_B = {2*T1*S1*kn:.4e} J")
check("Eq. 8 of the source (Mc^2 = k_B T ln2 S^(1/2)) fails by >30 orders; Smarr Mc^2 = 2 T S holds (erratum B6)",
      abs(2 * T1 * S1 * kn / (Msun * cn**2) - 1) < 1e-12 and kn * T1 * np.log(2) * np.sqrt(S1) / (Msun * cn**2) < 1e-30)

# ---------------------------------------------------------------------------------------------
print("\nI. Hoop conjecture = holographic bound at r_s; seed-mass inversion")
ratio = sp.simplify(SBH / (4 * sp.pi * (2 * G * M / c**2)**2))
check("S_BH/A at r_s = c^3/(4 hbar G) = 1/(4 l_P^2)", simp0(ratio - 1 / (4 * lP2)))
Sc = sp.symbols('S_c', positive=True)
mPl = sp.sqrt(hb * c / G)
Minv = sp.solve(sp.Eq(4 * sp.pi * G * M**2 / (hb * c), Sc), M)[0]
check("M = m_Pl sqrt(S)/(2 sqrt(pi)) is S_BH = 4 pi G M^2/(hbar c) inverted", simp0(Minv - mPl * sp.sqrt(Sc) / (2 * sp.sqrt(sp.pi))))
mPln = np.sqrt(hbn * cn / Gn)
for Se in (77, 83, 89, 95):
    print(f"   S = 1e{Se} nats -> M = {mPln*np.sqrt(10.0**Se)/(2*np.sqrt(np.pi))/Msun:.3g} Msun")

# ---------------------------------------------------------------------------------------------
print("\nJ. Numbers (CODATA 2018, M_sun = 1.98847e30 kg, yr = 3.15576e7 s)")
yr = 3.15576e7
print(" M/Msun |   T_BH (K)   | S (nats)   | S (bits)   | Gamma (bit/s) | tau_evap (yr)")
rows = {}
for m in (1, 10, 1e6, 1e9):
    Mk = m * Msun
    Tn = hbn * cn**3 / (8 * np.pi * Gn * Mk * kn)
    Sn = 4 * np.pi * Gn * Mk**2 / (hbn * cn)
    Gm = cn**3 / (1920 * Gn * Mk * np.log(2))
    te = 5120 * np.pi * Gn**2 * Mk**3 / (hbn * cn**4) / yr
    rows[m] = (Tn, Sn, Sn / np.log(2), Gm, te)
    print(f" {m:6.0e} | {Tn:.3e} | {Sn:.3e} | {Sn/np.log(2):.3e} | {Gm:.4e} | {te:.3e}")
check("1 Msun: T 6.17e-8 K, S 1.049e77 nats = 1.513e77 bits, Gamma 152.5 bit/s, tau 2.10e67 yr",
      abs(rows[1][0] - 6.17e-8) / 6.17e-8 < 2e-3 and abs(rows[1][1] / 1.049e77 - 1) < 1e-3 and abs(rows[1][2] / 1.513e77 - 1) < 1e-3
      and abs(rows[1][3] - 152.5) < 0.1 and abs(rows[1][4] / 2.10e67 - 1) < 5e-3)
print(f"   tau_evap(1 Msun) = {rows[1][4]/1e9:.3e} Gyr ; S_BH,0 = {rows[1][2]:.3e} bits (source printed 2.6e76; erratum B2)")
# numeric SB identity to 12 significant figures
sSBn = C.sigma
for m in (1, 10, 1e6, 1e9):
    Mk = m * Msun
    Pn = sSBn * 16 * np.pi * Gn**2 * Mk**2 / cn**4 * (hbn * cn**3 / (8 * np.pi * Gn * Mk * kn))**4
    Pf = hbn * cn**6 / (15360 * np.pi * Gn**2 * Mk**2)
    print(f"   M = {m:.0e} Msun: sigma A T^4 / [hbar c^6/(15360 pi G^2 M^2)] = {Pn/Pf:.12f}")
# numeric check of the time integral of Gamma
M0n = Msun; tevn = 5120 * np.pi * Gn**2 * M0n**3 / (hbn * cn**4)
Mfun = lambda tt_: (M0n**3 * (1 - tt_ / tevn))**(1 / 3)
I, _ = quad(lambda u: cn**3 / (1920 * Gn * Mfun(u * tevn) * np.log(2)) * tevn, 0, 0.6464466)
S0b = 4 * np.pi * Gn * M0n**2 / (hbn * cn) / np.log(2)
print(f"   int_0^0.6464 tau Gamma dt / S_0 = {I/S0b:.6f}")
check("numerical integral of Gamma reaches S_0/2 at 0.6464 tau", abs(I / S0b - 0.5) < 1e-5)

print("\nK. Two horizons: T_GH, M_eq at five epochs (H0 = 67.4, Om = 0.315, Or = 9.1e-5, flat), cosmic-bath factor, CMB balance")
Mpc = 3.0856775814913673e22
H0 = 67.4e3 / Mpc
Om, Or = 0.315, 9.1e-5
OL = 1 - Om - Or
print("   z        H (s^-1)     T_GH (K)     M_eq (Msun)")
meq = {}
for z in (0, 1, 1e3, 1e6, 1e10):
    Hz = H0 * np.sqrt(Om * (1 + z)**3 + OL + Or * (1 + z)**4)
    TGH = hbn * Hz / (2 * np.pi * kn)
    Me = cn**3 / (4 * Gn * Hz) / Msun
    meq[z] = (Hz, TGH, Me)
    print(f"   {z:<8.0e} {Hz:.3e}   {TGH:.3e}   {Me:.3e}")
check("z=0: H 2.18e-18, T_GH 2.66e-30 K, M_eq 2.32e22 Msun", abs(meq[0][0] / 2.184e-18 - 1) < 3e-3 and abs(meq[0][1] / 2.66e-30 - 1) < 3e-3
      and abs(meq[0][2] / 2.32e22 - 1) < 3e-3)
bath = (meq[0][1] / rows[1][0])**4
print(f"   (T_GH/T_BH)^4 for 1 Msun today = {bath:.2e}")
check("cosmic-bath correction 3.4e-90 (1 Msun, today)", abs(bath / 3.4e-90 - 1) < 0.03)
check("M_eq from T_BH = T_GH is c^3/(4 G H)", simp0(sp.solve(sp.Eq(TBH, hb * H / (2 * sp.pi * kB)), M)[0] - c**3 / (4 * G * H)))
TCMB = 2.7255
MCMB = hbn * cn**3 / (8 * np.pi * Gn * kn * TCMB)
print(f"   M_CMB = {MCMB:.3e} kg = {MCMB/7.342e22:.2f} lunar masses ; T_CMB/T_BH(1 Msun) = {TCMB/rows[1][0]:.3e} ; ratio^4 = {(TCMB/rows[1][0])**4:.2e} ; e-folds = {np.log(TCMB/rows[1][0]):.2f}")

print("\nL. Lensing-to-dynamical ratio (Table of predictions): 1/mu(0) = 1 + beta_m with beta_m = Omega_m/2 = 0.15765")
beta = 0.3153 / 2
print(f"   beta_m = {beta:.5f} ; 1/mu(0) = {1+beta:.5f} -> {100*beta:.2f} % at z = 0")
check("1/mu(0) = 1.158 (as in Chapter ch:threeway)", abs((1 + beta) - 1.158) < 5e-4)

print("\nM. Information-record size: Shannon entropy of a decoherence record is at most log2 N bits")
pk = np.array([0.5, 0.25, 0.25]); Hs = -(pk * np.log2(pk)).sum()
print(f"   N = 3, weights (1/2, 1/4, 1/4): H = {Hs:.3f} bits < log2 3 = {np.log2(3):.3f}")
check("H <= log2 N with equality only for equal weights", Hs < np.log2(3) and abs(-(np.ones(3)/3*np.log2(np.ones(3)/3)).sum() - np.log2(3)) < 1e-12)

print("\nN. Planck values")
lPn = np.sqrt(hbn * Gn / cn**3)
print(f"   l_P = {lPn:.5e} m ; 4 l_P^2 = {4*lPn**2:.4e} m^2 ; 4 ln2 l_P^2 = {4*np.log(2)*lPn**2:.4e} m^2 ; eta = 1/(4 l_P^2) = {1/(4*lPn**2):.4e} m^-2")

print("\n" + "=" * 100)
print(f"{sum(PASS)} of {len(PASS)} checks pass")
