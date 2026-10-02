#!/usr/bin/env python3
"""Checks every derivation step of Chapter 'IAM's Law' (docs/book/part1/p1_02_iams_law.tex), block by block.
Block tags [V1]..[V20] are the tags in the 'check' column of the equation table in MANIFEST.md.
Symbolic steps with sympy; numbers with numpy/scipy (constants from scipy.constants; Planck 2018 parameters as printed in the chapter).
Canon values: beta_m = Omega_m/2 = 0.15765 (Omega_m = 0.3153); H0 photon sector 67.16, matter sector 72.26 km/s/Mpc.
Run: python3 verify_iams_law_derivations.py  (about 20 s). Every check prints OK or FAIL."""
import runpy, io, contextlib, pathlib
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import scipy.constants as C

hbar, c, G, kB = C.hbar, C.c, C.G, C.k
Msun, Mpc = 1.98847e30, 3.0856775814913673e22
lP = np.sqrt(hbar * G / c**3)
Om, Ob, OL = 0.3153, 0.0493, 1 - 0.3153
bm = Om / 2
H0g, H0m = 67.16, 72.26
NFAIL = 0
def OK(flag):
    global NFAIL
    NFAIL += (not flag)
    return "OK" if flag else "FAIL"
def Hs(h): return h * 1e3 / Mpc
print("scipy.constants G =", G, " hbar =", hbar, f" lP = {lP:.6e} m")

# ------------------------------------------------------------------ V1 the law and the epoch cost
print("\n[V1] Cost per bit (Eqs. law, epochcost)")
TH = hbar * Hs(H0g) / (2 * np.pi * kB)
print(f"  T_H(H0={H0g}) = {TH:.4e} K; k_B T_H ln2 = {kB*TH*np.log(2):.4e} J = hbar H ln2/2pi: "
      + OK(np.isclose(kB * TH * np.log(2), hbar * Hs(H0g) * np.log(2) / (2 * np.pi))))
print(f"  Landauer at 300 K: k_B T ln2 = {kB*300*np.log(2)/C.e:.5f} eV")
M_ = sp.symbols('M', positive=True)
TBH = sp.Symbol('hbar', positive=True) * sp.Symbol('c', positive=True)**3 / (8 * sp.pi * sp.Symbol('G', positive=True) * M_ * sp.Symbol('k_B', positive=True))
print("  T_BH = hbar c^3/(8 pi G M k_B) decreases with M:", OK(sp.diff(TBH, M_).is_negative))

# ------------------------------------------------------------------ V2 decoherence and the bit
print("\n[V2] Decoherence (Eqs. entangle, rhodiag) and I = log2 N")
rng = np.random.default_rng(1)
N = 4
ci = rng.normal(size=N) + 1j * rng.normal(size=N); ci /= np.linalg.norm(ci)
def rhoS(overlap):
    Gm = np.full((N, N), overlap, complex); np.fill_diagonal(Gm, 1)     # <E_j|E_i>
    return np.outer(ci, ci.conj()) * Gm.T
r1, r0 = rhoS(1.0), rhoS(0.0)
print("  overlap 1: pure state, Tr rho^2 = %.3f; overlap 0: diagonal, off-diagonal max = %.1e" %
      (np.trace(r1 @ r1).real, np.abs(r0 - np.diag(np.diag(r0))).max()),
      OK(np.isclose(np.trace(r1 @ r1).real, 1) and np.allclose(np.diag(r0), np.abs(ci)**2)))
print("  diagonal = |c_i|^2, trace 1:", OK(np.isclose(np.trace(r0).real, 1)))
print("  I = log2 N for N = 2, 4, 8:", [np.log2(n) for n in (2, 4, 8)])

# ------------------------------------------------------------------ V3 virial theorem, binding identity, sign of Euler form
print("\n[V3] Virial theorem from Euler's theorem; binding identity")
x, y, z, k = sp.symbols('x y z k', positive=True)
r = sp.sqrt(x**2 + y**2 + z**2)
for deg, V in ((-1, 1 / r), (1, r), (2, r**2)):
    e = sp.simplify(x * sp.diff(V, x) + y * sp.diff(V, y) + z * sp.diff(V, z) - deg * V)
    print(f"  r.gradV = {deg:+d} V for V = r^{deg}:", OK(e == 0))
K, V = sp.symbols('K V')
sol = sp.solve(sp.Eq(2 * K, -1 * V), V)[0]
print("  2<K> = k<V>, k = -1  =>  V =", sol, OK(sol == -2 * K))
print("  printed form 2<K> + n<V> = 0 with n = -1 gives V =", sp.solve(sp.Eq(2 * K - V, 0), V)[0], "(wrong sign; book uses 2<K> = k<V>)")
Q = -(K + sol)
print("  Q = -E_f = <K> = |<V>|/2:", OK(sp.simplify(Q - K) == 0 and sp.simplify(Q + sol / 2) == 0))
Ry = C.physical_constants['Rydberg constant times hc in eV'][0]
print(f"  hydrogen: <K> = {Ry:.3f} eV, <V> = {-2*Ry:.3f} eV, photon from rest = {Ry:.3f} eV")

# ------------------------------------------------------------------ V4 Euclidean Rindler period and Unruh temperature
print("\n[V4] Euclidean Rindler horizon: period 2pi/kappa")
rho, kap, tau, P = sp.symbols('rho kappa tau P', positive=True)
# ds_E^2 = kappa^2 rho^2 dtau^2 + drho^2 : circumference of the circle rho = const over one period P, divided by 2 pi rho
ratio = sp.simplify((kap * rho * P) / (2 * sp.pi * rho))
Psol = sp.solve(sp.Eq(ratio, 1), P)[0]
print("  no conical singularity at rho = 0 requires P =", Psol, OK(sp.simplify(Psol - 2 * sp.pi / kap) == 0))
hb, cc, kb, GG = sp.symbols('hbar c k_B G', positive=True)
# kappa in s^-1 here (Euclidean period in time); T = hbar/(k_B P)
print("  T = hbar/(k_B P) = hbar kappa/(2 pi k_B):", OK(sp.simplify(hb / (kb * Psol) - hb * kap / (2 * sp.pi * kb)) == 0))
g_acc = 9.81
print(f"  Unruh T for a = 9.81 m/s^2: hbar a/(2 pi c k_B) = {hbar*g_acc/(2*np.pi*c*kB):.3e} K")

# ------------------------------------------------------------------ V5 Jacobson: null-tensor lemma, Raychaudhuri, coefficient, weak field
print("\n[V5] Jacobson construction")
gmn = sp.diag(-1, 1, 1, 1)
Xs = sp.symbols('X0:10'); idx = [(i, j) for i in range(4) for j in range(i, 4)]
X = sp.zeros(4)
for s_, (i, j) in zip(Xs, idx): X[i, j] = X[j, i] = s_
nulls = []
for v in [(1, 1, 0, 0), (1, -1, 0, 0), (1, 0, 1, 0), (1, 0, -1, 0), (1, 0, 0, 1), (1, 0, 0, -1),
          (sp.sqrt(2), 1, 1, 0), (sp.sqrt(2), 1, -1, 0), (sp.sqrt(2), 0, 1, 1), (sp.sqrt(2), 0, 1, -1),
          (sp.sqrt(2), 1, 0, 1), (sp.sqrt(2), 1, 0, -1)]:
    kv = sp.Matrix(v); nulls.append(sp.expand((kv.T * X * kv)[0]))
solX = sp.solve(nulls, Xs, dict=True)[0]
Xsol = X.subs(solX)
free = sorted(Xsol.free_symbols, key=str)
print("  X_ab k^a k^b = 0 for all null k  =>  X = f g:", OK(len(free) == 1 and sp.simplify(Xsol - Xsol[1, 1] * gmn) == sp.zeros(4)))
lam_, Rkk = sp.symbols('lambda R_kk')
theta = -lam_ * Rkk                                   # first-order solution of dtheta/dlambda = -R_kk, theta(0) = 0
print("  Raychaudhuri, first order: dtheta/dlambda = -R_kk:", OK(sp.diff(theta, lam_) == -Rkk))
eta = sp.symbols('eta', positive=True)
Gsol = sp.solve(sp.Eq(hb * eta / (2 * sp.pi), cc**3 / (8 * sp.pi * GG)), GG)[0]
print("  hbar eta/2pi = c^3/8piG  =>  G = c^3/(4 hbar eta):", OK(sp.simplify(Gsol - cc**3 / (4 * hb * eta)) == 0))
eta_num = c**3 / (4 * hbar * G)
print(f"  eta = c^3/(4 hbar G) = {eta_num:.6e} m^-2 = 1/(4 lP^2) = {1/(4*lP**2):.6e}:", OK(np.isclose(eta_num, 1 / (4 * lP**2))))
print("  dimensions: [c^3/(hbar G)] = m^-2;  [c^4/(hbar G)] = m^-1 s^-1 (printed c^4 form is not an inverse area)")
# weak field: R_00 for g = diag(-(1+2Phi/c^2), (1-2Phi/c^2) delta_ij), static, first order in Phi
t_, X1, X2, X3, eps = sp.symbols('t x1 x2 x3 epsilon')
Phi = sp.Function('Phi')(X1, X2, X3)
co = [t_, X1, X2, X3]
g = sp.diag(-(1 + 2 * eps * Phi), 1 - 2 * eps * Phi, 1 - 2 * eps * Phi, 1 - 2 * eps * Phi)
gi = g.inv()
Gam = [[[sum(gi[a, d] * (sp.diff(g[d, b], co[cc_]) + sp.diff(g[d, cc_], co[b]) - sp.diff(g[b, cc_], co[d])) for d in range(4)) / 2
         for cc_ in range(4)] for b in range(4)] for a in range(4)]
def Ric(b, d):
    return sum(sp.diff(Gam[a][b][d], co[a]) - sp.diff(Gam[a][b][a], co[d])
               + sum(Gam[a][a][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][a] for e in range(4)) for a in range(4))
R00 = sp.series(sp.simplify(Ric(0, 0)), eps, 0, 2).removeO()
lap = sum(sp.diff(Phi, v, 2) for v in (X1, X2, X3))
print("  R_00 = eps Lap(Phi) to first order (units c = 1):", OK(sp.simplify(R00 - eps * lap) == 0))
rho_, Gn = sp.symbols('rho G_N', positive=True)
R00_dust = 8 * sp.pi * Gn * (rho_ - sp.Rational(1, 2) * rho_)       # R_ab = 8piG (T_ab - T g_ab/2), dust, T = -rho, g_00 = -1
print("  trace-reversed field equation for dust: R_00 = 4 pi G rho  => Lap Phi = 4 pi G rho (8pi = 2 x 4pi):",
      OK(sp.simplify(R00_dust - 4 * sp.pi * Gn * rho_) == 0))

# ------------------------------------------------------------------ V6 one unit of entropy per 4 lP^2
print("\n[V6] Minimum area per unit of entropy")
Mb = sp.symbols('M', positive=True)
kap_a = cc**4 / (4 * GG * Mb)                         # surface gravity (acceleration) of Schwarzschild
A = 16 * sp.pi * GG**2 * Mb**2 / cc**4
print("  first law d(Mc^2) = kappa c^2 dA/(8 pi G):", OK(sp.simplify(kap_a * cc**2 * sp.diff(A, Mb) / (8 * sp.pi * GG) - cc**2) == 0))
dE = hb * kap / (2 * sp.pi * cc)                       # k_B T_Unruh with kappa an acceleration
dA = sp.simplify(8 * sp.pi * GG * dE / (kap * cc**2))
print("  dA_min =", dA, "= 4 lP^2, independent of kappa:", OK(sp.simplify(dA - 4 * hb * GG / cc**3) == 0))
print(f"  eta * dA_min = {eta_num*4*hbar*G/c**3:.6f} (one unit k_B, one nat); one bit: 4 ln2 lP^2 = {4*np.log(2):.4f} lP^2")

# ------------------------------------------------------------------ V7 Cai-Kim
print("\n[V7] Cai-Kim: Friedmann equations from the apparent horizon (hbar = c = k_B = 1)")
tt = sp.symbols('t'); Gs = sp.symbols('G', positive=True)
H = sp.Function('H')(tt); rho_t = sp.Function('rho')(tt); P_t = sp.Function('P')(tt)
Sgeo = sp.pi / (Gs * H**2); TH_ = H / (2 * sp.pi)
TdS = sp.simplify(TH_ * sp.diff(Sgeo, tt))
print("  T_H dS_geo/dt = -Hdot/(G H^2):", OK(sp.simplify(TdS + sp.diff(H, tt) / (Gs * H**2)) == 0))
rA = 1 / H; flux = 4 * sp.pi * rA**2 * (rho_t + P_t) * rA * H        # A_H (rho+P) rA H per unit time
Hdot = sp.solve(sp.Eq(flux, TdS), sp.diff(H, tt))[0]          # -dE (energy crossing) = T_H dS
print("  -dE = T_H dS_geo  =>  Hdot = -4 pi G (rho + P):", OK(sp.simplify(Hdot + 4 * sp.pi * Gs * (rho_t + P_t)) == 0))
expr = 2 * H * Hdot - sp.Rational(8, 3) * sp.pi * Gs * (-3 * H * (rho_t + P_t))
print("  with continuity, d/dt(H^2 - 8piG rho/3) = 0 (Lambda a constant of integration):", OK(sp.simplify(expr) == 0))
EMS = rA / (2 * Gs)
print("  Misner-Sharp energy r_A/2G = (4pi/3) r_A^3 rho on H^2 = 8piG rho/3:",
      OK(sp.simplify(EMS.subs(H, sp.sqrt(8 * sp.pi * Gs * rho_t / 3)) - sp.Rational(4, 3) * sp.pi * rA.subs(H, sp.sqrt(8 * sp.pi * Gs * rho_t / 3))**3 * rho_t) == 0))

# ------------------------------------------------------------------ V8 adding S_info
print("\n[V8] The record entropy in the first law")
Sdot = sp.symbols('Sdot_info'); Hs_, rho_s, P_s, a_ = sp.symbols('H rho P a', positive=True)
Hd2 = sp.solve(sp.Eq(4 * sp.pi * (rho_s + P_s) / Hs_**2, -sp.Symbol('Hd') / (Gs * Hs_**2) + Hs_ / (2 * sp.pi) * Sdot), sp.Symbol('Hd'))[0]
print("  Hdot = -4piG(rho+P) + (G/2pi) H^3 Sdot_info:", OK(sp.simplify(Hd2 - (-4 * sp.pi * Gs * (rho_s + P_s) + Gs / (2 * sp.pi) * Hs_**3 * Sdot)) == 0))
rx = sp.symbols('rho_x')
need = sp.solve(sp.Eq(Gs / (2 * sp.pi) * Hs_**3 * Sdot, -4 * sp.pi * Gs * (-rx / (3 * a_))), Sdot)[0]
print("  for rho_x + P_x = -rho_x/(3a) (w = -1 - 1/3a): Sdot_info =", need, OK(sp.simplify(need - 8 * sp.pi**2 * rx / (3 * a_ * Hs_**3)) == 0))
H0_, bms = sp.symbols('H_0 beta_m', positive=True)
dSdlna1 = sp.simplify((need / Hs_).subs({rx: 3 * H0_**2 * bms / (8 * sp.pi * Gs), Hs_: H0_, a_: 1}))
print("  dS_info/dln a at a = 1 =", dSdlna1, "= beta_m S_geo(1):", OK(sp.simplify(dSdlna1 - bms * sp.pi / (Gs * H0_**2)) == 0))
for h in (H0g, 67.36):
    Sg = np.pi * c**5 / (G * hbar * Hs(h)**2)        # in units of k_B
    print(f"  H0 = {h}: S_geo = {Sg:.3e} k_B; beta_m S_geo = {bm*Sg/np.log(2):.2e} bits per e-fold")

# ------------------------------------------------------------------ V9 the exponent n
print("\n[V9] Exponent n of the rate law")
n_ = sp.symbols('n')
pw = sp.powsimp(a_**-3 * a_**n_ / (a_**sp.Rational(-3, 2) * a_**3), force=True)
print("  dS/dln a ~ a^(n - 9/2):", OK(sp.simplify(pw / a_**(n_ - sp.Rational(9, 2))) == 1))
Sint = sp.integrate(a_**(n_ - sp.Rational(11, 2)), a_, conds='none')
print("  S ~ a^(n-9/2)/(n-9/2):", OK(sp.simplify(Sint - a_**(n_ - sp.Rational(9, 2)) / (n_ - sp.Rational(9, 2))) == 0))
nsol = sp.solve(sp.Eq(n_ - sp.Rational(9, 2), -1), n_)[0]
print("  n - 9/2 = -1  =>  n =", nsol, OK(nsol == sp.Rational(7, 2)))
print("  printed n = 5/2 gives S ~ a^", sp.Rational(5, 2) - sp.Rational(9, 2), "(not a^-1)")
print("  surface-density form: rho_m D^(n-1) f/(T_H a) ~ a^(n-7/2), constant only for n = 7/2:",
      OK(sp.solve(sp.Eq(-3 + (n_ - 1) + sp.Rational(3, 2) - 1, 0), n_)[0] == sp.Rational(7, 2)))
Or = 9.1e-5
def growth_lcdm(Om_, Or_, OL_):
    def H2(a): return Om_ / a**3 + Or_ / a**4 + OL_
    def rhs(lna, Y):
        a = np.exp(lna); h2 = H2(a); dlnh = -(1.5 * Om_ / a**3 + 2 * Or_ / a**4) / h2
        return [Y[1], -(2 + dlnh) * Y[1] + 1.5 * Om_ / a**3 / h2 * Y[0]]
    ai = 1e-6; yeq = ai * Om_ / Or_ if Or_ > 0 else None
    Y0 = [1 + 1.5 * yeq, 1.5 * yeq] if Or_ > 0 else [ai, ai]
    s = solve_ivp(rhs, [np.log(ai), np.log(3)], Y0, dense_output=True, rtol=1e-10, atol=1e-14)
    return s, H2
sol_g, H2g = growth_lcdm(Om, Or, 1 - Om - Or)
def slope(n, a1, a2, sol=sol_g, H2=H2g):
    aa = np.logspace(np.log10(a1), np.log10(a2), 60); D, Dp = sol.sol(np.log(aa)); f = Dp / D
    y = aa**-3 * D**n * f * np.sqrt(H2(aa))           # dS/dln a ~ rho_m D^n f /(T_H A_H) ~ a^-3 D^n f H
    return np.polyfit(np.log(aa), np.log(y), 1)[0]
for n in (2.5, 3.0, 3.5, 4.0):
    print(f"  full LCDM (Omega_r = 9.1e-5): n = {n}: power of dS/dln a, a 0.01-0.1: {slope(n,0.01,0.1):+.2f};  a 0.25-1: {slope(n,0.25,1):+.2f}")
p35 = slope(3.5, 0.01, 0.1)
print("  n = 7/2 matter era within 0.05 of -1:", OK(abs(p35 + 1) < 0.05))
nu_scal = sp.simplify(sp.diff(sp.log(sp.Symbol('delta_c') / (sp.Symbol('sigma') * sp.Symbol('D', positive=True))), sp.Symbol('D', positive=True)) * sp.Symbol('D', positive=True))
print("  Press-Schechter peak height nu = delta_c/(sigma D): dln nu/dln D =", nu_scal, OK(nu_scal == -1))

# ------------------------------------------------------------------ V10 activation function
print("\n[V10] Activation function")
ap = sp.symbols('aprime', positive=True)
lnE = sp.integrate(1 / ap**2, (ap, 1, a_))
print("  int_1^a da'/a'^2 = 1 - 1/a:", OK(sp.simplify(lnE - (1 - 1 / a_)) == 0))
print("  int_0^a diverges:", OK(sp.integrate(1 / ap**2, (ap, 0, a_)) == sp.oo))
E = sp.exp(1 - 1 / a_)
zz = sp.symbols('z', positive=True)
print("  E(z) = e^-z:", OK(sp.simplify(E.subs(a_, 1 / (1 + zz)) - sp.exp(-zz)) == 0),
      " E(1) = 1:", OK(E.subs(a_, 1) == 1), " E(a->0) = 0:", OK(sp.limit(E, a_, 0, '+') == 0), " E(inf) = e:", OK(sp.limit(E, a_, sp.oo) == sp.E))
print("  dE/da = E/a^2 > 0:", OK(sp.simplify(sp.diff(E, a_) - E / a_**2) == 0))
print("  E'' = 0 at a =", sp.solve(sp.simplify(sp.diff(E, a_, 2) / E), a_), " d(E/a)/da = 0 at a =", sp.solve(sp.simplify(sp.diff(E / a_, a_) / E), a_))
phi = 1 - 1 / sp.Function('a')(tt)
phid = sp.simplify(sp.diff(phi, tt) - sp.diff(sp.Function('a')(tt), tt) / sp.Function('a')(tt)**2)
print("  phi = 1 - 1/a obeys phidot = H/a:", OK(phid == 0))
print("  E at a = 1e-3:", float(sp.exp(1 - 1000)), "; 10/50/90 % of today at z =", [round(-np.log(p), 2) for p in (0.1, 0.5, 0.9)])

# ------------------------------------------------------------------ V11 variational route (minisuperspace)
print("\n[V11] Constrained action, minisuperspace")
Nf, af, phf, lf = [sp.Function(s)(tt) for s in ('N', 'a', 'phi', 'lambda')]
rm0, rL, r0 = sp.symbols('rho_m0 rho_L rho_0', positive=True)
L = (-sp.Rational(3, 8) / (sp.pi * Gs) * af * sp.diff(af, tt)**2 / Nf
     - Nf * af**3 * (rm0 / af**3 + rL + r0 * bms * sp.exp(phf))
     + lf * af**3 * (sp.diff(phf, tt) - sp.diff(af, tt) / af**2))
def EL(q):
    return sp.simplify(sp.diff(L, q) - sp.diff(sp.diff(L, sp.diff(q, tt)), tt))
eN = EL(Nf).subs(Nf, 1)
Hfun = sp.diff(af, tt) / af
fr = sp.simplify(sp.solve(eN, sp.diff(af, tt)**2)[0] / af**2)
print("  dN: H^2 = (8piG/3)(rho_m + rho_L + rho_0 beta_m e^phi):",
      OK(sp.simplify(fr - sp.Rational(8, 3) * sp.pi * Gs * (rm0 / af**3 + rL + r0 * bms * sp.exp(phf))) == 0))
print("  lambda absent from the dN equation:", OK(not eN.has(lf)))
el = EL(lf)
print("  dlambda: phidot = adot/a^2 = H/a:", OK(sp.simplify(el - af**3 * (sp.diff(phf, tt) - sp.diff(af, tt) / af**2)) == 0))
ephi = EL(phf).subs(Nf, 1).doit()
print("  dphi: d(lambda a^3)/dt = -a^3 rho_0 beta_m e^phi:", OK(sp.simplify(ephi - (-af**3 * r0 * bms * sp.exp(phf) - sp.diff(lf * af**3, tt))) == 0))
# da: compare with a barotropic fluid with rho(a) = rho_0 beta_m E(a) and P = -rho - rho/(3a)
ea = EL(af).subs(Nf, 1).doit()
lam_dot = sp.solve(ephi, sp.diff(lf, tt))[0]
ea2 = sp.simplify(ea.subs(sp.diff(lf, tt), lam_dot).subs(sp.diff(phf, tt), sp.diff(af, tt) / af**2))
rinfo = r0 * bms * sp.exp(phf)
Pinfo = -rinfo - rinfo / (3 * af)
# GR reference: L_ref = -(3/8piG) a adot^2 - a^3 (rho_m + rho_L + rho_info(a)) gives, by da, the same equation with 3 a^2 P_total on the right
Lr = -sp.Rational(3, 8) / (sp.pi * Gs) * af * sp.diff(af, tt)**2
grav = sp.simplify(sp.diff(Lr, af) - sp.diff(sp.diff(Lr, sp.diff(af, tt)), tt))
ref = grav + 3 * af**2 * (-rL + Pinfo)
print("  da (with dphi, dlambda): record sector acts with P_info = -rho_info(1 + 1/(3a)):", OK(sp.simplify(ea2 - ref) == 0))

# ------------------------------------------------------------------ V12 equation of state
print("\n[V12] w_info from continuity")
wexpr = -1 - sp.Rational(1, 3) * a_ * sp.diff(sp.log(E), a_)
print("  w = -1 - (1/3) dln rho/dln a = -1 - 1/(3a):", OK(sp.simplify(wexpr - (-1 - 1 / (3 * a_))) == 0),
      [round(float(wexpr.subs(a_, v)), 3) for v in (0.5, 1, 2)])

# ------------------------------------------------------------------ V13 coupling, mu, two Hubble rates
print("\n[V13] beta_m, mu(a), Hubble rates")
print(f"  beta_m = Omega_m/2 = {bm:.5f}:", OK(np.isclose(bm, 0.15765)))
H2 = lambda a: Om / a**3 + OL
Ea = lambda a: np.exp(1 - 1 / a)
mu = lambda a: H2(a) / (H2(a) + bm * Ea(a))
for zv in (0, 0.2, 0.3, 0.5, 0.7, 1, 2, 3):
    a = 1 / (1 + zv); print(f"  z = {zv}: mu = {mu(a):.4f}, H_m/H = {np.sqrt(1 + bm*Ea(a)/H2(a)):.4f}")
print(f"  mu(1) = 1/(1+beta_m) = {1/(1+bm):.4f}; mu0 = -beta/(1+beta) = {-bm/(1+bm):.5f}:", OK(np.isclose(mu(1), 1 / (1 + bm))))
print(f"  sqrt(1+beta_m) = {np.sqrt(1+bm):.5f}; 67.16 x = {H0g*np.sqrt(1+bm):.2f}:", OK(round(H0g * np.sqrt(1 + bm), 2) == 72.26))
print(f"  67.16 vs Planck 67.36 +- 0.54: {(H0g-67.36)/0.54:+.2f} sigma; 72.26 vs SH0ES 73.04 +- 1.04: {(H0m-73.04)/1.04:+.2f} sigma")
for lab, hv, sp_, sm_ in (("Palmese 2024", 75.46, 5.34, 5.39), ("Hotokezaka 2019", 68.9, 4.7, 4.6), ("Abbott 2017", 70.0, 12.0, 8.0)):
    d1 = (hv - H0g) / sm_; d2 = (hv - H0m) / (sm_ if hv > H0m else sp_)
    print(f"  siren {lab} {hv}: from 67.16 {d1:+.2f} sigma, from 72.26 {d2:+.2f} sigma")
# Weyl tensor of flat FRW in conformal time, g = a(tau)^2 eta: computed component by component
ta, q1, q2, q3 = sp.symbols('tau q1 q2 q3'); A_ = sp.Function('A')(ta); cq = [ta, q1, q2, q3]
gF = A_**2 * sp.diag(-1, 1, 1, 1); giF = gF.inv()
GF = [[[sp.simplify(sum(giF[a, d] * (sp.diff(gF[d, b], cq[e]) + sp.diff(gF[d, e], cq[b]) - sp.diff(gF[b, e], cq[d])) for d in range(4)) / 2)
        for e in range(4)] for b in range(4)] for a in range(4)]
def Riem(a, b, e, d):   # R^a_{b e d}
    return sp.diff(GF[a][b][d], cq[e]) - sp.diff(GF[a][b][e], cq[d]) + sum(GF[a][e][f] * GF[f][b][d] - GF[a][d][f] * GF[f][b][e] for f in range(4))
Rl = [[[[sp.simplify(sum(gF[a, f] * Riem(f, b, e, d) for f in range(4))) for d in range(4)] for e in range(4)] for b in range(4)] for a in range(4)]
RicF = sp.Matrix(4, 4, lambda b, d: sp.simplify(sum(Riem(a, b, a, d) for a in range(4))))
RsF = sp.simplify(sum(giF[b, d] * RicF[b, d] for b in range(4) for d in range(4)))
def Weyl(a, b, e, d):
    return sp.simplify(Rl[a][b][e][d] - (gF[a, e] * RicF[d, b] - gF[a, d] * RicF[e, b] - gF[b, e] * RicF[d, a] + gF[b, d] * RicF[e, a]) / 2
                       + RsF * (gF[a, e] * gF[d, b] - gF[a, d] * gF[e, b]) / 6)
wz = all(Weyl(a, b, e, d) == 0 for a in range(4) for b in range(4) for e in range(4) for d in range(4))
print("  Weyl tensor of flat FRW (g = a(tau)^2 eta), all 256 components zero:", OK(wz))

# ------------------------------------------------------------------ V14 growth, three implementations
print("\n[V14] Linear growth: three implementations of the record term")
def Dtoday(form):
    def rhs(lna, Y):
        a = np.exp(lna); h2 = H2(a); dlnh = -1.5 * Om / a**3 / h2; Om_a = Om / a**3 / h2
        hm2 = h2 + bm * Ea(a)
        if form == 'lcdm': return [Y[1], -(2 + dlnh) * Y[1] + 1.5 * Om_a * Y[0]]
        if form == 'i':    return [Y[1], -(2 + dlnh) * Y[1] + 1.5 * Om_a * mu(a) * Y[0]]
        if form == 'ii':   return [Y[1], -(dlnh + 2 * np.sqrt(hm2 / h2)) * Y[1] + 1.5 * Om_a * Y[0]]
        if form == 'iii':
            dhm2 = -3 * Om / a**3 + bm * Ea(a) / a                 # d(H_m^2/H0^2)/dln a
            return [Y[1], -(2 + 0.5 * dhm2 / hm2) * Y[1] + 1.5 * Om / a**3 / hm2 * Y[0]]
    s = solve_ivp(rhs, [np.log(1e-3), 0], [1e-3, 1e-3], rtol=1e-10, atol=1e-13)
    return s.y[0, -1]
D0 = Dtoday('lcdm')
gv = {f: 100 * (Dtoday(f) / D0 - 1) for f in ('i', 'ii', 'iii')}
print("  Delta D/D today: (i) G_eff = mu G %.2f %%, (ii) friction 2H_m %.2f %%, (iii) whole equation on H_m %.2f %%" % (gv['i'], gv['ii'], gv['iii']))
print("  matches -0.78 / -0.67 / -1.87:", OK(np.allclose([gv['i'], gv['ii'], gv['iii']], [-0.78, -0.67, -1.87], atol=0.01)))

# ------------------------------------------------------------------ V15 Euclid template (the book's only Euclid sensitivity statement)
print("\n[V15] Euclid sensitivity: rerun of verify_euclid_template.py")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runpy.run_path(str(pathlib.Path(__file__).resolve().parent / "verify_euclid_template.py"), run_name="__main__")
for line in buf.getvalue().splitlines(): print("  " + line)
print("  template script ran to completion (its own assert on f sigma8):", OK("all 0-2" in buf.getvalue()))

# ------------------------------------------------------------------ V16 black holes: saturation, equilibrium mass, rate
print("\n[V16] Holographic saturation, M_eq, Gamma")
SBH = 4 * sp.pi * GG * kb * Mb**2 / (hb * cc)
rs = 2 * GG * Mb / cc**2
print("  S_BH/(4 pi r_s^2) = k_B c^3/(4 hbar G) = k_B/(4 lP^2):", OK(sp.simplify(SBH / (4 * sp.pi * rs**2) - kb * cc**3 / (4 * hb * GG)) == 0))
Hh = sp.symbols('H', positive=True)
Meq = sp.solve(sp.Eq(hb * cc**3 / (8 * sp.pi * GG * Mb * kb), hb * Hh / (2 * sp.pi * kb)), Mb)[0]
print("  T_BH = T_H  =>  M_eq =", Meq, OK(sp.simplify(Meq - cc**3 / (4 * GG * Hh)) == 0))
for h in (H0g, 67.4): print(f"  M_eq(H0 = {h}) = {c**3/(4*G*Hs(h))/Msun:.3e} Msun")
print("  T_BH/T_H = M_eq/M:", OK(sp.simplify((hb * cc**3 / (8 * sp.pi * GG * Mb * kb)) / (hb * Hh / (2 * sp.pi * kb)) - Meq / Mb) == 0))
sig = sp.pi**2 * kb**4 / (60 * hb**3 * cc**2)
Tbh = hb * cc**3 / (8 * sp.pi * GG * Mb * kb)
Pw = sp.simplify(sig * A * Tbh**4)
print("  P = sigma A T^4 = hbar c^6/(15360 pi G^2 M^2):", OK(sp.simplify(Pw - hb * cc**6 / (15360 * sp.pi * GG**2 * Mb**2)) == 0))
print("  Gamma = P/(k_B T ln2) = c^3/(1920 G M ln2):", OK(sp.simplify(Pw / (kb * Tbh * sp.log(2)) - cc**3 / (1920 * GG * Mb * sp.log(2))) == 0))
print(f"  Gamma(1 Msun) = {c**3/(1920*G*Msun*np.log(2)):.1f} bits/s")

# ------------------------------------------------------------------ V17 vacuum term and the baryon relation
print("\n[V17] Vacuum term identity and Omega_b/Omega_m relation")
OLs, Obs, Oms, lPs, lHs = sp.symbols('Omega_L Omega_b Omega_m l_P l_H', positive=True)
rel = sp.solve(sp.Eq(2 / sp.pi * (lPs / lHs)**2 * sp.sqrt(OLs) * Obs / Oms, 3 * OLs / (8 * sp.pi) * (lPs / lHs)**2), Obs)[0]
print("  (2/pi)(lP/lH)^2 sqrt(OL) Ob/Om = (3 OL/8pi)(lP/lH)^2  =>  Ob/Om = (3/16) sqrt(OL):", OK(sp.simplify(rel / Oms - sp.Rational(3, 16) * sp.sqrt(OLs)) == 0))
print(f"  T_dS/T_H = sqrt(0.6846) = {np.sqrt(0.6846):.4f}")
lH = c / Hs(67.4)
print(f"  rho_L/rho_vac = (3 OL/8pi)(lP/lH)^2 at H0 = 67.4, OL = 0.6846: {3*0.6846/(8*np.pi)*(lP/lH)**2:.4e}")
meas = 3 * 0.6846 / (8 * np.pi) * (lP / lH)**2
base = 2 / np.pi * (lP / lH)**2 * Ob / Om; corr = base * np.sqrt(0.6846)
print(f"  baseline (2/pi)(lP/lH)^2 Ob/Om (Ob 0.0493, Om 0.3153) = {base:.4e} = {base/meas:.3f} x measured; with sqrt(OL): {corr:.4e} ({100*(corr/meas-1):+.2f} %)")
print("  matches 1.380e-123, 1.22x, 1.142e-123, +0.8 %:", OK(round(base * 1e123, 3) == 1.380 and round(corr * 1e123, 3) == 1.142 and abs(100 * (corr / meas - 1) - 0.8) < 0.05))
print(f"  eta = 273.9e-10 x 0.02232 = {273.9e-10*0.02232:.4e}; prior width ratio (0.040-0.010)/(0.025-0.020) = {(0.040-0.010)/(0.025-0.020):.1f}")

# ------------------------------------------------------------------ V18 Mahaffey number
print("\n[V18] Mahaffey number")
Mcell = 54000 / (C.R * 310.15)
print(f"  cell: 54 kJ/mol at 310.15 K: M = {Mcell:.2f}, /ln2 = {Mcell/np.log(2):.1f}:", OK(round(Mcell, 2) == 20.94))
print("  black hole: Mc^2/(k_B T_BH) = 8 pi G M^2/(hbar c) = 2 S_BH/k_B:",
      OK(sp.simplify(Mb * cc**2 / (kb * Tbh) - 8 * sp.pi * GG * Mb**2 / (hb * cc)) == 0 and sp.simplify(Mb * cc**2 / (kb * Tbh) - 2 * SBH / kb) == 0))
print(f"  1 Msun: {8*np.pi*G*Msun**2/(hbar*c):.2e}; 4.3e6 Msun: {8*np.pi*G*(4.3e6*Msun)**2/(hbar*c):.2e}")

# ------------------------------------------------------------------ V19 perturbations: Sigma = 1
print("\n[V19] Lensing potential")
Psi, PhiS = sp.symbols('Psi Phi')
print("  Psi = Phi  =>  (Psi + Phi)/2 = Psi (Sigma = 1):", OK(sp.simplify(((Psi + PhiS) / 2).subs(PhiS, Psi) - Psi) == 0))
print("  Omega_m^(m)/Omega_m = H^2/H_m^2 = mu:", OK(np.isclose((Om / H2(1)) / (Om / (H2(1) + bm)) , 1 + bm)))

# ------------------------------------------------------------------ V20 Planck units
print("\n[V20] eta in Planck units and 1/4 = 2pi/8pi")
print("  2 pi / 8 pi =", sp.Rational(2, 8), OK(sp.Rational(2, 8) == sp.Rational(1, 4)))
print(f"  lP = {lP:.5e} m; 4 lP^2 = {4*lP**2:.4e} m^2")

print("\nFAIL count:", NFAIL)
