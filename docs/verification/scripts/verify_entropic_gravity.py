#!/usr/bin/env python3
"""Recomputes every equation and number of the book chapter 'Entropy law or entropy source' (docs/book/part2/p2_03a_entropic_gravity.tex),
which carries the entropic-gravity note (March 2026, 10 pp) with errata EG1-EG8 applied.
Requires numpy, scipy, sympy. Run: python3 verify_entropic_gravity.py   (a few seconds)."""
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

Om = 0.3153                       # Planck 2018 (book canon)
OL = 1 - Om
b = Om / 2                        # beta_m = Omega_m / 2
Ea = lambda a: np.exp(1 - 1 / a)  # activation function

print("1. Horizon first law (Eq. eg:firstlaw)")
t = sp.symbols('t'); G = sp.symbols('G', positive=True)
H = sp.Function('H')(t); rho = sp.Function('rho')(t); P = sp.Function('P')(t)
r = 1 / H; V = sp.Rational(4, 3) * sp.pi * r**3; S = 4 * sp.pi * r**2 / (4 * G); W = (rho - P) / 2
cont = {sp.diff(rho, t): -3 * H * (rho + P)}
# (a) Cai-Kim: energy flux across the apparent horizon in dt, -dE = T_H dS with T_H = H/2pi
dE_flux = 4 * sp.pi * r**2 * (rho + P) * H * r          # A (rho+P) H r_A per unit time
sol = sp.solve(sp.Eq(dE_flux, (H / (2 * sp.pi)) * sp.diff(S, t)), sp.diff(H, t))
print("   Cai-Kim  -dE = T_H dS          ->  dH/dt =", sp.simplify(sol[0]))
# integrate with continuity: d/dt(H^2 - 8 pi G rho/3) = 0 when Hdot = -4 pi G (rho+P)
chk = sp.simplify((2 * H * sol[0] - sp.Rational(8, 3) * sp.pi * G * cont[sp.diff(rho, t)]))
print("   d/dt(H^2 - 8 pi G rho/3) =", chk, " -> H^2 = 8 pi G rho/3 + const (const = Lambda/3)")
# (b) unified first law with E = rho V inside the horizon, W = (rho-P)/2, T_h = kappa/2pi, kappa = -(1/r)(1 - rdot/(2 H r))
kap = -(1 / r) * (1 - sp.diff(r, t) / (2 * H * r))
for lab, T, sg in [("dE = (kappa/2pi) dS + W dV", kap / (2 * sp.pi), +1),
                   ("dE = (H/2pi) dS - W dV (printed form)", H / (2 * sp.pi), -1)]:
    ex = (sp.diff(rho * V, t) - (T * sp.diff(S, t) + sg * W * sp.diff(V, t))).subs(cont)
    s = [sp.simplify(x) for x in sp.solve(ex, sp.diff(H, t))]
    print(f"   {lab:40s} -> dH/dt = {s}")

print("2. Virial theorem (Eq. eg:virial): 2<T> = k <V> for V ~ r^k")
k, Vv = sp.symbols('k V')
print("   k = -1:  <T> =", sp.Rational(-1, 2) * Vv, " = |<V>|/2 for bound V < 0")

print("3. Coupling and the numbers that follow from it")
mu0 = -b / (1 + b)
print(f"   beta_m = Omega_m/2 = {b:.5f};  mu(z=0) = 1/(1+beta_m) = {1/(1+b):.4f};  mu0 = {mu0:.4f};  1 - mu(0) = {100*(1-1/(1+b)):.2f} %")
Hph, sHph = 67.16, 0.47
Hm = Hph * np.sqrt(1 + b); sHm = sHph * np.sqrt(1 + b)
print(f"   H0 matter = 67.16 sqrt(1+beta_m) = {Hm:.2f} +- {sHm:.2f}")
print(f"   photon 67.16 vs Planck 67.36 +- 0.54: {(Hph-67.36)/0.54:+.2f} sigma;  matter {Hm:.2f} vs SH0ES 73.04 +- 1.04: {(Hm-73.04)/1.04:+.2f} sigma")

print("4. Activation function (Eq. eg:Ea)")
x = sp.symbols('a', positive=True); E = sp.exp(1 - 1 / x)
print("   E(1) =", E.subs(x, 1), "; lim a->oo E =", sp.limit(E, x, sp.oo), f"; E(1)/e = {float(sp.exp(-1)):.4f}")
dlna = sp.simplify(x * sp.diff(E, x))                        # dE/dln a = E/a
print("   dE/dln a =", dlna, "; stationary at a =", sp.solve(sp.diff(dlna, x), x))
Cc, kk = sp.symbols('C k', positive=True)
fam = sp.exp(Cc - kk / x)
print("   family exp(C - k/a): E(1) = 1 and E(oo) = e give", sp.solve([sp.Eq(fam.subs(x, 1), 1), sp.Eq(sp.limit(fam, x, sp.oo), sp.E)], [Cc, kk], dict=True))
print("   d^2E/da^2 = 0 at a =", sp.solve(sp.simplify(sp.diff(E, x, 2) / E), x))
Hh = lambda a: np.sqrt(Om * a**-3 + OL)                      # H/H0, LambdaCDM
res = minimize_scalar(lambda a: -Hh(a) * Ea(a) / a, bounds=(0.1, 1.0), method='bounded')
print(f"   dE/dt = H E / a peaks at a = {res.x:.4f}, z = {1/res.x-1:.2f}")
print("   exponent: S_info ~ a^(n-9/2) must be ~ -1/a  ->  n =", sp.solve(sp.Symbol('n') - sp.Rational(9, 2) + 1)[0])

print("5. Growth: the friction form of Eq. eg:growth against the two chain implementations (fixed parameters, same early amplitude)")
E2L = lambda a: Om * a**-3 + OL
E2I = lambda a: E2L(a) + b * Ea(a)
mu = lambda a: E2L(a) / E2I(a)
def dlnH(f, a, h=1e-5):
    return 0.5 * (f(a * np.exp(h)) - f(a * np.exp(-h))) / (2 * h) / f(a)
def growth(mode):
    def rhs(l, y):
        a = np.exp(l); eL = E2L(a); kL = dlnH(E2L, a); src = 1.5 * Om * a**-3 / eL
        if mode == "LCDM":   return [y[1], -(2 + kL) * y[1] + src * y[0]]
        if mode == "L1_muG": return [y[1], -(2 + kL) * y[1] + src * mu(a) * y[0]]                  # G_eff = mu G (Level 1, MGCAMB)
        if mode == "L2_fric":return [y[1], -(kL + 2 * np.sqrt(E2I(a) / eL)) * y[1] + src * y[0]]   # 2 H_IAM delta-dot (Level 2)
        if mode == "note":   return [y[1], -(2 + b * Ea(a) + kL) * y[1] + src * y[0]]              # (2 + beta E) H delta-dot (Eq. eg:growth)
    a0 = 1e-3
    return solve_ivp(rhs, (np.log(a0), 0), [a0, a0], dense_output=True, rtol=1e-10, atol=1e-14)
gL = growth("LCDM")
for m in ("L1_muG", "L2_fric", "note"):
    g = growth(m)
    rr = [g.sol(np.log(1 / (1 + z)))[0] / gL.sol(np.log(1 / (1 + z)))[0] - 1 for z in (0, 0.5, 1, 2)]
    print(f"   {m:8s}: dD/D at z = 0, 0.5, 1, 2: " + ", ".join(f"{100*v:+.2f} %" for v in rr))
print(f"   friction coefficient today, in units of H0: (2 + beta_m) = {2+b:.4f};  2 sqrt(1+beta_m) = {2*np.sqrt(1+b):.4f};  LambdaCDM 2")
for z in (0.5, 1, 2):
    a = 1 / (1 + z)
    print(f"   z = {z}: extra friction / H:  note beta E = {b*Ea(a):.4f};  Level 2  2(sqrt(E2I/E2L) - 1) = {2*(np.sqrt(E2I(a)/E2L(a))-1):.4f}")

print("6. Information criteria: Delta AIC = Delta chi2_min + 2 Delta k")
for lab, dchi, dk in [("IAM Level 2, Planck", 0.54, 0), ("IAM Level 1, range", 0.56, 0), ("IAM Level 1, range", 1.73, 0)]:
    print(f"   {lab:22s}: Delta chi2 = {dchi:+.2f}, extra parameters {dk} -> Delta AIC = {dchi+2*dk:+.2f}")
print("   one extra parameter (Delta or delta): Delta AIC = Delta chi2 + 2")

print("7. Table eg:numbers: prediction against data (difference / data error)")
rows = [("mu0 vs DESI FS+BAO+CMB+DESY3 0.04 +- 0.22", mu0, 0.04, 0.22),
        ("mu0 vs ACT+WMAP+SDSS+SN 0.02 +- 0.19", mu0, 0.02, 0.19),
        ("Sigma0 0 vs DESI FS+BAO+CMB+DESY3 0.044 +- 0.047", 0.0, 0.044, 0.047),
        ("Sigma0 0 vs ACT+WMAP+SDSS+SN 0.021 +- 0.068", 0.0, 0.021, 0.068),
        ("sigma8 0.7998 vs KiDS-Legacy+DESY3+DESI+Pantheon+ 0.802 (+0.022 lower side -0.018)", 0.7998, 0.802, 0.018),
        ("H0 matter vs SH0ES 73.04 +- 1.04", Hm, 73.04, 1.04)]
for lab, p, d, s in rows:
    print(f"   {lab:80s}: {(p-d)/s:+.2f} sigma")

print("8. Lensing-to-dynamical mass in the Level 1 form, 1/mu(z)")
print("   " + ", ".join(f"z={z}: {1/mu(1/(1+z)):.3f}" for z in (0, 0.2, 0.5, 1, 2)))

print("9. Euclid (template comparison of docs/verification/scripts/verify_euclid_template.py, quoted)")
print("   template-equivalent mu0 over 0<z<2 = -0.072: 0.31 sigma at sigma(mu0)=0.23, 1.8 sigma at 0.04, 7 sigma at 0.01")
