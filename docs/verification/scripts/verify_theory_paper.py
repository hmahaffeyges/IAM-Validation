#!/usr/bin/env python3
"""Recomputes every number in THEORY_CHECK.md and EXPONENT_LINE_BY_LINE.md (IAM Theory paper, 14 Apr 2026).
Requires numpy, scipy, sympy. Run: python3 verify_theory_paper.py   (about 2 s)"""
import numpy as np, sympy as sp
from scipy.integrate import solve_ivp

Om, Orad = 0.315, 9.1e-5
OL = 1 - Om - Orad
b = Om / 2                                   # beta_m = Omega_m / 2
Ea = lambda a: np.exp(1 - 1 / a)             # activation function E(a) = exp(1 - 1/a)
E2L = lambda a: Om * a**-3 + Orad * a**-4 + OL
E2I = lambda a: E2L(a) + b * Ea(a)           # matter-sector expansion rate, Eq. (HIAM)

print("1. Activation function and coupling")
print(f"   beta_m = {b:.4f};  E(z=10) = {np.exp(-10):.2e};  E(z=2) = {np.exp(-2):.3f};  E(z=1) = {np.exp(-1):.3f}")
for z in (0, 0.5, 1.0):
    a = 1 / (1 + z); print(f"   mu(z={z}) = {E2L(a)/E2I(a):.3f}")
print(f"   mu0 = -beta/(1+beta) = {-b/(1+b):.3f}")

print("2. Exponent n (Eqs. 28-41): dS/dln a ∝ rho_m D^n f /(T_H A_H) with T_H ∝ H, A_H ∝ H^-2; exp(1-1/a) needs power -1")
def rhs(lna, y):
    a = np.exp(lna); h2 = E2L(a); dlnh = (-3*Om*a**-3 - 4*Orad*a**-4) / (2*h2)
    return [y[1], -(2 + dlnh) * y[1] + 1.5 * Om * a**-3 / h2 * y[0]]
lna = np.linspace(np.log(1e-3), 0, 4000); a = np.exp(lna)
s = solve_ivp(rhs, (lna[0], 0), [1e-3, 1e-3], t_eval=lna, rtol=1e-9)
D = s.y[0] / s.y[0][-1]; f = s.y[1] / s.y[0]; H = np.sqrt(E2L(a))
for n in (2.5, 3.0, 3.5):
    dS = Om * a**-3 * D**n * f / (H * H**-2)
    m1 = (a >= 0.01) & (a <= 0.1); m2 = (a >= 0.25) & (a <= 1)
    print(f"   n={n}: power in matter era {np.polyfit(lna[m1], np.log(dS[m1]), 1)[0]:+.2f}, late {np.polyfit(lna[m2], np.log(dS[m2]), 1)[0]:+.2f}")
print("   analytic: n - 9/2 = -1  ->  n =", sp.Rational(9, 2) - 1)

print("3. Equation of state (Eq. 68) and CPL")
x, O = sp.symbols("a Omega_m", positive=True)
w = (-(1 - O) + O/2 * sp.exp(1 - 1/x) * (-1 - 1/(3*x))) / ((1 - O) + O/2 * sp.exp(1 - 1/x))
dw = sp.simplify(sp.diff(w, x).subs(x, 1))
print(f"   w_info(1) = -4/3;  w_eff(1) = {float(w.subs({x:1, O:Om})):.3f};  dw/da|1 = {dw} = {float(dw.subs(O, Om)):+.4f};  w_a = -dw/da = {-float(dw.subs(O, Om)):+.3f}")
wf = sp.lambdify((x, O), w); A = np.linspace(0.5, 1, 200)
c = np.linalg.lstsq(np.vstack([np.ones_like(A), 1 - A]).T, wf(A, Om), rcond=None)[0]
print(f"   least-squares CPL over a 0.5-1: w0 = {c[0]:.3f}, wa = {c[1]:+.3f}")

print("4. Second order and bispectrum (Section 11.5): growth with 2H_IAM friction, unmodified Poisson source")
def grow(E2):
    def r(l, y):
        a = np.exp(l); e2 = E2(a); de = (E2(a*np.exp(1e-5)) - E2(a*np.exp(-1e-5))) / 2e-5
        k = 0.5 * de / e2; src = 1.5 * Om * a**-3 / e2
        return [y[1], -(2+k)*y[1] + src*y[0], y[3], -(2+k)*y[3] + src*y[2] - src*y[0]**2]
    a0 = 1e-3
    return solve_ivp(r, (np.log(a0), 0), [a0, a0, -3/7*a0**2, -6/7*a0**2], dense_output=True, rtol=1e-10, atol=1e-14)
gL, gI = grow(E2L), grow(E2I); r0 = gI.sol(0)[0] / gL.sol(0)[0]
for z in (0, 0.3, 0.5, 1.0):
    l = np.log(1/(1+z)); yL, yI = gL.sol(l), gI.sol(l)
    print(f"   z={z}: D1 ratio, same early amplitude {yI[0]/yL[0]:.3f} | same amplitude today {yI[0]/r0/yL[0]:.4f} -> B ∝ D^4 {(yI[0]/r0/yL[0])**4:.3f}")

print("5. Other numbers")
c_, G, Msun = 2.998e8, 6.674e-11, 1.989e30; H0 = 67.4e3 / 3.0857e22
print(f"   M_eq = c^3/(4 G H0) = {c_**3/(4*G*H0)/Msun:.2e} Msun;  H0_sirens = 67.4*sqrt(1+beta) = {67.4*np.sqrt(1+b):.2f};  67.161*sqrt(1+beta) = {67.161*np.sqrt(1+b):.2f}")
print(f"   Omega_m f_coll (f=0.62) = {Om*0.62:.3f} ({100*(0.62*2-1):.0f} % above Omega_m/2);  eta_vir = 1/(2 f_coll) = {1/(2*0.62):.3f}")
