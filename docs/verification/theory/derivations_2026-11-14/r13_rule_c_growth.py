#!/usr/bin/env python3
"""EXPLORATION (register R13 rule C; GitHub explore.yml), 2026-11-14 night. Growth with the book's closed form (p2_06 eq:l2_mu):
delta'' + (2 + dlnH/dlna) delta' = 1.5 Omega_m(a) mu(a) delta, mu = H^2/(H^2 + beta_m E(a) H0^2), E = exp(1 - 1/a), LCDM background (Planck 2018
Omega_m 0.3153). For beta_m = 0 (LCDM), Omega_m/2 (canon) and Omega_m^(3/2) (rule C): the D ratio to LCDM at z = 0, 0.5, 1, 2 (same primordial
amplitude; the canon row should reproduce the book's Table tab:l2_growth closed-form column 0.9922, 0.9978, 0.9994), f = dlnD/dlna, and S8 for a
Planck LCDM S8 of 0.830 scaled by the D ratio. Writes xout/r13_rule_c_growth_summary.txt."""
import os, numpy as np
from scipy.integrate import solve_ivp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
Om = 0.3153; OL = 1 - Om
H2 = lambda a: Om * a**-3 + OL
def grow(bm):
    def rhs(lna, y):
        a = np.exp(lna); D, Dp = y
        dlnH = -1.5 * Om * a**-3 / H2(a); mu = H2(a) / (H2(a) + bm * np.exp(1 - 1 / a)); Oma = Om * a**-3 / H2(a)
        return [Dp, -(2 + dlnH) * Dp + 1.5 * Oma * mu * D]
    a0 = 1e-3
    return solve_ivp(rhs, [np.log(a0), 0], [a0, a0], dense_output=True, rtol=1e-10, atol=1e-14)
B = {"LCDM": 0.0, "canon Omega_m/2": Om / 2, "rule C Omega_m^(3/2)": Om ** 1.5}
S = {k: grow(b) for k, b in B.items()}
L = ["  beta_m                       D ratio z=0   0.5      1        2       f(z=0)   S8 (0.830 x ratio)"]
for k, b in B.items():
    r = [S[k].sol(np.log(1 / (1 + z)))[0] / S["LCDM"].sol(np.log(1 / (1 + z)))[0] for z in (0, 0.5, 1, 2)]
    D, Dp = S[k].sol(0.0)
    L.append(f"  {k:24s} {b:6.4f}  {r[0]:.4f}   {r[1]:.4f}   {r[2]:.4f}   {r[3]:.4f}   {Dp / D:.4f}   {0.830 * r[0]:.4f}")
L += ["Reading guide (exploration): the canon row checks against the book's table; the rule C row shows how much more growth suppression the larger",
      "beta_m brings, beside the weak-lensing S8 values the book discusses (2-3 sigma below Planck)."]
open(os.path.join(OUT, "r13_rule_c_growth_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
