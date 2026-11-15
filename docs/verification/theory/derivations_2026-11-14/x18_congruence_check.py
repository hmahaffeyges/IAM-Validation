#!/usr/bin/env python3
"""EXPLORATION (register X18; runs on GitHub explore.yml), 2026-11-14 night. Reading: the record enters matter's CLOCK (the expansion theta_m = 3 H_m
of matter's own worldline congruence), not its source. Check (sympy): if matter's congruence feels the record as a component with the book's
rho_info = rho_c0 beta_m E(a), E = exp(1 - 1/a), w_info = -1 - 1/(3a) (p1_02 eq:law_winfo), then
 (1) the congruence's Friedmann form gives H_m^2 = H^2 + beta_m E H0^2 (the book's eq:Hm);
 (2) the timelike Raychaudhuri equation for that congruence, d theta/dt = -theta^2/3 - 4 pi G (rho + 3p), holds identically with the record term
     included (so (1) and (2) are one consistent clock);
 (3) the record term adds to the congruence's acceleration with the sign of a negative-pressure fluid (w < -1): it speeds matter's clock.
Background: flat LCDM for H (photon sector), H0 = 1 units. Writes xout/x18_congruence_summary.txt."""
import subprocess, sys
try:
    __import__("sympy")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import os, sympy as sp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
a, Om, bm, G = sp.symbols("a Omega_m beta_m G", positive=True)
OL = 1 - Om; rc0 = 3 / (8 * sp.pi * G)                 # H0 = 1
E = sp.exp(1 - 1 / a); w = -1 - 1 / (3 * a)
rho_m, rho_L, rho_i = rc0 * Om / a**3, rc0 * OL, rc0 * bm * E
p_m, p_L, p_i = 0, -rho_L, w * rho_i
H2 = Om / a**3 + OL; Hm2 = sp.Rational(8, 3) * sp.pi * G * (rho_m + rho_L + rho_i)
L = ["X18 congruence check (sympy):"]
L.append(f"  (0) record continuity d ln rho_info / d ln a = -3(1+w): {sp.simplify(a * sp.diff(sp.log(rho_i), a) + 3 * (1 + w)) == 0}")
L.append(f"  (1) H_m^2 - H^2 = beta_m E(a): {sp.simplify(Hm2 - H2 - bm * E) == 0}")
Hm = sp.sqrt(Hm2); dt = lambda f: Hm * a * sp.diff(f, a)          # d/dt along matter's clock: da/dt = a H_m
theta = 3 * Hm
ray = dt(theta) + theta**2 / 3 + 4 * sp.pi * G * ((rho_m + rho_L + rho_i) + 3 * (p_m + p_L + p_i))
L.append(f"  (2) Raychaudhuri residual for matter's congruence (should be 0): {sp.simplify(ray)}")
acc = sp.simplify(-sp.Rational(4, 3) * sp.pi * G * (rho_i + 3 * p_i))
L.append(f"  (3) record's contribution to (d^2 a/dt^2)/a: {acc}  (positive for a > 0: it speeds matter's clock)")
L.append("Reading guide (exploration): (1)-(2) say the book's H_m is exactly the self-consistent expansion of a congruence that feels the record as a fluid; "
         "the growth equation of that congruence then carries friction 2 H_m with the Poisson source unchanged (Sigma = 1). What stays open is WHY only "
         "matter's congruence feels it (the writing rule, R9/R12).")
open(os.path.join(OUT, "x18_congruence_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
