#!/usr/bin/env python3
"""EXPLORATION (register X15; GitHub explore.yml), 2026-11-14 night. The writing threshold M_min = 4 Omega_m sigma^3/(G H) (the Missing Satellites
paper's Eq. 11; its prefactor is 'not derived in the text', MISSING_SATELLITES_CHECK.md). Which halo condition does 4 Omega_m encode? sympy, for an
isothermal halo (V_c^2 = 2 sigma^2, M = 2 sigma^2 R / G):
 (a) as an overdensity Delta relative to the critical density: M = 4 sigma^3/(G H sqrt(Delta)), so 4 Omega_m <=> Delta = 1/Omega_m^2;
 (b) as a crossing time t = R/sigma: M = 2 sigma^3 t/G, so 4 Omega_m <=> t H = 2 Omega_m (= 4 beta_m with the canon beta_m);
 (c) against matter's own gravitational clock t_m = (8 pi G rho_m/3)^(-1/2) = 1/(sqrt(Omega_m) H) (R13 rule C): t/t_m = 2 Omega_m^(3/2).
Writes xout/x15_prefactor_summary.txt."""
import subprocess, sys
try:
    __import__("sympy")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import os, sympy as sp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
G, H, s, M, R, D, Om, t = sp.symbols("G H sigma M R Delta Omega_m t", positive=True)
rho_c = 3 * H**2 / (8 * sp.pi * G)
Rsol = sp.solve(sp.Eq(M, 2 * s**2 * R / G), R)[0]
Msol = sp.solve(sp.Eq(M, sp.Rational(4, 3) * sp.pi * D * rho_c * Rsol**3), M)
Mdelta = [m for m in Msol if m.is_positive is not False][0]
L = [f"(a) M(Delta) = {sp.simplify(Mdelta)}",
     f"    4 Omega_m sigma^3/(G H) at Delta = 1/Omega_m^2: {sp.simplify(Mdelta.subs(D, 1 / Om**2) - 4 * Om * s**3 / (G * H)) == 0}; Delta = {sp.N(1 / 0.3153**2, 4)} (critical), {sp.N(1 / 0.3153**3, 4)} (mean)"]
Mt = 2 * s**2 * (s * t) / G
L.append(f"(b) M(t) = {Mt}; 4 Omega_m sigma^3/(G H) at t = 2 Omega_m/H: {sp.simplify(Mt.subs(t, 2 * Om / H) - 4 * Om * s**3 / (G * H)) == 0}; t H = {sp.N(2 * 0.3153, 4)}")
tm = 1 / (sp.sqrt(Om) * H)
L.append(f"(c) (2 Omega_m/H) / t_m = {sp.simplify((2 * Om / H) / tm)} = {sp.N(2 * 0.3153**1.5, 4)}")
L.append("Reading guide (exploration): the prefactor as a condition on the halo (overdensity, crossing time, or crossing time against matter's own clock).")
open(os.path.join(OUT, "x15_prefactor_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
