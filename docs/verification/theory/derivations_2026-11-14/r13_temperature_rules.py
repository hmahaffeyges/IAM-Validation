#!/usr/bin/env python3
"""EXPLORATION (register R13; runs on GitHub explore.yml), 2026-11-14 night. Which temperature prices a written bit? Several candidate rules side by
side, each read for two sectors with the same rule: matter (X6 reads beta_m / Omega_m = 1/2) and Lambda (the book's chapter p2_12 uses the factor
T_dS / T_H = sqrt(Omega_L) on the Hubble capacity; the exponent closing it exactly is 0.521). Flat FRW, Planck 2018 densities. Factor F = (price
temperature x capacity) / (T_GH x S_Hubble), the canon's '1' = T_GH S_Hubble = M_H c^2 (eq:lam_smarr).
Rules: A the Hubble horizon at its Gibbons-Hawking temperature (the law's 'nearest surface at its local temperature', today's horizon);
B the Hubble sphere read as a saturated local surface (Schwarzschild radius = R_H): temperature T_GH/2, Hubble capacity;
C each sector at the Gibbons-Hawking temperature of the horizon that sector alone would make (H_X = sqrt(Omega_X) H), Hubble capacity;
D as C with that horizon's own capacity (S_H / Omega_X);  E = B within C (Schwarzschild temperature of the sector-alone horizon, Hubble capacity).
Writes xout/r13_temperature_rules_summary.txt."""
import subprocess, sys
try:
    __import__("sympy")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import os, sympy as sp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
Om, OL = sp.Rational(3153, 10000), sp.Rational(6846, 10000)
O = sp.symbols("Omega_X", positive=True)
rules = {"A  Hubble horizon, T_GH": sp.Integer(1), "B  Hubble sphere as Schwarzschild surface, T_GH/2": sp.Rational(1, 2),
         "C  sector-alone horizon temperature, Hubble capacity": sp.sqrt(O), "D  sector-alone horizon temperature and capacity": 1 / sp.sqrt(O),
         "E  Schwarzschild temperature of the sector-alone horizon, Hubble capacity": sp.sqrt(O) / 2}
tm, tl = sp.Rational(1, 2), sp.N(OL ** sp.Rational(521, 1000))
L = [f"targets: matter F = 1/2 (beta_m = Omega_m/2); Lambda F = sqrt(Omega_L) = {sp.N(sp.sqrt(OL)):.4f} (book) or Omega_L^0.521 = {tl:.4f} (exact closure)", ""]
L.append(f"  {'rule':72s} {'matter':>8s} {'/ 1/2':>7s} {'Lambda':>8s} {'/ sqrt':>7s}")
for k, f in rules.items():
    fm, fl = sp.N(f.subs(O, Om)), sp.N(f.subs(O, OL))
    L.append(f"  {k:72s} {fm:8.4f} {fm / tm:7.3f} {fl:8.4f} {fl / sp.sqrt(OL):7.3f}")
L += ["", "check: sector-alone H_X = sqrt(Omega_X) H for a flat sector of density Omega_X rho_c (sympy):",
      f"  {sp.simplify(sp.sqrt(8 * sp.pi * sp.Symbol('G') * O * 3 * sp.Symbol('H')**2 / (8 * sp.pi * sp.Symbol('G')) / 3) - sp.sqrt(O) * sp.Symbol('H')) == 0}",
      "Reading guide (exploration): which single rule comes closest for BOTH sectors, and by how much each misses; leads only."]
open(os.path.join(OUT, "r13_temperature_rules_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
