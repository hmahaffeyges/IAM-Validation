#!/usr/bin/env python3
"""EXPLORATION (register X1/R6; GitHub explore.yml), 2026-11-14 night. Each surface at its own established temperature T = hbar kappa/(2 pi k c):
T S / E and kappa R / c^2 for (i) a de Sitter / Hubble horizon (R = c/H, E = M_H c^2 with M_H = c^3/(2 G H), S = pi R^2/l_P^2), (ii) a Schwarzschild
black hole (R_s = 2GM/c^2, kappa = c^4/(4GM)), (iii) a particle on its Compton sphere with the Bekenstein count (R = hbar/mc, kappa = c^2/R).
sympy. Writes xout/x1_surface_gravity_summary.txt."""
import subprocess, sys
try:
    __import__("sympy")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import os, sympy as sp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
G, c, hb, k, H, M, m = sp.symbols("G c hbar k H M m", positive=True)
lP2 = hb * G / c**3
rows = []
R = c / H; kap = c * H; E = c**3 / (2 * G * H) * c**2; S = k * sp.pi * R**2 / lP2
rows.append(("de Sitter / Hubble horizon", kap, R, E, S))
R = 2 * G * M / c**2; kap = c**4 / (4 * G * M); E = M * c**2; S = k * sp.pi * R**2 / lP2
rows.append(("Schwarzschild black hole", kap, R, E, S))
R = hb / (m * c); kap = c**2 / R; E = m * c**2; S = 2 * sp.pi * k * E * R / (hb * c)
rows.append(("particle, Compton sphere, Bekenstein count", kap, R, E, S))
L = []
for name, kap, R, E, S in rows:
    T = hb * kap / (2 * sp.pi * k * c)
    L.append(f"{name:44s} kappa R / c^2 = {sp.simplify(kap * R / c**2)};  T S / E = {sp.simplify(T * S / E)}")
L.append("Reading guide (exploration): the ratio T S / E tracks kappa R / c^2 (1 for de Sitter-type surfaces, 1/2 for a Schwarzschild horizon).")
open(os.path.join(OUT, "x1_surface_gravity_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
