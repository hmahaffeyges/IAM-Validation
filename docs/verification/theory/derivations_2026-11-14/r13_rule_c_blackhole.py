#!/usr/bin/env python3
"""EXPLORATION (register R13; GitHub explore.yml), 2026-11-14 night. The established-physics check of rule C ('price at the Gibbons-Hawking
temperature of the horizon the writer's own density would make'): apply it to a Schwarzschild black hole, whose temperature is known (Hawking).
sympy: own mean density inside R_s, its own-density horizon rate H_X = sqrt(8 pi G rho/3), T_C = hbar H_X/(2 pi k), against T_Hawking =
hbar c^3/(8 pi G M k); and T S/(M c^2) for both (S = k pi R_s^2 c^3/(hbar G)). Also matter's own Smarr form: matter's mass in the Hubble sphere as a
black hole, (T_Hawking S)/(M_H c^2) = Omega_m/2 (the canon beta_m). Writes xout/r13_rule_c_blackhole_summary.txt."""
import subprocess, sys
try:
    __import__("sympy")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import os, sympy as sp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
G, M, c, hb, k, Om, H = sp.symbols("G M c hbar k Omega_m H", positive=True)
Rs = 2 * G * M / c**2; rho = M / (sp.Rational(4, 3) * sp.pi * Rs**3)
HX = sp.sqrt(8 * sp.pi * G * rho / 3)
TC = hb * HX / (2 * sp.pi * k); TH = hb * c**3 / (8 * sp.pi * G * M * k)
S = k * sp.pi * Rs**2 * c**3 / (hb * G)
L = [f"own-density horizon rate H_X = {sp.simplify(HX)}  (c / R_s = {sp.simplify(c / Rs)})",
     f"T_C / T_Hawking = {sp.simplify(TC / TH)}",
     f"T_C S / (M c^2) = {sp.simplify(TC * S / (M * c**2))};  T_Hawking S / (M c^2) = {sp.simplify(TH * S / (M * c**2))} (Smarr)"]
RH = c / H; MH = RH * c**2 / (2 * G); Mm = Om * MH
Rm = 2 * G * Mm / c**2; Sm = k * sp.pi * Rm**2 * c**3 / (hb * G); THm = hb * c**3 / (8 * sp.pi * G * Mm * k)
L.append(f"matter's mass in the Hubble sphere as its own black hole: T_Hawking S / (M_H c^2) = {sp.simplify(THm * Sm / (MH * c**2))}")
L.append("Reading guide (exploration): if T_C differs from Hawking's temperature, rule C as stated fails on the one horizon whose temperature is established.")
open(os.path.join(OUT, "r13_rule_c_blackhole_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
