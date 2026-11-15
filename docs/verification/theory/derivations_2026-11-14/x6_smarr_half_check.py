#!/usr/bin/env python3
"""EXPLORATION check (sympy; runs on GitHub explore.yml), register X6, 2026-11-14 evening. Verifies the hand algebra of the X6 lead:
(1) flat FRW: G M_H / (R_H c^2) = 1/2, so R_s(M_H) = R_H;  (2) T_Schwarzschild(R_H) = T_H / 2 with T_H = hbar H / (2 pi k);
(3) T_S * S_area(R_H) = M_H c^2 / 2 (Smarr);  T_H * S_area = M_H c^2 (the canon's '1');  (4) matter's share Omega_m of (3) over M_H c^2 = Omega_m / 2.
Also R6's held/paid ratio for a T ~ 1/R area-law surface: T S / Int T dS = 1/2. Writes xout/x6_smarr_half_summary.txt."""
import subprocess, sys
try:
    __import__("sympy")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)

import os, sympy as sp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
G, c, hbar, k, H, Om, R = sp.symbols("G c hbar k H Omega_m R", positive=True)
rho = 3 * H**2 / (8 * sp.pi * G); RH = c / H; MH = sp.Rational(4, 3) * sp.pi * RH**3 * rho
chk = {}
chk["(1) G M_H/(R_H c^2) = 1/2"] = sp.simplify(G * MH / (RH * c**2) - sp.Rational(1, 2)) == 0
chk["(1) R_s(M_H) = R_H"] = sp.simplify(2 * G * MH / c**2 - RH) == 0
TH = hbar * H / (2 * sp.pi * k); TS = hbar * c**3 / (8 * sp.pi * G * MH * k)
chk["(2) T_S(R_H) = T_H/2"] = sp.simplify(TS - TH / 2) == 0
S = k * c**3 * 4 * sp.pi * RH**2 / (4 * G * hbar)
chk["(3) T_S S = M_H c^2/2"] = sp.simplify(TS * S - MH * c**2 / 2) == 0
chk["(3) T_H S = M_H c^2"] = sp.simplify(TH * S - MH * c**2) == 0
chk["(4) Omega_m T_S S/(M_H c^2) = Omega_m/2"] = sp.simplify(Om * TS * S / (MH * c**2) - Om / 2) == 0
T_R = hbar * c / (4 * sp.pi * k * R); S_R = k * c**3 * sp.pi * R**2 / (G * hbar)
chk["R6: T S / Int_0^R T dS = 1/2"] = sp.simplify(T_R * S_R / sp.integrate(T_R * sp.diff(S_R, R), (R, 0, R)) - sp.Rational(1, 2)) == 0
L = ["X6 Smarr-half check (sympy):"] + [f"  {kk}: {'PASS' if v else 'FAIL'}" for kk, v in chk.items()]
L.append("ALL PASS" if all(chk.values()) else "SOME FAIL")
open(os.path.join(OUT, "x6_smarr_half_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
