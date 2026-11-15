#!/usr/bin/env python3
"""EXPLORATION (register X19; GitHub explore.yml), 2026-11-14 night. The book (p4_14 sec:qr_particles) prices a particle's Compton-sphere record (area
count pi (m_P/m)^2) at the COSMIC horizon's temperature and finds 5.6e5 m_e c^2. Under the nearest-surface reading (R13), the record sits on the
particle's own surface at that surface's temperature T = hbar c/(2 pi k lambdabar_C) (Unruh-Gibbons-Hawking form at radius lambdabar_C = hbar/mc).
sympy: T S / (m c^2) for (a) the area count and (b) the Bekenstein count S = 2 pi k E R/(hbar c) with E = m c^2, R = lambdabar_C (= 2 pi, the
value the book cites). Electron numbers for (a). Writes xout/x19_own_surface_summary.txt."""
import subprocess, sys
try:
    __import__("sympy")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import os, sympy as sp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
m, c, hb, k, G = sp.symbols("m c hbar k G", positive=True)
lam = hb / (m * c); T = hb * c / (2 * sp.pi * k * lam)
lP2 = hb * G / c**3; mP2 = hb * c / G
S_area = k * 4 * sp.pi * lam**2 / (4 * lP2); S_bek = 2 * sp.pi * k * (m * c**2) * lam / (hb * c)
L = [f"own-surface temperature k T = {sp.simplify(k * T)} (= m c^2 / 2 pi)",
     f"(a) area count S/k = {sp.simplify(S_area / k)} (= pi m_P^2/m^2: {sp.simplify(S_area / k - sp.pi * mP2 / m**2) == 0}); T S / m c^2 = {sp.simplify(T * S_area / (m * c**2))}",
     f"(b) Bekenstein count S/k = {sp.simplify(S_bek / k)}; T S / m c^2 = {sp.simplify(T * S_bek / (m * c**2))}"]
me, mPkg = 9.1093837e-31, 2.176434e-8
L.append(f"electron, (a): T S / m c^2 = (m_P/m)^2/2 = {(mPkg / me)**2 / 2:.3e}")
L.append("Reading guide (exploration): which count balances the rest energy when priced at the particle's own surface.")
open(os.path.join(OUT, "x19_own_surface_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
