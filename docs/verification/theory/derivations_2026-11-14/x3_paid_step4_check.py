#!/usr/bin/env python3
"""EXPLORATION check (sympy; GitHub explore.yml), register X3, 2026-11-14 evening. The book's four-step identity (p1_04) with Step 4 replaced by
the paid hold's steady state (p6_08 eq:middle_passage). Checks: (a) the held share per unit PAID, s(w) = ln(odds(w))/w, has weak-drive series
1/3 - w/36 + ...; (b) s is below 1/3 for all w > 0 (numerically on a grid); (c) per coin with two coins per restore, h(M) = 2 s(2M): 2/3 at M -> 0,
1/2 near M = 1.82, 0.1464 at M = 20.94; (d) the settled power-law share (k+2)/2 is 1/2 at k = -1 (Euler's theorem; Step 4 of p1_04).
Writes xout/x3_paid_step4_summary.txt."""
import subprocess, sys
try:
    __import__("sympy")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sympy"], check=True)
import os, sympy as sp
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
w, M, k = sp.symbols("w M k", positive=True)
odds = (w / 2) / (1 - (1 - sp.exp(-w)) / w); s = sp.log(odds) / w
ser = sp.series(s, w, 0, 3).removeO()
L = [f"(a) weak-drive series of s(w): {sp.simplify(ser)}"]
grid = [10 ** (e / 20) for e in range(-80, 61)]
vals = [float(s.subs(w, g)) for g in grid]
L.append(f"(b) max of s over w = 1e-4 .. 1e3: {max(vals):.6f} (below 1/3: {max(vals) < 1 / 3})")
h = lambda m: float(2 * s.subs(w, 2 * m))
root = sp.nsolve(2 * s.subs(w, 2 * M) - sp.Rational(1, 2), M, 1.8)
L.append(f"(c) per coin h(M): M=0.001 {h(0.001):.4f}; h = 1/2 at M = {float(root):.3f}; M = 20.94 {h(20.94):.4f}; per unit paid at M = 1.82: {h(1.82) / 2:.4f}")
L.append(f"(d) settled share (k+2)/2 at k = -1: {sp.Rational(-1 + 2, 2)}")
open(os.path.join(OUT, "x3_paid_step4_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
