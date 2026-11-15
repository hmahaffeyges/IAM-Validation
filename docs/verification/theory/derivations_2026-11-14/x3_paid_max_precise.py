#!/usr/bin/env python3
"""EXPLORATION (register X3; GitHub explore.yml), 2026-11-14 night. x3_paid_step4_check.py found max s(w) = 0.333375 on a float grid, just above the
series limit 1/3 (s = 1/3 - w/36 + w^2/810 - ...). Re-evaluate s(w) = ln[(w/2)/(1 - (1 - e^-w)/w)]/w at 50 digits (mpmath) on the same grid, and
report the true maximum and where it sits. Writes xout/x3_paid_max_summary.txt."""
import subprocess, sys
try:
    __import__("mpmath")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mpmath"], check=True)
import os, mpmath as mp
mp.mp.dps = 50
OUT = os.path.abspath("xout"); os.makedirs(OUT, exist_ok=True)
s = lambda w: mp.log((w / 2) / (1 - (1 - mp.e**(-w)) / w)) / w
grid = [mp.mpf(10) ** (mp.mpf(k) / 100) for k in range(-400, 301)]
vals = [(s(w), w) for w in grid]
m, wm = max(vals)
L = [f"max s over w = 1e-4 .. 1e3 at 50 digits: {mp.nstr(m, 20)} at w = {mp.nstr(wm, 6)}; 1/3 - max = {mp.nstr(mp.mpf(1)/3 - m, 6)}",
     f"s(1e-4) = {mp.nstr(s(mp.mpf('1e-4')), 20)}; series 1/3 - w/36 + w^2/810 at 1e-4: {mp.nstr(mp.mpf(1)/3 - mp.mpf('1e-4')/36 + mp.mpf('1e-8')/810, 20)}",
     f"monotone decreasing over the grid: {all(vals[i][0] > vals[i + 1][0] for i in range(len(vals) - 1))}"]
open(os.path.join(OUT, "x3_paid_max_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
