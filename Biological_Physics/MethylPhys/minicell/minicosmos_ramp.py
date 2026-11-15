#!/usr/bin/env python3
"""minicosmos_ramp.py (2026-11-15; register R17; exploration). Observability of the rate exponent n, and through it of c (the final size of the
record), via the shape of the f sigma8 deficit ('the ramp', book p2_09b: 3.1 % at z = 0.15, 1.9 % at 0.35, 0.9 % at 0.65, 0.4 % at 1.05, 0.1 % at
1.55; 'at 1 % per bin over fifteen bins of width 0.1 in 0.1 < z < 1.6 ... 5.0 sigma from LambdaCDM').
For n = 3, 3.5, 4 (closure B, feedback on; minicosmos_loop.py) the ramp D_IAM f_IAM / (D_LCDM f_LCDM) - 1 at the 15 bin centres, same primordial
amplitude; then a Fisher estimate of sigma(n) at 1 % per bin, and sigma(c) = |dc/dn| sigma(n). Self-check: the canon E(a) with D^3.5 must reproduce
the book's ramp numbers within the book's own reduced-growth caveat. Writes xout/minicosmos_ramp_summary.txt on GitHub."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import minicosmos_loop as ML

ZB = np.round(np.arange(0.15, 1.6, 0.1), 2)            # 15 bin centres
def ramp_from(D, Doff):
    lna = np.log(ML.A)
    f, foff = np.gradient(np.log(D), lna), np.gradient(np.log(Doff), lna)
    r = (D / D[0]) * f / ((Doff / Doff[0]) * foff) - 1
    return np.interp(1 / (1 + ZB), ML.A, r)

ML.set_background(om0=0.3153)
Doff, _ = ML.growth(np.zeros_like(ML.A), True)
Dcanon, _ = ML.growth(ML.BETA_M * np.exp(1 - 1 / ML.A), True)
canon = ramp_from(Dcanon, Doff)
out = {}
for n in (3.0, 3.25, 3.5, 3.75, 4.0):
    o = ML.loop(f"D^{n}", "B", feedback=True); r = ML.report(o)
    out[n] = (ramp_from(o["D"], Doff), r["c_direct"], r["beta"])
L = ["RAMP: f sigma8 deficit (IAM / LambdaCDM - 1), %, at the 15 bins of 0.1 < z < 1.6",
     "  z      " + " ".join(f"{z:6.2f}" for z in ZB),
     "  canon  " + " ".join(f"{100 * v:6.2f}" for v in canon) + "   (book p2_09b: -3.1 at 0.15, -1.9 at 0.35, -0.9 at 0.65, -0.4 at 1.05, -0.1 at 1.55)"]
for n, (rp, c, b) in out.items(): L.append(f"  n={n:<4} " + " ".join(f"{100 * v:6.2f}" for v in rp) + f"   c {c:.3f}  beta {b:.3f}")
dr = (out[3.75][0] - out[3.25][0]) / 0.5; dc = (out[3.75][1] - out[3.25][1]) / 0.5
sig_n = 1 / np.sqrt(np.sum((dr / 0.01) ** 2)); chi_l = np.sqrt(np.sum((out[3.5][0] / 0.01) ** 2))
L += ["", f"Fisher at 1 % per bin: sigma(n) = {sig_n:.2f}; dc/dn = {dc:.3f}, so sigma(c) = {abs(dc) * sig_n:.3f} around c = {out[3.5][1]:.3f}",
      f"Separation of the n = 3.5 ramp from LambdaCDM at 1 % per bin: {chi_l:.1f} sigma (book: 5.0 for the canon E(a))",
      f"Can full DESI tell c = 1 (canon) from the computed c? c = 1 would need n far outside 3-4; at sigma(c) = {abs(dc) * sig_n:.3f} the separation is "
      f"{(1 - out[3.5][1]) / max(abs(dc) * sig_n, 1e-9):.1f} sigma IF closure B holds (the closure is a separate, model-level lever)."]
txt = "\n".join(L); print(txt)
if os.environ.get("GITHUB_ACTIONS"): os.makedirs("xout", exist_ok=True); open("xout/minicosmos_ramp_summary.txt", "w").write(txt + "\n")
