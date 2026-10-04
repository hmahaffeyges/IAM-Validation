"""part4/fig_landauer_temperatures (Chapter 'From the horizon to the nucleus', Figure fig:landauerT).

E_bit = k_B T ln2 at the surfaces of the book: the de Sitter horizon at the Gibbons-Hawking temperature hbar H0/(2 pi k_B) with
H0 = 67.16 km/s/Mpc (photon sector, Level 2 chain, as the caption states); a one-solar-mass black hole at its Hawking temperature
hbar c^3/(8 pi G M k_B); a transmon at 20 mK (the chapter's value); the CMB today, 2.7255 K (Fixsen 2009); a cell nucleus at T_cell.
k_B and T_cell from CANON/iam_canon.json; hbar, c, G from CODATA 2018. Run: python docs/book/figscripts/fig_p4_landauer_temperatures.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()
CANON = json.load(open(S.REPO / "CANON" / "iam_canon.json"))["constants"]
kB = CANON["k_B"]["value"]
Mpc = 3.0856775814913673e22
H0 = 67.16e3 / Mpc                      # photon-sector H0, Level 2 chain (canon), as the caption states
T_GH = sc.hbar * H0 / (2 * np.pi * kB)
Msun = 1.98847e30                       # IAU nominal solar mass parameter / G (CODATA G)
T_BH = sc.hbar * sc.c**3 / (8 * np.pi * sc.G * Msun * kB)
T_cell = CANON["T_cell"]["value"]

PTS = [("de Sitter horizon\n(Gibbons-Hawking)", T_GH, (8, 6), "left"), ("solar-mass\nblack hole", T_BH, (8, -14), "left"),
       ("transmon\n(20 mK)", 0.020, (-8, 8), "right"), ("CMB today\n(2.7255 K)", 2.7255, (14, -30), "left"),
       ("cell nucleus\n(37 \u00b0C)", T_cell, (-8, 8), "right")]
T = np.logspace(-31, 4, 50)
fig, ax = plt.subplots(figsize=(0.9 * S.TEXTW, 2.7))
ax.plot(T, kB * T * np.log(2), color=S.GR, lw=0.9)
for lab, t, off, ha in PTS:
    e = kB * t * np.log(2)
    ax.plot([t], [e], "o", color=S.ALT if "cell" in lab else S.IAM, ms=5 if "cell" in lab else 4)
    ax.annotate(lab, (t, e), xytext=off, textcoords="offset points", fontsize=6.3, ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4", shrinkA=1, shrinkB=3) if "CMB" in lab else None)
    print(f"{lab.splitlines()[0]}: T = {t:.4g} K, E_bit = {e:.3e} J")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-32, 1e5); ax.set_ylim(1e-55, 1e-17); ax.set_yticks([1e-52, 1e-44, 1e-36, 1e-28, 1e-20]); ax.set_xticks([1e-30, 1e-24, 1e-18, 1e-12, 1e-6, 1e0])
ax.set_xlabel("temperature of the encoding surface (K)"); ax.set_ylabel(r"Landauer cost per bit, $k_BT\ln2$ (J)")
ax.set_title("One cost per bit; the surface's temperature sets its size")
S.save(fig, "part4", "fig_landauer_temperatures")
