"""part4/fig_scale_ladder (Chapter 'From the horizon to the nucleus', Figure fig:scaleladder).

Three encoding surfaces by temperature and capacity in bits. Cosmic horizon: Gibbons-Hawking temperature and area/(4 l_P^2 ln2) at the
Hubble radius c/H0, H0 = 67.16 km/s/Mpc (photon sector, Level 2 chain). Solar-mass black hole: Hawking temperature and Bekenstein-Hawking
entropy 4 pi G M^2/(hbar c) in bits. Cell: T_cell and one bit per CpG on one strand of hg19, 28,217,448 sites.
k_B, T_cell from CANON/iam_canon.json; hbar, c, G from CODATA 2018. Run: python docs/book/figscripts/fig_p4_scale_ladder.py
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

lP2 = sc.hbar * sc.G / sc.c**3
N_hor = np.pi * (sc.c / H0)**2 / lP2 / np.log(2)
N_bh = 4 * np.pi * sc.G * Msun**2 / (sc.hbar * sc.c) / np.log(2)
N_cell = 28217448
PTS = [("cosmic horizon", T_GH, N_hor, S.IAM), ("solar-mass black hole", T_BH, N_bh, S.ALT2), ("one cell's methylome", T_cell, N_cell, S.ALT)]
fig, ax = plt.subplots(figsize=(0.78 * S.TEXTW, 2.8))
for lab, t, n, col in PTS:
    ax.plot([t], [n], "o", color=col, ms=6)
    ha = "right" if "cell" in lab else "left"
    ax.annotate(f"{lab}\nT = {S.sci(t)} K, N = {S.sci(n)} bits", (t, n), xytext=(-8 if ha == "right" else 8, 6 if "cell" in lab else 0),
                textcoords="offset points", fontsize=6.3, ha=ha, va="bottom" if "cell" in lab else "center")
    print(f"{lab}: T = {t:.3e} K, N = {n:.3e} bits")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-31, 1e4); ax.set_ylim(1e3, 1e130)
ax.set_xlabel("surface temperature (K)"); ax.set_ylabel("bits the surface holds")
ax.set_title("Same accounting, very different surfaces")
S.save(fig, "part4", "fig_scale_ladder")
