"""Part 2, Chapter 'Missing satellites: two mechanisms tested' (p2_19_missing_satellites.tex).
Census and equations as docs/verification/scripts/verify_obs_chapters.py sections A and F, on
docs/verification/observations/data/lvdb_dwarf_mw.csv (Local Volume Database, Pace et al. 2025).
fig_sat_census: velocity dispersions of the 54 Milky Way satellites with a measured value or an upper limit, sorted, against the
  4 km/s floor of Mechanism B.
fig_sat_mechanisms: (a) Mechanism A, Press-Schechter abundance change |Delta ln n| = |nu^2 - 1| x 0.78 % at fixed mass for the growth change
  of the exact coupling, against the order-of-magnitude deficit (ln 10); (b) Mechanism B, M_min(sigma) = 4 Om sigma^3/(G H0) and the
  dynamical-time form sqrt(6/pi) sigma^3/(G H0), with the 4 km/s point.
"""
import sys, csv, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import _bookstyle as S
import _cosmo as K
import matplotlib.pyplot as plt

S.apply()
rows = list(csv.DictReader(open(S.REPO / "docs/verification/observations/data/lvdb_dwarf_mw.csv")))
kin = [r for r in rows if r["vlos_sigma"] or r["vlos_sigma_ul"]]
val = lambda r: float(r["vlos_sigma"]) if r["vlos_sigma"] else float(r["vlos_sigma_ul"])
kin.sort(key=val)
below = [r for r in kin if val(r) < 4]
print(f"{len(rows)} satellites; {len(kin)} with dispersion or limit; {len(below)} below 4 km/s; "
      f"{sum(1 for r in below if r['vlos_sigma'])} resolved, {sum(1 for r in below if not r['vlos_sigma'])} limits")
fig, ax = plt.subplots(figsize=(S.TEXTW, 3.0))
for i, r in enumerate(kin):
    v = val(r); col = S.DATA if v < 4 else S.GR
    if r["vlos_sigma"]:
        em = float(r["vlos_sigma_em"] or 0); ep = float(r["vlos_sigma_ep"] or 0)
        ax.errorbar(i, v, yerr=[[em], [ep]], fmt="o", color=col, ms=3, lw=0.7, capsize=0)
    else:
        ax.plot(i, v, "v", color=col, ms=4)
ax.axhline(4, color=S.IAM, lw=1.0, ls="--")
ax.text(len(kin) - 0.5, 3.6, "Mechanism B floor, 4 km s$^{-1}$", fontsize=7, color=S.IAM, ha="right", va="top")
ax.text(1, 9.5, f"{len(below)} of {len(kin)} below the floor", fontsize=7, color=S.DATA)
lab = [r for r in below if r["name"] in ("Tucana V", "Crater II", "Segue 2", "Tucana III")]
for r in lab:
    i = kin.index(r); top = val(r) + float(r["vlos_sigma_ep"] or 0)
    if r["name"] == "Tucana V":
        ax.annotate(r["name"], (i, val(r)), xytext=(-3, 0), textcoords="offset points", fontsize=6, ha="right", va="center")
    else:
        ax.annotate(r["name"], (i, top), xytext=(0, 4), textcoords="offset points", fontsize=6, ha="center", rotation=90, va="bottom")
ax.set_yscale("log"); ax.set_ylim(0.6, 30); ax.set_xlim(-6, len(kin))
ax.set_yticks([1, 2, 4, 10, 20]); ax.set_yticklabels(["1", "2", "4", "10", "20"])
ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
ax.set_xlabel("Milky Way satellites, ordered by velocity dispersion (circles: measured, 68 %; triangles: upper limits)")
ax.set_ylabel("velocity dispersion (km s$^{-1}$)")
ax.set_title("The census rejects the dispersal floor")
S.save(fig, "part2", "fig_sat_census")

# ---------------- mechanisms -----------------
eps = K.amp_deficit(0.0)          # 0.78 %
nu = np.linspace(0.05, 3, 300)
G = 6.674e-11; Msun = 1.98892e30; Om = 0.3153; H0s = 67.36e3 / 3.0857e22
sg = np.linspace(1, 20, 300)
M4 = lambda s: 4 * Om * (s * 1e3)**3 / (G * H0s) / Msun
Md = lambda s: np.sqrt(6 / np.pi) * (s * 1e3)**3 / (G * H0s) / Msun
print(f"eps {eps:.3f} %; M_min(4) 10^{np.log10(M4(4)):.2f}; direct 10^{np.log10(Md(4)):.2f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33))
a1.plot(nu, abs(nu**2 - 1) * eps, color=S.IAM, lw=1.6, label=f"Mechanism A, growth $-${eps:.2f} %")
a1.axhline(100 * np.log(10), color=S.DATA, lw=1.0, ls="--", label="deficit of 10$\\times$ ($\\ln10$)")
a1.axvspan(0.05, 1, color=S.LIGHT, alpha=0.35, lw=0); a1.text(0.1, 1.2, "satellite halos\n$\\nu<1$", fontsize=7, color=S.GR)
a1.set_yscale("log"); a1.set_xlim(0, 3); a1.set_ylim(1e-2, 1e3)
a1.set_xlabel("peak height $\\nu=\\delta_c/\\sigma_M$"); a1.set_ylabel("$|\\Delta\\ln n|$ (%)")
a1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), fontsize=6.5)
a1.set_title("Mechanism A is too small")
S.panel_letter(a1, "a", dx=-0.14)
a2.plot(sg, np.log10(M4(sg)), color=S.IAM, lw=1.6, label="$M_{\\min}=4\\Omega_m\\sigma^3/(GH_0)$")
a2.plot(sg, np.log10(Md(sg)), color=S.ALT, lw=1.2, ls="--", label="$\\sqrt{6/\\pi}\\,\\sigma^3/(GH_0)$")
a2.plot(4, np.log10(M4(4)), "o", color=S.IAM, ms=4.5)
a2.annotate(f"4 km s$^{{-1}}$: $10^{{{np.log10(M4(4)):.2f}}}\\,M_\\odot$", (4, np.log10(M4(4))), xytext=(8, -4), textcoords="offset points", fontsize=7, va="top")
a2.set_xscale("log"); a2.set_xlim(1, 20); a2.set_xticks([1, 2, 4, 10, 20]); a2.set_xticklabels(["1", "2", "4", "10", "20"])
a2.set_xlabel("velocity dispersion $\\sigma$ (km s$^{-1}$)"); a2.set_ylabel("$\\log_{10}(M_{\\min}/M_\\odot)$")
a2.legend(loc="upper left", fontsize=6.5)
a2.set_title("Mechanism B: the minimum mass")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_sat_mechanisms")
