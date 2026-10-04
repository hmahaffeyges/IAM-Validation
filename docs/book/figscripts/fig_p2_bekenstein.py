"""Part 2, Chapters 'Black-hole horizons' (p3_01_blackholes.tex) and 'The horizon coefficient' (p3_01a_bekenstein.tex).
fig_bh_transfer: (a) entropy carried off by black-body evaporation, S_tr/S_0 = 1 - (1 - t/tau)^(2/3), with the entropy left on the horizon
  and the upper envelope min(S_tr, S_BH(t)) that bounds the fine-grained entropy of the radiation (Page); (b) mass, temperature and bit rate.
fig_rindler_cone: (a) the Euclidean Rindler plane: with theta = kappa tau / c of period 2 pi the tip is smooth; any other period leaves a cone;
  (b) the area one nat (k_B) and one bit (k_B ln 2) occupy on horizons of very different surface gravity: 4 l_P^2 and 4 ln2 l_P^2 for all.
Constants CODATA 2018 (scipy). Checks: docs/verification/scripts/verify_bekenstein_bh.py."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
import _bookstyle as S
import matplotlib.pyplot as plt

hbar, c, G, k = C.hbar, C.c, C.G, C.k
Msun, Mpc, ln2 = 1.98847e30, 3.0856775814913673e22, np.log(2)
lP2 = hbar * G / c**3
S.apply()

# ---------------- fig_bh_transfer -----------------
x = np.linspace(0, 1, 2001)
Str = 1 - (1 - x)**(2 / 3)
Sleft = (1 - x)**(2 / 3)
env = np.minimum(Str, Sleft)
xh = 1 - 2**-1.5
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.38))
a1.plot(x, Str, color=S.IAM, label="carried off, $S_{\\rm tr}/S_0$ (first law)")
a1.plot(x, Sleft, color=S.GR, ls="--", label="left on the horizon, $S_{\\rm BH}(t)/S_0$")
a1.fill_between(x, 0, env, color=S.SKY, alpha=0.35, lw=0)
a1.plot(x, env, color=S.DATA, lw=1.0, label="bound on the fine-grained entropy")
a1.axvline(xh, color=S.LIGHT, lw=0.7, ls=":")
a1.text(xh + 0.02, 0.06, "0.646$\\,\\tau$", fontsize=7, color=S.GR)
a1.set_xlim(0, 1); a1.set_ylim(0, 1.12)
a1.set_xlabel("$t/\\tau_{\\rm evap}$"); a1.set_ylabel("entropy / initial horizon entropy")
a1.legend(loc="upper left", fontsize=6.5, handlelength=1.8)
a1.set_title("Entropy carried off and entropy left")
S.panel_letter(a1, "a", dx=-0.16)
xm = x[:-1]
a2.plot(xm, (1 - xm)**(1 / 3), color=S.GR, label="$M/M_0$")
a2.plot(xm, (1 - xm)**(-1 / 3), color=S.IAM, label="$T_{\\rm BH}/T_0=\\Gamma/\\Gamma_0$")
a2.set_yscale("log"); a2.set_xlim(0, 1); a2.set_ylim(0.1, 20)
a2.set_xlabel("$t/\\tau_{\\rm evap}$"); a2.set_ylabel("ratio to initial value")
a2.legend(loc="upper left", fontsize=6.5)
a2.set_title("Smaller and hotter, it writes faster")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part3", "fig_bh_transfer")

# ---------------- fig_rindler_cone -----------------
fig, (b1, b2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.8), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.25]))
b1.set_aspect("equal")
for r in (0.33, 0.66, 1.0):
    th = np.linspace(0, 2 * np.pi, 400)
    b1.plot(r * np.cos(th), r * np.sin(th), color=S.IAM, lw=0.8)
for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    b1.plot([0, np.cos(ang)], [0, np.sin(ang)], color=S.LIGHT, lw=0.5)
# a deficit wedge drawn shaded: period 2 pi - delta would leave this wedge missing
dlt = 0.9
wed = np.linspace(-dlt / 2, dlt / 2, 60)
b1.fill(np.r_[0, 1.35 * np.cos(wed + 3 * np.pi / 2) + 0, 0], np.r_[0, 1.35 * np.sin(wed + 3 * np.pi / 2), 0], color=S.DATA, alpha=0.18, lw=0)
b1.plot(0, 0, "o", color=S.GR, ms=3)
b1.annotate("horizon, $\\rho=0$", xy=(0, 0), xytext=(0.25, 1.22), fontsize=6.5, color=S.GR,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GR))
b1.text(0, -1.52, "a period other than $2\\pi$ removes\n(or adds) a wedge: a conical tip", fontsize=6.5, ha="center", va="top", color=S.DATA)
b1.text(1.06, -0.22, "$\\theta=\\kappa\\tau/c$", fontsize=7, color=S.IAM)
b1.set_xlim(-1.45, 1.6); b1.set_ylim(-2.05, 1.45); b1.axis("off")
b1.set_title("Euclidean Rindler plane", loc="center")
S.panel_letter(b1, "a", dx=0.0, dy=1.0)
# (b) area per nat and per bit on horizons of very different kappa
labels, kap = [], []
for name, kk in [("Planck", c**2 / np.sqrt(lP2)),
                 ("1 M$_\\odot$", c**4 / (4 * G * Msun)),
                 ("Sgr A*", c**4 / (4 * G * 4.3e6 * Msun)),
                 ("M87*", c**4 / (4 * G * 6.5e9 * Msun)),
                 ("cosmic", c * 67.4e3 / Mpc)]:
    labels.append(name); kap.append(kk)
kap = np.array(kap)
dE_nat = hbar * kap / (2 * np.pi * c)
dA_nat = 8 * np.pi * G * dE_nat / (kap * c**2) / lP2
dA_bit = dA_nat * ln2
xs = np.arange(len(kap))
b2.plot(xs, dA_nat, "o", color=S.IAM, ms=4, label="one nat ($k_B$): $4\\,\\ell_P^2$")
b2.plot(xs, dA_bit, "s", color=S.ALT, ms=4, label="one bit ($k_B\\ln 2$): $4\\ln 2\\,\\ell_P^2$")
b2.axhline(4, color=S.IAM, lw=0.6, ls="--"); b2.axhline(4 * ln2, color=S.ALT, lw=0.6, ls=":")
b2.set_xticks(xs); b2.set_xticklabels([f"{l}\n{S.sci(kk, 2)}" for l, kk in zip(labels, kap)], fontsize=6)
b2.set_xlabel("horizon and its surface gravity $\\kappa$ (m s$^{-2}$)")
b2.set_xlim(-0.6, len(kap) - 0.4); b2.set_ylim(0, 6.2)
b2.set_ylabel("horizon area per unit, $\\delta A/\\ell_P^2$")
b2.legend(loc="upper center", fontsize=6.5, ncol=2, bbox_to_anchor=(0.56, 1.0))
b2.set_title("The same area at every surface gravity")
S.panel_letter(b2, "b", dx=-0.12)
print("dA per nat / l_P^2:", dA_nat, " per bit:", dA_bit)
S.save(fig, "part3", "fig_rindler_cone")
