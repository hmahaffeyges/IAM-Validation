"""Part 2, Chapter 'The equation of state through time, and the far future' (p2_20_wz_far_future.tex).
Equations and inputs as docs/verification/scripts/verify_wz_far_future.py and verify_obs_chapters.py section G
(H0 67.4, Om 0.315, OL 0.685, beta_m = 0.3153/2, E(a) = exp(1 - 1/a); DESI DR2 CPL fits, arXiv:2503.14738).
fig_wz_history: (a) w_info(z) = -1 - (1+z)/3 and its saturation value; (b) the weight of the informational density,
  rho_info/rho_Lambda and rho_info/(rho_Lambda + rho_info), against z, past and future (a up to 100).
fig_wz_cpl_clocks: (a) the CPL image (w0, wa) = (-4/3, -1/3) of w_info beside the three DESI DR2 + CMB + supernova fits
  (1 sigma bars on each axis; the fits are made with light-ruler distances, on which the term predicts w = -1);
  (b) the writing rate of E(a) in three clocks, each normalised to its maximum: per unit a, per e-fold, per unit cosmic time.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
H0, Om, OL = 67.4, 0.315, 0.685; bm = 0.3153 / 2; Gyr = 977.792 / H0
E = lambda a: np.exp(1 - 1 / a); H = lambda a: np.sqrt(Om * a**-3 + OL)
w = lambda a: -1 - 1 / (3 * a)
r = lambda a: bm * E(a) / OL
zp = np.linspace(0, 4, 300); ap = 1 / (1 + zp)
af = np.logspace(np.log10(0.2), 2, 400)
print(f"z=0: w {w(1):.3f}, rho_info/rho_L {r(1):.4f}, share {r(1)/(1+r(1)):.3f}; z=3: {r(0.25):.4f}; saturation {bm*np.e/OL:.3f}, share {bm*np.e/OL/(1+bm*np.e/OL):.3f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.33))
a1.plot(zp, w(ap), color=S.IAM, lw=1.6, label="$w_{\\rm info}=-1-(1+z)/3$")
a1.axhline(-1, color=S.GR, lw=0.8, ls="--"); a1.text(3.95, -0.97, "$w=-1$ (saturation, $a\\to\\infty$)", fontsize=7, color=S.GR, ha="right", va="bottom")
for zz in (0, 1, 2, 3):
    a = 1 / (1 + zz); a1.plot(zz, w(a), "o", color=S.IAM, ms=3.5)
    a1.annotate(f"{w(a):.3f}".replace("-", "\u2212"), (zz, w(a)), xytext=(5, 2), textcoords="offset points", fontsize=7, color=S.IAM)
a1.set_xlim(0, 4); a1.set_ylim(-2.8, -0.8)
a1.set_xlabel("redshift $z$"); a1.set_ylabel("equation of state $w_{\\rm info}$")
a1.legend(loc="lower left")
a1.set_title("Strongly phantom only where it is small")
S.panel_letter(a1, "a", dx=-0.14)
a2.plot(af, r(af), color=S.IAM, lw=1.6, label="$\\rho_{\\rm info}/\\rho_\\Lambda$")
a2.plot(af, r(af) / (1 + r(af)), color=S.ALT, lw=1.4, ls="--", label="$\\rho_{\\rm info}/(\\rho_\\Lambda+\\rho_{\\rm info})$")
a2.axhline(bm * np.e / OL, color=S.IAM, lw=0.6, ls=":"); a2.axvline(1, color=S.LIGHT, lw=0.6, ls=":")
a2.text(1.1, 0.71, "today", fontsize=7, color=S.GR)
a2.plot(1, r(1), "o", color=S.IAM, ms=3.5)
a2.annotate(f"today {r(1):.3f}", (1, r(1)), xytext=(8, -3), textcoords="offset points", fontsize=7, color=S.IAM, ha="left", va="top")
a2.text(95, bm * np.e / OL + 0.015, f"saturation {bm*np.e/OL:.3f}", fontsize=7, color=S.IAM, ha="right")
a2.plot(0.25, r(0.25), "o", color=S.IAM, ms=3.5); a2.annotate(f"$z$ = 3: {r(0.25):.3f}", (0.25, r(0.25)), xytext=(0.215, 0.30), textcoords="data", fontsize=7, color=S.IAM, arrowprops=dict(arrowstyle="-", color=S.IAM, lw=0.5))
a2.set_xscale("log"); a2.set_xticks([0.2, 0.5, 1, 3, 10, 30, 100]); a2.set_xticklabels(["0.2", "0.5", "1", "3", "10", "30", "100"])
a2.set_xlim(0.2, 100); a2.set_ylim(0, 0.75)
a2.set_xlabel("scale factor $a$"); a2.set_ylabel("weight of the informational density")
a2.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
a2.set_title("Its weight, past and future")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_wz_history")

# ---------------- CPL and three clocks -----------------
desi = (("DESI + CMB + Pantheon+", -0.838, 0.055, -0.62, 0.21), ("DESI + CMB + Union3", -0.667, 0.088, -1.09, 0.29),
        ("DESI + CMB + DES Y5", -0.752, 0.057, -0.86, 0.22))
for nm, w0, s0, wa, sa in desi:
    print(f"{nm}: w0 {abs(-4/3-w0)/s0:.1f} sigma, wa {abs(-1/3-wa)/sa:.1f} sigma")
aa = np.linspace(0.12, 3, 600)
ra = E(aa) / aa**2; rl = E(aa) / aa; rt = E(aa) / aa * H(aa)
pk = {nm: minimize_scalar(fn, bounds=(0.05, 5), method="bounded").x for nm, fn in
      (("a", lambda a: -E(a) / a**2), ("ln a", lambda a: -E(a) / a), ("t", lambda a: -E(a) / a * H(a)))}
tb = lambda a: quad(lambda x: 1 / (x * H(x)), 1e-8, a, limit=200)[0] * Gyr
print("peaks:", {k: round(v, 3) for k, v in pk.items()}, f"; z_t {1/pk['t']-1:.2f}, {tb(1)-tb(pk['t']):.1f} Gyr ago")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.8), gridspec_kw=dict(wspace=0.33))
cols = (S.DATA, S.ALT2, S.GOLD)
for (nm, w0, s0, wa, sa), col in zip(desi, cols):
    a1.errorbar(w0, wa, xerr=s0, yerr=sa, fmt="s", color=col, ms=4, lw=0.9, capsize=1.5, label=nm)
a1.plot(-4 / 3, -1 / 3, "o", color=S.IAM, ms=6, label="CPL image of $w_{\\rm info}$")
a1.plot(-1, 0, "+", color=S.GR, ms=9, mew=1.4, label="$\\Lambda$")
a1.axvline(-1, color=S.LIGHT, lw=0.6, ls=":"); a1.axhline(0, color=S.LIGHT, lw=0.6, ls=":")
a1.set_xlim(-1.5, -0.45); a1.set_ylim(-1.6, 0.5)
a1.set_xlabel("$w_0$"); a1.set_ylabel("$w_a$")
a1.legend(loc="lower left", fontsize=6)
a1.set_title("CPL image against DESI DR2 (not a test)")
S.panel_letter(a1, "a", dx=-0.14)
for y, col, ls, lab, key in ((ra, S.ALT, "--", "per unit $a$, $E/a^2$: peak $z$ = 1", "a"), (rl, S.IAM, "-", "per e-fold, $E/a$: peak today", "ln a"),
                             (rt, S.DATA, "-", f"per unit time, $HE/a$: peak $z$ = {1/pk['t']-1:.2f}", "t")):
    a2.plot(aa, y / y.max(), color=col, lw=1.5, ls=ls, label=lab)
    a2.axvline(pk[key], color=col, lw=0.6, ls=":")
a2.set_xlim(0.12, 3); a2.set_ylim(0, 1.3)
a2.set_xlabel("scale factor $a$"); a2.set_ylabel("writing rate / its maximum")
a2.legend(loc="upper right", fontsize=6.5)
a2.set_title("Three clocks, three peaks")
S.panel_letter(a2, "b", dx=-0.14)
S.save(fig, "part2", "fig_wz_cpl_clocks")
