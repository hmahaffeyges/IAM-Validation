"""Part 2, Chapter 'Horizon thermodynamics and gravitational decoherence' (p2_03_theory.tex), derivation figures.
fig_theory_mu_sigma: (a) mu(z) = H^2/(H^2 + beta_m E H0^2) and the MGCAMB form 1 + mu0 Omega_DE(a)/Omega_DE; (b) where model classes sit in the
  (mu0, Sigma0) plane in the quasi-static regime (f(R) and normal-branch DGP: mu > 1, Sigma = 1; self-accelerating DGP: mu < 1, Sigma = 1, ghost);
  (c) E(a) = exp(1 - 1/a) with mu(a). Canon background Omega_m = 0.3153, beta_m = Omega_m/2 (CANON/iam_canon.json).
fig_record_fit: the accumulated record I(a) = int R/(T_H A_H) dt for R = Omega_m(a) f D^n, normalised today, against exp(alpha - beta/a);
  full LambdaCDM background (Omega_m 0.315, Omega_r 9.1e-5). Same numbers as docs/verification/scripts/verify_theory_derivations.py section 12.
Run from any directory: python docs/book/figscripts/fig_p2_theory_derivations.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import curve_fit
import _bookstyle as S
import matplotlib.pyplot as plt

Om, OL = 0.3153, 0.6847
bm = Om / 2
E = lambda a: np.exp(1 - 1 / a)
H2 = lambda a: Om * a**-3 + OL
mu = lambda a: H2(a) / (H2(a) + bm * E(a))
mu0 = -bm / (1 + bm)
mu_mg = lambda a: 1 + mu0 * (OL / H2(a)) / OL

S.apply()
fig, axs = plt.subplots(1, 3, figsize=(S.TEXTW, 2.3), gridspec_kw={"wspace": 0.55})
# (a)
ax = axs[0]
z = np.linspace(0, 3, 600); a = 1 / (1 + z)
ax.axhline(1, color=S.GR, lw=0.8)
ax.plot(z, mu(a), color=S.IAM, lw=1.6, label="exact")
ax.plot(z, mu_mg(a), color=S.ALT, lw=1.0, ls="--", label="MGCAMB form")
for zz in (0.0, 0.5, 1.0):
    ax.plot([zz], [mu(1 / (1 + zz))], "o", ms=3, color=S.IAM)
    ax.text(zz + 0.08, mu(1 / (1 + zz)) - 0.012, f"{mu(1/(1+zz)):.3f}", fontsize=6, color=S.IAM, va="top")
ax.text(2.9, 1.004, "GR, $\\Lambda$CDM", fontsize=6, color=S.GR, ha="right", va="bottom")
ax.set_xlabel("redshift $z$"); ax.set_ylabel("$\\mu(z)$"); ax.set_ylim(0.84, 1.03); ax.set_xlim(0, 3)
ax.legend(loc="lower right", fontsize=6, handlelength=1.6)
S.panel_letter(ax, "a", dx=-0.30, dy=1.06)
# (b)
ax = axs[1]
ax.axhline(0, color=S.LIGHT, lw=0.6); ax.axvline(0, color=S.LIGHT, lw=0.6)
ax.plot([mu0], [0], "*", ms=9, color=S.IAM)
ax.text(mu0, 0.035, "IAM", fontsize=6, color=S.IAM, ha="center", va="bottom")
ax.plot([0], [0], "o", ms=4, color=S.GR)
ax.text(0.01, -0.035, "GR", fontsize=6, color=S.GR, ha="left", va="top")
ax.annotate("", xy=(0.25, 0), xytext=(0.03, 0), arrowprops=dict(arrowstyle="->", color=S.ALT, lw=0.9))
ax.text(0.14, 0.03, "$f(R)$, nDGP", fontsize=6, color=S.ALT, ha="center", va="bottom")
ax.text(-0.15, -0.12, "sDGP (ghost):\nalso on $\\Sigma_0=0$", fontsize=6, color=S.ALT2, ha="center", va="top")
ax.set_xlim(-0.35, 0.3); ax.set_ylim(-0.25, 0.2)
ax.set_xlabel("$\\mu_0$"); ax.set_ylabel("$\\Sigma_0$")
S.panel_letter(ax, "b", dx=-0.30, dy=1.06)
# (c)
ax = axs[2]
aa = np.linspace(0.05, 2.5, 600)
ax.plot(aa, E(aa), color=S.IAM, lw=1.4)
ax.axhline(np.e, color=S.LIGHT, lw=0.6, ls="--")
ax.text(2.45, np.e - 0.06, "$e$ = 2.718", fontsize=6, color=S.GR, va="top", ha="right")
for zz in (2, 1, 0):
    ax.plot([1 / (1 + zz)], [E(1 / (1 + zz))], "o", ms=3, color=S.IAM)
    ax.text(1 / (1 + zz) + 0.1, E(1 / (1 + zz)) - {0: 0.1, 1: 0.12, 2: 0.06}[zz], f"z = {zz}: {E(1/(1+zz)):.3f}", fontsize=6, color=S.IAM, va="center", ha="left")
ax.set_xlabel("scale factor $a$"); ax.set_ylabel("$E(a)$", color=S.IAM); ax.set_ylim(0, 3.2); ax.set_xlim(0, 2.5)
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(aa, mu(aa), color=S.GOLD, lw=1.0, ls="--")
ax2.set_ylabel("$\\mu(a)$, dashed", color=S.GOLD); ax2.set_ylim(0.5, 1.02)
ax2.tick_params(labelsize=6)
S.panel_letter(ax, "c", dx=-0.30, dy=1.06)
S.save(fig, "part2", "fig_theory_mu_sigma")

# record fit figure
Om2, Or = 0.315, 9.1e-5; OL2 = 1 - Om2 - Or
E2 = lambda x: Om2 / x**3 + Or / x**4 + OL2
def rhs(l, y):
    x = np.exp(l); e2 = E2(x); dlnH = 0.5 * (-3 * Om2 / x**3 - 4 * Or / x**4) / e2
    return [y[1], -(2 + dlnH) * y[1] + 1.5 * (Om2 / x**3 / e2) * y[0]]
lg = np.linspace(np.log(1e-5), np.log(2.0), 40001)
s = solve_ivp(rhs, [lg[0], lg[-1]], [1.0, 0.0], t_eval=lg, rtol=1e-10, atol=1e-13)
ag = np.exp(lg); i1 = np.argmin(abs(ag - 1)); Dg = s.y[0] / s.y[0][i1]; fg = s.y[1] / s.y[0]; Omag = Om2 / ag**3 / E2(ag)
m = (ag >= 0.15) & (ag <= 2.0)
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.6))
cols = {2.5: S.ALT2, 3.5: S.IAM, 4.0: S.ALT}
for nn, col in cols.items():
    I = cumulative_trapezoid(Dg**nn * Omag * fg / ag, ag, initial=0); y = I / np.interp(1.0, ag, I)
    p, _ = curve_fit(lambda x, al, be: np.exp(al - be / x), ag[m], y[m], p0=[1, 1])
    ax.plot(ag[m], y[m], color=col, lw=1.5 if nn == 3.5 else 1.0)
    ax.text(2.02, {2.5: 1.12, 3.5: 1.33, 4.0: 1.55}[nn], f"$n$ = {nn:g}: $\\alpha$ = {p[0]:.2f}, $\\beta$ = {p[1]:.2f}", fontsize=6, color=col, va="center")
    print(f"n = {nn}: alpha {p[0]:.3f} beta {p[1]:.3f}")
ax.plot(ag[m], E(ag[m]), color=S.GR, lw=0.8, ls="--")
ax.text(0.2, 1.9, "dashed: $E(a)=\\exp(1-1/a)$", fontsize=6, color=S.GR)
ax.set_xlabel("scale factor $a$"); ax.set_ylabel("$I(a)/I(1)$"); ax.set_xlim(0.15, 2.0); ax.set_ylim(0, 3.0)
S.save(fig, "part2", "fig_record_fit")
