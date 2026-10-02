"""Part 2, Chapter 'Horizon thermodynamics and gravitational decoherence' (p2_03_theory.tex).
fig_exponent: dS_info/dln a = rho_m D^n f/(T_H A_H) with T_H ~ H, A_H ~ H^-2, full LambdaCDM background (Om 0.315, Or 9.1e-5),
normalised at a = 1, for n = 2.5, 3, 3.5, 4; the activation function needs a^-1 (method of docs/verification/scripts/verify_theory_paper.py).
fig_w_eff: w_info(a), the combined dark-sector w_eff(a) and its CPL forms: tangent at a = 1 (w0 = -1.062, wa = -0.012) and least squares over 0.5 <= a <= 1."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
import _bookstyle as S
import matplotlib.pyplot as plt

Om, Orad = 0.315, 9.1e-5; OL = 1 - Om - Orad; b = Om / 2
E2L = lambda a: Om * a**-3 + Orad * a**-4 + OL
def rhs(lna, y):
    a = np.exp(lna); h2 = E2L(a); dlnh = (-3 * Om * a**-3 - 4 * Orad * a**-4) / (2 * h2)
    return [y[1], -(2 + dlnh) * y[1] + 1.5 * Om * a**-3 / h2 * y[0]]
lna = np.linspace(np.log(1e-3), 0, 4000); a = np.exp(lna)
s = solve_ivp(rhs, (lna[0], 0), [1e-3, 1e-3], t_eval=lna, rtol=1e-9)
D = s.y[0] / s.y[0][-1]; f = s.y[1] / s.y[0]; H = np.sqrt(E2L(a))
S.apply()
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.8))
m1 = (a >= 0.01) & (a <= 0.1)
cols = {2.5: S.ALT2, 3.0: S.GOLD, 3.5: S.IAM, 4.0: S.ALT}
for n, col in cols.items():
    dS = Om * a**-3 * D**n * f / (H * H**-2); dS = dS / dS[-1]
    sl = np.polyfit(lna[m1], np.log(dS[m1]), 1)[0]
    print(f"n = {n}: matter-era power {sl:+.2f}")
    ax.plot(a, dS, color=col, lw=1.6 if n == 3.5 else 1.0)
    i0 = np.argmin(abs(a - 0.0032))
    ax.text(0.0032, dS[i0] * (1.35 if n != 4.0 else 0.55), f"$n$ = {n:g} (power {sl:+.2f})", color=col, fontsize=7, va="bottom" if n != 4.0 else "top")
ax.plot(a, a**-1.0, color=S.GR, lw=0.8, ls="--")
ax.text(0.95, 0.55, "dashed: $a^{-1}$, needed by $E(a)$", fontsize=7, color=S.GR, va="top", ha="right")
ax.axvspan(0.01, 0.1, color=LIGHT if False else S.LIGHT, alpha=0.25, lw=0)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.003, 1); ax.set_ylim(0.12, 2e6)
ax.text(0.0105, 0.2, "matter era (fit)", fontsize=7, color=S.GR, ha="left", va="bottom")
ax.set_xlabel("scale factor $a$"); ax.set_ylabel("$dS_{\\rm info}/d\\ln a$ (normalised today)")
ax.set_title("$n = 7/2$ gives the $a^{-1}$ the activation needs")
S.save(fig, "part2", "fig_exponent")

x = np.linspace(0.3, 1.6, 400)
w_info = -1 - 1 / (3 * x)
Ea = np.exp(1 - 1 / x)
w_eff = (-(1 - Om) + b * Ea * w_info) / ((1 - Om) + b * Ea)
w0 = (-(1 - Om) + b * (-4 / 3)) / ((1 - Om) + b); wa = -Om**2 / (3 * (2 - Om)**2)
A = np.linspace(0.5, 1, 200); wA = (-(1 - Om) + b * np.exp(1 - 1 / A) * (-1 - 1 / (3 * A))) / ((1 - Om) + b * np.exp(1 - 1 / A))
cf = np.linalg.lstsq(np.vstack([np.ones_like(A), 1 - A]).T, wA, rcond=None)[0]
print(f"w_eff(1) = {w0:.3f}; tangent wa = {wa:+.4f}; least squares w0 = {cf[0]:.3f}, wa = {cf[1]:+.3f}")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.33))
a1.plot(x, w_info, color=S.IAM, lw=1.5); a1.plot(x, w_eff, color=S.ALT, lw=1.5)
a1.axhline(-1, color=S.GR, lw=0.8, ls="--")
a1.text(1.55, -1.45, "$w_{\\rm info}=-1-1/3a$", color=S.IAM, fontsize=7, ha="right", va="top")
a1.text(1.55, -1.09, "$w_{\\rm eff}$ (vacuum + record)", color=S.ALT, fontsize=7, ha="right", va="top")
a1.axvline(1, color=S.LIGHT, lw=0.6, ls=":")
a1.set_xlim(0.3, 1.6); a1.set_ylim(-2.2, -0.9)
a1.set_xlabel("scale factor $a$"); a1.set_ylabel("equation of state $w$")
a1.set_title("Phantom record term, sum near $-1$")
S.panel_letter(a1, "a", dx=-0.16)
a2.plot(x, w_eff, color=S.ALT, lw=1.6)
a2.plot(x, w0 + wa * (1 - x), color=S.GR, lw=1.0, ls="--")
a2.plot(x, cf[0] + cf[1] * (1 - x), color=S.GOLD, lw=1.0, ls="-.")
a2.text(0.32, -1.098, f"tangent CPL: $w_0$ = {w0:.3f}, $w_a$ = {wa:+.3f}", fontsize=7, color=S.GR)
a2.text(0.32, -1.106, f"fit over 0.5–1: $w_0$ = {cf[0]:.3f}, $w_a$ = {cf[1]:+.3f}", fontsize=7, color=S.GOLD, va="top")
a2.set_xlim(0.3, 1.6); a2.set_ylim(-1.115, -1.0); a2.set_xticks([0.4, 0.8, 1.2, 1.6])
a2.set_xlabel("scale factor $a$"); a2.set_ylabel("$w_{\\rm eff}$")
a2.set_title("CPL forms: $w_a$ is small")
S.panel_letter(a2, "b", dx=-0.2)
S.save(fig, "part2", "fig_w_eff")
