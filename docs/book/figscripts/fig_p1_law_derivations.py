"""Part 1, Chapter 'IAM's Law' (p1_02_iams_law.tex).
fig_law_exponent: the power p of dS_info/dln a ~ a^p for n = 5/2, 3, 7/2, 4 (rate law I ~ rho_m D^n f H, accumulation per unit horizon
area at the horizon temperature), with the full LambdaCDM growth factor (Omega_m = 0.3153, Omega_r = 9.1e-5). E(a) = exp(1 - 1/a) needs p = -1.
fig_law_mu: mu(a) = H^2/(H^2 + beta_m E(a) H0^2) and H_m/H against redshift, beta_m = Omega_m/2 = 0.15765.
Numbers printed here are the ones quoted in the captions (checked also in docs/verification/scripts/verify_iams_law_derivations.py [V9], [V13])."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
Om, Or = 0.3153, 9.1e-5
OL = 1 - Om - Or
bm = Om / 2
H2 = lambda a: Om / a**3 + Or / a**4 + OL


def rhs(lna, Y):
    a = np.exp(lna); h2 = H2(a); dlnh = -(1.5 * Om / a**3 + 2 * Or / a**4) / h2
    return [Y[1], -(2 + dlnh) * Y[1] + 1.5 * Om / a**3 / h2 * Y[0]]


ai = 1e-6; y0 = ai * Om / Or                       # Meszaros growing mode D = 1 + 3y/2 deep in the radiation era
sol = solve_ivp(rhs, [np.log(ai), np.log(3)], [1 + 1.5 * y0, 1.5 * y0], dense_output=True, rtol=1e-10, atol=1e-14)
aa = np.logspace(-2.5, np.log10(2), 400)
D, Dp = sol.sol(np.log(aa)); f = Dp / D

fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.42))
cols = {2.5: S.GR, 3.0: S.ALT2, 3.5: S.IAM, 4.0: S.GOLD}
for n, col in cols.items():
    y = aa**-3 * D**n * f * np.sqrt(H2(aa))          # dS/dln a ~ rho_m D^n f /(T_H A_H) ~ a^-3 D^n f H
    p = np.gradient(np.log(y), np.log(aa))
    a1.plot(aa, p, color=col, lw=1.6 if n == 3.5 else 1.0, ls="-" if n == 3.5 else "--")
    i = np.argmin(abs(aa - 0.02))
    lab = {2.5: "$n=5/2$", 3.0: "$n=3$", 3.5: "$n=7/2$", 4.0: "$n=4$"}[n]
    a1.text(0.0042, p[np.argmin(abs(aa - 0.0042))] + 0.07, lab, color=col, fontsize=7, va="bottom")
    m = (aa >= 0.01) & (aa <= 0.1); q = (aa >= 0.25) & (aa <= 1)
    print(f"n = {n}: fitted power a 0.01-0.1 {np.polyfit(np.log(aa[m]), np.log(y[m]), 1)[0]:+.2f}; a 0.25-1 {np.polyfit(np.log(aa[q]), np.log(y[q]), 1)[0]:+.2f}")
a1.axhline(-1, color=S.LIGHT, lw=0.9, ls=":")
a1.text(1.9, -0.97, "needed for $E=\\exp(1-1/a)$", fontsize=7, ha="right", va="bottom", color=S.GR)
a1.set_xscale("log"); a1.set_xlim(3e-3, 2); a1.set_ylim(-3.2, 0.2); a1.set_yticks([-3, -2, -1, 0])
a1.set_xlabel("scale factor $a$"); a1.set_ylabel("local power $p$ of $dS_{\\rm info}/d\\ln a$")
a1.set_title("$p=-1$ in the matter era only for $n=7/2$")
S.panel_letter(a1, "a", dx=-0.15)

zz = np.linspace(0, 3, 301); a = 1 / (1 + zz)
E = np.exp(1 - 1 / a); h2 = Om / a**3 + (1 - Om)        # background without radiation, as in the chains' mu(a)
mu = h2 / (h2 + bm * E); r = np.sqrt(1 + bm * E / h2)
a2.plot(zz, mu, color=S.IAM, lw=1.6)
a2.plot(zz, r, color=S.ALT, lw=1.2, ls="--")
a2.axhline(1, color=S.LIGHT, lw=0.8, ls=":")
for zv in (0, 0.5, 1):
    av = 1 / (1 + zv); hv = Om / av**3 + 1 - Om; mv = hv / (hv + bm * np.exp(1 - 1 / av))
    a2.plot(zv, mv, "o", color=S.IAM, ms=3.5)
    a2.annotate(f"{mv:.3f}", (zv, mv), xytext=(4, -9), textcoords="offset points", fontsize=7)
    print(f"z = {zv}: mu = {mv:.4f}, H_m/H = {np.sqrt(1 + bm*np.exp(1-1/av)/hv):.4f}")
a2.text(2.9, 0.975, "$\\mu(z)$", color=S.IAM, fontsize=8, ha="right", va="top")
a2.text(2.9, 1.012, "$H_m/H$", color=S.ALT, fontsize=8, ha="right", va="bottom")
a2.set_xlim(0, 3); a2.set_ylim(0.84, 1.10)
a2.set_xlabel("redshift $z$"); a2.set_ylabel("ratio")
a2.set_title("Matter feels the record term late")
S.panel_letter(a2, "b", dx=-0.15)
S.save(fig, "part1", "fig_law_derivations")
