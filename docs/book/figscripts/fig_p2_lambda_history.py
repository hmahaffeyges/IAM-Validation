"""Part 2, Chapter 'The cosmological constant over cosmic history' (p2_12b_lambda_history.tex), Fig. fig:cc_history.
(a) The integrand of the history integral as printed, (Ob/Otot)(a) (l_P/l_H(a))^2 / (a^2 H/H0), per unit ln a and in units of
    (l_P/l_H0)^2, from a_EW = 2.3e-15 to a = 1 (Planck 2018 densities, radiation 9.22e-5): the earliest epoch dominates.
(b) The activation-weighted coefficients int (H/H0)^p dE for p = -2..2 on the 18th-chain Omega_m = 0.3198, against the
    required K = (3 OL/8 pi)/(Ob/Om) = 0.523.
Same inputs as docs/verification/scripts/verify_lambda_baryon_book.py, section C.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import quad
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
Ob, Om, OL = 0.0493, 0.3153, 0.6846
Orad = 2.473e-5 / 0.6736**2 * (1 + 0.2271 * 3.046)
fb = Ob / Om; Kreq = 3 * OL / (8 * np.pi) / fb
Hn = lambda a: np.sqrt(Om / a**3 + Orad / a**4 + OL)
fbt = lambda a: (Ob / a**3) / (Om / a**3 + Orad / a**4 + OL)
integrand = lambda a: fbt(a) * Hn(a) / a**2          # (l_P/l_H(a))^2 = (l_P/l_H0)^2 (H/H0)^2, divided by a^2 H/H0
lnA = np.linspace(np.log(2.3e-15), 0, 200001); A = np.exp(lnA)
perln = integrand(A) * A
K = np.trapezoid(perln, lnA) / fb
aeq = Orad / Om
print(f"K (as printed) = {K:.2e}; required {Kreq:.3f}; a_eq = {aeq:.2e}")

Omx = 0.3198
Hx = lambda a: np.sqrt(Omx / a**3 + 1 - Omx); dE = lambda a: np.exp(1 - 1 / a) / a**2
ps = [-2, -1, 0, 1, 2]
vals = [quad(lambda a: Hx(a)**p * dE(a), 1e-6, 1, limit=200)[0] for p in ps]
print("activation-weighted:", [round(v, 3) for v in vals])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.7), gridspec_kw=dict(wspace=0.36, width_ratios=[1.25, 1]))
a1.loglog(A[::200], perln[::200], color=S.IAM, lw=1.5)
a1.axvline(aeq, color=S.LIGHT, lw=0.8, ls="--")
a1.text(aeq * 1.6, 1e20, "matter-radiation\nequality", fontsize=6.5, color=S.GR, va="center", ha="left")
a1.axvline(2.3e-15, color=S.LIGHT, lw=0.8, ls=":")
a1.text(4e-15, 1e5, "printed lower limit\n$a_{\\rm EW}=2.3\\times10^{-15}$", fontsize=6.5, color=S.GR, va="center", ha="left")
a1.text(0.97, 0.95, f"integral: $K$ = {S.sci(K, 2)}\nrequired: $K$ = {Kreq:.3f}", transform=a1.transAxes, fontsize=7,
        ha="right", va="top", color=S.IAM)
a1.set_xlim(1e-15, 1.5); a1.set_ylim(1e-3, 1e30)
a1.set_xlabel("scale factor $a$")
a1.set_ylabel("integrand per unit $\\ln a$ [$(l_P/l_H)^2$]")
a1.set_title("As printed: set by the lower limit")
S.panel_letter(a1, "a", dx=-0.16)

x = np.arange(len(ps))
a2.semilogy(x, vals, "o", color=S.IAM, ms=5)
for xi, v in zip(x, vals):
    a2.annotate(f"{v:.3f}" if v < 1.5 else f"{v:.2f}", (xi, v), xytext=(6, -10) if v < Kreq else (6, 2),
                textcoords="offset points", fontsize=7, color=S.IAM)
a2.axhline(Kreq, color=S.DATA, lw=0.9, ls="--")
a2.text(4.5, Kreq * 0.80, f"required {Kreq:.3f}", fontsize=7, color=S.DATA, ha="right", va="top")
a2.set_xticks(x); a2.set_xticklabels([f"$p={p}$" for p in ps], fontsize=7)
a2.set_xlim(-0.5, 4.6); a2.set_ylim(0.3, 10)
a2.set_yticks([0.5, 1, 2, 5]); a2.set_yticklabels(["0.5", "1", "2", "5"])
a2.set_ylabel("$\\int_0^1 (H/H_0)^p\\,dE$")
a2.set_title("Activation weighting")
S.panel_letter(a2, "b", dx=-0.16)
S.save(fig, "part2", "fig_cc_history")
