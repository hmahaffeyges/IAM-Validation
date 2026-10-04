"""part5/fig_T1_xqp (Chapter 'The superconducting qubit as an encoding surface', Figure fig:T1xqp).

Quasiparticle-limited T1 from the chapter's Eq. eq:catelani, Gamma_1 = x_qp (omega_q/pi) sqrt(2 Delta/(hbar omega_q))
(Catelani et al. 2011, PRB 84, 064517), for Al (Delta = 182 ueV) at the two frequencies of Table tab:T1xqp (4 and 5 GHz).
Band: the measured background of best-isolated devices, 1e-8 to 1e-6 (Riste et al. 2013, Nat. Commun. 4, 1913), as in the table.
Dashed: the 1 ms working target. Constants: CODATA 2018 (scipy.constants).
Run: python docs/book/figscripts/fig_p3_T1_xqp.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()

DELTA = 182e-6 * sc.e
def T1(x, f):
    w = 2 * np.pi * f
    return 1 / (x * w / np.pi * np.sqrt(2 * DELTA / (sc.hbar * w)))

x = np.logspace(-9, -5, 200)
t5 = T1(1e-7, 5e9) * 1e3
x1ms = 1e-7 * t5                                  # T1 scales as 1/x: 1 ms needs x <= this
fig, ax = plt.subplots(figsize=(0.75 * S.TEXTW, 2.9))
ax.axvspan(1e-8, 1e-6, color=S.LIGHT, alpha=0.35, lw=0)
ax.text(1e-7, 2.2e-3, "measured background,\nbest-isolated devices", ha="center", va="bottom", fontsize=6, color=S.GR)
ax.plot(x, T1(x, 5e9) * 1e3, color=S.IAM, label=r"$\omega_q/2\pi$ = 5 GHz")
ax.plot(x, T1(x, 4e9) * 1e3, color=S.ALT, label=r"$\omega_q/2\pi$ = 4 GHz")
ax.axhline(1.0, color=S.DATA, lw=0.8, ls="--")
ax.text(9e-6, 1.25, "1 ms working target", fontsize=6.5, color=S.DATA, va="bottom", ha="right")
ax.plot([1e-7], [t5], "o", color=S.IAM, ms=4)
ax.annotate(f"{t5:.2f} ms at $x_{{\\rm qp}}=10^{{-7}}$", (1e-7, t5), xytext=(8, 6), textcoords="offset points", fontsize=6.5, color=S.IAM)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-9, 1e-5); ax.set_ylim(1e-3, 1e2)
ax.set_xlabel(r"quasiparticle density $x_{\rm qp}$"); ax.set_ylabel(r"$T_1^{\rm qp}$ ceiling (ms)")
ax.legend(loc="upper right")
ax.set_title(f"At $x_{{\\rm qp}}=10^{{-7}}$ a 5 GHz Al transmon is capped near {t5:.2f} ms")
print(f"T1(1e-7, 5 GHz) = {t5:.3f} ms; 1 ms needs x_qp <= {x1ms:.2e}; T1(1e-7, 4 GHz) = {T1(1e-7, 4e9)*1e3:.3f} ms")
S.save(fig, "part5", "fig_T1_xqp")
