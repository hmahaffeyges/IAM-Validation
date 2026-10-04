"""part5/fig_timescales (Chapter 'The superconducting qubit as an encoding surface', Figure fig:timescales).

Timescales at an Al/AlOx junction, as the chapter gives them: hbar/Delta_Al = 3.6 ps (Delta_Al = 182 ueV, calculated); the qubit
period at 5 GHz, 0.2 ns; quasiparticle recombination, tens of ns to ms depending on density; TLS switching, 1-100 us (book working
value 30 us, Chapter ch:xqp); quasiparticle lifetime ~100 us (Chapter ch:xqp); T1 of the best transmons, 0.3-0.5 ms (Place et al. 2021,
Nat. Commun. 12, 1779; Wang et al. 2022, npj Quantum Inf. 8, 3); charge-parity switching, ~1 ms (Riste et al. 2013, Nat. Commun. 4, 1913).
Run: python docs/book/figscripts/fig_p3_timescales.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()

tphi = sc.hbar / (182e-6 * sc.e)
ROWS = [  # label, point (s), range (s) or None
    (r"$\hbar/\Delta_{\rm Al}$, condensate phase", tphi, None),
    ("qubit period, 5 GHz", 1 / 5e9, None),
    ("quasiparticle recombination", None, (2e-8, 1e-3)),
    (r"TLS switching ($\tau_{\rm TLS}$)", 30e-6, (1e-6, 1e-4)),
    (r"quasiparticle lifetime $\tau_{\rm qp}$", 100e-6, None),
    (r"$T_1$, best transmons", None, (3e-4, 5e-4)),
    ("charge-parity switching", 1e-3, None)]
fig, ax = plt.subplots(figsize=(0.52 * S.TEXTW + 0.4, 2.7))
for i, (lab, p, r) in enumerate(ROWS):
    if r:
        ax.plot(r, [i, i], color=S.IAM, lw=3, solid_capstyle="butt")
    if p:
        ax.plot([p], [i], "o", color=S.IAM, ms=4)
    left = r[0] if r else p
    ax.text(left, i + 0.28, lab, fontsize=6, va="bottom", ha="left" if left < 1e-5 else "right" if left > 3e-4 else "center")
ax.set_xscale("log"); ax.set_xlim(1e-12, 1e-2); ax.set_ylim(-0.6, len(ROWS) - 0.1)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("time (s)")
dec = (np.log10(1e-6 / tphi), np.log10(1e-4 / tphi))
ax.set_title(f"{dec[0]:.0f} to {dec[1]:.0f} decades from phase restoration to a TLS flip")
print(f"hbar/Delta = {tphi*1e12:.2f} ps; decades to 1 and 100 us: {dec[0]:.2f}, {dec[1]:.2f}")
S.save(fig, "part5", "fig_timescales")
