"""Part 3 figures (qubits and chips). Values are those printed in the chapters (quoted as published or calibrated there) or
computed from them here; no curve is typed by hand.
p3_02 fig_xqp_sites: x_qp = 2 N tau_qp/(tau_TLS n_cp V), n_cp = 4e6 um^-3, tau_TLS = 30 us, tau_qp = 100 us (Eq. eq:xqp; verify_xqp.py).
p3_02 fig_xqp_thermal: equilibrium x_qp(T) = sqrt(2 pi k_B T/Delta) exp(-Delta/k_B T), Delta_Al = 182 ueV, against the measured background.
p3_08 fig_imr90_channels: IMR90 Met-A ranges per channel (proliferating held out, senescent, SV40-immortalised; WGBS GSE48580).
p3_08 fig_holding_energy: E_hold measured from the copy error against the Hopfield range k_B T ln(selectivity) of DNMT1."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
import _bookstyle as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

S.apply()
# ---------------- x_qp vs active sites -----------------
ncp, tT, tq = 4e6, 30.0, 100.0
N = np.logspace(0, 5, 200)
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.7))
ax.axhspan(1e-8, 1e-6, color=S.DATA, alpha=0.15, lw=0); ax.axhline(1e-7, color=S.DATA, lw=0.8, ls="--")
ax.text(1.3, 1.25e-7, "measured floor, $10^{-7}$", fontsize=7, color=S.DATA, va="bottom")
for V, ls in ((1e3, "-"), (1e4, "--"), (1e5, ":")):
    x = 2 * N * tq / (tT * ncp * V)
    Nn = 1e-7 * ncp * V * tT / (2 * tq)
    ax.plot(N, x, color=S.IAM, lw=1.4, ls=ls, label=f"island $10^{int(np.log10(V))}$ µm³: {Nn:.0f} sites at $10^{{-7}}$")
    ax.plot(Nn, 1e-7, "o", color=S.IAM, ms=4)
    print(f"V {V:.0e}: sites for 1e-7 = {Nn:.0f}; x per site {2*tq/(tT*ncp*V):.1e}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 1e5); ax.set_ylim(1e-12, 1e-5)
ax.legend(loc="lower right", fontsize=7)
ax.set_xlabel("active fluctuators $N$ on the island"); ax.set_ylabel("quasiparticle fraction $x_{\\rm qp}$")
ax.set_title("Hundreds to thousands of sites hold the floor")
S.save(fig, "part3", "fig_xqp_sites")

k = C.k; Delta = 182e-6 * C.e
T = np.linspace(0.010, 0.300, 400); xth = np.sqrt(2 * np.pi * k * T / Delta) * np.exp(-Delta / (k * T))
x15 = np.sqrt(2 * np.pi * k * 0.015 / Delta) * np.exp(-Delta / (k * 0.015))
Tx = T[np.argmin(abs(np.log10(xth) + 7))]
print(f"Delta/kT at 15 mK = {Delta/(k*0.015):.1f}; x_th(15 mK) = {x15:.2e}; x_th = 1e-7 at T = {Tx*1e3:.0f} mK")
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 2.7))
ax.plot(T * 1e3, xth, color=S.GR, lw=1.5)
ax.axhspan(1e-8, 1e-6, color=S.DATA, alpha=0.2, lw=0); ax.axhspan(1e-6, 1e-5, color=S.DATA, alpha=0.08, lw=0)
ax.text(5, 5e-9, "measured, best-isolated devices", fontsize=7, color=S.DATA, ha="left", va="top")
ax.text(5, 2e-5, "less protected devices", fontsize=7, color=S.DATA, ha="left", va="bottom")
ax.plot(15, x15, "o", color=S.GR, ms=4); ax.annotate(f"15 mK: {S.sci(x15, 1)}", (15, x15), xytext=(6, 0), textcoords="offset points", fontsize=7, va="center")
ax.text(120, 1e-40, "thermal equilibrium, Al ($\\Delta$ = 182 µeV)", fontsize=7, color=S.GR)
ax.set_yscale("log"); ax.set_xlim(0, 300); ax.set_ylim(1e-66, 1)
ax.set_yticks([1e-60, 1e-45, 1e-30, 1e-15, 1])
ax.set_xlabel("temperature (mK)"); ax.set_ylabel("quasiparticle fraction $x_{\\rm qp}$")
ax.set_title("The measured background is 55 orders above equilibrium")
S.save(fig, "part3", "fig_xqp_thermal")

# ---------------- IMR90 channels -----------------
states = [("proliferating (held out)", [(0.991, 1.005), (0.991, 1.016), (0.997, 1.002)]),
          ("senescent", [(0.941, 1.016), (0.685, 0.695), (0.874, 0.925)]),
          ("SV40-immortalised", [(1.077, 1.120), (0.587, 0.664), (0.965, 0.975)])]
chan = [("methylated channel", S.IAM), ("unmethylated channel", S.ALT), ("both", S.GR)]
fig, ax = plt.subplots(figsize=(0.8 * S.TEXTW, 2.4))
ax.axvspan(0.95, 1.05, color=S.LIGHT, alpha=0.35, lw=0); ax.axvline(1, color=S.GR, lw=0.8)
ax.text(1.0, 2.62, "Normal 0.95–1.05", fontsize=7, color=S.GR, ha="center", va="bottom")
for i, (st, rr) in enumerate(states[::-1]):
    for j, ((lo, hi), (cn, col)) in enumerate(zip(rr, chan)):
        yy = i + 0.22 * (1 - j)
        ax.plot([lo, hi], [yy, yy], color=col, lw=3.0, solid_capstyle="butt")
        ax.plot([lo, hi], [yy, yy], "|", color=col, ms=5)
for cn, col in chan:
    ax.plot([], [], color=col, lw=3, label=cn)
ax.legend(loc="upper left", fontsize=7)
ax.set_yticks(range(3)); ax.set_yticklabels([s[0] for s in states[::-1]], fontsize=7)
ax.set_xlim(0.45, 1.2); ax.set_ylim(-0.45, 2.9)
ax.set_xlabel("Met-A, each culture against the proliferating cultures")
ax.set_title("The two channels err in opposite directions")
S.save(fig, "part3", "fig_imr90_channels")

# ---------------- holding energy against Hopfield -----------------
sel = [("purified enzyme, 7–21×", 7, 21), ("Goyal et al. 2006, 30–40×", 30, 40), ("across flanking sequences, 80×", 80, 80)]
fig, ax = plt.subplots(figsize=(0.62 * S.TEXTW, 1.9))
for i, (lab, lo, hi) in enumerate(sel[::-1]):
    a, b = np.log(lo), np.log(hi); print(f"{lab}: {a:.2f}-{b:.2f} kT")
    if hi > lo:
        ax.plot([a, b], [i, i], color=S.DATA, lw=3.0, solid_capstyle="butt")
    else:
        ax.plot(a, i, "s", color=S.DATA, ms=5)
ax.axvline(3.41, color=S.IAM, lw=1.2); ax.axvline(3.77, color=S.IAM, lw=1.0, ls="--")
ax.text(3.41 - 0.05, 2.55, "3.41", fontsize=7, color=S.IAM, ha="right", va="bottom"); ax.text(3.77 + 0.05, 2.55, "3.77", fontsize=7, color=S.IAM, ha="left", va="bottom")
ax.set_yticks(range(3)); ax.set_yticklabels([s[0] for s in sel[::-1]], fontsize=7)
ax.set_xlim(1.5, 5.0); ax.set_ylim(-0.5, 2.95)
ax.set_xlabel("energy gap $k_BT\\ln$(selectivity) or measured $E_{\\rm hold}$ ($k_BT$)")
ax.set_title("Measured holding energy inside DNMT1's range")
S.save(fig, "part3", "fig_holding_energy")
