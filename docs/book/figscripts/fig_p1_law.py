"""Part 1, Chapter 'IAM's Law' (p1_02_iams_law.tex).
fig_activation: (a) E(a) = exp(1 - 1/a) and the record written per e-fold, dE/dln a = E/a; (b) w_info(a) = -1 - 1/(3a).
fig_mahaffey_number: M = E_drive/(k_B T) for the systems the book evaluates: cell (ATP, 54 kJ/mol at 310.15 K), two chips
(whole-chip TDP/(N f) at T_j = 75 C, inputs as printed in p3_06 and p3_08), black holes (M c^2/(k_B T_BH) = 2 S_BH/k_B)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, scipy.constants as C
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
# ---------------- activation function -----------------
E = lambda a: np.exp(1 - 1 / a)
a = np.linspace(0.05, 5, 600)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.6), gridspec_kw=dict(wspace=0.32))
a1.plot(a, E(a), color=S.IAM, lw=1.6)
a1.plot(a, E(a) / a, color=S.ALT, lw=1.2, ls="--")
a1.axhline(np.e, color=S.LIGHT, lw=0.8, ls=":")
a1.text(4.95, np.e - 0.08, "$e$ = 2.718 (record complete)", fontsize=7, ha="right", va="top", color=S.GR)
a1.plot(1, 1, "o", color=S.IAM, ms=4); a1.plot(0.5, E(0.5), "o", mfc="white", mec=S.IAM, ms=4)
a1.annotate("today, $E=1$;\nrate per e-fold peaks", (1, 1), xytext=(1.4, 0.12), textcoords="data", fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GR))
a1.annotate("inflection of $E$\nat $a=1/2$", (0.5, E(0.5)), xytext=(0.08, 1.75), fontsize=7,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=S.GR))
a1.text(3.2, E(3.2) + 0.08, "$E(a)$", color=S.IAM, fontsize=8, va="bottom")
a1.text(3.4, E(3.4) / 3.4 + 0.07, "$dE/d\\ln a = E/a$", color=S.ALT, fontsize=8, va="bottom")
a1.set_xlim(0, 5); a1.set_ylim(0, 3.0)
a1.set_xlabel("scale factor $a$"); a1.set_ylabel("activation")
a1.set_title("Written fastest per e-fold today")
S.panel_letter(a1, "a", dx=-0.13)
aa = np.logspace(np.log10(0.25), 1, 300)
w = lambda a: -1 - 1 / (3 * a)
a2.plot(aa, w(aa), color=S.IAM, lw=1.6)
a2.axhline(-1, color=S.GR, lw=0.8, ls="--"); a2.text(9.5, -0.98, "$w=-1$", fontsize=7, ha="right", va="bottom", color=S.GR)
for x in (0.5, 1, 2):
    a2.plot(x, w(x), "o", color=S.IAM, ms=4)
    a2.annotate(f"{w(x):.2f}", (x, w(x)), xytext=(5, -2), textcoords="offset points", fontsize=7, va="top")
a2.set_xscale("log"); a2.set_xticks([0.25, 0.5, 1, 2, 5, 10]); a2.set_xticklabels(["0.25", "0.5", "1", "2", "5", "10"])
a2.set_ylim(-2.45, -0.85)
a2.set_xlabel("scale factor $a$"); a2.set_ylabel("$w_{\\rm info}$")
a2.set_title("$w_{\\rm info}<-1$ at every finite $a$")
S.panel_letter(a2, "b", dx=-0.15)
print("w_info at 0.5, 1, 2:", [round(w(x), 3) for x in (0.5, 1, 2)], " E(0.5) =", round(E(0.5), 4))
S.save(fig, "part1", "fig_activation")

# ---------------- Mahaffey number -----------------
k, hbar, c, G = C.k, C.hbar, C.c, C.G; Msun = 1.98847e30
Tj = 348.15
M_cell = 54000 / (C.R * 310.15)
M_9950 = 170 / (20.6e9 * 4.3e9) / (k * Tj)         # p3_08 inputs
M_bh = lambda m: 8 * np.pi * G * (m * Msun)**2 / (hbar * c)
rows = [("living cell (ATP at 310 K)", M_cell, S.ALT),
        ("CMOS chip, AMD Ryzen 9 9950X (whole-chip average)", M_9950, S.GOLD), ("black hole, 1 M$_\\odot$", M_bh(1.0), S.IAM),
        ("black hole, Sgr A$^*$", M_bh(4.3e6), S.IAM)]
for r in rows:
    print(f"{r[0]:42s} M = {r[1]:.4g}  (Landauer units {r[1]/np.log(2):.4g})")
fig, ax = plt.subplots(figsize=(S.TEXTW, 2.2))
for i, (lab, M, col) in enumerate(rows[::-1]):
    ax.plot([1, M], [i, i], color=S.LIGHT, lw=0.8, zorder=1)
    ax.plot(M, i, "o", color=col, ms=5, zorder=3)
    txt = f"{M:.4g}" if M < 1e4 else S.sci(M, 2)
    ax.annotate(txt, (M, i), xytext=(6, 0), textcoords="offset points", fontsize=7, va="center")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7)
ax.set_xscale("log"); ax.set_xlim(0.5, 1e95)
ax.set_xticks([1, 1e20, 1e40, 1e60, 1e80])
ax.axvline(1, color=S.GR, lw=0.8, ls=":")
ax.set_xlabel("Mahaffey number $\\mathcal{M}=E_{\\rm drive}/k_BT$  (1 = drive equal to the thermal quantum)")
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_title("Every record-keeping system runs far above the thermal quantum")
S.save(fig, "part1", "fig_mahaffey_number")
