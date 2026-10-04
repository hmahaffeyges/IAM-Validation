"""part2/star_gauge (Chapter 'Black-hole horizons', Figure fig:stargauge).

Compact stars on the same gauge: A = mass / the typical remnant of the same kind as built (white dwarfs 0.6 Msun, the measured DA mean
0.593 Msun, Kepler et al. 2007; neutron stars 1.4 Msun). Surface full at the Chandrasekhar mass 1.44 Msun (A = 2.40) and the TOV limit taken
as ~2.3 Msun (A ~ 1.64); beyond, a black hole. Masses as in the chapter's table: Sun's future white dwarf 0.54 (Sackmann et al. 1993),
Procyon B 0.592 (Bond et al. 2015), Sirius B 1.02 (Bond et al. 2017), IK Pegasi B 1.15 (Landsman et al. 1993), PSR J0740+6620 2.08
(Fonseca et al. 2021). Each row has its own linear scale. Run: python docs/book/figscripts/fig_p2_star_gauge.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()

from matplotlib.colors import LinearSegmentedColormap
CM = LinearSegmentedColormap.from_list("g", [(0, "#56B4E9"), (0.3, "#3cb371"), (0.62, "#f0b070"), (0.85, "#d55e6a"), (1, "#5a1a1a")])
ROWS = [dict(name="White dwarfs: held up by electron pressure", typ=0.6, full=1.44, lo=0.4, ticks=[0.5, 1, 1.5, 2],
             stars=[("Sun's future WD", 0.54, "isolated"), ("Procyon B", 0.592, "wide binary"), ("Sirius B", 1.02, "wide binary"),
                    ("IK Pegasi B", 1.15, "21.7-day binary")], fullname="Chandrasekhar 1.44 M$_\\odot$"),
        dict(name="Neutron stars: held up by neutron pressure", typ=1.4, full=2.3, lo=0.4, ticks=[0.5, 1, 1.25, 1.5],
             stars=[("typical NS", 1.4, ""), ("PSR J0740+6620", 2.08, "")], fullname="TOV $\\approx$ 2.3 M$_\\odot$")]
fig, axs = plt.subplots(2, 1, figsize=(S.TEXTW, 4.0), gridspec_kw=dict(hspace=0.9))
for ax, R_ in zip(axs, ROWS):
    Af = R_["full"] / R_["typ"]; hi = Af * 1.3
    ax.imshow(np.linspace(0, 1, 400)[None, :], extent=(R_["lo"], Af, 0, 1), aspect="auto", cmap=CM)
    ax.add_patch(plt.Rectangle((Af, 0), hi - Af, 1, fc="#111111", ec="none"))
    ax.text((Af + hi) / 2, 0.5, "black hole", color="white", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.plot([Af, Af], [-0.1, 1.1], color="#a01020", lw=2, ls="--"); ax.plot([1, 1], [-0.1, 1.1], color="#1a6b3a", lw=2)
    ax.set_xlim(R_["lo"], hi); ax.set_ylim(-2.3, 1.0); ax.set_yticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks(R_["ticks"] + [round(Af, 2)]); ax.tick_params(axis="x", length=0, pad=2)
    ax.xaxis.set_ticks_position("top"); ax.set_xticklabels([])
    for t in R_["ticks"] + [Af]:
        ax.text(t, -0.08, f"{t:g}" if t != Af else f"{Af:.2f}", ha="center", va="top", fontsize=6)
    ax.text(R_["lo"], 1.75, R_["name"], fontsize=7.5, fontweight="bold", va="bottom", transform=ax.transData)
    ax.text(1, 1.12, f"1 = typical ({R_['typ']} M$_\\odot$)", ha="center", va="bottom", fontsize=6.5, color="#1a6b3a")
    ax.text(Af + 0.01 * (hi - R_["lo"]), 1.12, f"surface full {Af:.2f}\n{R_['fullname']}", ha="left", va="bottom", fontsize=6.5, color="#a01020")
    As = [m / R_["typ"] for _, m, _ in R_["stars"]]; span = hi - R_["lo"]
    for i, (n, m, note) in enumerate(R_["stars"]):
        A = As[i]; print(f"{n}: A = {A:.3f}")
        ax.plot([A], [-0.42], "^", color=S.IAM, ms=6)
        lev = -0.75 - 0.75 * (i % 2)
        near_next = i + 1 < len(As) and As[i + 1] - A < 0.12 * span
        near_prev = i > 0 and A - As[i - 1] < 0.12 * span
        ha = "right" if near_next else ("left" if near_prev else "center")
        ax.plot([A, A], [-0.45, lev + 0.05], color=S.IAM, lw=0.4)
        ax.text(A + {"right": -0.004, "left": 0.004, "center": 0}[ha] * span, lev, f"{n}" + (f", {note}" if note else "") + f"\n{m} M$_\\odot$, A = {A:.2f}", ha=ha, va="top", fontsize=5.8, color=S.IAM)
fig.text(0.5, 0.0, "A = mass / the typical remnant of the same kind (each row its own scale)", ha="center", fontsize=6.5)
S.save(fig, "part2", "star_gauge")
