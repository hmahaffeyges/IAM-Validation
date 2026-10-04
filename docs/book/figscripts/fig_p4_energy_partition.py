"""part6/fig_energy_partition (Chapter 'Two ledgers and the virial balance', Figure fig:partition).

(a) Virial theorem for an inverse-square force: 2K + U = 0, so U = -2K and E = K + U = -K. (b) Slow contraction of a self-gravitating
body: of the released |dU|, half goes into heat (dK = -dU/2) and half is radiated (-dE = -dU/2). (c) Smarr relation for a Schwarzschild
black hole, M c^2 = 2 T_H S, so T_H S = M c^2 / 2 (Smarr 1973). Derived; no data. Run: python docs/book/figscripts/fig_p4_energy_partition.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()

fig, axs = plt.subplots(1, 3, figsize=(S.TEXTW, 2.4), gridspec_kw=dict(wspace=0.7))
K = 1.0; U = -2 * K; E = K + U
axs[0].bar(["K", "U", "E"], [K, U, E], color=[S.GOLD, S.IAM, S.GR], width=0.6); axs[0].axhline(0, color="black", lw=0.6)
axs[0].set_ylabel("energy (units of K)"); axs[0].set_title("Virial: $U=-2K$, $E=-K$")
axs[1].bar(["into heat\n($\\Delta K$)", "radiated\n($-\\Delta E$)"], [0.5, 0.5], color=[S.GOLD, S.ALT2], width=0.6)
axs[1].set_ylim(0, 0.7); axs[1].set_ylabel(r"fraction of $|\Delta U|$ released"); axs[1].set_title("Contraction: half and half")
axs[2].bar(["$Mc^2$", "$T_HS$"], [1.0, 0.5], color=[S.GR, S.IAM], width=0.6)
axs[2].set_ylim(0, 1.15); axs[2].set_ylabel("units of $Mc^2$"); axs[2].set_title("Horizon: $Mc^2=2T_HS$")
for a, l in zip(axs, "abc"):
    S.panel_letter(a, l, dx=-0.32, dy=1.10)
S.save(fig, "part6", "fig_energy_partition")
