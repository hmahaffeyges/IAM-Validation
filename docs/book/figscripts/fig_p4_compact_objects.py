"""part6/fig_compact_objects (Chapter 'Translation', Figure fig:compact).

Load ratio = core mass / saturation mass: 1.44 Msun for white dwarfs (Chandrasekhar 1931) and ~2.3 Msun for neutron stars (the book's
TOV value; Rezzolla et al. 2018 give M_TOV <~ 2.16 +0.17/-0.15). Masses as tabulated in Chapter ch:blackholes: the Sun's future white dwarf
0.54 (Sackmann et al. 1993), Procyon B 0.592 (Bond et al. 2015), Sirius B 1.02 (Bond et al. 2017), IK Pegasi B 1.15 (Landsman et al. 1993),
a typical neutron star 1.4, PSR J0740+6620 2.08 (Fonseca et al. 2021). Run: python docs/book/figscripts/fig_p4_compact_objects.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()

MCH, MTOV = 1.44, 2.3
OBJ = [("Sun's future white dwarf", 0.54, MCH), ("Procyon B", 0.592, MCH), ("Sirius B", 1.02, MCH), ("IK Pegasi B", 1.15, MCH),
       ("typical neutron star", 1.4, MTOV), ("PSR J0740+6620", 2.08, MTOV)]
fig, ax = plt.subplots(figsize=(0.52 * S.TEXTW + 0.6, 2.5))
for i, (n, m, ms) in enumerate(OBJ):
    r = m / ms
    ax.plot([0, r], [i, i], color=S.LIGHT, lw=0.8); ax.plot([r], [i], "o", color=S.IAM if ms == MCH else S.ALT2, ms=5)
    ax.text(r + 0.03, i, f"{r:.2f}", va="center", fontsize=6.5) if r < 0.85 else ax.text(r, i + 0.22, f"{r:.2f}", va="bottom", ha="center", fontsize=6.5)
    print(f"{n}: {m}/{ms} = {r:.3f}")
ax.axvline(1, color=S.DATA, lw=1)
ax.text(1.02, 2.5, "saturation mass\n(Chandrasekhar / TOV)", fontsize=6, color=S.DATA, va="center")
ax.set_yticks(range(len(OBJ))); ax.set_yticklabels([o[0] for o in OBJ], fontsize=6.5)
ax.set_xlim(0, 1.45); ax.set_ylim(-0.6, len(OBJ) - 0.4); ax.set_xlabel("core mass / saturation mass")
ax.set_title("Compact objects on their own load ratio")
S.save(fig, "part6", "fig_compact_objects")
