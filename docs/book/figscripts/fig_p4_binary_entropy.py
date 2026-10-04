"""part6/fig_binary_entropy (Chapter 'The methylome as an encoding surface', Figure fig:binent).

H(beta) = -beta log2 beta - (1-beta) log2(1-beta) (the chapter's Eq. eq:H). A site held near 0 or 1 carries little entropy; a coin flip
carries one bit. H is symmetric, so a site held unmethylated and a site held methylated reach the same entropy (the two channels):
beta = 0.1 and 0.9 both carry H = 0.469 bits. Both moves toward one half raise H. Derived; no data.
Run: python docs/book/figscripts/fig_p4_binary_entropy.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()

H = lambda b: -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
b = np.linspace(1e-6, 1 - 1e-6, 801)
fig, ax = plt.subplots(figsize=(0.95 * S.TEXTW, 2.6))
ax.plot(b, H(b), color="black", lw=1.3)
ax.plot([0.5], [1.0], "o", color=S.GR, ms=4); ax.text(0.5, 1.03, "one bit: coin flip, $\\beta=0.5$", ha="center", va="bottom", fontsize=6.5, color=S.GR)
h1 = H(0.1)
ax.plot([0.1, 0.9], [h1, h1], color=S.IAM, lw=0.8, ls=":")
ax.plot([0.1], [h1], "o", color=S.ALT, ms=5); ax.plot([0.9], [h1], "o", color=S.DATA, ms=5)
ax.text(0.5, h1 + 0.02, f"same entropy, {h1:.3f} bits", ha="center", va="bottom", fontsize=6.5, color=S.IAM)
ax.annotate("", (0.27, H(0.27)), (0.12, H(0.12) - 0.01), arrowprops=dict(arrowstyle="->", color=S.ALT, lw=0.9))
ax.annotate("", (0.73, H(0.73)), (0.88, H(0.88) - 0.01), arrowprops=dict(arrowstyle="->", color=S.DATA, lw=0.9))
ax.text(0.16, 0.10, "held unmethylated:\ngaining methylation raises $H$", fontsize=6.5, color=S.ALT, va="bottom")
ax.text(0.84, 0.10, "held methylated:\nlosing methylation raises $H$", fontsize=6.5, color=S.DATA, va="bottom", ha="right")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.12)
ax.set_xlabel(r"methylation fraction $\beta$ at a site"); ax.set_ylabel(r"entropy $H(\beta)$ (bits)")
ax.set_title("A site held near 0 or 1 carries little entropy; a coin flip carries one bit")
S.save(fig, "part6", "fig_binary_entropy")
