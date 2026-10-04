"""p4_16 fig_cell_count_floor: the cell-count floor, Eq. eq:cellcount, written per copy of a site.
sigma = sqrt(beta (1 - beta) / n) with n = 2N copies for N diploid genomes; beta = 0.7.
A plasma genome equivalent is one haploid genome, i.e. one copy of each site (Sender et al. 2024, eLife 12:RP89321,
about 1e3 per mL), so a draw gives about 1e3-1e4 copies. Array noise 0.01 and 0.02 drawn for comparison.  [calculated]
Run: python docs/book/figscripts/fig_p4_cellcount.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import _bookstyle as S
import matplotlib.pyplot as plt

S.apply()
BETA = 0.7
n = np.logspace(1, 6, 200)
sig = np.sqrt(BETA * (1 - BETA) / n)
for nn in (1e2, 1e3, 1e4):
    print(f"n = {nn:.0e} copies: sigma = {np.sqrt(BETA * (1 - BETA) / nn):.4f}")
fig, ax = plt.subplots(figsize=(0.8 * S.TEXTW, 2.6))
ax.plot(n, sig, color="k", lw=1.4, label=r"copy-number (binomial) floor, $\beta$ = 0.7")
for y in (0.01, 0.02):
    ax.axhline(y, color=S.GR, lw=0.8, ls=":"); ax.text(15, y * 1.08, f"array noise {y:.2f}", fontsize=7, color=S.GR, va="bottom")
for x, lab in ((1e3, "1,000 copies"), (1e4, "~10,000 copies\n(plasma draw)")):
    ax.axvline(x, color=S.ALT, lw=1.0); ax.text(x * 1.15, 0.09, lab, fontsize=7, color=S.ALT, va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(7, 1.5e6); ax.set_ylim(2e-4, 0.15)
ax.set_xlabel("copies of the site in the specimen ($2N$ for $N$ diploid genomes)")
ax.set_ylabel(r"SD of measured $\beta$ at one site")
ax.legend(loc="lower left", fontsize=7)
ax.set_title("The cell-count floor: the methylome's cosmic variance")
S.save(fig, "part6", "fig_cell_count_floor")
