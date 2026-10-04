"""Figures for ch:landauer and ch:fixedorigin (Part 4).

    python docs/book/figscripts/fig_p4_landauer_metrology.py

Outputs (docs/book/figures/part4/): fig_cell_budget, fig_division_floor, fig_p4_02_operating,
fig_p4_15b_transfer, fig_p4_15b_lowsignal (.pdf + .png).
Data read from docs/book/figscripts/cell_data/ (copied there from the retired kit/results/; PROC_TARE_01_per_array.parquet,
FINDING_GSE125105_controls.csv). Constants recomputed as in Biological_Physics/Landauer_Metrology/verify_landauer_metrology.py.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _bookstyle as bs

bs.apply()
KIT = bs.REPO / "docs/book/figscripts/cell_data"   # copied from Biological_Physics/MethylPhys/kit/results/ (retired 2026-10-03)
kB, NA = 1.380649e-23, 6.02214076e23
Tb, ln2 = 310.15, math.log(2)
M = 54000 / (kB * NA * Tb)
EBIT = kB * Tb * ln2
EATP = 54000 / NA
N_CPG = 28_217_448

# 1. Energy per event at body temperature, in k_B T
fig, ax = plt.subplots(figsize=(0.62 * bs.TEXTW, 1.9))
labels = ["Landauer floor,\none bit", "holding energy\nper site (molecules)", "one ATP\n(cytosolic)"]
vals = [ln2, 3.41, M]
cols = [bs.GR, bs.DATA, bs.IAM]
ax.barh(range(3), vals, color=cols, height=0.55)
for i, v in enumerate(vals):
    ax.text(v + 0.4, i, f"{v:.2f} $k_BT$  ({v/ln2:.2f} in Landauer units)", va="center", fontsize=6)
ax.set_yticks(range(3), labels)
ax.set_xlim(0, 34)
ax.set_xlabel(r"energy per event ($k_BT$ at 310.15 K)")
fig.tight_layout()
bs.save(fig, "part4", "fig_cell_budget")

# 2. Landauer floor for one rewrite of the pattern vs number of sites counted
fig, ax = plt.subplots(figsize=(0.62 * bs.TEXTW, 2.3))
N = np.logspace(5, 8, 200)
ax.loglog(N, N * EBIT / EATP, color=bs.IAM, label=r"Landauer floor $N\,k_BT\ln2$")
ax.loglog(N, 0.70 * N, color=bs.DATA, ls="--", label="chemical cost, ≥ 1 ATP per methylated site (70 %)")
for n, txt in ((N_CPG, "all CpGs\n28,217,448"),):
    ax.axvline(n, color=bs.LIGHT, lw=0.6)
    ax.plot([n], [n * EBIT / EATP], "o", color=bs.IAM, ms=3)
    ax.text(n * 0.85, 2.2e3, txt, ha="right", fontsize=6)
ax.text(N_CPG * 1.1, N_CPG * EBIT / EATP * 0.55, f"{N_CPG*EBIT/EATP:.2e} ATP".replace("e+05", "×10⁵"), fontsize=6)
ax.set_xlabel("sites counted, $N$")
ax.set_ylabel("ATP equivalents (54 kJ/mol)")
ax.set_ylim(1e3, 2e8)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
bs.save(fig, "part4", "fig_division_floor")

# 3. The operating ratio for three substrates
fig, ax = plt.subplots(figsize=(0.62 * bs.TEXTW, 1.9))
rows = [("aluminium transmon\nat its gap temperature", ln2, ln2), ("human cell, one ATP\nat 310 K", M, M),
        ("CMOS logic, AMD 9950X\nat 348.15 K (range)", 399, 411)]
for i, (lab, lo, hi) in enumerate(rows):
    ax.plot([lo, hi], [i, i], color=bs.IAM, lw=4, solid_capstyle="butt")
    ax.plot([0.5 * (lo + hi)], [i], "o", color=bs.IAM, ms=3)
    ax.text(hi * 1.25, i, (f"{lo:.3f}" if lo < 1 else f"{lo:.2f}" if lo < 100 else f"{lo:.0f}–{hi:.0f}")
            + f"  ({lo/ln2:.2f}" * (lo < 100) + (f"  ({lo/ln2:.0f}–{hi/ln2:.0f}" if lo >= 100 else "") + " in Landauer units)",
            va="center", fontsize=6)
ax.axvline(ln2, color=bs.LIGHT, lw=0.6)
ax.set_xscale("log")
ax.set_xlim(0.3, 3e4)
ax.set_yticks(range(3), [r[0] for r in rows])
ax.set_ylim(-0.6, 2.6)
ax.set_xlabel(r"$\mathcal{M}=E_{\rm drive}/k_BT$")
fig.tight_layout()
bs.save(fig, "part4", "fig_p4_02_operating")

# 4. Three laboratories on one scale
d = pd.read_parquet(KIT / "PROC_TARE_01_per_array.parquet")
labs = [("GSE87571", "Uppsala\n(n = 732)"), ("GSE42861", "Karolinska\n(n = 12)"), ("GSE111629", "UCLA\n(n = 12)")]
fig, ax = plt.subplots(figsize=(0.6 * bs.TEXTW, 2.3))
ax.axhspan(0.95, 1.05, color=bs.ALT, alpha=0.12, lw=0)
ax.axhline(1.0, color=bs.GR, lw=0.6)
rng = np.random.default_rng(0)
for i, (lab, nm) in enumerate(labs):
    x = d.loc[d.lab == lab, "A_raw"].to_numpy()
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(x)), x, s=3 if len(x) > 100 else 8, color=bs.DATA, alpha=0.45, lw=0)
    ax.plot([i - 0.25, i + 0.25], [np.median(x)] * 2, color=bs.IAM, lw=1.4)
    ax.text(i + 0.28, np.median(x), f"{np.median(x):.3f}", va="center", fontsize=6, color=bs.IAM)
ax.set_xticks(range(3), [n for _, n in labs])
ax.set_xlim(-0.5, 2.7)
ax.set_ylabel("immune $A$, mapped (identity sites, 450K)")
ax.text(2.65, 1.045, "Normal 0.95–1.05", ha="right", va="top", fontsize=6, color=bs.ALT)
fig.tight_layout()
bs.save(fig, "part4", "fig_p4_15b_transfer")

# 5. The low-signal laboratory: control-probe intensities and probes at background
c = pd.read_csv(KIT / "FINDING_GSE125105_controls.csv").groupby("lab").median(numeric_only=True)
order = [("GSE87571", "Uppsala"), ("GSE42861", "Karolinska"), ("GSE111629", "UCLA"), ("GSE125105", "Munich")]
feats = [("nonpoly_G", "non-polym. G"), ("nonpoly_R", "non-polym. R"), ("bsII_R", "bisulfite II R"),
         ("hyb_G", "hybridisation G"), ("neg_G", "negative G")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(bs.TEXTW, 2.3), gridspec_kw=dict(width_ratios=[3, 1.2]))
w = 0.2
colors = [bs.IAM, bs.SKY, bs.ALT, bs.DATA]
for j, (lab, nm) in enumerate(order):
    a1.bar(np.arange(len(feats)) + (j - 1.5) * w, [c.loc[lab, f] for f, _ in feats], w, color=colors[j], label=nm)
a1.set_yscale("log")
a1.set_xticks(range(len(feats)), [n for _, n in feats])
a1.set_ylabel("median intensity (3 arrays)")
a1.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
bs.panel_letter(a1, "a")
a2.bar(range(4), [c.loc[l, "poobah_fail%"] for l, _ in order], color=colors)
a2.set_xticks(range(4), [n for _, n in order], rotation=30, ha="right")
a2.set_ylabel("probes at background (%)")
bs.panel_letter(a2, "b", dx=-0.3)
fig.tight_layout()
bs.save(fig, "part4", "fig_p4_15b_lowsignal")
