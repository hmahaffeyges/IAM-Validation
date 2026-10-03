#!/usr/bin/env python3
"""Figure: estimated vs true fraction per cell group and method. Run from the zip root: python code/make_figure.py"""
import json, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
W = pd.read_csv("results/fractions_groups_wide.csv"); S = pd.read_csv("results/samples_and_truth.csv").set_index("gsm")
M = pd.read_csv("results/metrics_by_set_selected_methods.csv"); ch = json.load(open("results/nilc_choice.json"))
SHORT = {"NNLS8": "NNLS8", "ATLAS_a": "Atl-a", "ATLAS_b": "Atl-b", "ATLAS_c": "Atl-c", "ATLAS_e": "Atl-e", ch["NILC_atlas_chosen"]: "NILC-c", ch["NILC_e_chosen"]: "NILC-e"}
inv = {v: k for k, v in SHORT.items()}; figm = list(SHORT.values())
W = W[W.set.isin(["FACS", "MIX18", "MIX22", "MIX12", "LONG"]) & W.method.isin(SHORT)]
grps = ["NEU", "MONO", "B", "NK", "CD4T", "CD8T", "EOS", "BASO"]
setcol = {"FACS": "#000000", "LONG": "#7f7f7f", "MIX18": "#0072B2", "MIX22": "#E69F00", "MIX12": "#009E73"}
fig, axs = plt.subplots(len(figm), len(grps), figsize=(9, 7.9), gridspec_kw=dict(wspace=0.32, hspace=0.28))
for j, g in enumerate(grps):
    L = min(1.0, np.nanmax(np.r_[S[g].dropna().values, W[g].dropna().values]) * 1.08)
    for i, m in enumerate(figm):
        ax = axs[i, j]; X = W[W.method == inv[m]]; ax.plot([0, L], [0, L], color="#bbbbbb", lw=0.6)
        for st, c in setcol.items():
            Y = X[X.set == st]; t = S.loc[Y.gsm, g].values.astype(float); e = Y[g].values; ok = np.isfinite(t) & np.isfinite(e)
            ax.scatter(t[ok], e[ok], s=7, color=c, lw=0)
        mm = M[(M.m == m) & (M.group == g) & M.set.isin(["FACS", "MIX18", "MIX22", "MIX12"])]
        if len(mm): ax.text(0.04, 0.96, f"RMSE<={mm.rmse.max():.3f}", transform=ax.transAxes, va="top", fontsize=6)
        ax.set_xlim(-0.02 * L, L); ax.set_ylim(-0.08 * L, 1.02 * L)
fig.savefig("fig_est_vs_true_rebuilt.png", dpi=200)
