"""fig_sky_cd: genome-distance correlation for single neutrophil arrays (box job f754cf20, remote_jobs/gate/cd_neut.py)."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
T = pd.read_csv("cd_neut.csv"); T["d"] = np.sqrt(T.d_lo * T.d_hi)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for a, g in T[T.quantity == "beta"].groupby("array"):
    axs[0].plot(g.d, g.C, color="#4c72b0", lw=0.8, alpha=0.8)
axs[0].set_xscale("log"); axs[0].set_xlabel("distance between two CpGs (bp)"); axs[0].set_ylabel("correlation of $\\beta$")
axs[0].set_title("Within one healthy neutrophil (6 arrays)", loc="left"); axs[0].axhline(0, color="black", lw=0.4)
axs[0].text(-0.13, 1.04, "a", transform=axs[0].transAxes, fontsize=10, fontweight="bold")
for q, col, lab in (("z_healthy", "#4c72b0", "healthy array"), ("z_local5pct", "#c44e52", "same array, damage in 10 regions")):
    g = T[T.quantity == q]; axs[1].plot(g.d, g.C, color=col, lw=1.2, label=lab)
axs[1].set_xscale("log"); axs[1].set_xlabel("distance between two CpGs (bp)"); axs[1].set_ylabel("correlation of residual $z$")
axs[1].set_title("Residual against five healthy arrays", loc="left"); axs[1].axhline(0, color="black", lw=0.4); axs[1].legend(frameon=False, fontsize=7)
axs[1].text(-0.13, 1.04, "b", transform=axs[1].transAxes, fontsize=10, fontweight="bold")
fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.17, wspace=0.28)
fig.savefig("fig_sky_cd.pdf"); fig.savefig("fig_sky_cd.png"); print("ok")
