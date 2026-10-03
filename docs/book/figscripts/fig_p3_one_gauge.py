import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import math
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

NAVY = "#1b2a41"
HEALTH = "#1d7a46"
FLOOR = "#5b5b5b"
FAIL = "#b2182b"

cm_cell = LinearSegmentedColormap.from_list("c", [(0, "#5b8def"), (0.42, "#9fd3c7"), (0.5, "#2fbf71"), (0.58, "#9fd3c7"), (0.78, "#f6b26b"), (1, "#d6455d")])
cm_dev = LinearSegmentedColormap.from_list("d", [(0, "#3fa7d6"), (0.5, "#2fbf71"), (0.75, "#f6b26b"), (1, "#d6455d")])

e = lambda p: -math.log(1 - p)

G = [
    dict(title="Cell · neutrophil · IAM-A", sub="copy error at each CpG, read on single molecules", lim=(1/5.6, 5.6), hmin=1/1.099, cmap=cm_cell,
         band=(0.95, 1.05), hmin_lab="H$_{\\min}$ 0.910\nphysics floor ε$_0$", fail=4.452, fail_lab="surface full 4.45",
         zones=[(1/5.6, 1/1.099, "thermal\nkicks win"), (1.35, 4.0, "more error · breach, cancer: to be measured →")],
         marks=[(1.0, "healthy granulocytes, 3 donors", "#1d7a46")],
         ticks=[0.3, 0.5, 1, 2, 4.45]),
    dict(title="Qubit · Quantinuum Helios · two-qubit gate", sub="gate error ε = −ln(1 − p)", lim=(0.07, 1/0.07), hmin=e(1e-4)/e(7.9e-4), cmap=cm_dev,
         band=None, hmin_lab="H$_{\\min}$ 0.13\nmaterial floor (Ba$^+$)", fail=e(1e-2)/e(7.9e-4), fail_lab="error correction\nfails · p = 1 %",
         zones=[(0.07, e(1e-4)/e(7.9e-4), "thermal\nkicks win"), (e(1e-4)/e(7.9e-4), 1, "←  better than as built"), (1, e(1e-2)/e(7.9e-4), "more gate error  →")],
         marks=[(1.0, "as built  ·  p = 7.9 × 10$^{-4}$", "#1d7a46")],
         ticks=[0.1, 0.3, 1, 3, 10]),
    dict(title="Chip · AMD Ryzen 9 9950X · switching energy", sub="E = TDP / (N f); earlier AMD chips on the same scale", lim=(1/6000, 6000), hmin=1/576.0, cmap=cm_dev,
         band=None, hmin_lab="H$_{\\min}$ 0.0017\nLandauer floor k$_B$T ln 2", fail=None,
         zones=[(1/6000, 1/576, "thermal\nkicks win"), (1/576, 1, "←  closer to Landauer"), (1, 6000, "more energy per switch  →")],
         marks=[(1.0, "9950X (2024)\nas built", "#1d7a46"), (1650/576, "   Ryzen 7 1800X (2017)", "#c46a1b"), (105002/576, "Athlon 64\n(2003)", "#b23a48")],
         ticks=[0.001, 0.01, 0.1, 1, 10, 100, 1000]),
]

def draw_onegauge(path):
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 6.3))
    fig.suptitle("One gauge for every record IAM reads", fontsize=11, fontweight="bold", color=NAVY, y=0.995)
    fig.text(0.5, 0.955, "1 = healthy, in the middle   ·   H$_{\\min}$ = the floor where thermal kicks win; nothing reads below it   ·   above 1 = more error",
             ha="center", fontsize=7.5, color="0.35", style="italic")
    for i, (ax, g) in enumerate(zip(axs, G)):
        lo, hi = g["lim"]
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1.0, 1.0)
        edges = np.logspace(np.log10(lo), np.log10(hi), 501)
        mid = np.sqrt(edges[1:] * edges[:-1])
        u = (np.log(mid) - np.log(lo)) / (np.log(hi) - np.log(lo))
        ax.pcolormesh(edges, [0, 0.5], u[None, :], cmap=g["cmap"], shading="flat", zorder=1, rasterized=True)
        ax.add_patch(Rectangle((lo, 0), g["hmin"] - lo, 0.5, facecolor="#2d2d2d", alpha=0.85, hatch="////", edgecolor="#888", lw=0, zorder=2))
        if g["band"]:
            ax.add_patch(Rectangle((g["band"][0], 0), g["band"][1] - g["band"][0], 0.5, fill=False, edgecolor="white", lw=1.8, zorder=3))
        ax.plot([1, 1], [-0.04, 0.58], color="white", lw=4.5, zorder=4)
        ax.plot([1, 1], [-0.04, 0.58], color="#1d7a46", lw=2.2, zorder=5)
        ax.plot([g["hmin"]] * 2, [-0.04, 0.58], color="black", lw=2.4, zorder=5)
        for a, b, lab in g["zones"]:
            ax.text(math.sqrt(a * b), 0.25, lab, ha="center", va="center", fontsize=7 if "\n" in lab else 7.5,
                    color="white", fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#00000066")])
        ax.annotate(g["hmin_lab"].replace("\n", "  ·  "), xy=(g["hmin"], 0.58), xytext=(2, 3),
                    textcoords="offset points", ha="left", va="bottom", fontsize=7, color="black", fontweight="bold")
        if g["fail"]:
            ax.plot([g["fail"]] * 2, [-0.04, 0.58], color="#b2182b", lw=2, ls=(0, (3, 2)), zorder=5)
            ax.annotate(g["fail_lab"].replace("\n", " "), xy=(g["fail"], 0.58), xytext=(-2, 3),
                        textcoords="offset points", ha="right", va="bottom", fontsize=7, color="#b2182b", fontweight="bold")
        for v in g["ticks"]:
            ax.plot([v, v], [0, -0.06], color="0.45", lw=0.8)
            ax.text(v, -0.09, f"{v:g}", ha="center", va="top", fontsize=6.8, color="0.4")
        for x, lab, c in g["marks"]:
            ax.plot([x], [-0.42], marker="^", ms=8, color=c, zorder=7)
            if lab.startswith("   "):
                ax.annotate(lab.strip(), xy=(x, -0.42), xytext=(7, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=6.8, color=c)
            else:
                ax.text(x, -0.55, lab, ha="center", va="top", fontsize=6.8, color=c)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)
        ax.set_yticks([])
        [ax.spines[s].set_visible(False) for s in ax.spines]
        ax.text(0, 0.99, g["title"], transform=ax.transAxes, fontsize=9, fontweight="bold", color=NAVY, va="top")
        ax.text(1, 0.99, g["sub"], transform=ax.transAxes, fontsize=6.8, color="0.42", va="top", ha="right", style="italic")
    fig.text(0.5, 0.01, "A = reading ÷ the same system's healthy reading  (log scale, each gauge centred on 1)",
             ha="center", fontsize=8, color=NAVY)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05, hspace=0.18)
    fig.savefig(path + ".png", dpi=220)
    fig.savefig(path + ".pdf")
    return fig

fig = draw_onegauge(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "part3", "one_gauge"))