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
_h, _k = 6.62607015e-34, 1.380649e-23
_peq = 1 / (1 + math.exp(_h * 5e9 / (_k * 0.035)))
QF = _peq * 40e-9 / 68e-6 / e(1e-3)          # thermal floor of the transmon reading on its own gauge (6.2e-4)

G = [
    dict(title="Cell · neutrophil · IAM-A", sub="copy error at each CpG, read on single molecules", lim=(1/5.6, 5.6), hmin=1/5.6, cmap=cm_cell,
         band=(0.95, 1.05), hmin_lab="H$_{\\min}$ 1×10⁻⁷, off the left edge\n(thermal kicks win against one ATP)", fail=1/(1.1492*0.20433), fail_lab=f"surface full {1/(1.1492*0.20433):.2f}",
         zones=[(1.35, 4.0, "more error · breach, cancer: to be measured →")],
         marks=[(1.0, "healthy granulocytes, 3 donors", "#1d7a46"), (1/1.1492, "   H$_{\\rm ref}$ 0.870, average healthy cell type", "#555555")],
         ticks=[0.3, 0.5, 1, 2, 4.26]),
    # Qubit: transmon reading of Chapter ch:ascoreqc (5 GHz, own 35 mK, T1 68 us, 40 ns gate, p = 1e-3, illustrative);
    # thermal floor p_eq t_g / T1 with p_eq = 1/(1+exp(hf/kT)) (Chapter ch:qplatforms).
    dict(title="Qubit · transmon reading · two-qubit gate", sub="ε = −ln(1 − p); 5 GHz at its own 35 mK, T1 68 µs, 40 ns gate, p = 10⁻³",
         lim=(1/12000, 12000), hmin=QF, cmap=cm_dev,
         band=None, hmin_lab=f"H$_{{\\min}}$ {QF:.1e}\nthermal floor".replace("e-0", "e-"), fail=e(1e-2)/e(1e-3), fail_lab="error correction\nfails · p = 1 %",
         zones=[(1/12000, QF, "thermal\nkicks win"), (QF, 1, "←  room above the floor"), (1, e(1e-2)/e(1e-3), "more gate error  →")],
         marks=[(1.0, "as built  ·  p = 10$^{-3}$", "#1d7a46")],
         ticks=[0.001, 0.01, 0.1, 1, 10, 100, 1000]),
    dict(title="Chip · AMD Ryzen 9 9950X · switching energy", sub="E = TDP / (N f); the chip as built against k_B T_j ln 2", lim=(1/6000, 6000), hmin=1/576.0, cmap=cm_dev,
         band=None, hmin_lab="H$_{\\min}$ 0.0017\nLandauer floor k$_B$T ln 2", fail=None,
         zones=[(1/6000, 1/576, "thermal\nkicks win"), (1/576, 1, "←  closer to Landauer"), (1, 6000, "more energy per switch  →")],
         marks=[(1.0, "9950X (2024)\nas built", "#1d7a46")],
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

fig = draw_onegauge(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures", "part7", "one_gauge"))