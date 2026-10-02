"""Shared style and helpers for the book figures (Parts 1, 2, 3 and 5).

Every figure script in this folder imports this module. Run any script from any directory:
    python docs/book/figscripts/fig_p1_ladder.py
Outputs go to docs/book/figures/<part>/<name>.pdf and a .png preview.
Style: role-mapped font ladder (8/7/6 pt), open frame, frameless legends, Type-42 fonts,
CVD-safe palette (Okabe-Ito). Figures are drawn at their printed width (text width 6.2 in).
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # docs/book/figscripts
BOOK = HERE.parent                              # docs/book
REPO = BOOK.parent.parent                       # repository root
TEXTW = 6.2                                     # inches, A4 with 2.6 cm margins

# Okabe-Ito colours. One meaning per colour across the book figures.
IAM = "#0072B2"       # the informational term / IAM prediction (focal series)
GR = "#4D4D4D"        # general relativity / LambdaCDM / standard comparator
DATA = "#D55E00"      # measurements
ALT = "#009E73"       # a second model form or second sector
ALT2 = "#CC79A7"      # a third series
SKY = "#56B4E9"       # light tint of the focal hue
GOLD = "#E69F00"      # a fourth series
LIGHT = "#BDBDBD"     # reference lines, bands


def apply():
    """Publication rcParams (same mechanics as the figure-style rules)."""
    rc = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.size": 1.6, "ytick.minor.size": 1.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.titleweight": "normal", "axes.titlelocation": "left",
        "lines.linewidth": 1.2, "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans",
    }
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.10, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def sci(x, digits=2):
    """LaTeX-free scientific notation for annotations, e.g. 1.5x10^77 -> '1.5×10⁷⁷'."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10**e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.{digits-1}f}×10{sup}"


def overlaps(fig):
    """Render-then-verify bbox check: overlapping visible text boxes, and text crossing a spine
    (tick labels on their own spine excluded). Returns a list of string pairs."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible() and t.get_alpha() != 0]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes for s in ax.spines.values() if s.get_visible()]
    tick = {ax: set(ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both")) for ax in fig.axes}
    alltick = set().union(*tick.values()) if tick else set()
    out = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if ba.overlaps(bb):
                out.append((a.get_text()[:30], b.get_text()[:30]))
    for t, bt in texts:
        if t in alltick:
            continue
        for s, bs in spines:
            if bt.overlaps(bs) and t not in tick.get(s.axes, set()):
                out.append((t.get_text()[:30], "spine"))
    return out


def save(fig, part, name):
    """Save PDF + PNG preview into docs/book/figures/<part>/ and report the overlap check."""
    d = BOOK / "figures" / part
    d.mkdir(parents=True, exist_ok=True)
    ov = overlaps(fig)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"saved figures/{part}/{name}.pdf  overlaps={len(ov)}" + ("" if not ov else f"  {ov}"))
    return ov
