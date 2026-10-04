"""On the shoulders of giants: who found each piece, and when.

Every year on the figure is read from the bibliography files (docs/book/iam.bib);
nothing is typed in. Rows are the threads of the chapter p0_giants.tex. Output: figures/front/fig_giants_timeline.{pdf,png}.
Run: python docs/book/figscripts/fig_p0_giants_timeline.py
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bookstyle as bs
import matplotlib.pyplot as plt

THREADS = [  # (row label, colour, [(bib key, label on the figure)])
    ("gravity", bs.GR, [("Clausius1870", "Clausius"), ("Einstein1916", "Einstein"), ("Hubble1929", "Hubble")]),
    ("the surface", bs.IAM, [("Bekenstein1973", "Bekenstein"), ("Hawking1975", "Hawking"),
                             ("GibbonsHawking1977", "Gibbons–Hawking"), ("Susskind1995", "Susskind")]),
    ("the price", bs.DATA, [("Shannon1948", "Shannon"), ("Landauer1961", "Landauer"), ("Bennett1982", "Bennett"),
                            ("Berut2012", "Bérut et al.")]),
    ("the engine", bs.ALT, [("Jacobson1995", "Jacobson"), ("CaiKim2005", "Cai–Kim")]),
    ("the record", bs.ALT2, [("Zurek1981", "Zurek"), ("Diosi1987", "Diósi"), ("Penrose1996", "Penrose"),
                             ("Zurek2009", "quantum Darwinism")]),
    ("the sky", bs.SKY, [("AlpherHerman1948", "Alpher–Herman"), ("PenziasWilson1965", "Penzias–Wilson"),
                         ("Smoot1992", "COBE"), ("Planck2018VI", "Planck")]),
    ("the cell", bs.GOLD, [("Hotchkiss1948", "Hotchkiss"), ("HollidayPugh1975", "Holliday–Pugh"),
                           ("Bestor1988", "Bestor"), ("Frommer1992", "Frommer"), ("Sanchez2016", "Sanchez–Mackenzie"),
                           ("Loyfer2023", "Loyfer et al.")]),
]


LEVELS = (0.16, -0.20, 0.44, -0.48)  # label offsets in row units; each label's year identifies its dot


def bib_years():
    years = {}
    for name in ("iam.bib",):
        text = (bs.BOOK / name).read_text(encoding="utf-8")
        for m in re.finditer(r"@\w+\{([^,\s]+),(.*?)\n\}", text, re.S):
            y = re.search(r"\byear\s*=\s*\{?(\d{4})", m.group(2))
            if y:
                years[m.group(1)] = int(y.group(1))
    return years


# Reprints whose bibliography year is the reprint year are not used on this figure (Schwarzschild, Wheeler, Schrodinger).

def main():
    bs.apply()
    years = bib_years()
    fig, ax = plt.subplots(figsize=(bs.TEXTW, 4.4))
    ax.set_xlim(1860, 2030)
    fig.canvas.draw()
    n = len(THREADS)
    allyears = []
    for i, (row, col, items) in enumerate(THREADS):
        y0 = n - 1 - i
        ys = [years[k] for k, _ in items]
        allyears += ys
        ax.plot([min(ys), max(ys)], [y0, y0], color=col, lw=0.8, alpha=0.6, zorder=1)
        ax.scatter(ys, [y0] * len(ys), s=14, color=col, zorder=2, edgecolor="white", linewidth=0.4)
        placed = []
        for (k, lab), yr in sorted(zip(items, ys), key=lambda t: t[1]):
            for dy in LEVELS:  # first level whose box clears every label already placed in this row
                t = ax.text(yr, y0 + dy, f"{lab} {yr}", fontsize=5.6, ha="center",
                            va="bottom" if dy > 0 else "top", color="#222222")
                bb = t.get_window_extent(fig.canvas.get_renderer())
                if not any(bb.overlaps(o) for o in placed):
                    placed.append(bb)
                    break
                t.remove()
        ax.text(1858, y0, row, fontsize=7, ha="right", va="center", color=col)
    ax.set_xlim(1860, 2030)
    ax.set_ylim(-0.8, n - 0.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("year of publication")
    span = max(allyears) - min(allyears)
    print(f"years {min(allyears)}-{max(allyears)} ({span} years), {len(allyears)} works")
    return fig


if __name__ == "__main__":
    fig = main()
    bs.save(fig, "front", "fig_giants_timeline")
