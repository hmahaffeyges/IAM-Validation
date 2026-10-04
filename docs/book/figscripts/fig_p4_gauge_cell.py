"""part4/fig_gauge_cell (Chapter 'One gauge for a cell', Figure fig:gauge).

One gauge, four readings, for neutrophils. Each row: the healthy reference 1 in the middle; the left half runs linearly from the floor to 1,
the right half on a log scale from 1 to the far end. Values:
  * Normal 0.95-1.05 (CANON Normal_band, Met-A and IAM-A);
  * IAM-A floor H_min = 1/P_neutrophil (CANON P_neutrophil_IAM_A); full surface 1/(P H(eps0)) with eps0 = CANON eps0_meth;
  * Met-A floor not yet defined; full surface 1/H_ref with H_ref = CANON Met_A_floor_EPIC_neutrophil (every identity site at a coin flip);
  * Met-A C-score: floor 0, far end = block size / healthy baseline (neutrophil_reference_v1_1.json: 50 / 1.1104);
  * bars: measured ranges over six held-out healthy arrays and the same arrays with known damage, read from the chain's sky run
    Biological_Physics/MethylPhys/reference_floors_v1/sky/sky_neut6_stats.csv (Chapter ch:skytools);
  * IAM-A C-score: not built.
Run: python docs/book/figscripts/fig_p4_gauge_cell.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()
CANON = json.load(open(S.REPO / "CANON" / "iam_canon.json"))["constants"]
kB = CANON["k_B"]["value"]

import csv
from matplotlib.patches import Rectangle
H = lambda b: -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
P = CANON["P_neutrophil_IAM_A"]["value"]; eps0 = CANON["eps0_meth"]["value"]; Href = CANON["Met_A_floor_EPIC_neutrophil"]["value"]
lo_n, hi_n = CANON["Normal_band"]["value"]
ref = json.load(open(S.REPO / "Biological_Physics/MethylPhys/chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_1.json"))
C_far = ref["clustering_block"] / ref["healthy_clustering_median"]
iama_floor, iama_full, meta_full = 1 / P, 1 / (P * H(eps0)), 1 / Href
st = list(csv.DictReader(open(S.REPO / "Biological_Physics/MethylPhys/reference_floors_v1/sky/sky_neut6_stats.csv")))
rng = lambda m, col: (min(float(r[col]) for r in st if r["map"] == m), max(float(r[col]) for r in st if r["map"] == m))
print(f"IAM-A floor {iama_floor:.3f}, IAM-A full {iama_full:.2f}, Met-A full {meta_full:.2f}, C-score far end {C_far:.1f}")
print("Met-A healthy", rng("healthy", "MetA_6000"), "blur2pct", rng("blur2pct", "MetA_6000"), "C healthy", rng("healthy", "C_6000"), "C local", rng("local5pct", "C_6000"))

def X(v, floor, full):
    """gauge position in [0, 1]: linear floor..1 on [0, 0.5], log 1..full on [0.5, 1]."""
    v = np.asarray(v, float)
    return np.where(v <= 1, 0.5 * (v - floor) / (1 - floor), 0.5 + 0.5 * np.log(np.maximum(v, 1)) / np.log(full))

fig, ax = plt.subplots(figsize=(S.TEXTW, 3.1)); ax.set_xlim(-0.24, 1.02); ax.set_ylim(-0.6, 4.1); ax.axis("off")
x0, w, h = 0.0, 1.0, 0.18
ax.text(0.25, 4.0, "less error than healthy (floor to 1)", ha="center", fontsize=6.5, color=S.IAM)
ax.text(0.75, 4.0, "more error than healthy (1 to full, log scale)", ha="center", fontsize=6.5, color=S.DATA)
def row(y, name, floor, full, floor_lab, full_lab, normal=True, floor_known=True):
    ax.text(-0.23, y, name, fontsize=7.5, fontweight="bold", va="center")
    ax.add_patch(Rectangle((0, y - h / 2), 0.5, h, fc="#eef3f8" if floor_known else "#f2f2f2", ec="none", hatch=None if floor_known else "////"))
    ax.add_patch(Rectangle((0.5, y - h / 2), 0.5, h, fc="#fbeeee", ec="none"))
    if normal:
        a, b = X([lo_n, hi_n], floor if floor_known else 0.9, full)
        if not floor_known:
            a = 0.5 - 0.5 * (1 - lo_n) / 0.1 * 0.1                     # left half has no scale yet: mark 0.95 at its nominal place
        ax.add_patch(Rectangle((a, y - h / 2), b - a, h, fc="#a8d5a2", ec="none"))
    ax.add_patch(Rectangle((0, y - h / 2), 1, h, fc="none", ec="#555555", lw=0.6))
    ax.plot([0.5, 0.5], [y - h * 0.8, y + h * 0.8], color="black", lw=1.2); ax.text(0.5, y - h * 0.9, "1", ha="center", va="top", fontsize=6.5)
    ax.text(0, y - h * 0.9, floor_lab, ha="left", va="top", fontsize=6.3)
    ax.text(1, y - h * 0.9, full_lab, ha="right", va="top", fontsize=6.3)
    return lambda v: X(v, floor if floor_known else 0.9, full)

def bar(xf, y, r, col, lab, dy, ha="center"):
    a, b = xf(r); ax.add_patch(Rectangle((a, y - 0.035), max(b - a, 0.004), 0.07, fc=col, ec="none"))
    ax.annotate(lab, ((a + b) / 2, y + (h / 2 if dy > 0 else -h / 2)), xytext=(0, dy), textcoords="offset points", fontsize=6, color=col, ha=ha,
                va="bottom" if dy > 0 else "top", arrowprops=dict(arrowstyle="-", lw=0.5, color=col))

y = 3.3; xf = row(y, "Met-A", 0.9, meta_full, r"$H_{\rm min}$ not yet defined", f"surface full {meta_full:.2f}", floor_known=False)
ax.text(0.98, y + h * 0.7, "breach and cancer region: to be measured", ha="right", va="bottom", fontsize=6, color=S.DATA)
r = rng("healthy", "MetA_6000"); bar(xf, y, r, S.IAM, f"healthy held-out\n{r[0]:.3f}\u2013{r[1]:.3f}", -14, "right")
r = rng("blur2pct", "MetA_6000"); bar(xf, y, r, S.DATA, f"2 % blur {r[0]:.3f}\u2013{r[1]:.3f}", -14, "left")
y = 2.2; xf = row(y, "IAM-A", iama_floor, iama_full, rf"$H_{{\rm min}}$ {iama_floor:.3f}", f"surface full {iama_full:.2f}")
ax.text(0.98, y + h * 0.7, "breach and cancer region: to be measured", ha="right", va="bottom", fontsize=6, color=S.DATA)
y = 1.1; xf = row(y, "Met-A C-score", 0.0, C_far, "0", f"every block as one {C_far:.0f}", normal=False)
r = rng("healthy", "C_6000"); bar(xf, y, r, S.IAM, f"healthy {r[0]:.2f}\u2013{r[1]:.2f}", 7, "right")
r = rng("local5pct", "C_6000"); bar(xf, y, r, S.DATA, f"5 % blur in 10 regions\n{r[0]:.1f}\u2013{r[1]:.1f}", 8)
y = 0.0; ax.text(-0.23, y, "IAM-A C-score", fontsize=7.5, fontweight="bold", va="center")
ax.add_patch(Rectangle((0, y - h / 2), 1, h, fc="#f2f2f2", ec="#999999", lw=0.6, hatch="////")); ax.text(0.5, y, "not built", ha="center", va="center", fontsize=6.5, color=S.GR)
S.save(fig, "part4", "fig_gauge_cell")
