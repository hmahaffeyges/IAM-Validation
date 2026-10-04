"""part4/fig_v5 (Chapter 'The reference: purified healthy cells on the specimen's platform', Figure fig:v5).

Coverage of held-out observations by the atlas v2 90 % predictive interval, read from the pre-registered outcome record
Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md (B1 overall; B2 by kind of data). Band: the pre-registered pass range 85-95 %.
Measured. Run: python docs/book/figscripts/fig_p4_v5.py
"""
import sys, json, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import _bookstyle as S

S.apply()

import re
rec = (S.REPO / "Biological_Physics/MethylPhys/doors/PROC_V5_HELDOUT_OUTCOME.md").read_text()
m = re.search(r"covers \*\*([\d.]+) %\*\* of ([\d,]+) held-out", rec)
ROWS = [("all", float(m.group(1)), m.group(2))]
for kind in ("WGBS", "array", "pooled"):
    k = re.search(kind + r" ([\d.]+) % \(([\d,]+)\)", rec); ROWS.append((kind, float(k.group(1)), k.group(2)))
assert ROWS[0][2] == "342,716", ROWS
fig, ax = plt.subplots(figsize=(0.75 * S.TEXTW, 2.4))
ax.axvspan(85, 95, color=S.ALT, alpha=0.15, lw=0); ax.axvline(90, color="black", lw=0.8)
for i, (lab, v, n) in enumerate(reversed(ROWS)):
    ax.plot([v], [i], "o", color=S.ALT, ms=6); ax.text(v + 0.3, i, f"{v:.2f} %  (n = {n})", va="center", fontsize=6.5)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in reversed(ROWS)])
ax.set_xlim(84, 99); ax.set_ylim(-0.6, len(ROWS) - 0.4)
ax.set_xlabel("held-out values inside the atlas's 90 % interval (%)")
ax.set_title("The atlas's stated uncertainty holds on data it never saw")
print(ROWS)
S.save(fig, "part4", "fig_v5")
