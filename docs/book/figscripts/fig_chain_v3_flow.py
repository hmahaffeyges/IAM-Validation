#!/usr/bin/env python3
"""Chain v3 flow (neutrophils, EPIC v1). Stage names and rules as in
Biological_Physics/MethylPhys/chain/conductor_v3.py, stage_m_met_a.py, stage_q_iam_a.py and MethylPhys_Interface/run_sample.py.
Writes figures/part4/fig_chain_v3_flow.pdf (run from docs/book)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROW1 = [("Stage 0", "intake\ncall rate, controls,\nsex check, hash"),
        ("Stage 1", "calibration\nnoob $\\beta$; probes at\nbackground removed"),
        ("platform", "EPIC v1 only;\n450K and EPIC v2\nrefused"),
        ("Stage A", "composition\n8 blood groups,\n963 markers (NNLS)"),
        ("Stage M", "Met-A on 6,000\nidentity sites;\nnoise index $N$"),
        ("Stage MC", "Met-A C-score\n50-site blocks;\nno band yet"),
        ("Stage T", "same-run tare\n$\\geq3$ references;\n$A_{\\rm rel}$, detection limit"),
        ("report", "one HTML page,\nJSON bundle,\nledger row")]
fig, ax = plt.subplots(figsize=(13.5, 3.4))
ax.set_xlim(0, 8); ax.set_ylim(0, 3.3); ax.axis("off")
w, h, y = 0.9, 1.45, 1.6
for k, (t, s) in enumerate(ROW1):
    x = k + 0.05
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc="#eef3f8", ec="#3b5b7a", lw=1.0))
    ax.text(x + w / 2, y + h - 0.22, t, ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(x + w / 2, y + 0.55, s, ha="center", va="center", fontsize=7.0, linespacing=1.25)
    if k < len(ROW1) - 1:
        ax.annotate("", xy=(k + 1.05, y + h / 2), xytext=(x + w, y + h / 2), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
# Stage Q branch: sequencing input
xq = 4.07
ax.add_patch(FancyBboxPatch((xq, 0.12), w * 2.6, 1.0, boxstyle="round,pad=0.02", fc="#f6f1e8", ec="#7a5b3b", lw=1.0))
ax.text(xq + w * 1.3, 0.88, "Stage Q (sequencing reads)", ha="center", va="center", fontsize=9.0, weight="bold")
ax.text(xq + w * 1.3, 0.42, "IAM-A = $H(\\varepsilon)/(P\\,H(\\varepsilon_0))$; pipeline loyfer_pat_v1;\n$\\geq100{,}000$ opportunities", ha="center", va="center", fontsize=7.2)
ax.annotate("", xy=(7.05 + w / 2, y), xytext=(xq + w * 2.6, 0.62), arrowprops=dict(arrowstyle="->", lw=0.9, color="#7a5b3b", connectionstyle="arc3,rad=0.15"))
ax.text(3.5 * 1.0 + 0.5, 3.22, "Stage A runs on whole blood only; isolated neutrophils go from the platform check to Stage M against their own floor",
        ha="center", va="center", fontsize=7.4, style="italic")
fig.savefig("figures/part4/fig_chain_v3_flow.pdf", bbox_inches="tight")
fig.savefig("figures/part4/fig_chain_v3_flow.png", dpi=150, bbox_inches="tight")
