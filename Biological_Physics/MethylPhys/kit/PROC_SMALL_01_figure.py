
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

apply_figure_style(sizes=(8, 7, 6))
res = json.load(open("handoff/small01_results.json"))
thr = res["_threshold"]
HOSTS = ["GSM2333901", "GSM2333905", "GSM1051533"]
F = [0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
BLUE, GREY = "#1a5fb4", "#9a9a9a"

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
for ax, cls, letter in ((axes[0], "secretory", "a"), (axes[1], "cycling", "b")):
    for cfg, col, lab in (("BASE", GREY, "as commissioned"),
                          ("GLS", BLUE, "inverse-variance weights")):
        m = res["mixtures"][cfg]
        lo, hi, mid = [], [], []
        for f in F:
            ts = [m[f"{h}|{cls}|{f}"]["t"] for h in HOSTS]
            lo.append(min(ts)); hi.append(max(ts)); mid.append(float(np.mean(ts)))
        x = [f * 100 for f in F]
        ax.fill_between(x, lo, hi, color=col, alpha=0.22, lw=0, zorder=2)
        ax.plot(x, mid, color=col, lw=1.5, marker="o", ms=3.2, label=lab, zorder=3)
        ax.axhline(thr[cfg][cls], color=col, lw=0.9, ls=":", zorder=1)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    first = {"secretory": 2.0, "cycling": 2.0}[cls]
    ax.annotate("detected from 2 %", xy=(2.0, thr["GLS"][cls]), xytext=(2.4, 20),
                fontsize=6, color=BLUE, ha="left",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7, shrinkA=0, shrinkB=2))
    if cls == "secretory":
        ax.annotate("only from 5 %", xy=(5.0, thr["BASE"][cls]), xytext=(5.6, -12),
                    fontsize=6, color="#6e6e6e", ha="left",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.7, shrinkA=0, shrinkB=2))
    ax.set_xscale("log")
    ax.set_xticks([0.25, 1, 5, 20]); ax.set_xticklabels(["0.25", "1", "5", "20"])
    ax.set_xlabel("fraction mixed in (%)")
    ax.set_title({"secretory": "secretory: 5 % \u2192 2 %",
                  "cycling": "cycling: 2 % either way"}[cls], loc="left")
    ax.margins(0.06)
    panel_letter(ax, letter)
axes[0].set_ylabel("detection statistic  t")
axes[1].legend(frameon=False, loc="lower right", fontsize=6, handlelength=1.4, borderpad=0.1)

ax = axes[2]
held = {g: v for g, v in json.load(open("handoff/small01_heldout.json")).items() if "error" not in v}
rng = np.random.default_rng(7)
for i, cls in enumerate(("secretory", "cycling")):
    groups = [("threshold\nset (38)", [res["nulls"]["GLS"][g]["t"][cls] for g in res["nulls"]["GLS"]], "#7aa6d4"),
              (f"held out\n({len(held)})", [held[g][cls]["t"] for g in held if held[g][cls]["t"] is not None], BLUE)]
    for j, (lab, vals, col) in enumerate(groups):
        x = i * 2.4 + j * 1.05
        ax.scatter(x + rng.normal(0, 0.075, len(vals)), vals, s=5, color=col, alpha=0.75, lw=0, zorder=3)
        ax.plot([x - 0.22, x + 0.22], [np.median(vals)] * 2, color="0.2", lw=1.2, zorder=4)
        if i == 0:
            ax.text(x, -31.5, lab, fontsize=6, ha="center", va="top", color=col)
    ax.plot([i * 2.4 - 0.4, i * 2.4 + 1.45], [thr["GLS"][cls]] * 2, color="#c01c28", lw=1.0, zorder=5)
    ax.text(i * 2.4 + 0.52, 8.2, "threshold", fontsize=6, color="#c01c28", ha="center", va="top")
    ax.text(i * 2.4 + 0.52, -36.5, cls, fontsize=7, ha="center", va="top")
ax.set_xlim(-0.8, 4.6); ax.set_ylim(-30, 10)
ax.set_xticks([]); ax.set_ylabel("detection statistic  t")
ax.set_title("healthy blood sits far below it", loc="left")
panel_letter(ax, "c")
for a in axes:
    set_frame(a)
fig.tight_layout()
fig.savefig("PROC_SMALL_01_detection.png", dpi=300, bbox_inches="tight")

r = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
ov = [(a.get_text()[:22], b.get_text()[:22]) for i, (a, ba) in enumerate(texts) for b, bb in texts[i+1:] if ba.overlaps(bb)]
print("overlaps:", len(ov)); [print("  ", o) for o in ov[:6]]
print("held-out:", len(held), {c: sum(1 for g in held if held[g][c]["detected"]) for c in ("secretory", "cycling")})
