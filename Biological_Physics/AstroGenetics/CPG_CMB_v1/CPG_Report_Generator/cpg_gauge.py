#!/usr/bin/env python3
"""
cpg_gauge.py — Stage 9 report-builder cellular reference gauge (Appendix A1).

Renders the A-score CALIBRATION SCALE (the "cellular thermometer") for the report.
This is a fixed reference scale — it carries NO patient markers; the patient's per-cell
positions are shown separately in Appendix A2 (cellular departure ranking).

Single source of truth
----------------------
The tier boundaries are READ FROM tier_breakpoints.json at render time, so the gauge can
never drift out of sync with the canonical v1.3 scheme.

Vocabulary
----------
Customer six-tier only: SUPPRESSED / NORMAL / ELEVATED / WARBURG (line) / SIGNIFICANTLY
ELEVATED / BREACH. (The engine five-tier labels are intentionally not used.)

Framing
-------
The axis is zoomed to the decision-relevant window (~0.90–1.20) so the NORMAL / ELEVATED /
SIGNIFICANTLY ELEVATED bands are legible rather than squished; only a sliver of SUPPRESSED
and a modest BREACH zone are shown. In the live report this gauge is stacked next to the
star gauge (figC_cosmic_gauge.pdf) — same ruler, same physics, gravitational saturation
(A_IAM=1) mapping to the cellular breach at 1.10.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Decision-relevant window: stretch normal/elevated/sig-elevated, trim suppressed & breach.
AXIS_MIN, AXIS_MAX = 0.90, 1.20

TIER_COLORS = {
    "SUPPRESSED": "#aec7e8",
    "NORMAL": "#b5e3b5",
    "ELEVATED": "#ffe2a8",
    "SIGNIFICANTLY_ELEVATED": "#f4c06a",
    "BREACH": "#e08a7c",
}

CUSTOMER_LABELS = {
    "SUPPRESSED": "SUPPRESSED",
    "NORMAL": "NORMAL\nhealthy range",
    "ELEVATED": "ELEVATED",
    "SIGNIFICANTLY_ELEVATED": "SIGNIFICANTLY\nELEVATED",
    "BREACH": "BREACH",
}

PARTITION_ORDER = ["SUPPRESSED", "NORMAL", "ELEVATED", "SIGNIFICANTLY_ELEVATED", "BREACH"]


def load_tier_scheme(tier_breakpoints_path):
    """Parse tier_breakpoints.json into the partition the gauge needs.

    Returns: partitions [(tier_id, lo, hi)], warburg_line, breach_line, malignant/senescent.
    """
    d = json.load(open(tier_breakpoints_path))
    ts = d["tier_system_v1_2"]
    partitions = []
    for t in ts["tiers"]:
        if t.get("is_boundary_line"):          # WARBURG_TRANSITION is a line, not a band
            continue
        rng = t.get("a_score_range")
        if not rng:
            continue
        lo = rng["min"]
        hi = rng.get("max_exclusive", rng.get("max_inclusive", AXIS_MAX))
        partitions.append((t["tier_id"], lo, hi))
    clusters = ts.get("reference_clusters_past_breach", {})
    return {
        "partitions": partitions,
        "warburg_line": ts.get("warburg_line_value", 1.07),
        "breach_line": ts.get("breach_line_value", 1.10),
        "senescent": clusters.get("senescent_cells"),
        "malignant": clusters.get("malignant_cells"),
    }


def render_reference_gauge(tier_breakpoints_path,
                           out_path,
                           title="A-score reference gauge \u2014 the calibration scale",
                           dpi=160):
    """Render the calibration scale (no patient markers). Output format follows the file
    extension (.svg for the report, .png for review)."""
    scheme = load_tier_scheme(tier_breakpoints_path)

    fig, ax = plt.subplots(figsize=(12, 3.3))

    # Coloured tier bands (clipped to the printed window).
    band_mids = {}
    for tier_id, lo, hi in scheme["partitions"]:
        lo_c, hi_c = max(lo, AXIS_MIN), min(hi, AXIS_MAX)
        if hi_c <= lo_c:
            continue
        ax.add_patch(Rectangle((lo_c, 0.0), hi_c - lo_c, 1.0,
                               facecolor=TIER_COLORS.get(tier_id, "#dddddd"),
                               edgecolor="white", linewidth=1.5, zorder=1))
        band_mids[tier_id] = (lo_c + hi_c) / 2.0

    # NORMAL and BREACH bands are wide enough to label inside.
    for tier_id in ("NORMAL", "BREACH"):
        if tier_id in band_mids:
            ax.text(band_mids[tier_id], 0.5, CUSTOMER_LABELS[tier_id],
                    ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="#222222", zorder=3)

    # Narrow bands (SUPPRESSED, ELEVATED, SIGNIFICANTLY ELEVATED) labelled BELOW the bar,
    # staggered across two levels with leader lines so each stays tied to its own zone.
    below = [t for t in ("SUPPRESSED", "ELEVATED", "SIGNIFICANTLY_ELEVATED") if t in band_mids]
    y_levels = [-0.20, -0.40]
    for i, tier_id in enumerate(below):
        xc = band_mids[tier_id]
        yl = y_levels[i % 2]
        ax.plot([xc, xc], [-0.02, yl + 0.05], color="#999999", lw=0.7, zorder=2)
        ax.text(xc, yl, CUSTOMER_LABELS[tier_id], ha="center", va="top",
                fontsize=8.5, fontweight="bold", color="#444444", zorder=3)

    # A = 1.0 healthy reference (solid black), labelled above.
    ax.axvline(1.00, color="black", linewidth=2.0, zorder=4)
    ax.text(1.00, 1.30, "A = 1.0\nhealthy reference", ha="center", va="bottom",
            fontsize=8, fontweight="bold")

    # Warburg line (dashed) — left of the line so it does not collide with the breach label.
    wl = scheme["warburg_line"]
    ax.axvline(wl, color="#c8771f", linewidth=1.8, linestyle="--", zorder=4)
    ax.text(wl - 0.004, 1.06, f"{wl:.2f} Warburg:\nmetabolic strategy\nmust change",
            ha="right", va="bottom", fontsize=7.5, color="#c8771f", fontweight="bold")

    # Breach line label — right of the breach edge.
    bl = scheme["breach_line"]
    ax.text(bl + 0.004, 1.06, f"{bl:.2f}\nbreach", ha="left", va="bottom",
            fontsize=7.5, color="#8a3326", fontweight="bold")

    # Active-malignancy reference (clusters past breach) — annotated near the right edge,
    # with the true canonical cluster range (the clusters themselves sit further right).
    mal = scheme.get("malignant")
    if mal:
        ax.text(AXIS_MAX - 0.005, 1.06,
                f"active malignancy\n(clusters {mal['a_low']:.2f}-{mal['a_high']:.2f}) \u2192",
                ha="right", va="bottom", fontsize=7.5, color="#b03020", fontweight="bold")

    # Axis cosmetics.
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(-0.52, 1.55)
    ax.set_yticks([])
    ax.set_xticks([round(x, 2) for x in _frange(AXIS_MIN, AXIS_MAX, 0.05)])
    ax.tick_params(axis="x", labelsize=8.5)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("Architectural A-score  (mean of H(\u03b2) / H_min(class) over panel CpGs)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=16)
    footer = ("Below 1.0 = suppressed / inversion.  At 1.0 = healthy reference.  "
              f"Above = drifting; {bl:.2f} = breach.  Same physics, same ruler as a stellar core "
              "(see the star gauge alongside).")
    fig.text(0.5, 0.005, footer, ha="center", fontsize=7.5, style="italic", color="#555555")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _tier_for(a, partitions):
    """Canonical tier_id for an A-score from the JSON partition (consistency-by-construction)."""
    for tier_id, lo, hi in partitions:
        if lo <= a < hi:
            return tier_id
    return "BREACH" if a >= partitions[-1][1] else "SUPPRESSED"


def render_cellular_departure_ranking(cells,
                                      tier_breakpoints_path,
                                      out_path,
                                      title="Cellular departure ranking \u2014 top cells by magnitude",
                                      dpi=160):
    """Appendix A2 \u2014 horizontal bar chart of the top-N cells by A-score departure, drawn
    against the reference-gauge tier zones (same boundaries as A1, read from the same JSON).

    cells: list of dicts with keys: rank, cell_type, class (short), a_score, and optionally
           ci_lo / ci_hi. Bars run from the A=1.0 baseline to each cell's A-score, coloured by
           the tier the score falls in; 95% CI shown as a whisker. Rank 1 at top.
    """
    scheme = load_tier_scheme(tier_breakpoints_path)
    partitions = scheme["partitions"]
    cells = sorted(cells, key=lambda c: c["rank"])
    n = len(cells)

    lo_vals = [c.get("ci_lo", c["a_score"]) for c in cells]
    hi_vals = [c.get("ci_hi", c["a_score"]) for c in cells]
    x_lo = min(0.99, min(lo_vals) - 0.005)
    x_hi = max(1.10, max(hi_vals) + 0.010)

    fig, ax = plt.subplots(figsize=(11, 0.42 * n + 1.6))

    # Tier zones shaded behind the bars.
    for tier_id, lo, hi in partitions:
        lo_c, hi_c = max(lo, x_lo), min(hi, x_hi)
        if hi_c > lo_c:
            ax.axvspan(lo_c, hi_c, color=TIER_COLORS.get(tier_id, "#dddddd"), alpha=0.30, zorder=0)

    y = list(range(n))[::-1]  # rank 1 at top
    for yi, c in zip(y, cells):
        a = c["a_score"]
        color = TIER_COLORS[_tier_for(a, partitions)]
        ax.barh(yi, a - 1.00, left=1.00, height=0.62, color=color,
                edgecolor="#666666", linewidth=0.5, zorder=2)
        if "ci_lo" in c and "ci_hi" in c:
            ax.plot([c["ci_lo"], c["ci_hi"]], [yi, yi], color="#333333", lw=1.0, zorder=3)
            for xx in (c["ci_lo"], c["ci_hi"]):
                ax.plot([xx, xx], [yi - 0.12, yi + 0.12], color="#333333", lw=1.0, zorder=3)
        ax.text(max(a, c.get("ci_hi", a)) + 0.0015, yi, f"{a:.3f}", va="center", ha="left",
                fontsize=7.5, fontweight="bold", color="#222222", zorder=4)

    ax.axvline(1.00, color="black", lw=1.6, zorder=4)
    ax.axvline(scheme["warburg_line"], color="#c8771f", lw=1.5, ls="--", zorder=4)
    if scheme["breach_line"] <= x_hi:
        ax.axvline(scheme["breach_line"], color="#8a3326", lw=1.3, ls=":", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{c['rank']}. {c['cell_type']} ({c.get('class','')})" for c in cells], fontsize=8)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("Architectural A-score  (1.00 = healthy class baseline)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.text(0.5, 0.006,
             f"Bars coloured by tier (zones shaded behind). Whiskers = 95% CI. "
             f"Warburg {scheme['warburg_line']:.2f} dashed; breach {scheme['breach_line']:.2f} dotted.",
             ha="center", fontsize=7.5, style="italic", color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _frange(lo, hi, step):
    x = lo
    while x <= hi + 1e-9:
        yield x
        x += step


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render the CPG A-score reference gauge from tier_breakpoints.json")
    ap.add_argument("--tier-breakpoints", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = render_reference_gauge(args.tier_breakpoints, args.out)
    print(f"wrote {p}")
