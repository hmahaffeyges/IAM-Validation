#!/usr/bin/env python3
"""
cpg_gauge.py — Stage 9 report-builder cellular gauge.

Renders the "cellular thermometer" gauge (the figB layout) for the doctor/patient
report. The tier boundaries are READ FROM tier_breakpoints.json at render time, so the
gauge can never drift out of sync with the canonical tier scheme — there is exactly one
source of truth for the breakpoints.

AstroGenetics framing
---------------------
The cell gauge is paired in the report with the star gauge (figC_cosmic_gauge.pdf), which
plots real stellar A_IAM values on the SAME ruler: gravitational saturation (A_IAM = 1:
Chandrasekhar / TOV / Schwarzschild) maps to the cellular breach at 1.10, because collapse
and cellular breach are the same no-return event. The star gauge is a fixed reference
(its values come from Heath's derived A_IAM rescaling) and is NOT regenerated here.

Vocabulary
----------
Two label sets are supported (the report can show one or both):
  * 'customer' (default): SUPPRESSED / NORMAL / ELEVATED / WARBURG line / SIGNIFICANTLY_ELEVATED / BREACH
  * 'engine'  (clinician): BELOW NORMAL / NORMAL / MARGINAL / DETECTABLE / BREACH (figB labels)

Usage
-----
    from cpg_gauge import render_cellular_gauge
    render_cellular_gauge(
        markers=[("immune", 1.054), ("secretory", 1.021)],
        tier_breakpoints_path=".../tier_breakpoints.json",
        out_path="immune_gauge.pdf",
        vocabulary="customer",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Axis extent of the printed ruler.
AXIS_MIN, AXIS_MAX = 0.80, 1.35

# Tier fill colours (shared by both vocabularies; keyed by the canonical partition tier).
TIER_COLORS = {
    "SUPPRESSED": "#aec7e8",            # blue
    "NORMAL": "#b5e3b5",               # green
    "ELEVATED": "#ffe2a8",             # light amber
    "SIGNIFICANTLY_ELEVATED": "#f4c06a",  # amber
    "BREACH": "#e08a7c",               # red
}

# Customer label per partition tier; engine label per partition tier (figB conventions).
CUSTOMER_LABELS = {
    "SUPPRESSED": "SUPPRESSED\nsuppressed",
    "NORMAL": "NORMAL\nhealthy range",
    "ELEVATED": "ELEVATED",
    "SIGNIFICANTLY_ELEVATED": "SIGNIFICANTLY\nELEVATED",
    "BREACH": "BREACH",
}
ENGINE_LABELS = {
    "SUPPRESSED": "BELOW NORMAL\nsuppressed",
    "NORMAL": "NORMAL\nhealthy range",
    "ELEVATED": "MARGINAL",
    "SIGNIFICANTLY_ELEVATED": "DETECTABLE",
    "BREACH": "BREACH",
}

PARTITION_ORDER = ["SUPPRESSED", "NORMAL", "ELEVATED", "SIGNIFICANTLY_ELEVATED", "BREACH"]


def load_tier_scheme(tier_breakpoints_path: str | Path) -> dict:
    """Parse tier_breakpoints.json into the partition the gauge needs.

    Returns dict with: partitions [(tier_id, lo, hi)], warburg_line, breach_line,
    and reference clusters (senescent / malignant) if present.
    """
    d = json.load(open(tier_breakpoints_path))
    ts = d["tier_system_v1_2"]
    partitions = []
    for t in ts["tiers"]:
        if t.get("is_boundary_line"):     # WARBURG_TRANSITION is a line, not a band
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


def render_cellular_gauge(markers,
                          tier_breakpoints_path: str | Path,
                          out_path: str | Path,
                          vocabulary: str = "customer",
                          title: str = "How the cellular thermometer reads cellular state",
                          dpi: int = 160) -> Path:
    """Render the cellular gauge.

    markers: iterable of (label, a_score) to place arrows on the ruler. May be empty
             (renders the reference scale only).
    """
    scheme = load_tier_scheme(tier_breakpoints_path)
    labels = ENGINE_LABELS if vocabulary == "engine" else CUSTOMER_LABELS

    fig, ax = plt.subplots(figsize=(13, 3.6))
    band_lo, band_hi = 0.0, 1.0

    # Coloured tier bands (clipped to the printed ruler extent).
    for tier_id, lo, hi in scheme["partitions"]:
        lo_c, hi_c = max(lo, AXIS_MIN), min(hi, AXIS_MAX)
        if hi_c <= lo_c:
            continue
        ax.add_patch(Rectangle((lo_c, band_lo), hi_c - lo_c, band_hi - band_lo,
                               facecolor=TIER_COLORS.get(tier_id, "#dddddd"),
                               edgecolor="white", linewidth=1.5, zorder=1))
        ax.text((lo_c + hi_c) / 2, 0.5, labels.get(tier_id, tier_id),
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="#222222", zorder=3)

    # A = 1.0 healthy reference (solid black).
    ax.axvline(1.00, color="black", linewidth=2.0, zorder=4)
    ax.text(1.00, -0.16, "A = 1.0\nhealthy reference", ha="center", va="top",
            fontsize=8, fontweight="bold")

    # Warburg line (dashed) — the metabolic-strategy-change boundary.
    wl = scheme["warburg_line"]
    ax.axvline(wl, color="#c8771f", linewidth=1.8, linestyle="--", zorder=4)
    ax.text(wl - 0.006, 1.10, f"{wl:.2f} Warburg:\nmetabolic strategy\nmust change",
            ha="right", va="bottom", fontsize=7.5, color="#c8771f", fontweight="bold")

    # Breach line annotation.
    bl = scheme["breach_line"]
    ax.text(bl + 0.006, 1.10, f"{bl:.2f}\nbreach", ha="left", va="bottom",
            fontsize=7.5, color="#8a3326", fontweight="bold")

    # Malignancy / senescence reference clusters past breach.
    mal = scheme.get("malignant")
    if mal:
        ax.text(min(mal["a_high"], AXIS_MAX) - 0.02, 1.12,
                f"active malignancy\n(clusters {mal['a_low']:.2f}-{mal['a_high']:.2f})",
                ha="center", va="bottom", fontsize=7.5, color="#b03020", fontweight="bold")

    # Patient markers (sorted; labels staggered across y-levels so clustered markers don't collide).
    y_levels = [-0.19, -0.33, -0.47]
    for i, (label, a) in enumerate(sorted(markers, key=lambda m: m[1])):
        a_clip = min(max(a, AXIS_MIN), AXIS_MAX)
        yl = y_levels[i % len(y_levels)]
        ax.annotate("", xy=(a_clip, 0.02), xytext=(a_clip, -0.10),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2), zorder=5)
        ax.plot([a_clip, a_clip], [-0.10, yl + 0.05], color="#999999", lw=0.6, zorder=4)
        ax.text(a_clip, yl, f"{label}  A={a:.3f}", ha="center", va="top",
                fontsize=7.5, color="#333333", zorder=5)

    # Axis cosmetics.
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(-0.58, 1.45)
    ax.set_yticks([])
    ax.set_xticks([round(x, 2) for x in _frange(AXIS_MIN, AXIS_MAX, 0.05)])
    ax.tick_params(axis="x", labelsize=8)
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("Architectural A-score  (mean of H(\u03b2) / H_min(class) over panel CpGs)", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=18)
    footer = ("Below 1.0 = suppressed / inversion.  At 1.0 = healthy reference.  "
              f"Above = drifting; {bl:.2f} = breach.  Same physics, same ruler as a stellar core "
              "(see star gauge).")
    fig.text(0.5, 0.01, footer, ha="center", fontsize=7.5, style="italic", color="#555555")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
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
    ap = argparse.ArgumentParser(description="Render the CPG cellular gauge from tier_breakpoints.json")
    ap.add_argument("--tier-breakpoints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocabulary", default="customer", choices=["customer", "engine"])
    args = ap.parse_args()
    # Demo markers (subtle-drift archetype): immune slightly elevated, others normal.
    demo = [("immune", 1.054), ("secretory", 1.021), ("terminal", 0.992)]
    p = render_cellular_gauge(demo, args.tier_breakpoints, args.out, vocabulary=args.vocabulary)
    print(f"wrote {p}")
