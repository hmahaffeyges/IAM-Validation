#!/usr/bin/env python3
"""
STEP 7 / CHECK D — Breast pre-dx scoring against IAMAtlas
============================================================

Re-scores GSE51057 and GSE51032 breast pre-diagnostic cohorts using the
Xu 2020 98-CpG directional panel (rule A, frozen ±1 per CpG) anchored
against IAMAtlas v0.1 immune brightness.

INPUTS
======
1. IAMAtlas_v0_1.csv                                  (production matrix)
2. xu2020_breast_directional_RuleA.json               (98 CpGs, ±1 direction)
3. Per-sample β at panel CpGs (from VAL-047 Tightening fresh):
     VAL047_samples_GSE51057.csv  (n=730 with TtD windows)
     VAL047_samples_GSE51032.csv  (n=857 with TtD windows + colorectal)

PER-SAMPLE CSVs FORMAT (from VAL047_tightening_fresh.py):
  sample_id, age, sex, case_status, ttd_years, cancer_type, ...,
  cg00008800, cg00011346, ...  (one column per panel CpG)

METHOD
======
For each sample, compute IAMAtlas-anchored directional A-score:
  A_dir_iam = mean over panel of: direction_i × (β_i - iamatlas_immune_mean_i)

For each TtD window (0-2, 2-5, 5-10, >10 yr, all_pre_dx), compute Cohen's d
of A_dir_iam between cases and controls. Compare against the headline
numbers from Phase 9/12:

   GSE51057 breast >10yr:    d = +1.85  (target: preserve direction, magnitude similar)
   GSE51057 breast 5-10yr:   d = +0.76
   GSE51057 breast all:      d = +0.44
   GSE51032 breast >10yr:    d = +1.34
   GSE51032 breast 5-10yr:   d = +0.94
   GSE51032 breast all:      d = +0.71
   GSE51032 colorectal >10yr: d = -0.80  (sign-flipped; framework calls this normal)
   GSE51032 colorectal all:   d = -0.55

OUTPUT
======
step_7_chk_d_breast_iam_scoring.md

Date: 2026-05-04
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict


# TtD windows (FROZEN per VAL-047 invariants)
TTD_WINDOWS = [
    ("0-2 yr",      0.0,   2.0),
    ("2-5 yr",      2.0,   5.0),
    ("5-10 yr",     5.0,  10.0),
    (">10 yr",     10.0, 999.0),
    ("all_pre_dx",  0.0, 999.0),
]


def load_iamatlas_immune(matrix_path: Path) -> dict:
    out = {}
    with open(matrix_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpg = row["cpg_id"]
            mean = row.get("immune_mean", "NA")
            if mean not in ("NA", "", None):
                try:
                    out[cpg] = float(mean)
                except ValueError:
                    pass
    return out


def cohen_d(a, b) -> float:
    if len(a) < 2 or len(b) < 2: return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    pooled = math.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / (len(a) + len(b) - 2))
    if pooled == 0: return float("nan")
    return (ma - mb) / pooled


def score_sample(beta_at_panel: dict, panel: list, iamatlas_immune: dict) -> tuple:
    """Returns (A_dir_iam, n_used, n_atlas_hits)."""
    score = 0.0
    n_used = 0
    n_atlas_hits = 0
    for entry in panel:
        cpg = entry["cpg"]
        direction = entry["direction"]
        beta = beta_at_panel.get(cpg)
        if beta is None: continue
        try: beta = float(beta)
        except (ValueError, TypeError): continue
        if not (0 <= beta <= 1): continue
        atlas_mean = iamatlas_immune.get(cpg)
        if atlas_mean is None:
            # CpG not in matrix — fall back to flat (β / H_min) which is mathematically
            # equivalent to using atlas_mean = 0 in the residual form, but we'd rather skip
            continue
        n_atlas_hits += 1
        score += direction * (beta - atlas_mean)
        n_used += 1
    if n_used == 0:
        return (float("nan"), 0, 0)
    return (score / n_used, n_used, n_atlas_hits)


def process_cohort(name: str, samples_csv: Path, panel: list,
                   iamatlas_immune: dict, cancer_filter: str = None) -> dict:
    """Process one cohort (GSE51057 or GSE51032)."""
    if not samples_csv.exists():
        return {"name": name, "status": "MISSING", "path": str(samples_csv)}

    panel_cpgs = set(e["cpg"] for e in panel)

    # Group sample scores by TtD window × case/control
    scores_by_group = defaultdict(lambda: {"case": [], "control": []})
    n_total = 0
    n_scored = 0

    with open(samples_csv) as f:
        reader = csv.DictReader(f)
        # Identify panel-CpG columns vs metadata columns
        panel_cols = [c for c in reader.fieldnames if c in panel_cpgs]
        meta_cols = [c for c in reader.fieldnames if c not in panel_cpgs]

        for row in reader:
            n_total += 1
            # Filter by cancer type if requested (e.g., colorectal-only on GSE51032)
            cancer_type = (row.get("cancer_type") or "").lower()
            if cancer_filter and cancer_filter.lower() not in cancer_type:
                continue
            # Determine case/control
            case_status = (row.get("case_status") or row.get("status") or "").lower()
            is_case = "case" in case_status or case_status == "1"
            is_control = "control" in case_status or case_status == "0"
            if not (is_case or is_control): continue
            # TtD years
            try:
                ttd = float(row.get("ttd_years") or row.get("ttd") or "nan")
            except (ValueError, TypeError):
                ttd = float("nan")
            # Build per-CpG β dict
            beta_at_panel = {}
            for c in panel_cols:
                v = row.get(c)
                if v not in (None, "", "NA"):
                    try: beta_at_panel[c] = float(v)
                    except ValueError: pass
            score, n_used, n_atlas_hits = score_sample(beta_at_panel, panel, iamatlas_immune)
            if math.isnan(score): continue
            n_scored += 1
            # Bin into TtD windows for cases; controls go into all windows
            if is_control:
                for label, _, _ in TTD_WINDOWS:
                    scores_by_group[label]["control"].append(score)
            else:
                if math.isnan(ttd): continue
                for label, lo, hi in TTD_WINDOWS:
                    if lo <= ttd < hi:
                        scores_by_group[label]["case"].append(score)

    # Compute Cohen's d per window
    per_window = {}
    for label, _, _ in TTD_WINDOWS:
        cases = scores_by_group[label]["case"]
        ctrls = scores_by_group[label]["control"]
        # For 'all_pre_dx', deduplicate controls (we added them to every window)
        if label == "all_pre_dx":
            ctrls = list(set(ctrls))
        per_window[label] = {
            "n_case": len(cases),
            "n_control": len(ctrls),
            "case_mean": statistics.mean(cases) if cases else float("nan"),
            "control_mean": statistics.mean(ctrls) if ctrls else float("nan"),
            "cohen_d": cohen_d(cases, ctrls),
        }

    return {
        "name": name,
        "cancer_filter": cancer_filter,
        "status": "COMPLETE",
        "n_total": n_total,
        "n_scored": n_scored,
        "per_window": per_window,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="IAMAtlas_v0_1.csv")
    parser.add_argument("--panel", default="xu2020_breast_directional_RuleA.json")
    parser.add_argument("--gse51057", default="VAL047_samples_GSE51057.csv")
    parser.add_argument("--gse51032", default="VAL047_samples_GSE51032.csv")
    parser.add_argument("--report", default="step_7_chk_d_breast_iam_scoring.md")
    args = parser.parse_args()

    print("=" * 72)
    print("STEP 7 / CHECK D — Breast pre-dx scoring vs IAMAtlas")
    print("=" * 72)

    iamatlas_immune = load_iamatlas_immune(Path(args.matrix))
    print(f"\nIAMAtlas immune brightness: {len(iamatlas_immune)} CpGs")
    with open(args.panel) as f:
        panel_obj = json.load(f)
    panel = panel_obj["cpgs"]
    print(f"Panel: {len(panel)} CpGs (Xu 2020 directional rule A)")

    # Coverage check: how many panel CpGs are in the matrix
    panel_in_matrix = sum(1 for e in panel if e["cpg"] in iamatlas_immune)
    print(f"Panel ∩ IAMAtlas: {panel_in_matrix}/{len(panel)} CpGs anchored")

    # Process each cohort
    runs = []
    runs.append(process_cohort("GSE51057_breast", Path(args.gse51057), panel, iamatlas_immune))
    runs.append(process_cohort("GSE51032_breast", Path(args.gse51032), panel, iamatlas_immune,
                                cancer_filter="breast"))
    runs.append(process_cohort("GSE51032_colorectal", Path(args.gse51032), panel, iamatlas_immune,
                                cancer_filter="colorect"))

    # Print
    for r in runs:
        print(f"\n{r['name']}: {r.get('status')}")
        if r["status"] != "COMPLETE": continue
        print(f"  n_scored = {r['n_scored']}")
        for label, _, _ in TTD_WINDOWS:
            w = r["per_window"][label]
            print(f"  {label:<12}  n_case={w['n_case']:>4}  n_ctrl={w['n_control']:>4}  d={w['cohen_d']:+.3f}")

    # Report
    with open(args.report, "w") as f:
        f.write("# Step 7 / Check D — Breast pre-dx IAMAtlas scoring\n\n")
        f.write(f"**Date:** 2026-05-04\n")
        f.write(f"**Panel:** Xu 2020 directional Rule A, n={len(panel)} CpGs ({panel_in_matrix} in IAMAtlas)\n")
        f.write(f"**Method:** A_dir_iam = mean over panel of `direction × (β − IAMAtlas_immune_mean)`\n\n")
        f.write("Headline benchmarks (must preserve direction, magnitude similar):\n")
        f.write("- GSE51057 breast >10yr: d = +1.85\n")
        f.write("- GSE51032 breast >10yr: d = +1.34\n")
        f.write("- GSE51032 colorectal >10yr: d = −0.80 (sign-flipped, expected)\n\n")

        for r in runs:
            f.write(f"## {r['name']}\n\n")
            if r["status"] != "COMPLETE":
                f.write(f"**Status:** {r['status']} — {r.get('path','')}\n\n")
                continue
            f.write(f"- n_scored = {r['n_scored']} samples\n")
            if r.get("cancer_filter"):
                f.write(f"- cancer filter: `{r['cancer_filter']}`\n")
            f.write("\n| TtD window | n cases | n controls | Case mean | Control mean | Cohen's d |\n|---|---|---|---|---|---|\n")
            for label, _, _ in TTD_WINDOWS:
                w = r["per_window"][label]
                f.write(f"| {label} | {w['n_case']} | {w['n_control']} | "
                        f"{w['case_mean']:+.4f} | {w['control_mean']:+.4f} | "
                        f"**{w['cohen_d']:+.3f}** |\n")
            f.write("\n")

        f.write("## Decision\n\n")
        f.write("- **PASS:** All windows preserve sign of headline; >10yr |d| within 30% of benchmark.\n")
        f.write("- **PASS+:** IAM scoring lifts d above benchmark — matrix improves discrimination.\n")
        f.write("- **INVESTIGATE:** Sign flip in any breast cohort window OR colorectal goes positive.\n")

    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
