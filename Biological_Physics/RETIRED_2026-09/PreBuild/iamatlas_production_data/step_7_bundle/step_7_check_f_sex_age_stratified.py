#!/usr/bin/env python3
"""
STEP 7 / CHECK F — Sex-stratified + age-decade-stratified validation
======================================================================

Implements CHK-A11 (sex-stratified) and CHK-A12 (age-decade-stratified)
sub-checks Heath flagged in response to recent literature reporting
sex-specific drift in epigenetic age clocks and intervention responses.

QUESTIONS
=========
CHK-A11: Does the IAMAtlas immune scoring preserve direction in BOTH male
         and female sub-cohorts, or does sex-specific calibration drift
         compromise discrimination in one stratum?

CHK-A12: Does the per-decade healthy A_immune mean drift smoothly with age,
         or are there cliffs that would force per-decade tier thresholds?

INPUTS
======
1. AIBL + AddNeuroMed cohort metadata (sex, age, case/control, A_immune)
   from Step 7 / Check C output OR re-derived here from manifests
2. GSE51057 / GSE51032 per-sample CSVs (have age, sex, case/control)

METHOD
======
For each cohort, partition samples on sex AND on age decade:
  - male AD vs male HC: Cohen's d
  - female AD vs female HC: Cohen's d
  - 50-59 HC mean / sd, 60-69 HC mean / sd, 70-79 HC mean / sd, 80+ HC mean / sd

Compare male d vs female d. If they differ by >40%, flag for sex-specific
panel work. Compare per-decade HC means; if monotonic drift exceeds 0.05
between adjacent decades, flag for per-decade tier-threshold calibration.

OUTPUT
======
step_7_chk_f_sex_age_stratified.md
step_7_chk_f_age_decade_baselines.csv  (HC mean per decade per cohort)

Date: 2026-05-04
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict


def cohen_d(a, b) -> float:
    if len(a) < 2 or len(b) < 2: return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    pooled = math.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / (len(a) + len(b) - 2))
    if pooled == 0: return float("nan")
    return (ma - mb) / pooled


def age_decade(age) -> str:
    """Bin age (numeric) into 10-yr decade label."""
    try:
        a = float(age)
    except (ValueError, TypeError):
        return "unknown"
    if a < 50: return "<50"
    if a < 60: return "50-59"
    if a < 70: return "60-69"
    if a < 80: return "70-79"
    if a < 90: return "80-89"
    return "90+"


def normalize_sex(s: str) -> str:
    if not s: return "unknown"
    s = str(s).lower().strip()
    if s in ("m", "male", "1"): return "M"
    if s in ("f", "female", "2"): return "F"
    return "unknown"


def normalize_status(s: str) -> str:
    if not s: return "unknown"
    s = str(s).lower()
    if any(k in s for k in ["alzheimer", "ad ", "dementia", "mci", "case", "cancer"]): return "case"
    if any(k in s for k in ["healthy", "control", "hc", "normal"]) or s == "0": return "control"
    if s == "1": return "case"
    return "unknown"


def stratify_cohort(samples: list, group_label: str) -> dict:
    """
    samples: list of dicts with keys: sample_id, sex, age, case_status, a_score
    group_label: name for the report
    
    Returns nested dict of stratified Cohen's d and per-decade baselines.
    """
    by_sex = defaultdict(lambda: {"case": [], "control": []})
    by_decade = defaultdict(list)  # decade -> list of HC scores

    n_total = len(samples)
    n_used = 0
    for s in samples:
        a = s.get("a_score")
        if a is None or (isinstance(a, float) and math.isnan(a)): continue
        sex = normalize_sex(s.get("sex"))
        status = normalize_status(s.get("case_status"))
        if status not in ("case", "control"): continue
        if sex in ("M", "F"):
            by_sex[sex][status].append(a)
        if status == "control":
            d = age_decade(s.get("age"))
            by_decade[d].append(a)
        n_used += 1

    sex_strat = {}
    for sx in ("M", "F"):
        cases = by_sex[sx]["case"]
        ctrls = by_sex[sx]["control"]
        sex_strat[sx] = {
            "n_case": len(cases),
            "n_control": len(ctrls),
            "case_mean": statistics.mean(cases) if cases else float("nan"),
            "control_mean": statistics.mean(ctrls) if ctrls else float("nan"),
            "cohen_d": cohen_d(cases, ctrls),
        }

    decade_baselines = {}
    for d in ["<50", "50-59", "60-69", "70-79", "80-89", "90+", "unknown"]:
        scores = by_decade[d]
        if not scores:
            decade_baselines[d] = {"n": 0, "mean": float("nan"), "sd": float("nan")}
            continue
        decade_baselines[d] = {
            "n": len(scores),
            "mean": statistics.mean(scores),
            "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        }

    return {
        "cohort": group_label,
        "n_total": n_total,
        "n_used": n_used,
        "sex_stratified": sex_strat,
        "decade_baselines_HC": decade_baselines,
    }


def load_aibl_addneuromed_results(check_c_per_sample_csv: Path) -> list:
    """Load per-sample A-scores from Check C if it produced them; otherwise empty."""
    if not check_c_per_sample_csv.exists():
        return []
    samples = []
    with open(check_c_per_sample_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try: a = float(row.get("a_score") or row.get("a_immune") or "nan")
            except: a = float("nan")
            samples.append({
                "sample_id": row.get("sample_id"),
                "sex": row.get("sex"),
                "age": row.get("age"),
                "case_status": row.get("case_status") or row.get("disease_status"),
                "a_score": a,
                "cohort": row.get("cohort", ""),
            })
    return samples


def load_breast_cohort_csv(samples_csv: Path) -> list:
    """Load GSE51057 / GSE51032 per-sample CSV from VAL-047 Tightening fresh."""
    if not samples_csv.exists():
        return []
    samples = []
    with open(samples_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # VAL-047 CSVs may not have a pre-computed a_score column;
            # if so, this Check F doesn't have what it needs from this file.
            # Still extract metadata.
            try: a = float(row.get("a_score") or row.get("a_immune") or "nan")
            except: a = float("nan")
            samples.append({
                "sample_id": row.get("sample_id") or row.get("gsm"),
                "sex": row.get("sex") or "F",  # breast cohorts default to female
                "age": row.get("age"),
                "case_status": row.get("case_status"),
                "a_score": a,
            })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_c_per_sample", default="step_7_chk_c_per_sample.csv",
                        help="Optional: per-sample A-score CSV emitted by Check C (if you upgrade Check C to emit one)")
    parser.add_argument("--gse51057", default="VAL047_samples_GSE51057.csv")
    parser.add_argument("--gse51032", default="VAL047_samples_GSE51032.csv")
    parser.add_argument("--report", default="step_7_chk_f_sex_age_stratified.md")
    parser.add_argument("--decade_csv", default="step_7_chk_f_age_decade_baselines.csv")
    args = parser.parse_args()

    print("=" * 72)
    print("STEP 7 / CHECK F — Sex + age-decade stratification")
    print("=" * 72)

    # Try to load each cohort
    cohorts_data = []
    aibl_addn = load_aibl_addneuromed_results(Path(args.check_c_per_sample))
    if aibl_addn:
        # Group by cohort label
        by_cohort = defaultdict(list)
        for s in aibl_addn:
            by_cohort[s.get("cohort", "AIBL_or_AddNeuroMed")].append(s)
        for label, samples in by_cohort.items():
            cohorts_data.append((label, samples))
    else:
        print("\n[INFO] Check C per-sample CSV not found at "
              f"{args.check_c_per_sample}.")
        print("       To enable AIBL/AddNeuroMed sex+age stratification, upgrade Check C to emit per-sample A-scores")
        print("       OR provide the file with columns: sample_id, sex, age, case_status, cohort, a_score")

    gse51057 = load_breast_cohort_csv(Path(args.gse51057))
    gse51032 = load_breast_cohort_csv(Path(args.gse51032))
    if gse51057: cohorts_data.append(("GSE51057_breast", gse51057))
    if gse51032: cohorts_data.append(("GSE51032", gse51032))

    if not cohorts_data:
        print("\nNo cohort data available. Skipping. Re-run after Check C/D produce per-sample CSVs.")
        sys.exit(0)

    # Stratify each
    results = []
    for label, samples in cohorts_data:
        print(f"\n--- Stratifying {label} (n={len(samples)}) ---")
        r = stratify_cohort(samples, label)
        results.append(r)
        # Print sex-stratified
        for sx in ("M", "F"):
            s = r["sex_stratified"][sx]
            print(f"  {sx}:  n_case={s['n_case']:>4}  n_ctrl={s['n_control']:>4}  d={s['cohen_d']:+.3f}")
        # Print decade baselines
        for d in ["50-59", "60-69", "70-79", "80-89"]:
            db = r["decade_baselines_HC"][d]
            print(f"  HC decade {d}: n={db['n']:>3}  mean A_immune={db['mean']:.4f}  sd={db['sd']:.4f}")

    # Report
    with open(args.report, "w") as f:
        f.write("# Step 7 / Check F — Sex + age-decade stratification\n\n")
        f.write("**Date:** 2026-05-04\n")
        f.write("**Purpose:** Verify IAMAtlas-anchored A_immune scoring is robust across sex and age strata. Surfaces calibration cliffs that would justify per-stratum panels (CHK-A11) or per-decade tier thresholds (CHK-A12).\n\n")
        f.write("Recent literature has flagged sex-specific drift in epigenetic age clocks and intervention responses. The framework's universal panels (VAL-053 confirmed unified > sex-specific for AD) need replication on this matrix.\n\n")

        for r in results:
            f.write(f"## {r['cohort']}\n\n")
            f.write(f"- n_total = {r['n_total']}, n_used = {r['n_used']}\n\n")
            f.write("### Sex-stratified Cohen's d\n\n")
            f.write("| Sex | n cases | n controls | Case mean | Control mean | Cohen's d |\n|---|---|---|---|---|---|\n")
            for sx in ("M", "F"):
                s = r["sex_stratified"][sx]
                f.write(f"| {sx} | {s['n_case']} | {s['n_control']} | "
                        f"{s['case_mean']:.4f} | {s['control_mean']:.4f} | "
                        f"**{s['cohen_d']:+.3f}** |\n")
            md = r["sex_stratified"]["M"]["cohen_d"]
            fd = r["sex_stratified"]["F"]["cohen_d"]
            if not (math.isnan(md) or math.isnan(fd)) and abs(md - fd) > 0.4 * max(abs(md), abs(fd)):
                f.write(f"\n**FLAG:** Male/Female Cohen's d differ by >40% — consider sex-specific panels for this disease.\n")
            f.write("\n### HC age-decade baselines\n\n")
            f.write("| Decade | n | A_immune mean | sd |\n|---|---|---|---|\n")
            for d in ["<50", "50-59", "60-69", "70-79", "80-89", "90+"]:
                db = r["decade_baselines_HC"][d]
                if db["n"] == 0: continue
                f.write(f"| {d} | {db['n']} | {db['mean']:.4f} | {db['sd']:.4f} |\n")
            # Check for cliffs
            decades_ordered = ["50-59", "60-69", "70-79", "80-89"]
            means = [r["decade_baselines_HC"][d]["mean"] for d in decades_ordered]
            cliff = False
            for i in range(len(means) - 1):
                if not (math.isnan(means[i]) or math.isnan(means[i+1])):
                    if abs(means[i+1] - means[i]) > 0.05:
                        f.write(f"\n**FLAG:** Decade {decades_ordered[i]} → {decades_ordered[i+1]} mean A_immune jumps by {means[i+1] - means[i]:+.4f} (>0.05 threshold). Consider per-decade tier thresholds.\n")
                        cliff = True
            if not cliff:
                f.write(f"\nNo decade-cliff flagged. Smooth aging drift acceptable for unified tier thresholds.\n")
            f.write("\n")

        f.write("## Decision\n\n")
        f.write("- **PASS:** No sex split flagged AND no decade cliffs. IAMAtlas calibration is robust; unified scoring acceptable.\n")
        f.write("- **PARTIAL:** One flag raised. Document in card, plan per-stratum panel work for next version.\n")
        f.write("- **INVESTIGATE:** Multiple flags. Hold deployment; build per-stratum panels before EDEAR commercial launch.\n")

    # Decade baselines CSV (used by EDEAR scoring engine for per-decade tier thresholds)
    with open(args.decade_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "decade", "n_HC", "mean_A_immune", "sd_A_immune"])
        for r in results:
            for d, db in r["decade_baselines_HC"].items():
                if db["n"] == 0: continue
                w.writerow([r["cohort"], d, db["n"], f"{db['mean']:.4f}", f"{db['sd']:.4f}"])

    print(f"\nReport: {args.report}")
    print(f"Decade baselines: {args.decade_csv}")


if __name__ == "__main__":
    import sys
    main()
