#!/usr/bin/env python3
"""
VAL-047 TIGHTENING — PATCH AGE + RE-RUN ANALYSES FROM CSV
==========================================================
Reads the per-sample CSVs already on disk in ~/Downloads/, fixes the age column
(previous regex grabbed 'age at menarche' instead of 'age'), and re-runs all three
tightening analyses without re-parsing the matrix.

INPUT (already exist):
  ~/Downloads/VAL047_samples_GSE51057.csv
  ~/Downloads/VAL047_samples_GSE51032.csv

OUTPUT:
  ~/Downloads/VAL047_samples_GSE51057_fixed.csv
  ~/Downloads/VAL047_samples_GSE51032_fixed.csv
  ~/Downloads/VAL047_tightening_results_v2.json
"""

import ast
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# PATHS — confirmed
# ============================================================================

DOWN = Path("/Users/hmahaffeyges/Downloads/4-21-26")
CSV_57 = DOWN / "VAL047_samples_GSE51057.csv"
CSV_32 = DOWN / "VAL047_samples_GSE51032.csv"
HEALTHY_BASELINES_PATH = Path("/Users/hmahaffeyges/Downloads/GAPE Work 4-17-26/Evidece 4-18-26/HEALTHY_BASELINES.json")

OUT_57 = DOWN / "VAL047_samples_GSE51057_fixed.csv"
OUT_32 = DOWN / "VAL047_samples_GSE51032_fixed.csv"
OUT_JSON = DOWN / "VAL047_tightening_results_v2.json"

H_MIN_IMMUNE = 0.838889

TtD_WINDOWS = [
    ("0-2 yr",     0.0,   2.0),
    ("2-5 yr",     2.0,   5.0),
    ("5-10 yr",    5.0,  10.0),
    (">10 yr",    10.0, 999.0),
    ("all_pre_dx", 0.0, 999.0),
]

# ============================================================================
# AGE PARSER — STRICT
# ============================================================================
# Match exactly "age:" (with optional whitespace), NOT "age at menarche:" etc.
# The negative lookahead requires that the colon comes right after "age" with
# only whitespace between.

AGE_RE = re.compile(r"^\s*age\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)

def parse_age_from_raw(raw_str):
    """raw_str is the str() form of a python list, e.g. "['gender: F', 'age: 54.322', ...]" """
    if not isinstance(raw_str, str): return float("nan")
    try:
        items = ast.literal_eval(raw_str)
    except Exception:
        return float("nan")
    if not isinstance(items, list): return float("nan")
    for item in items:
        if not isinstance(item, str): continue
        m = AGE_RE.match(item)
        if m:
            try: return float(m.group(1))
            except Exception: return float("nan")
    return float("nan")

# ============================================================================
# HELPERS
# ============================================================================

def summarize(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"),
                    p10=float("nan"), p25=float("nan"), p50=float("nan"),
                    p75=float("nan"), p90=float("nan"))
    return dict(
        n=int(len(x)), mean=float(np.mean(x)),
        sd=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        p10=float(np.percentile(x, 10)), p25=float(np.percentile(x, 25)),
        p50=float(np.percentile(x, 50)), p75=float(np.percentile(x, 75)),
        p90=float(np.percentile(x, 90)),
    )

def cohens_d(cases, controls):
    cases = np.asarray(cases, dtype=float); cases = cases[~np.isnan(cases)]
    controls = np.asarray(controls, dtype=float); controls = controls[~np.isnan(controls)]
    n1, n2 = len(cases), len(controls)
    if n1 < 2 or n2 < 2: return float("nan")
    s1 = float(np.std(cases, ddof=1)); s2 = float(np.std(controls, ddof=1))
    pooled = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled == 0: return 0.0
    return float((np.mean(cases) - np.mean(controls)) / pooled)

def age_decade(age):
    try: age = float(age)
    except Exception: return None
    if math.isnan(age) or age < 0: return None
    if age >= 90: return "90+"
    return f"{int(age // 10) * 10}-{int(age // 10) * 10 + 9}"

def percentile_in_reference(value, p10, p25, p50, p75, p90):
    if value is None or (isinstance(value, float) and math.isnan(value)): return float("nan")
    anchors = sorted([(10, p10), (25, p25), (50, p50), (75, p75), (90, p90)], key=lambda t: t[1])
    if value <= anchors[0][1]: return float(anchors[0][0])
    if value >= anchors[-1][1]: return float(anchors[-1][0])
    for i in range(len(anchors) - 1):
        p_lo, v_lo = anchors[i]; p_hi, v_hi = anchors[i + 1]
        if v_lo <= value <= v_hi:
            if v_hi == v_lo: return (p_lo + p_hi) / 2
            return float(p_lo + (value - v_lo) / (v_hi - v_lo) * (p_hi - p_lo))
    return float("nan")

# ============================================================================
# LOAD + PATCH
# ============================================================================

print("=" * 78)
print("VAL-047 TIGHTENING v2 — patched age parser, re-run from CSVs")
print("=" * 78)

with open(HEALTHY_BASELINES_PATH) as f:
    baselines = json.load(f)
immune_tbl = baselines["tables"]["immune"]
print(f"Healthy baselines loaded: {len(immune_tbl)} age decades (immune class)")
print()

print(f"[load] {CSV_57.name}")
df57 = pd.read_csv(CSV_57)
print(f"  rows: {len(df57)}  columns: {list(df57.columns)}")
print(f"  OLD age summary (from broken regex):")
print(f"    median={df57['age'].median():.1f}  mean={df57['age'].mean():.1f}  max={df57['age'].max():.1f}")
df57["age"] = df57["raw"].apply(parse_age_from_raw)
print(f"  NEW age summary (after patch):")
print(f"    median={df57['age'].median():.1f}  mean={df57['age'].mean():.1f}  max={df57['age'].max():.1f}")
df57.to_csv(OUT_57, index=False)
print(f"  wrote: {OUT_57.name}")
print()

print(f"[load] {CSV_32.name}")
df32 = pd.read_csv(CSV_32)
print(f"  rows: {len(df32)}  columns: {list(df32.columns)}")
print(f"  OLD age summary (from broken regex):")
print(f"    median={df32['age'].median():.1f}  mean={df32['age'].mean():.1f}  max={df32['age'].max():.1f}")
df32["age"] = df32["raw"].apply(parse_age_from_raw)
print(f"  NEW age summary (after patch):")
print(f"    median={df32['age'].median():.1f}  mean={df32['age'].mean():.1f}  max={df32['age'].max():.1f}")
df32.to_csv(OUT_32, index=False)
print(f"  wrote: {OUT_32.name}")
print()

# ============================================================================
# ANALYSIS 1
# ============================================================================

print("=" * 78)
print("ANALYSIS 1 — Mean A-score (Kresovich-538) per TtD window")
print("=" * 78)

def window_analysis(df, case_filter, label):
    ctrl = df[df["status"] == "control"]
    ctrl_A = ctrl["A_kresovich"].dropna().values
    ctrl_summary = summarize(ctrl_A)
    out = {"label": label, "controls": ctrl_summary, "windows": {}}
    for name, lo, hi in TtD_WINDOWS:
        cases = df[case_filter & (df["ttd_years"] >= lo) & (df["ttd_years"] < hi)]
        if len(cases) == 0:
            out["windows"][name] = {"n_cases": 0}; continue
        case_A = cases["A_kresovich"].dropna().values
        ages = cases["age"].dropna().values
        cs = summarize(case_A)
        out["windows"][name] = dict(
            n_cases=int(len(cases)), case_A_mean=cs["mean"], case_A_sd=cs["sd"],
            case_A_p50=cs["p50"], ctrl_A_mean=ctrl_summary["mean"], ctrl_A_sd=ctrl_summary["sd"],
            delta_A=cs["mean"] - ctrl_summary["mean"], cohens_d=cohens_d(case_A, ctrl_A),
            age_mean=float(np.mean(ages)) if len(ages) else float("nan"),
            age_median=float(np.median(ages)) if len(ages) else float("nan"),
            age_min=float(np.min(ages)) if len(ages) else float("nan"),
            age_max=float(np.max(ages)) if len(ages) else float("nan"),
        )
    return out

def print_window_table(result):
    ctrl = result["controls"]
    print(f"\n[{result['label']}]")
    print(f"  controls: n={ctrl['n']}  A_mean={ctrl['mean']:.4f}  A_p10={ctrl['p10']:.4f}  A_p90={ctrl['p90']:.4f}")
    print(f"  {'window':12} {'n':>4}  {'A_case':>8}  {'A_ctrl':>8}  {'delta':>8}  {'d':>7}  {'age_med':>7}  {'age_rng':>13}")
    for name, _, _ in TtD_WINDOWS:
        d = result["windows"].get(name, {})
        if d.get("n_cases", 0) == 0: continue
        rng = f"[{d['age_min']:.0f}-{d['age_max']:.0f}]"
        print(f"  {name:12} {d['n_cases']:>4}  {d['case_A_mean']:>8.4f}  {d['ctrl_A_mean']:>8.4f}  "
              f"{d['delta_A']:>+8.4f}  {d['cohens_d']:>+7.3f}  {d['age_median']:>7.1f}  {rng:>13}")

ph9_breast = window_analysis(df57, (df57["status"] == "case") & (df57["cancer_type"] == "primary"),
                              "Phase 9 breast (GSE51057)")
ph12_breast = window_analysis(df32, (df32["status"] == "case") & (df32["cancer_type"] == "primary"),
                               "Phase 12 breast (GSE51032)")
ph12_colorectal = window_analysis(df32, (df32["status"] == "case") & (df32["cancer_type"] == "secondary"),
                                   "Phase 12 colorectal (GSE51032)")
for r in [ph9_breast, ph12_breast, ph12_colorectal]:
    print_window_table(r)

# ============================================================================
# ANALYSIS 2 — within-cohort percentile (NOT vs HEALTHY_BASELINES)
# ============================================================================
# Because the Kresovich-538 panel sits in a CpG region with mean A ~ 0.44 rather
# than the broader-immune-class baseline mean A ~ 0.95, comparison against
# HEALTHY_BASELINES.json gives meaningless saturated percentiles. Instead, we
# use AGE-MATCHED CONTROL DECILES from the same cohort (GSE51032 controls or
# GSE51057 controls) as the reference distribution.

print()
print("=" * 78)
print("ANALYSIS 2 — Case A-scores vs age-matched WITHIN-COHORT controls")
print("=" * 78)
print("""
  Reference: controls from the same cohort, stratified by age decade.
  Why: the Kresovich-538 panel's A-score scale (~0.44) is not comparable to
  HEALTHY_BASELINES.json (~0.95, full-immune-class). Within-cohort age-matched
  controls give a clean apples-to-apples comparison.
  pct>=P90 = fraction of cases at/above 90th percentile of age-matched controls
  pct<=P10 = fraction of cases at/below 10th percentile of age-matched controls
""")

def build_age_decile_table(df_controls):
    """Compute control A-score percentiles per age decade."""
    out = {}
    df_controls = df_controls.copy()
    df_controls["age_decade"] = df_controls["age"].apply(age_decade)
    for ad, group in df_controls.groupby("age_decade"):
        if ad is None: continue
        A = group["A_kresovich"].dropna().values
        if len(A) < 5: continue  # need minimum N for stable percentiles
        out[ad] = dict(n=int(len(A)),
                       A_p10=float(np.percentile(A, 10)),
                       A_p25=float(np.percentile(A, 25)),
                       A_p50=float(np.percentile(A, 50)),
                       A_p75=float(np.percentile(A, 75)),
                       A_p90=float(np.percentile(A, 90)))
    return out

def percentile_analysis(df, case_filter, control_table, label):
    out = {"label": label, "windows": {}}
    for name, lo, hi in TtD_WINDOWS:
        cases = df[case_filter & (df["ttd_years"] >= lo) & (df["ttd_years"] < hi)]
        if len(cases) == 0:
            out["windows"][name] = {"n_cases": 0}; continue
        pcts = []; missing_decade = 0
        for _, row in cases.iterrows():
            ad = age_decade(row.get("age"))
            if ad is None or ad not in control_table:
                missing_decade += 1; continue
            A = row.get("A_kresovich")
            if A is None or (isinstance(A, float) and math.isnan(A)): continue
            ref = control_table[ad]
            pcts.append(percentile_in_reference(A, ref["A_p10"], ref["A_p25"],
                         ref["A_p50"], ref["A_p75"], ref["A_p90"]))
        if not pcts:
            out["windows"][name] = {"n_cases": int(len(cases)),
                                     "n_missing_age_match": int(missing_decade)}
            continue
        p = np.asarray(pcts)
        out["windows"][name] = dict(
            n_cases=int(len(cases)), n_scored=int(len(p)),
            n_missing_age_match=int(missing_decade),
            pct_mean=float(np.mean(p)), pct_median=float(np.median(p)),
            pct_p25=float(np.percentile(p, 25)), pct_p75=float(np.percentile(p, 75)),
            frac_ge_p90=float(np.mean(p >= 90) * 100),
            frac_le_p10=float(np.mean(p <= 10) * 100),
        )
    return out

def print_pct(r):
    print(f"\n[{r['label']}]")
    print(f"  {'window':12} {'n':>4}  {'med pct':>8}  {'IQR':>12}  {'>=P90 %':>8}  {'<=P10 %':>8}")
    for name, _, _ in TtD_WINDOWS:
        d = r["windows"].get(name, {})
        if d.get("n_cases", 0) == 0: continue
        if "pct_median" not in d:
            print(f"  {name:12} {d['n_cases']:>4}  (no age-matched controls)")
            continue
        iqr = f"[{d['pct_p25']:.0f},{d['pct_p75']:.0f}]"
        print(f"  {name:12} {d['n_cases']:>4}  {d['pct_median']:>8.1f}  {iqr:>12}  "
              f"{d['frac_ge_p90']:>8.1f}  {d['frac_le_p10']:>8.1f}")

ctrl_decile_57 = build_age_decile_table(df57[df57["status"] == "control"])
ctrl_decile_32 = build_age_decile_table(df32[df32["status"] == "control"])

print(f"\n  GSE51057 control age decade table:")
for ad, v in sorted(ctrl_decile_57.items()):
    print(f"    {ad:8} n={v['n']:>3}  A_p10={v['A_p10']:.4f}  A_p50={v['A_p50']:.4f}  A_p90={v['A_p90']:.4f}")
print(f"\n  GSE51032 control age decade table:")
for ad, v in sorted(ctrl_decile_32.items()):
    print(f"    {ad:8} n={v['n']:>3}  A_p10={v['A_p10']:.4f}  A_p50={v['A_p50']:.4f}  A_p90={v['A_p90']:.4f}")

pct_ph9  = percentile_analysis(df57, (df57["status"]=="case")&(df57["cancer_type"]=="primary"),
                                ctrl_decile_57, "Phase 9 breast (GSE51057)")
pct_ph12 = percentile_analysis(df32, (df32["status"]=="case")&(df32["cancer_type"]=="primary"),
                                ctrl_decile_32, "Phase 12 breast (GSE51032)")
pct_crc  = percentile_analysis(df32, (df32["status"]=="case")&(df32["cancer_type"]=="secondary"),
                                ctrl_decile_32, "Phase 12 colorectal (GSE51032)")
for r in [pct_ph9, pct_ph12, pct_crc]:
    print_pct(r)

# ============================================================================
# ANALYSIS 3 — colorectal C18/C19/C20 stratification (unchanged from v1)
# ============================================================================

print()
print("=" * 78)
print("ANALYSIS 3 — Colorectal subtype stratification (C18 / C19 / C20)")
print("=" * 78)

ctrl32 = df32[df32["status"] == "control"]
ctrl_A32 = ctrl32["A_kresovich"].dropna().values
ctrl_sum = summarize(ctrl_A32)
print(f"\n  controls (shared): n={ctrl_sum['n']}  A_mean={ctrl_sum['mean']:.4f}  A_sd={ctrl_sum['sd']:.4f}")
print(f"\n  {'subtype':10} {'n':>4}  {'A_case':>8}  {'delta_A':>9}  {'cohens_d':>9}  {'age_med':>7}  {'TtD_med':>7}")

subtype_out = {"controls": ctrl_sum, "subtypes": {}}
for prefix in ["C18", "C19", "C20"]:
    sub = df32[df32["icd_code"].fillna("").astype(str).str.startswith(prefix) &
               (df32["status"] == "case")]
    if len(sub) == 0:
        subtype_out["subtypes"][prefix] = {"n": 0}
        print(f"  {prefix:10} {'n/a':>4}"); continue
    A = sub["A_kresovich"].dropna().values
    ages = sub["age"].dropna().values
    ttds = sub["ttd_years"].dropna().values
    cs = summarize(A)
    d = dict(
        n=int(len(sub)), case_A_mean=cs["mean"], case_A_sd=cs["sd"], case_A_p50=cs["p50"],
        delta_A=cs["mean"] - ctrl_sum["mean"], cohens_d=cohens_d(A, ctrl_A32),
        age_median=float(np.median(ages)) if len(ages) else float("nan"),
        age_min=float(np.min(ages)) if len(ages) else float("nan"),
        age_max=float(np.max(ages)) if len(ages) else float("nan"),
        ttd_median=float(np.median(ttds)) if len(ttds) else float("nan"),
    )
    subtype_out["subtypes"][prefix] = d
    print(f"  {prefix:10} {d['n']:>4}  {d['case_A_mean']:>8.4f}  {d['delta_A']:>+9.4f}  "
          f"{d['cohens_d']:>+9.3f}  {d['age_median']:>7.1f}  {d['ttd_median']:>7.1f}")

pooled_A = df32[(df32["cancer_type"]=="secondary") & (df32["status"]=="case")]["A_kresovich"].dropna().values
if len(pooled_A) > 0:
    cs = summarize(pooled_A)
    p = dict(n=int(len(pooled_A)), case_A_mean=cs["mean"], delta_A=cs["mean"]-ctrl_sum["mean"],
             cohens_d=cohens_d(pooled_A, ctrl_A32))
    subtype_out["pooled"] = p
    print(f"  {'POOLED':10} {p['n']:>4}  {p['case_A_mean']:>8.4f}  {p['delta_A']:>+9.4f}  {p['cohens_d']:>+9.3f}")

# ============================================================================
# SAVE
# ============================================================================

combined = {
    "meta": {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "version": "v2 - age regex patched, percentile vs within-cohort controls",
        "H_min_immune": H_MIN_IMMUNE,
        "panel": "Kresovich-538",
    },
    "counts": {
        "GSE51057": {
            "total": int(len(df57)),
            "controls": int((df57["status"]=="control").sum()),
            "breast_cases": int(((df57["status"]=="case") & (df57["cancer_type"]=="primary")).sum()),
        },
        "GSE51032": {
            "total": int(len(df32)),
            "controls": int((df32["status"]=="control").sum()),
            "breast_cases": int(((df32["status"]=="case") & (df32["cancer_type"]=="primary")).sum()),
            "colorectal_cases": int(((df32["status"]=="case") & (df32["cancer_type"]=="secondary")).sum()),
        },
    },
    "control_decile_tables": {
        "GSE51057_immune_by_age_decade": ctrl_decile_57,
        "GSE51032_immune_by_age_decade": ctrl_decile_32,
    },
    "analysis_1_mean_Ascore_by_window": {
        "phase9_breast": ph9_breast,
        "phase12_breast": ph12_breast,
        "phase12_colorectal": ph12_colorectal,
    },
    "analysis_2_percentile_within_cohort": {
        "phase9_breast": pct_ph9,
        "phase12_breast": pct_ph12,
        "phase12_colorectal": pct_crc,
    },
    "analysis_3_colorectal_subtypes": subtype_out,
}
with open(OUT_JSON, "w") as f:
    json.dump(combined, f, indent=2, default=str)

print()
print("=" * 78)
print(f"[save] {OUT_JSON}")
print("=" * 78)
print()
print("Counts:")
print(f"  GSE51057: {combined['counts']['GSE51057']}")
print(f"  GSE51032: {combined['counts']['GSE51032']}")
print()
print("Done. Paste output to Walther.")
