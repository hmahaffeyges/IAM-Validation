#!/usr/bin/env python3
"""
VAL-047 TIGHTENING — Self-Contained Re-Analysis
================================================
Reads GSE51057 (Phase 9) and GSE51032 (Phase 12) series matrices directly.
Computes per-sample immune-class A-scores with the Kresovich-538 panel.
Parses age, time-to-diagnosis, and ICD code from sample metadata headers.
Runs three tightening analyses in one pass:

  (1) Mean A-score per TtD window, both cohorts, both cancers
  (2) A-score placed vs age-matched healthy percentile bands
  (3) Colorectal C18 / C19 / C20 subtype stratification

Outputs:
  - ~/Downloads/VAL047_tightening_results.json
  - Human-readable summary to stdout

No dependency on prior Phase 9 / Phase 12 JSON outputs.
All paths confirmed with `find` 2026-04-21.
"""

import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIG — paths confirmed via find on 2026-04-21
# ============================================================================

GSE51032_MATRIX = Path("/Users/hmahaffeyges/IAMPerformance/GSE51032_replication/GSE51032_series_matrix.txt.gz")
GSE51057_MATRIX = Path("/Users/hmahaffeyges/IAMPerformance/VAL047 Testing Suite (10 Year detection)/GSE51057_series_matrix.txt.gz")
HEALTHY_BASELINES_PATH = Path("/Users/hmahaffeyges/Downloads/GAPE Work 4-17-26/Evidece 4-18-26/HEALTHY_BASELINES.json")
KRESOVICH_PANEL_PATH   = Path("/Users/hmahaffeyges/IAMPerformance/VAL047 Testing Suite (10 Year detection)/kresovich_100_cpgs.json")

OUT_DIR = Path.home() / "Downloads"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "VAL047_tightening_results.json"

H_MIN_IMMUNE = 0.838889

TtD_WINDOWS = [
    ("0-2 yr",     0.0,   2.0),
    ("2-5 yr",     2.0,   5.0),
    ("5-10 yr",    5.0,  10.0),
    (">10 yr",    10.0, 999.0),
    ("all_pre_dx", 0.0, 999.0),
]

# ============================================================================
# HELPERS
# ============================================================================

def H_binary(beta):
    if beta is None or not (0.0 < beta < 1.0) or math.isnan(beta):
        return 0.0
    return -beta * math.log2(beta) - (1.0 - beta) * math.log2(1.0 - beta)

def A_score(beta):
    return H_binary(beta) / H_MIN_IMMUNE

def summarize(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0, mean=float("nan"), sd=float("nan"),
                    p10=float("nan"), p25=float("nan"), p50=float("nan"),
                    p75=float("nan"), p90=float("nan"))
    return dict(
        n   = int(len(x)),
        mean= float(np.mean(x)),
        sd  = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        p10 = float(np.percentile(x, 10)),
        p25 = float(np.percentile(x, 25)),
        p50 = float(np.percentile(x, 50)),
        p75 = float(np.percentile(x, 75)),
        p90 = float(np.percentile(x, 90)),
    )

def cohens_d(cases, controls):
    cases    = np.asarray(cases,    dtype=float); cases    = cases[~np.isnan(cases)]
    controls = np.asarray(controls, dtype=float); controls = controls[~np.isnan(controls)]
    n1, n2 = len(cases), len(controls)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = float(np.std(cases,    ddof=1))
    s2 = float(np.std(controls, ddof=1))
    pooled = math.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(cases) - np.mean(controls)) / pooled)

def age_decade(age):
    try:
        age = float(age)
    except Exception:
        return None
    if math.isnan(age) or age < 0:
        return None
    if age >= 90:
        return "90+"
    d = int(age // 10) * 10
    return f"{d}-{d + 9}"

def percentile_in_reference(value, ref_p10, ref_p25, ref_p50, ref_p75, ref_p90):
    """Piecewise-linear interpolation across 5 anchor percentiles."""
    if value is None or math.isnan(value):
        return float("nan")
    anchors = sorted([(10, ref_p10), (25, ref_p25), (50, ref_p50),
                      (75, ref_p75), (90, ref_p90)], key=lambda t: t[1])
    if value <= anchors[0][1]:  return float(anchors[0][0])
    if value >= anchors[-1][1]: return float(anchors[-1][0])
    for i in range(len(anchors) - 1):
        p_lo, v_lo = anchors[i]
        p_hi, v_hi = anchors[i + 1]
        if v_lo <= value <= v_hi:
            if v_hi == v_lo: return (p_lo + p_hi) / 2
            frac = (value - v_lo) / (v_hi - v_lo)
            return float(p_lo + frac * (p_hi - p_lo))
    return float("nan")

# ============================================================================
# LOAD PANEL + BASELINES
# ============================================================================

print("=" * 78)
print("VAL-047 TIGHTENING — single-pass re-analysis")
print("=" * 78)

with open(KRESOVICH_PANEL_PATH) as f:
    kresovich_data = json.load(f)

# Handle either list or dict-with-list format
if isinstance(kresovich_data, list):
    kresovich_cpgs = set(kresovich_data)
elif isinstance(kresovich_data, dict):
    for key in ("cpgs", "panel", "cpg_ids", "probes"):
        if key in kresovich_data and isinstance(kresovich_data[key], list):
            kresovich_cpgs = set(kresovich_data[key])
            break
    else:
        # Flatten: take all string values that look like cg########
        kresovich_cpgs = set()
        for v in kresovich_data.values():
            if isinstance(v, list):
                kresovich_cpgs.update(x for x in v if isinstance(x, str) and x.startswith("cg"))
else:
    sys.exit(f"[FATAL] Cannot parse kresovich panel format: {type(kresovich_data)}")

print(f"Kresovich panel loaded: {len(kresovich_cpgs)} CpGs")

with open(HEALTHY_BASELINES_PATH) as f:
    baselines = json.load(f)
immune_tbl = baselines["tables"]["immune"]
print(f"Healthy baselines loaded: {len(immune_tbl)} age decades (immune class)")
print()

# ============================================================================
# MATRIX PARSER
# ============================================================================

SAMPLE_META_KEYS = [
    "!Sample_geo_accession",
    "!Sample_title",
    "!Sample_characteristics_ch1",
    "!Sample_source_name_ch1",
]

def read_series_matrix(path):
    """
    Parse a GEO series matrix .txt.gz file.
    Returns:
      meta_df  : DataFrame indexed by GSM with sample metadata fields
      beta_df  : DataFrame indexed by CpG, columns = GSM samples (filtered to Kresovich panel)
    """
    print(f"[parse] {path.name} ...")
    meta_rows = {}
    data_lines = []
    in_data = False
    header = None

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_data = True
                header = next(f).rstrip("\n").split("\t")
                continue
            if line.startswith("!series_matrix_table_end"):
                in_data = False
                continue
            if in_data:
                if not line: continue
                fields = line.split("\t")
                cpg = fields[0].strip('"')
                if cpg in kresovich_cpgs:
                    data_lines.append(fields)
                continue
            if line.startswith("!Sample_"):
                key = line.split("\t", 1)[0]
                vals = [v.strip('"') for v in line.split("\t")[1:]]
                meta_rows.setdefault(key, []).append(vals)

    if header is None:
        sys.exit(f"[FATAL] No matrix data block found in {path}")

    samples = [h.strip('"') for h in header[1:]]

    # Flatten characteristics: multiple !Sample_characteristics_ch1 lines exist,
    # each with the same column count. Concatenate into list-of-lists per sample.
    per_sample_meta = {gsm: {"characteristics": []} for gsm in samples}
    for key, rows in meta_rows.items():
        for row in rows:
            if key == "!Sample_characteristics_ch1":
                for i, gsm in enumerate(samples):
                    if i < len(row):
                        per_sample_meta[gsm]["characteristics"].append(row[i])
            elif key == "!Sample_title":
                for i, gsm in enumerate(samples):
                    if i < len(row):
                        per_sample_meta[gsm]["title"] = row[i]
            elif key == "!Sample_source_name_ch1":
                for i, gsm in enumerate(samples):
                    if i < len(row):
                        per_sample_meta[gsm]["source_name"] = row[i]
            elif key == "!Sample_geo_accession":
                pass  # samples list already has GSM IDs

    # Build beta matrix
    if not data_lines:
        sys.exit(f"[FATAL] No Kresovich panel CpGs found in {path}")

    cpg_ids = [row[0].strip('"') for row in data_lines]
    betas = []
    for row in data_lines:
        row_vals = []
        for v in row[1:]:
            v = v.strip('"')
            if v in ("", "NA", "NaN", "null"):
                row_vals.append(np.nan)
            else:
                try:
                    row_vals.append(float(v))
                except ValueError:
                    row_vals.append(np.nan)
        betas.append(row_vals)
    beta_df = pd.DataFrame(betas, index=cpg_ids, columns=samples)

    meta_df = pd.DataFrame.from_dict(per_sample_meta, orient="index")
    print(f"  samples: {len(samples)}   CpGs (panel ∩ matrix): {len(cpg_ids)}")
    return meta_df, beta_df

# ============================================================================
# METADATA EXTRACTION
# ============================================================================

def extract_fields(characteristics):
    """Pull age, gender, status, TtD, ICD out of !Sample_characteristics_ch1 lines."""
    out = {
        "age": float("nan"),
        "gender": None,
        "status": None,        # 'case' | 'control'
        "ttd_years": float("nan"),
        "icd_code": None,
        "cancer_site": None,
        "raw": characteristics,
    }
    if not characteristics:
        return out
    for c in characteristics:
        if c is None: continue
        cl = c.lower().strip()
        # age
        m = re.search(r"age[^:=]*[:=]\s*([0-9]+(?:\.[0-9]+)?)", cl)
        if m:
            out["age"] = float(m.group(1))
        # gender
        m = re.search(r"gender[^:=]*[:=]\s*([a-zA-Z]+)", cl)
        if m:
            out["gender"] = m.group(1).lower()
        # status / disease state
        if re.search(r"(disease state|case/control|status)[^:=]*[:=]", cl):
            after = cl.split(":", 1)[-1].strip() if ":" in cl else cl.split("=", 1)[-1].strip()
            if any(x in after for x in ["control", "healthy", "non-cancer"]):
                out["status"] = "control"
            elif any(x in after for x in ["case", "cancer"]):
                out["status"] = "case"
        # ICD-10
        m = re.search(r"icd[^:=]*[:=]\s*([A-Z][0-9]+(?:\.[0-9]+)?)", c, re.IGNORECASE)
        if m:
            out["icd_code"] = m.group(1).upper().strip(".")
        # Alternate form: characteristic line literally contains "C50" or "C18-C20"
        if out["icd_code"] is None:
            m2 = re.search(r"\b(C[0-9]{2,3})(?:\.[0-9]+)?\b", c)
            if m2:
                out["icd_code"] = m2.group(1).upper()
        # cancer site / type
        m = re.search(r"(cancer type|cancer site|tumour site|tumor site|tissue)[^:=]*[:=]\s*(.+)$", cl)
        if m:
            out["cancer_site"] = m.group(2).strip()
        # time to diagnosis
        m = re.search(r"(time to diagnosis|time_to_diagnosis|ttd|years to diagnosis|follow[- ]up time)[^:=]*[:=]\s*([-0-9]+(?:\.[0-9]+)?)", cl)
        if m:
            try:
                out["ttd_years"] = float(m.group(2))
            except Exception:
                pass
    return out

# ============================================================================
# BUILD PER-SAMPLE FRAMES
# ============================================================================

def build_per_sample(meta_df, beta_df, cohort_tag):
    """Compute per-sample immune A-score + attach extracted metadata."""
    rows = []
    for gsm in beta_df.columns:
        panel = beta_df[gsm].dropna().values
        if len(panel) < 10:
            A = float("nan")
        else:
            A_vals = [A_score(b) for b in panel]
            A = float(np.mean(A_vals))
        chars = meta_df.loc[gsm, "characteristics"] if gsm in meta_df.index else []
        fields = extract_fields(chars)
        fields.update({
            "gsm": gsm,
            "cohort": cohort_tag,
            "A_kresovich": A,
            "n_cpgs_used": int(len(panel)),
        })
        rows.append(fields)
    df = pd.DataFrame(rows)
    return df

# ============================================================================
# CLASSIFY CASES vs CONTROLS BASED ON ICD + SITE
# ============================================================================

def classify_cohort(df, primary_icd_prefix, secondary_icd_prefixes=None):
    """
    Assign 'status' and 'cancer_type' columns.
    primary_icd_prefix: e.g. 'C50' for breast
    secondary_icd_prefixes: e.g. ['C18', 'C19', 'C20'] for colorectal
    """
    secondary = secondary_icd_prefixes or []
    df = df.copy()
    def classify(row):
        icd = (row.get("icd_code") or "")
        site = (row.get("cancer_site") or "").lower()
        if icd.startswith(primary_icd_prefix):
            return ("case", "primary")
        for p in secondary:
            if icd.startswith(p):
                return ("case", "secondary")
        if row.get("status") == "control": return ("control", None)
        if row.get("status") == "case":    return ("case", "other")
        # No explicit status → heuristic
        if icd == "" or icd is None:
            return ("control", None)
        return ("case", "other")
    df["status"], df["cancer_type"] = zip(*df.apply(classify, axis=1))
    return df

# ============================================================================
# READ BOTH MATRICES
# ============================================================================

print("[1/4] Reading GSE51057 (Phase 9 — breast-focused nested case-control)...")
meta57, beta57 = read_series_matrix(GSE51057_MATRIX)
samples_57 = build_per_sample(meta57, beta57, cohort_tag="GSE51057")
samples_57 = classify_cohort(samples_57, primary_icd_prefix="C50")
print(f"  per-sample A-scores computed: {len(samples_57)}")
print()

print("[2/4] Reading GSE51032 (Phase 12 — mixed cancer nested case-control)...")
meta32, beta32 = read_series_matrix(GSE51032_MATRIX)
samples_32 = build_per_sample(meta32, beta32, cohort_tag="GSE51032")
samples_32 = classify_cohort(samples_32, primary_icd_prefix="C50",
                              secondary_icd_prefixes=["C18", "C19", "C20"])
print(f"  per-sample A-scores computed: {len(samples_32)}")
print()

# Save per-sample frames for the record
samples_57.to_csv(OUT_DIR / "VAL047_samples_GSE51057.csv", index=False)
samples_32.to_csv(OUT_DIR / "VAL047_samples_GSE51032.csv", index=False)
print(f"  per-sample CSVs written to {OUT_DIR}/")
print()

# ============================================================================
# ANALYSIS 1 — MEAN A-SCORE PER TtD WINDOW
# ============================================================================

print("=" * 78)
print("ANALYSIS 1 — Mean A-score (Kresovich-538) per TtD window")
print("=" * 78)

def window_analysis(df, case_filter, label):
    """
    df: per-sample frame
    case_filter: boolean Series selecting cases for this analysis
    """
    ctrl = df[df["status"] == "control"]
    ctrl_A = ctrl["A_kresovich"].dropna().values
    ctrl_summary = summarize(ctrl_A)
    out = {"label": label, "controls": ctrl_summary, "windows": {}}
    for name, lo, hi in TtD_WINDOWS:
        cases = df[case_filter & (df["ttd_years"] >= lo) & (df["ttd_years"] < hi)]
        if len(cases) == 0:
            out["windows"][name] = {"n_cases": 0}
            continue
        case_A = cases["A_kresovich"].dropna().values
        ages   = cases["age"].dropna().values
        cs = summarize(case_A)
        out["windows"][name] = dict(
            n_cases     = int(len(cases)),
            case_A_mean = cs["mean"],
            case_A_sd   = cs["sd"],
            case_A_p50  = cs["p50"],
            ctrl_A_mean = ctrl_summary["mean"],
            ctrl_A_sd   = ctrl_summary["sd"],
            delta_A     = cs["mean"] - ctrl_summary["mean"],
            cohens_d    = cohens_d(case_A, ctrl_A),
            age_mean    = float(np.mean(ages)) if len(ages) else float("nan"),
            age_median  = float(np.median(ages)) if len(ages) else float("nan"),
        )
    return out

def print_window_table(result):
    ctrl = result["controls"]
    print(f"\n[{result['label']}]")
    print(f"  controls: n={ctrl['n']}  A_mean={ctrl['mean']:.4f}  "
          f"A_p10={ctrl['p10']:.4f}  A_p90={ctrl['p90']:.4f}")
    print(f"  {'window':12} {'n':>4}  {'A_case':>8}  {'A_ctrl':>8}  {'delta':>8}  "
          f"{'d':>7}  {'age_med':>7}")
    for name, _, _ in TtD_WINDOWS:
        d = result["windows"].get(name, {})
        if d.get("n_cases", 0) == 0:
            continue
        print(f"  {name:12} {d['n_cases']:>4}  "
              f"{d['case_A_mean']:>8.4f}  {d['ctrl_A_mean']:>8.4f}  "
              f"{d['delta_A']:>+8.4f}  {d['cohens_d']:>+7.3f}  "
              f"{d['age_median']:>7.1f}")

ph9_breast = window_analysis(
    samples_57,
    (samples_57["status"] == "case") & (samples_57["cancer_type"] == "primary"),
    "Phase 9 breast (GSE51057)"
)
ph12_breast = window_analysis(
    samples_32,
    (samples_32["status"] == "case") & (samples_32["cancer_type"] == "primary"),
    "Phase 12 breast (GSE51032)"
)
ph12_colorectal = window_analysis(
    samples_32,
    (samples_32["status"] == "case") & (samples_32["cancer_type"] == "secondary"),
    "Phase 12 colorectal (GSE51032)"
)

for result in [ph9_breast, ph12_breast, ph12_colorectal]:
    print_window_table(result)

# ============================================================================
# ANALYSIS 2 — PERCENTILE vs AGE-MATCHED HEALTHY
# ============================================================================

print()
print("=" * 78)
print("ANALYSIS 2 — A-scores vs age-matched healthy percentile bands")
print("=" * 78)
print("""
  Interpretation:
    median pct = median percentile of cases relative to AGE-MATCHED healthy distribution
    pct>=P90  = fraction of cases sitting at/above 90th percentile of age-matched healthy
    pct<=P10  = fraction of cases sitting at/below 10th percentile of age-matched healthy
""")

def percentile_analysis(df, case_filter, label):
    out = {"label": label, "windows": {}}
    for name, lo, hi in TtD_WINDOWS:
        cases = df[case_filter & (df["ttd_years"] >= lo) & (df["ttd_years"] < hi)]
        if len(cases) == 0:
            out["windows"][name] = {"n_cases": 0}
            continue
        pcts = []
        for _, row in cases.iterrows():
            ad = age_decade(row.get("age"))
            if ad is None or ad not in immune_tbl: continue
            A = row.get("A_kresovich")
            if A is None or math.isnan(A): continue
            ref = immune_tbl[ad]
            pcts.append(percentile_in_reference(A,
                          ref["A_p10"], ref["A_p25"], ref["A_p50"],
                          ref["A_p75"], ref["A_p90"]))
        if not pcts:
            out["windows"][name] = {"n_cases": int(len(cases)), "no_age_match": True}
            continue
        p = np.asarray(pcts)
        out["windows"][name] = dict(
            n_cases     = int(len(cases)),
            n_scored    = int(len(p)),
            pct_mean    = float(np.mean(p)),
            pct_median  = float(np.median(p)),
            pct_p25     = float(np.percentile(p, 25)),
            pct_p75     = float(np.percentile(p, 75)),
            frac_ge_p90 = float(np.mean(p >= 90) * 100),
            frac_le_p10 = float(np.mean(p <= 10) * 100),
        )
    return out

def print_percentile_table(result):
    print(f"\n[{result['label']}]")
    print(f"  {'window':12} {'n':>4}  {'med pct':>8}  {'IQR':>12}  {'>=P90 %':>8}  {'<=P10 %':>8}")
    for name, _, _ in TtD_WINDOWS:
        d = result["windows"].get(name, {})
        if d.get("n_cases", 0) == 0:
            continue
        if d.get("no_age_match"):
            print(f"  {name:12} {d['n_cases']:>4}  (no age metadata)")
            continue
        iqr = f"[{d['pct_p25']:.0f},{d['pct_p75']:.0f}]"
        print(f"  {name:12} {d['n_cases']:>4}  "
              f"{d['pct_median']:>8.1f}  {iqr:>12}  "
              f"{d['frac_ge_p90']:>8.1f}  {d['frac_le_p10']:>8.1f}")

pct_ph9  = percentile_analysis(samples_57, (samples_57["status"]=="case")&(samples_57["cancer_type"]=="primary"),   "Phase 9 breast (GSE51057)")
pct_ph12 = percentile_analysis(samples_32, (samples_32["status"]=="case")&(samples_32["cancer_type"]=="primary"),   "Phase 12 breast (GSE51032)")
pct_crc  = percentile_analysis(samples_32, (samples_32["status"]=="case")&(samples_32["cancer_type"]=="secondary"), "Phase 12 colorectal (GSE51032)")

for r in [pct_ph9, pct_ph12, pct_crc]:
    print_percentile_table(r)

# ============================================================================
# ANALYSIS 3 — C18 / C19 / C20 SUBTYPE STRATIFICATION
# ============================================================================

print()
print("=" * 78)
print("ANALYSIS 3 — Colorectal subtype stratification (C18 / C19 / C20)")
print("=" * 78)

subtype_out = {"controls": None, "subtypes": {}}
ctrl32 = samples_32[samples_32["status"] == "control"]
ctrl_A32 = ctrl32["A_kresovich"].dropna().values
ctrl_sum = summarize(ctrl_A32)
subtype_out["controls"] = ctrl_sum
print(f"\n  controls (shared): n={ctrl_sum['n']}  A_mean={ctrl_sum['mean']:.4f}  "
      f"A_sd={ctrl_sum['sd']:.4f}")
print(f"\n  {'subtype':10} {'n':>4}  {'A_case':>8}  {'delta_A':>9}  {'cohens_d':>9}  "
      f"{'age_med':>7}  {'TtD_med':>7}")

def subtype_row(df, prefix):
    sub = df[df["icd_code"].fillna("").astype(str).str.startswith(prefix) &
             (df["status"] == "case")]
    if len(sub) == 0:
        return {"n": 0}
    A = sub["A_kresovich"].dropna().values
    cs = summarize(A)
    ages = sub["age"].dropna().values
    ttds = sub["ttd_years"].dropna().values
    return dict(
        n            = int(len(sub)),
        case_A_mean  = cs["mean"],
        case_A_sd    = cs["sd"],
        case_A_p50   = cs["p50"],
        delta_A      = cs["mean"] - ctrl_sum["mean"],
        cohens_d     = cohens_d(A, ctrl_A32),
        age_median   = float(np.median(ages)) if len(ages) else float("nan"),
        ttd_median   = float(np.median(ttds)) if len(ttds) else float("nan"),
    )

for prefix in ["C18", "C19", "C20"]:
    d = subtype_row(samples_32, prefix)
    subtype_out["subtypes"][prefix] = d
    if d["n"] == 0:
        print(f"  {prefix:10} {'n/a':>4}")
        continue
    print(f"  {prefix:10} {d['n']:>4}  "
          f"{d['case_A_mean']:>8.4f}  {d['delta_A']:>+9.4f}  "
          f"{d['cohens_d']:>+9.3f}  "
          f"{d['age_median']:>7.1f}  {d['ttd_median']:>7.1f}")

# pooled sanity check
pooled_cases = samples_32[(samples_32["cancer_type"] == "secondary") & (samples_32["status"] == "case")]
pooled_A = pooled_cases["A_kresovich"].dropna().values
if len(pooled_A) > 0:
    cs = summarize(pooled_A)
    d_pooled = dict(
        n           = int(len(pooled_A)),
        case_A_mean = cs["mean"],
        delta_A     = cs["mean"] - ctrl_sum["mean"],
        cohens_d    = cohens_d(pooled_A, ctrl_A32),
    )
    subtype_out["pooled"] = d_pooled
    print(f"  {'POOLED':10} {d_pooled['n']:>4}  "
          f"{d_pooled['case_A_mean']:>8.4f}  {d_pooled['delta_A']:>+9.4f}  "
          f"{d_pooled['cohens_d']:>+9.3f}")

# ============================================================================
# SAVE
# ============================================================================

combined = {
    "meta": {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "H_min_immune":  H_MIN_IMMUNE,
        "panel":         "Kresovich-538",
        "panel_size":    len(kresovich_cpgs),
        "matrices": {
            "GSE51057": str(GSE51057_MATRIX),
            "GSE51032": str(GSE51032_MATRIX),
        },
        "baselines": str(HEALTHY_BASELINES_PATH),
    },
    "counts": {
        "GSE51057": {
            "total":          int(len(samples_57)),
            "controls":       int((samples_57["status"] == "control").sum()),
            "breast_cases":   int(((samples_57["status"] == "case") & (samples_57["cancer_type"] == "primary")).sum()),
            "other_cases":    int(((samples_57["status"] == "case") & (samples_57["cancer_type"] == "other")).sum()),
        },
        "GSE51032": {
            "total":            int(len(samples_32)),
            "controls":         int((samples_32["status"] == "control").sum()),
            "breast_cases":     int(((samples_32["status"] == "case") & (samples_32["cancer_type"] == "primary")).sum()),
            "colorectal_cases": int(((samples_32["status"] == "case") & (samples_32["cancer_type"] == "secondary")).sum()),
            "other_cases":      int(((samples_32["status"] == "case") & (samples_32["cancer_type"] == "other")).sum()),
        },
    },
    "analysis_1_mean_Ascore_by_window": {
        "phase9_breast":      ph9_breast,
        "phase12_breast":     ph12_breast,
        "phase12_colorectal": ph12_colorectal,
    },
    "analysis_2_percentile_vs_healthy": {
        "phase9_breast":      pct_ph9,
        "phase12_breast":     pct_ph12,
        "phase12_colorectal": pct_crc,
    },
    "analysis_3_colorectal_subtypes": subtype_out,
}

with open(OUT_JSON, "w") as f:
    json.dump(combined, f, indent=2, default=str)

print()
print("=" * 78)
print(f"[save] JSON written to: {OUT_JSON}")
print(f"[save] Per-sample CSVs in: {OUT_DIR}")
print("=" * 78)
print()
print("Counts summary:")
print(f"  GSE51057: {combined['counts']['GSE51057']}")
print(f"  GSE51032: {combined['counts']['GSE51032']}")
print()
print("Done. Copy the terminal output above and paste back to Walther.")
