#!/usr/bin/env python3
"""
T1 — GSE40279 (Hannum 2013) US healthy reference baseline
==========================================================

Purpose
-------
Sanity check: confirm the Xu-538 immune-class A-score distribution in 656
healthy US adults (Hannum 2013) is consistent with the A-score distribution
in EPIC-Italy healthy controls. This tests whether H_min(immune) = 0.838889
is a population-independent calibration or a European-specific artifact.

This is NOT a case-control test. Hannum is healthy only. The output is a
distribution and a comparison against the EPIC-Italy control means.

Prediction
----------
The Hannum A-score distribution should sit near the EPIC-Italy control
means (GSE51057 controls: 0.4370; GSE51032 controls: 0.4356). Cohen's d
between Hannum and either EPIC-Italy control group should be < 0.3.

Inputs
------
--matrix_path : GSE40279_series_matrix.txt.gz (downloaded from GEO)
--panel       : xu538_breast_panel.json (SHA ada672960563...4a6d6)
--output_dir  : directory to write per-sample CSV + distribution JSON

Invariants
----------
H_MIN_IMMUNE = 0.838889
RANDOM_SEED  = 20260420
AGE_REGEX    = strict colon-anchored (no "age at menarche" capture)

Citation
--------
Hannum G, Guinney J, Zhao L, et al. "Genome-wide methylation profiles
reveal quantitative views of human aging rates." Mol Cell 49(2):359-367,
2013. PMID 23177740. doi:10.1016/j.molcel.2012.10.016

Panel source
------------
Xu Z, Sandler DP, Taylor JA. "Blood DNA Methylation and Breast Cancer:
A Prospective Case-Cohort Analysis in the Sister Study." JNCI 112(1):
87-94, 2020. PMID 30989176. doi:10.1093/jnci/djz065
"""

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# INVARIANTS — FIXED across all cohorts
# ============================================================================

H_MIN_IMMUNE = 0.838889
RANDOM_SEED  = 20260420

# Strict colon-anchored age regex.
# This requires the characteristic line to be LITERALLY "age: <number>" —
# not "age at menarche: 13" or "age at first period: 11", which would
# overwrite a real age of 54 with 13. This bug was documented twice
# in the EPIC-Italy work.
AGE_REGEX = re.compile(
    r"^\s*age\s*(?:\([^)]*\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)

# EPIC-Italy control A-score means (from VAL047 Tightening v2 results JSON).
# Used for the cross-population comparison at the end.
EPIC_ITALY_CONTROL_MEANS = {
    "GSE51057_controls": 0.4370,   # Phase 9
    "GSE51032_controls": 0.4356,   # Phase 12
}
EPIC_ITALY_CONTROL_SD_APPROX = 0.030   # from Tightening v2 histograms; used only
                                       # if we need an approximate Cohen's d
                                       # without the raw control arrays.

np.random.seed(RANDOM_SEED)

# ============================================================================
# HELPERS
# ============================================================================

def H_binary(beta):
    """Binary Shannon entropy of a beta value. Returns 0 for beta in {0, 1, NaN}."""
    if beta is None or not (0.0 < beta < 1.0) or math.isnan(beta):
        return 0.0
    return -beta * math.log2(beta) - (1.0 - beta) * math.log2(1.0 - beta)

def A_score(beta):
    return H_binary(beta) / H_MIN_IMMUNE

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = float(np.std(a, ddof=1))
    s2 = float(np.std(b, ddof=1))
    pooled = math.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)

def cohens_d_approx(mean_a, mean_b, sd_shared):
    """Approx Cohen's d when only means + a shared SD estimate are available."""
    if sd_shared is None or sd_shared == 0:
        return float("nan")
    return (mean_a - mean_b) / sd_shared

# ============================================================================
# PANEL LOADER
# ============================================================================

def load_panel(panel_path):
    with open(panel_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        for key in ("cpgs", "panel", "cpg_ids", "probes"):
            if key in data and isinstance(data[key], list):
                return set(data[key])
        # Fallback: flatten all string cg* values
        cpgs = set()
        for v in data.values():
            if isinstance(v, list):
                cpgs.update(x for x in v if isinstance(x, str) and x.startswith("cg"))
        if cpgs:
            return cpgs
    sys.exit(f"[FATAL] Cannot parse panel file format: {type(data)}")

# ============================================================================
# MATRIX PARSER
# ============================================================================

def read_series_matrix(path, panel_cpgs):
    """
    Parse a GEO series matrix .txt.gz file.
    Returns (per_sample_meta dict, beta_df filtered to panel CpGs only).
    """
    print(f"[parse] {path.name} ...", flush=True)
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
                if cpg in panel_cpgs:
                    data_lines.append(fields)
                continue
            if line.startswith("!Sample_"):
                key = line.split("\t", 1)[0]
                vals = [v.strip('"') for v in line.split("\t")[1:]]
                meta_rows.setdefault(key, []).append(vals)

    if header is None:
        sys.exit(f"[FATAL] No matrix data block in {path}")

    samples = [h.strip('"') for h in header[1:]]

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

    if not data_lines:
        sys.exit(f"[FATAL] No panel CpGs found in {path}")

    cpg_ids = [row[0].strip('"') for row in data_lines]
    betas = []
    for row in data_lines:
        row_vals = []
        for v in row[1:]:
            v = v.strip('"')
            if v in ("", "NA", "NaN", "null"):
                row_vals.append(np.nan)
            else:
                try:    row_vals.append(float(v))
                except ValueError: row_vals.append(np.nan)
        betas.append(row_vals)
    beta_df = pd.DataFrame(betas, index=cpg_ids, columns=samples)
    print(f"  samples: {len(samples)}   panel CpGs in matrix: {len(cpg_ids)}", flush=True)
    return per_sample_meta, beta_df

# ============================================================================
# METADATA EXTRACTION — strict regex only
# ============================================================================

def parse_age_strict(characteristics):
    """Match only lines of the form 'age: <number>'. No 'age at X' matches."""
    if not characteristics:
        return float("nan")
    for c in characteristics:
        if c is None: continue
        m = AGE_REGEX.match(c)
        if m:
            return float(m.group(1))
    return float("nan")

def parse_gender(characteristics):
    if not characteristics:
        return None
    for c in characteristics:
        if c is None: continue
        m = re.match(r"^\s*gender\s*:\s*([A-Za-z]+)\s*$", c, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return None

def parse_source(characteristics, source_name):
    """Try to identify the biosample source (expect whole blood / PBMC for Hannum)."""
    candidates = []
    if source_name:
        candidates.append(source_name)
    if characteristics:
        for c in characteristics:
            if c is None: continue
            m = re.match(r"^\s*(source|tissue|cell\s*type)\s*:\s*(.+)\s*$", c, re.IGNORECASE)
            if m:
                candidates.append(m.group(2))
    for cand in candidates:
        s = cand.lower()
        if "whole blood" in s or "pbmc" in s or "peripheral" in s or "wb" == s.strip():
            return "whole_blood_or_pbmc"
    return (candidates[0] if candidates else None)

def parse_ethnicity(characteristics):
    if not characteristics:
        return None
    for c in characteristics:
        if c is None: continue
        m = re.match(r"^\s*(ethnicity|race|plate)\s*:\s*(.+)\s*$", c, re.IGNORECASE)
        if m and m.group(1).lower() in ("ethnicity", "race"):
            return m.group(2).strip().lower()
    return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="T1 Hannum 2013 baseline (GSE40279)")
    ap.add_argument("--matrix_path", required=True,
                    help="Path to GSE40279_series_matrix.txt.gz")
    ap.add_argument("--panel", required=True,
                    help="Path to xu538_breast_panel.json")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to write per-sample CSV and distribution JSON")
    args = ap.parse_args()

    matrix_path = Path(args.matrix_path)
    panel_path  = Path(args.panel)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Reproducibility anchors ------------------------------------------
    print("=" * 78)
    print("T1 GSE40279 — Hannum 2013 US healthy baseline, Xu-538 panel")
    print("=" * 78)
    print(f"Random seed:      {RANDOM_SEED}")
    print(f"H_min(immune):    {H_MIN_IMMUNE}")
    print(f"Age regex:        {AGE_REGEX.pattern}")
    print()
    print("Computing SHA-256 of inputs...", flush=True)
    matrix_sha = sha256_of_file(matrix_path)
    panel_sha  = sha256_of_file(panel_path)
    print(f"  matrix ({matrix_path.name}): {matrix_sha}")
    print(f"  panel  ({panel_path.name}):  {panel_sha}")
    print()

    # ---- Load panel -------------------------------------------------------
    panel_cpgs = load_panel(panel_path)
    print(f"Panel CpGs loaded: {len(panel_cpgs)}")

    # ---- Parse matrix -----------------------------------------------------
    meta, beta_df = read_series_matrix(matrix_path, panel_cpgs)

    # ---- Per-sample A-scores ---------------------------------------------
    print("Computing per-sample A-scores...", flush=True)
    rows = []
    for gsm in beta_df.columns:
        betas = beta_df[gsm].to_numpy(dtype=float)
        hs = [H_binary(b) for b in betas]
        hs = [h for h in hs if h > 0]   # drop zeros (invalid / 0/1 betas)
        n_valid = len(hs)
        a_score_val = (sum(hs) / n_valid / H_MIN_IMMUNE) if n_valid > 0 else float("nan")

        m = meta.get(gsm, {})
        chars = m.get("characteristics", [])
        rows.append({
            "gsm":          gsm,
            "title":        m.get("title"),
            "source_name":  m.get("source_name"),
            "age":          parse_age_strict(chars),
            "gender":       parse_gender(chars),
            "ethnicity":    parse_ethnicity(chars),
            "source":       parse_source(chars, m.get("source_name")),
            "n_cpgs_valid": n_valid,
            "A_score":      a_score_val,
            "raw_characteristics": "|".join([c for c in chars if c is not None]),
        })
    df = pd.DataFrame(rows)

    # ---- Write per-sample CSV --------------------------------------------
    csv_path = out_dir / "GSE40279_per_sample_A.csv"
    df.to_csv(csv_path, index=False)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV:   {csv_path}")
    print(f"  rows: {len(df)}   sha256: {csv_sha}")
    print()

    # ---- Age and valid-CpG sanity -----------------------------------------
    n_with_age = int(df["age"].notna().sum())
    print(f"[sanity] samples with parsed age:  {n_with_age} / {len(df)}")
    if n_with_age > 0:
        print(f"         age: median={df['age'].median():.1f}  "
              f"mean={df['age'].mean():.1f}  "
              f"min={df['age'].min():.0f}  max={df['age'].max():.0f}")
    if df["age"].max() > 100 or (df["age"] < 20).sum() > n_with_age * 0.1:
        print("  [WARNING] age distribution looks suspicious — "
              "verify the age regex did not match wrong characteristic line")
    valid_panel_cpgs = beta_df.shape[0]
    print(f"[sanity] panel CpGs present in matrix: {valid_panel_cpgs} / {len(panel_cpgs)}")
    print()

    # ---- Distribution summary --------------------------------------------
    A = df["A_score"].to_numpy(dtype=float)
    A = A[~np.isnan(A)]
    dist = {
        "n":     int(len(A)),
        "mean":  float(np.mean(A)),
        "sd":    float(np.std(A, ddof=1)),
        "median":float(np.median(A)),
        "p10":   float(np.percentile(A, 10)),
        "p25":   float(np.percentile(A, 25)),
        "p50":   float(np.percentile(A, 50)),
        "p75":   float(np.percentile(A, 75)),
        "p90":   float(np.percentile(A, 90)),
        "min":   float(np.min(A)),
        "max":   float(np.max(A)),
    }
    print("A-score distribution:")
    for k, v in dist.items():
        print(f"  {k:>8s}: {v}")
    print()

    # ---- Cross-population comparison vs EPIC-Italy controls ---------------
    # We have the control means (0.4370, 0.4356) but NOT the per-sample
    # control A-score arrays in this session — they sit on Heath's machine.
    # So we compute the mean-difference-over-SD approximation here and note
    # that the exact Cohen's d requires the raw control arrays.
    vs_epic = {}
    for name, mean_ctrl in EPIC_ITALY_CONTROL_MEANS.items():
        d_approx = cohens_d_approx(
            mean_a     = dist["mean"],
            mean_b     = mean_ctrl,
            sd_shared  = EPIC_ITALY_CONTROL_SD_APPROX,
        )
        vs_epic[name] = {
            "EPIC_control_mean":  mean_ctrl,
            "Hannum_mean":        dist["mean"],
            "delta_mean":         dist["mean"] - mean_ctrl,
            "Cohens_d_approx":    d_approx,
            "approx_SD_used":     EPIC_ITALY_CONTROL_SD_APPROX,
            "note": ("approximation using shared-SD proxy; "
                     "exact d requires EPIC-Italy per-sample control arrays"),
        }
    print("vs EPIC-Italy healthy controls (approximation):")
    for name, block in vs_epic.items():
        print(f"  {name}: delta_mean={block['delta_mean']:+.4f}   "
              f"d_approx={block['Cohens_d_approx']:+.3f}")
    print()

    # ---- Acceptance criterion --------------------------------------------
    d_max = max(abs(v["Cohens_d_approx"]) for v in vs_epic.values()
                if not math.isnan(v["Cohens_d_approx"]))
    verdict = "PASS" if d_max < 0.3 else ("INVESTIGATE" if d_max < 0.5 else "FAIL")
    print(f"Acceptance criterion: |Cohen's d approx| < 0.3 vs EPIC-Italy controls")
    print(f"  max |d_approx|:  {d_max:.3f}")
    print(f"  verdict:         {verdict}")
    print()

    # ---- Write distribution JSON ------------------------------------------
    out_json = {
        "cohort":          "GSE40279_Hannum2013_US_healthy",
        "purpose":         "cross-population baseline check, not case-control",
        "n_samples_total": int(len(df)),
        "n_samples_valid_A": int(len(A)),
        "panel":           "xu538_breast",
        "panel_n_cpgs":    len(panel_cpgs),
        "panel_cpgs_in_matrix": valid_panel_cpgs,
        "H_min_immune":    H_MIN_IMMUNE,
        "random_seed":     RANDOM_SEED,
        "age_regex":       AGE_REGEX.pattern,
        "input_sha256": {
            "matrix_file": matrix_sha,
            "panel_file":  panel_sha,
        },
        "output_sha256": {
            "per_sample_csv": csv_sha,
        },
        "A_score_distribution": dist,
        "age_summary": {
            "n_with_parsed_age": n_with_age,
            "median": (float(df['age'].median()) if n_with_age else None),
            "mean":   (float(df['age'].mean())   if n_with_age else None),
            "min":    (float(df['age'].min())    if n_with_age else None),
            "max":    (float(df['age'].max())    if n_with_age else None),
        },
        "vs_EPIC_Italy_controls": vs_epic,
        "acceptance_criterion": {
            "threshold":       "|Cohen's d approx| < 0.3",
            "max_abs_d_approx": d_max,
            "verdict":         verdict,
        },
    }
    json_path = out_dir / "GSE40279_distribution_comparison.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    json_sha = sha256_of_file(json_path)
    print(f"Distribution JSON:  {json_path}")
    print(f"  sha256: {json_sha}")
    print()
    print("T1 complete.")

if __name__ == "__main__":
    main()
