#!/usr/bin/env python3
"""
===============================================================================
VAL-056 — Lung-EPIC Card Validation Pipeline (Phase 9/12-equivalent)
===============================================================================

Parameterized Phase 9/12-equivalent validation script for the lung-epic card.
Runs the identical universal pipeline used in VAL-047 Phase 9 (GSE51057 breast)
and Phase 12 (GSE51032 breast + colorectal) on a lung cancer cohort.

USAGE:
    python3 val056_lung_epic_validation.py \\
        --matrix <path_to_450k_or_EPIC_series_matrix.txt.gz> \\
        --metadata <path_to_metadata_csv> \\
        --output_dir <path_to_output_dir> \\
        --cohort_label <cohort_name> \\
        --seed 20260420

METADATA CSV REQUIRED COLUMNS:
    sample_id        — matches matrix column headers
    case_control     — 1 = case, 0 = control
    age_at_blood     — age in years at blood draw
    sex              — M / F
    smoking_status   — never / former / current
    pack_years       — float; NaN if unknown
    time_to_dx_years — years from blood draw to lung cancer diagnosis (NaN for controls)
    histology        — NSCLC_adeno / NSCLC_squamous / SCLC / mixed / NA_control

VALIDATION TIER ACHIEVED IS DETERMINED BY:
    - If n_cases ≥ 100 AND pre-dx metadata available → cohort_screening_validated
    - If n_cases ≥ 100 AND cross-cohort replication exists → cross_platform_validated
    - If n_cases < 100 OR at-diagnosis only → exploratory or direction_check_only

STATUS AS OF 2026-04-24: NO SUITABLE PUBLIC COHORT AVAILABLE.
    Script is authored and tested against the VAL-047 Phase 9/12 pipeline logic.
    When a lung cohort becomes accessible (CLUE II public release, UK Biobank
    approval via TODO 8.2, or another public GSE deposit), this script is ready
    to run.

SOURCE PAPER CITATIONS:
    Xu-538 panel: Xu Z, Sandler DP, Taylor JA. J Natl Cancer Inst 2020;112:87-94.
                  DOI: https://doi.org/10.1093/jnci/djz065
    Moss 2018:    Moss J et al. Nat Commun 2018;9:5068.
                  DOI: https://doi.org/10.1038/s41467-018-07466-6
    VAL-046:      UK Biobank pre-dx lung (gated; see README_MASTER_v2.1)
    VAL-041:      Stage 2 Moss NNLS localization validation (IAM-Validation repo)

===============================================================================
"""

import argparse
import hashlib
import json
import gzip
import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ── CANONICAL CONSTANTS (frozen from README_MASTER_v2.1 THE UNIVERSAL RULE) ────

H_MIN_IMMUNE = 0.838889        # G-003b MCMC posterior, R-hat < 1.001
H_MIN_CYCLING_METHYL = 0.856055  # For Stage 2 lung_epithelial scoring
HEALTHY_LUNG_BETA = 0.738       # Moss 2018 Table S1
XU538_PANEL_SHA = "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6"

# Age-matched baseline (80-cell reference, decade-stratified)
# Source: Hannum 2013, Horvath 2013, Roadmap 2015, Moss 2018, Lister 2013, Alisch 2012
AGE_BASELINE = {
    # decade: (A_mean, A_sd)
    "40-49": (0.9421, 0.0365),
    "50-59": (0.9538, 0.0372),
    "60-69": (0.9652, 0.0380),
    "70-79": (0.9764, 0.0387),
    "80-89": (0.9873, 0.0394),
    "90+":   (0.9996, 0.0415),
}

# Tier thresholds (positive-direction, same as breast card)
TIERS = [
    ("NORMAL",      0.00, 1.01),
    ("MARGINAL",    1.01, 1.05),
    ("DETECTABLE",  1.05, 1.07),
    ("URGENT",      1.07, 1.10),
    ("FLOOR_BREACH", 1.10, float("inf")),
]

# Time-to-diagnosis windows (matching VAL-047 Phase 9/12 windows)
TTD_WINDOWS = [
    ("0-2 yr",  0.0,  2.0),
    ("2-5 yr",  2.0,  5.0),
    ("5-10 yr", 5.0, 10.0),
    (">10 yr", 10.0, float("inf")),
]


# ── UTILITY FUNCTIONS ─────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def shannon_entropy(beta):
    """H(β) = -β log2(β) - (1-β) log2(1-β), with NaN-safe handling."""
    beta = np.clip(beta, 1e-9, 1 - 1e-9)
    return -(beta * np.log2(beta) + (1 - beta) * np.log2(1 - beta))


def a_score(beta, h_min):
    """A = H(β) / H_min(class), the IAM canonical A-score."""
    return shannon_entropy(beta) / h_min


def age_decade(age):
    if age < 50:  return "40-49"
    if age < 60:  return "50-59"
    if age < 70:  return "60-69"
    if age < 80:  return "70-79"
    if age < 90:  return "80-89"
    return "90+"


def tier_call(a_score_value):
    for name, lo, hi in TIERS:
        if lo <= a_score_value < hi:
            return name
    return "NORMAL"


def cohens_d(a, b):
    """Standard Cohen's d with pooled SD."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled


def permutation_p(case_scores, ctrl_scores, n_perm=10000, seed=20260420):
    """One-sided permutation test for case > control (lung expected positive)."""
    rng = np.random.default_rng(seed)
    observed_d = cohens_d(case_scores, ctrl_scores)
    if np.isnan(observed_d):
        return np.nan, np.nan
    pooled = np.concatenate([case_scores, ctrl_scores])
    n_case = len(case_scores)
    greater = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_d = cohens_d(pooled[:n_case], pooled[n_case:])
        if perm_d >= observed_d:
            greater += 1
    p = (greater + 1) / (n_perm + 1)
    return observed_d, p


def bootstrap_ci_d(case_scores, ctrl_scores, n_boot=10000, seed=20260420):
    """95% bootstrap CI on Cohen's d."""
    rng = np.random.default_rng(seed)
    ds = []
    n_case = len(case_scores)
    n_ctrl = len(ctrl_scores)
    for _ in range(n_boot):
        case_idx = rng.integers(0, n_case, n_case)
        ctrl_idx = rng.integers(0, n_ctrl, n_ctrl)
        d = cohens_d(case_scores[case_idx], ctrl_scores[ctrl_idx])
        if not np.isnan(d):
            ds.append(d)
    if len(ds) < 100:
        return np.nan, np.nan
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


# ── PANEL LOADING ─────────────────────────────────────────────────────────────

def load_xu538_panel(panel_path):
    """Load the Xu-538 panel and verify SHA-256."""
    with open(panel_path, "rb") as f:
        panel_bytes = f.read()
    sha = hashlib.sha256(panel_bytes).hexdigest()
    if sha != XU538_PANEL_SHA:
        raise ValueError(
            f"Panel SHA mismatch. Expected {XU538_PANEL_SHA}, got {sha}. "
            "The Xu-538 panel file has been modified — abort."
        )
    panel = json.loads(panel_bytes)
    return panel["cpgs"], sha


# ── MATRIX LOADING ────────────────────────────────────────────────────────────

def load_series_matrix(matrix_path, cpg_list):
    """
    Load a GEO series matrix .txt.gz and extract β values for the panel CpGs.
    Returns a DataFrame indexed by CpG_id with columns = sample IDs.

    This function assumes standard GEO series matrix format. Adapt if the
    input format differs.
    """
    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    rows = []
    sample_ids = None
    with opener(matrix_path, "rt") as f:
        in_matrix = False
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_matrix = True
                header_line = next(f).rstrip("\n").split("\t")
                sample_ids = [s.strip('"') for s in header_line[1:]]
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_matrix:
                fields = line.split("\t")
                cpg = fields[0].strip('"')
                if cpg in set(cpg_list):
                    values = [float(v) if v and v != "NA" else np.nan
                              for v in fields[1:]]
                    rows.append([cpg] + values)

    if not rows:
        raise ValueError(f"No panel CpGs found in matrix at {matrix_path}")
    df = pd.DataFrame(rows, columns=["cpg_id"] + sample_ids)
    df = df.set_index("cpg_id")
    return df


# ── STAGE 1 COMPUTATION ───────────────────────────────────────────────────────

def compute_stage1_scores(beta_df, metadata_df):
    """
    For each sample, compute A_immune_pooled on the Xu-538 panel.
    Returns metadata_df with A_immune_pooled, tier, and age_decade columns added.
    """
    # Sample-level A-score: mean of A(β) across all panel CpGs present
    sample_ids = beta_df.columns.tolist()
    a_scores = {}
    for sid in sample_ids:
        beta_vals = beta_df[sid].dropna().values
        if len(beta_vals) < 50:  # QC: require ≥50 panel CpGs present
            a_scores[sid] = np.nan
            continue
        a_vals = a_score(beta_vals, H_MIN_IMMUNE)
        a_scores[sid] = float(np.mean(a_vals))

    result = metadata_df.copy()
    result["A_immune_pooled"] = result["sample_id"].map(a_scores)
    result["age_decade"] = result["age_at_blood"].apply(age_decade)
    result["tier"] = result["A_immune_pooled"].apply(
        lambda x: tier_call(x) if not np.isnan(x) else "QC_FAIL"
    )
    return result


# ── WINDOW ANALYSIS ───────────────────────────────────────────────────────────

def analyze_windows(metadata_with_scores, n_perm=10000, n_boot=10000, seed=20260420):
    """
    Compute per-window Cohen's d, permutation p-value, and bootstrap CI.
    Matches VAL-047 Phase 9/12 methodology exactly.
    """
    results = {}
    controls = metadata_with_scores[
        (metadata_with_scores["case_control"] == 0) &
        metadata_with_scores["A_immune_pooled"].notna()
    ]["A_immune_pooled"].values

    results["all_controls"] = {
        "n": int(len(controls)),
        "mean_A": float(np.mean(controls)) if len(controls) else np.nan,
        "sd_A":   float(np.std(controls, ddof=1)) if len(controls) > 1 else np.nan,
    }

    cases_all = metadata_with_scores[
        (metadata_with_scores["case_control"] == 1) &
        metadata_with_scores["A_immune_pooled"].notna()
    ]

    for label, lo, hi in TTD_WINDOWS:
        window_cases = cases_all[
            (cases_all["time_to_dx_years"] >= lo) &
            (cases_all["time_to_dx_years"] < hi)
        ]["A_immune_pooled"].values

        if len(window_cases) < 3:
            results[label] = {
                "n_cases": int(len(window_cases)),
                "status": "insufficient_n",
                "cohens_d": None, "perm_p": None, "ci_95": None,
            }
            continue

        d, p = permutation_p(window_cases, controls, n_perm=n_perm, seed=seed)
        lo_ci, hi_ci = bootstrap_ci_d(window_cases, controls, n_boot=n_boot, seed=seed)
        results[label] = {
            "n_cases": int(len(window_cases)),
            "cohens_d": float(d),
            "perm_p": float(p),
            "ci_95": [float(lo_ci), float(hi_ci)],
        }

    # Pooled all-pre-dx
    all_pre_dx = cases_all["A_immune_pooled"].values
    if len(all_pre_dx) >= 3:
        d, p = permutation_p(all_pre_dx, controls, n_perm=n_perm, seed=seed)
        lo_ci, hi_ci = bootstrap_ci_d(all_pre_dx, controls, n_boot=n_boot, seed=seed)
        results["all_pre_dx"] = {
            "n_cases": int(len(all_pre_dx)),
            "cohens_d": float(d),
            "perm_p": float(p),
            "ci_95": [float(lo_ci), float(hi_ci)],
        }

    return results


# ── SMOKING STRATIFICATION ─────────────────────────────────────────────────────

def analyze_by_smoking(metadata_with_scores, n_perm=10000, n_boot=10000, seed=20260420):
    """
    Separate analysis stratified by smoking status. Critical for lung.
    """
    results = {}
    for status in ["never", "former", "current"]:
        sub = metadata_with_scores[metadata_with_scores["smoking_status"] == status]
        cases = sub[(sub["case_control"] == 1) & sub["A_immune_pooled"].notna()]["A_immune_pooled"].values
        ctrls = sub[(sub["case_control"] == 0) & sub["A_immune_pooled"].notna()]["A_immune_pooled"].values
        if len(cases) < 3 or len(ctrls) < 3:
            results[status] = {"n_cases": int(len(cases)), "n_controls": int(len(ctrls)),
                               "status": "insufficient_n"}
            continue
        d, p = permutation_p(cases, ctrls, n_perm=n_perm, seed=seed)
        lo_ci, hi_ci = bootstrap_ci_d(cases, ctrls, n_boot=n_boot, seed=seed)
        results[status] = {
            "n_cases": int(len(cases)),
            "n_controls": int(len(ctrls)),
            "cohens_d": float(d),
            "perm_p": float(p),
            "ci_95": [float(lo_ci), float(hi_ci)],
        }
    return results


# ── HISTOLOGY STRATIFICATION ──────────────────────────────────────────────────

def analyze_by_histology(metadata_with_scores, n_perm=10000, n_boot=10000, seed=20260420):
    """
    Separate analysis for NSCLC adenocarcinoma vs squamous cell carcinoma.
    """
    results = {}
    controls = metadata_with_scores[
        (metadata_with_scores["case_control"] == 0) &
        metadata_with_scores["A_immune_pooled"].notna()
    ]["A_immune_pooled"].values

    for histo in ["NSCLC_adeno", "NSCLC_squamous", "SCLC"]:
        cases = metadata_with_scores[
            (metadata_with_scores["histology"] == histo) &
            metadata_with_scores["A_immune_pooled"].notna()
        ]["A_immune_pooled"].values
        if len(cases) < 3:
            results[histo] = {"n_cases": int(len(cases)), "status": "insufficient_n"}
            continue
        d, p = permutation_p(cases, controls, n_perm=n_perm, seed=seed)
        lo_ci, hi_ci = bootstrap_ci_d(cases, controls, n_boot=n_boot, seed=seed)
        results[histo] = {
            "n_cases": int(len(cases)),
            "cohens_d": float(d),
            "perm_p": float(p),
            "ci_95": [float(lo_ci), float(hi_ci)],
        }
    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VAL-056 Lung-EPIC Validation")
    parser.add_argument("--matrix", required=True, help="Path to GEO series matrix .txt.gz")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--panel", required=True, help="Path to Xu-538 panel JSON")
    parser.add_argument("--output_dir", required=True, help="Directory for output JSON")
    parser.add_argument("--cohort_label", required=True, help="Cohort name (e.g., CLUE_II, UKB_lung)")
    parser.add_argument("--seed", type=int, default=20260420, help="RNG seed")
    parser.add_argument("--n_perm", type=int, default=10000, help="Permutation replicates")
    parser.add_argument("--n_boot", type=int, default=10000, help="Bootstrap replicates")
    args = parser.parse_args()

    start_time = time.time()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[VAL-056] Lung-EPIC validation pipeline")
    print(f"[VAL-056] Cohort: {args.cohort_label}")
    print(f"[VAL-056] Matrix: {args.matrix}")
    print(f"[VAL-056] Metadata: {args.metadata}")
    print(f"[VAL-056] Seed: {args.seed}")

    # Compute matrix SHA for locking
    print("[VAL-056] Computing matrix SHA-256...")
    matrix_sha = sha256_file(args.matrix)
    print(f"[VAL-056] Matrix SHA: {matrix_sha}")

    # Load panel
    print("[VAL-056] Loading Xu-538 panel...")
    cpg_list, panel_sha = load_xu538_panel(args.panel)
    print(f"[VAL-056] Panel SHA verified: {panel_sha}")
    print(f"[VAL-056] Panel size: {len(cpg_list)} CpGs")

    # Load matrix
    print("[VAL-056] Loading series matrix...")
    beta_df = load_series_matrix(args.matrix, cpg_list)
    print(f"[VAL-056] Matrix loaded: {beta_df.shape[0]} CpGs × {beta_df.shape[1]} samples")

    # Load metadata
    print("[VAL-056] Loading metadata...")
    metadata_df = pd.read_csv(args.metadata)
    required_cols = ["sample_id", "case_control", "age_at_blood", "sex",
                     "smoking_status", "time_to_dx_years"]
    for col in required_cols:
        if col not in metadata_df.columns:
            raise ValueError(f"Metadata CSV missing required column: {col}")
    if "histology" not in metadata_df.columns:
        metadata_df["histology"] = "NA_unknown"
    if "pack_years" not in metadata_df.columns:
        metadata_df["pack_years"] = np.nan
    print(f"[VAL-056] Metadata loaded: n={len(metadata_df)}")

    # Compute Stage 1 scores
    print("[VAL-056] Computing Stage 1 A_immune_pooled...")
    scored = compute_stage1_scores(beta_df, metadata_df)
    n_qc_fail = (scored["tier"] == "QC_FAIL").sum()
    print(f"[VAL-056] Samples with valid A-score: {(scored['tier'] != 'QC_FAIL').sum()}")
    print(f"[VAL-056] Samples failing QC (< 50 panel CpGs): {n_qc_fail}")

    # Window analysis
    print("[VAL-056] Running time-to-diagnosis window analysis...")
    window_results = analyze_windows(scored, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)

    # Smoking stratification
    print("[VAL-056] Running smoking-stratified analysis...")
    smoking_results = analyze_by_smoking(scored, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)

    # Histology stratification
    print("[VAL-056] Running histology-stratified analysis...")
    histology_results = analyze_by_histology(scored, n_perm=args.n_perm, n_boot=args.n_boot, seed=args.seed)

    # Assemble results
    runtime_s = time.time() - start_time
    result_json = {
        "val_id": "VAL-056",
        "card_id": "lung-epic",
        "card_version": "v0.1",
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "cohort_label": args.cohort_label,
        "cohort_matrix_sha256": matrix_sha,
        "panel_id": "Xu2020_breast_cancer_replicated_full",
        "panel_sha256": panel_sha,
        "h_min_immune": H_MIN_IMMUNE,
        "h_min_source": "G-003b MCMC posterior, R-hat < 1.001",
        "rng_seed": args.seed,
        "n_perm": args.n_perm,
        "n_boot": args.n_boot,
        "runtime_seconds": round(runtime_s, 2),
        "sample_counts": {
            "total_loaded": int(len(metadata_df)),
            "valid_A_score": int((scored["tier"] != "QC_FAIL").sum()),
            "qc_fail": int(n_qc_fail),
            "cases": int((scored["case_control"] == 1).sum()),
            "controls": int((scored["case_control"] == 0).sum()),
        },
        "window_analysis": window_results,
        "smoking_stratified": smoking_results,
        "histology_stratified": histology_results,
    }

    out_path = out_dir / f"VAL056_lung_epic_{args.cohort_label}_results.json"
    with open(out_path, "w") as f:
        json.dump(result_json, f, indent=2, default=str)
    print(f"[VAL-056] Results written: {out_path}")
    print(f"[VAL-056] Runtime: {runtime_s:.1f}s")


if __name__ == "__main__":
    main()
