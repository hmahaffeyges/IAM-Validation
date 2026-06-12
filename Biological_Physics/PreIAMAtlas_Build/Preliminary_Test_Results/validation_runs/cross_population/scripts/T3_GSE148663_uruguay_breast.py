#!/usr/bin/env python3
"""
T3 — GSE148663 (Cayota 2020) Uruguayan sporadic breast cancer
==============================================================

Cohort
------
32 peripheral blood leukocyte samples, Uruguayan population, GPL13534 450K.

  disease state: sporadic breast cancer    n = 22
  disease state: healthy control           n = 10

All uniform cell type ('Peripheral blood leukocytes'), single population,
no biosource heterogeneity — the cleanest case/control design of the three
new cohorts.

What this test does and does not do
-----------------------------------
These are ALREADY-DIAGNOSED sporadic breast cancer patients (GEO metadata
does not report time since diagnosis, stage, or treatment status). The
comparison is:

  "Can the Xu-538 immune-class A-score discriminate diagnosed sporadic
   breast cancer cases from matched healthy controls in an independent
   Latin American (Uruguayan) population?"

This is a RETROSPECTIVE case-control comparison, not the pre-diagnostic
time-to-diagnosis test EPIC-Italy runs. A positive d here says the panel
detects something in blood at the time of (or after) diagnosis. It does
not directly replicate the EPIC-Italy >10yr pre-diagnostic finding, but
it DOES test whether the panel generalizes across populations and whether
the signal direction (elevated in cases) holds in Latin American blood.

Sample size (n=22 vs n=10) is small — bootstrap CIs will be wide.

Citation
--------
Fernandez-Calero T, Davyt M, Perelmuter K, et al. "DNA methylation of
leukocytes from sporadic breast cancer patients and healthy controls in
Latin American population." Series GSE148663, submitted 2020.
PubMed ID 33145876.

Invariants: H_MIN_IMMUNE = 0.838889; RANDOM_SEED = 20260420; Xu-538 panel.
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
# INVARIANTS
# ============================================================================

H_MIN_IMMUNE = 0.838889
RANDOM_SEED  = 20260420

AGE_REGEX = re.compile(
    r"^\s*age\s*(?:\([^)]*\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)

np.random.seed(RANDOM_SEED)

# ============================================================================
# HELPERS — identical to T1/T2
# ============================================================================

def H_binary(beta):
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

def permutation_p(cases, controls, n_perm=10000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(cases,    dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    d_obs = cohens_d(a, b)
    if math.isnan(d_obs):
        return float("nan"), d_obs
    combined = np.concatenate([a, b])
    n_a = len(a)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        ca, cb = combined[:n_a], combined[n_a:]
        dd = cohens_d(ca, cb)
        if abs(dd) >= abs(d_obs):
            count_ge += 1
    return float((count_ge + 1) / (n_perm + 1)), float(d_obs)

def bootstrap_d_ci(cases, controls, n_boot=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 1)
    a = np.asarray(cases,    dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    ds = []
    for _ in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        dd = cohens_d(ra, rb)
        if not math.isnan(dd):
            ds.append(dd)
    if not ds:
        return float("nan"), float("nan")
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(lo), float(hi)

# ============================================================================
# PANEL LOADER — identical
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
        cpgs = set()
        for v in data.values():
            if isinstance(v, list):
                cpgs.update(x for x in v if isinstance(x, str) and x.startswith("cg"))
        if cpgs:
            return cpgs
    sys.exit(f"[FATAL] Cannot parse panel: {type(data)}")

# ============================================================================
# MATRIX PARSER — identical
# ============================================================================

def read_series_matrix(path, panel_cpgs):
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
                try: row_vals.append(float(v))
                except ValueError: row_vals.append(np.nan)
        betas.append(row_vals)
    beta_df = pd.DataFrame(betas, index=cpg_ids, columns=samples)
    print(f"  samples: {len(samples)}   panel CpGs in matrix: {len(cpg_ids)}", flush=True)
    return per_sample_meta, beta_df

# ============================================================================
# METADATA EXTRACTION — Uruguayan cohort
# ============================================================================

def parse_disease_state(characteristics):
    if not characteristics:
        return None
    for c in characteristics:
        if c is None: continue
        m = re.match(r"^\s*disease\s*state\s*:\s*(.+)\s*$", c, re.IGNORECASE)
        if m:
            v = m.group(1).strip().lower()
            if "breast cancer" in v:
                return "case"
            if "healthy" in v or "control" in v:
                return "control"
            return v  # unexpected — pass through
    return None

def parse_individual(characteristics):
    if not characteristics: return None
    for c in characteristics:
        if c is None: continue
        m = re.match(r"^\s*individual\s*:\s*(.+)\s*$", c, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def parse_population(characteristics):
    if not characteristics: return None
    for c in characteristics:
        if c is None: continue
        m = re.match(r"^\s*population\s*:\s*(.+)\s*$", c, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def parse_age_strict(characteristics):
    if not characteristics:
        return float("nan")
    for c in characteristics:
        if c is None: continue
        m = AGE_REGEX.match(c)
        if m:
            return float(m.group(1))
    return float("nan")

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="T3 GSE148663 Uruguayan sporadic breast")
    ap.add_argument("--matrix_path", required=True)
    ap.add_argument("--panel",       required=True)
    ap.add_argument("--output_dir",  required=True)
    args = ap.parse_args()

    matrix_path = Path(args.matrix_path)
    panel_path  = Path(args.panel)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("T3 GSE148663 — Uruguayan sporadic breast, Xu-538 panel")
    print("=" * 78)
    print(f"Random seed:   {RANDOM_SEED}")
    print(f"H_min(immune): {H_MIN_IMMUNE}")
    print()

    matrix_sha = sha256_of_file(matrix_path)
    panel_sha  = sha256_of_file(panel_path)
    print(f"  matrix sha256: {matrix_sha}")
    print(f"  panel  sha256: {panel_sha}")
    print()

    panel_cpgs = load_panel(panel_path)
    print(f"Panel CpGs loaded: {len(panel_cpgs)}")

    meta, beta_df = read_series_matrix(matrix_path, panel_cpgs)

    print("Computing per-sample A-scores...", flush=True)
    rows = []
    for gsm in beta_df.columns:
        betas = beta_df[gsm].to_numpy(dtype=float)
        hs = [H_binary(b) for b in betas]
        hs = [h for h in hs if h > 0]
        n_valid = len(hs)
        a_val = (sum(hs)/n_valid/H_MIN_IMMUNE) if n_valid else float("nan")

        m = meta.get(gsm, {})
        chars = m.get("characteristics", [])
        rows.append({
            "gsm":           gsm,
            "title":         m.get("title"),
            "source_name":   m.get("source_name"),
            "status":        parse_disease_state(chars),
            "individual":    parse_individual(chars),
            "population":    parse_population(chars),
            "age":           parse_age_strict(chars),
            "n_cpgs_valid":  n_valid,
            "A_score":       a_val,
            "raw_characteristics": "|".join([c for c in chars if c is not None]),
        })
    df = pd.DataFrame(rows)

    csv_path = out_dir / "GSE148663_per_sample_A.csv"
    df.to_csv(csv_path, index=False)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV:  {csv_path}")
    print(f"  rows: {len(df)}   sha256: {csv_sha}")
    print()

    print("Cohort composition (status × source):")
    comp = df.groupby(["source_name","status"]).size().reset_index(name="n")
    for _, r in comp.iterrows():
        print(f"  {r['source_name']:<35s}  status={r['status']:<10s}  n={r['n']}")
    print()

    # Primary: case vs control on all 32 samples (single uniform source)
    cases_a    = df[df["status"]=="case"   ]["A_score"].to_numpy(dtype=float)
    controls_a = df[df["status"]=="control"]["A_score"].to_numpy(dtype=float)
    cases_a    = cases_a[~np.isnan(cases_a)]
    controls_a = controls_a[~np.isnan(controls_a)]

    n_c = int(len(cases_a)); n_k = int(len(controls_a))
    d = cohens_d(cases_a, controls_a) if (n_c>=2 and n_k>=2) else float("nan")
    p_perm, _ = (permutation_p(cases_a, controls_a, n_perm=10000, seed=RANDOM_SEED)
                 if (n_c>=2 and n_k>=2) else (float("nan"), float("nan")))
    ci_lo, ci_hi = (bootstrap_d_ci(cases_a, controls_a, n_boot=2000, seed=RANDOM_SEED)
                    if (n_c>=2 and n_k>=2) else (float("nan"), float("nan")))

    case_mean  = float(np.mean(cases_a))    if n_c else float("nan")
    case_sd    = float(np.std(cases_a,   ddof=1)) if n_c>=2 else float("nan")
    ctrl_mean  = float(np.mean(controls_a)) if n_k else float("nan")
    ctrl_sd    = float(np.std(controls_a, ddof=1)) if n_k>=2 else float("nan")

    print("PRIMARY — sporadic breast cancer vs healthy control (peripheral blood leukocytes):")
    print(f"  n_cases={n_c}  n_controls={n_k}")
    print(f"  case_mean_A = {case_mean:.4f}  (sd {case_sd:.4f})")
    print(f"  ctrl_mean_A = {ctrl_mean:.4f}  (sd {ctrl_sd:.4f})")
    print(f"  Cohen's d = {d:+.3f}")
    print(f"  permutation p (10,000 shuffles) = {p_perm:.4f}")
    print(f"  bootstrap 95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print()

    # Direction verdict
    if math.isnan(d):
        verdict = "INSUFFICIENT_N"
    elif d > 0.5:   verdict = "STRONG_POSITIVE"
    elif d > 0.2:   verdict = "MODEST_POSITIVE"
    elif d > -0.2:  verdict = "NULL"
    elif d > -0.5:  verdict = "MODEST_INVERSE"
    else:           verdict = "STRONG_INVERSE"
    print(f"Direction verdict: {verdict}")
    print("NOTE: These are already-diagnosed cases, not pre-diagnostic.")
    print("      This tests panel generalization across populations at time of diagnosis,")
    print("      not replication of the EPIC-Italy pre-diagnostic >10yr pattern.")
    print()

    # JSON output
    result = {
        "cohort":        "GSE148663_Fernandez-Calero2020_Uruguayan_sporadic_breast",
        "purpose":       ("cross-population case/control discrimination test at "
                          "time of diagnosis — NOT pre-diagnostic"),
        "cohort_design_note": ("Peripheral blood leukocytes from sporadic breast cancer "
                               "patients vs healthy controls in Uruguayan population. "
                               "No time-to-diagnosis, age, or stage info in GEO metadata."),
        "n_samples_total": int(len(df)),
        "panel":              "xu538_breast",
        "panel_n_cpgs":       len(panel_cpgs),
        "panel_cpgs_in_matrix": int(beta_df.shape[0]),
        "H_min_immune":       H_MIN_IMMUNE,
        "random_seed":        RANDOM_SEED,
        "input_sha256": {
            "matrix_file": matrix_sha,
            "panel_file":  panel_sha,
        },
        "output_sha256": {
            "per_sample_csv": csv_sha,
        },
        "cohort_composition": [
            {"source_name": r["source_name"], "status": r["status"], "n": int(r["n"])}
            for _, r in comp.iterrows()
        ],
        "primary_analysis": {
            "n_cases":            n_c,
            "n_controls":         n_k,
            "case_A_mean":        case_mean,
            "case_A_sd":          case_sd,
            "control_A_mean":     ctrl_mean,
            "control_A_sd":       ctrl_sd,
            "Cohens_d":           d,
            "permutation_p_10000": p_perm,
            "bootstrap_95CI_lo":  ci_lo,
            "bootstrap_95CI_hi":  ci_hi,
        },
        "direction_verdict": verdict,
    }
    json_path = out_dir / "GSE148663_case_control.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    json_sha = sha256_of_file(json_path)
    print(f"Output JSON:  {json_path}")
    print(f"  sha256: {json_sha}")
    print()
    print("T3 complete.")

if __name__ == "__main__":
    main()
