#!/usr/bin/env python3
"""
T2 — GSE104942 (Joo 2018) Australian HBOC familial breast cancer
=================================================================

Cohort
------
210 samples from 25 Australian multi-case breast cancer families, no known
BRCA1/BRCA2 mutations. Peripheral blood DNA. GPL13534 (Infinium 450K).

Design (per Joo et al., Nat Commun 9:867, 2018; doi:10.1038/s41467-018-03058-6)
-----------------------------------------------------------------------------
Family-based segregation study of constitutional / heritable methylation
marks that may predispose to breast cancer. Status is "affected" or
"unaffected" at a single assessment — the GEO metadata does NOT include
time-to-diagnosis, age, or ICD code per sample.

  Status:  87 affected  /  123 unaffected   (n = 210)

What this test does and does not do
-----------------------------------
This is NOT a pre-diagnostic blood test in the EPIC-Italy sense. There is
no TtD information. "Affected" samples are a mix of women who may have
been diagnosed years before blood draw, shortly before, or between sample
and follow-up. Without TtD we cannot replicate the monotonic >10yr → 0-2yr
pattern seen in EPIC-Italy.

This test asks a different, weaker question:
  "Does the Xu-538 immune-class A-score show a case-vs-control signal
   at the constitutional/familial level in an Australian non-BRCA HBOC
   pedigree?"

A positive d would indicate that the panel captures some heritable or
long-term methylation difference between women who develop breast cancer
and their relatives who do not. A null would NOT falsify the EPIC-Italy
pre-diagnostic finding — it would indicate the panel does not discriminate
at the constitutional level, which is a different claim entirely.

Biosource filtering
-------------------
GSE104942 mixes five biosource types. Only the two that approximate whole
blood immune composition are used for the primary analysis:

  INCLUDED:
    - blood pellet                 (136 samples; cell pellet post-plasma)
    - whole peripheral blood       ( 55 samples; WPB pre-fractionation)

  EXCLUDED:
    - EBV transformed lymphoblastoid (13)  — immortalized B-cell lines
    - non lymphocyte blood fraction  ( 5)  — alters immune class ratio
    - buffy coat                     ( 1)  — leukocyte-enriched, not comparable

A secondary analysis on the full 210 samples is also reported for
transparency, but it should NOT be used as primary evidence.

Citation
--------
Joo JE, Dowty JG, Milne RL, et al. "Heritable DNA methylation marks
associated with susceptibility to breast cancer." Nat Commun 9:867, 2018.
PMID 29491469. doi:10.1038/s41467-018-03058-6

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
# INVARIANTS — identical to T1
# ============================================================================

H_MIN_IMMUNE = 0.838889
RANDOM_SEED  = 20260420

# Same extended age regex as T1. GSE104942 has no age metadata so this
# regex will match nothing in this cohort — kept for consistency.
AGE_REGEX = re.compile(
    r"^\s*age\s*(?:\([^)]*\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    re.IGNORECASE,
)

# Biosource filter (blood-pellet + whole peripheral blood)
PRIMARY_SOURCES = ("blood pellet", "whole peripheral blood")

np.random.seed(RANDOM_SEED)

# ============================================================================
# HELPERS — identical primitives as T1
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
    """Two-sided permutation p on Cohen's d."""
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
    p = (count_ge + 1) / (n_perm + 1)
    return float(p), float(d_obs)

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
# PANEL LOADER — identical to T1
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
    sys.exit(f"[FATAL] Cannot parse panel file: {type(data)}")

# ============================================================================
# MATRIX PARSER — identical to T1
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
# METADATA EXTRACTION — specific to GSE104942 (status only)
# ============================================================================

def parse_status(characteristics):
    if not characteristics:
        return None
    for c in characteristics:
        if c is None: continue
        m = re.match(r"^\s*status\s*:\s*(affected|unaffected)\s*$", c, re.IGNORECASE)
        if m:
            return m.group(1).lower()
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

def classify_source(source_name):
    if source_name is None:
        return "unknown"
    s = source_name.strip().lower()
    if s in ("blood pellet", "whole peripheral blood"):
        return s
    if "lymphoblastoid" in s or "lcl" in s:
        return "lcl"
    if "non lymphocyte" in s or "non-lymphocyte" in s:
        return "non_lymph"
    if "buffy" in s:
        return "buffy"
    return s

def run_case_control(df, name, seed=RANDOM_SEED):
    """Run Cohen's d, permutation p, bootstrap 95% CI on cases vs controls."""
    cases_a    = df[df["status"]=="affected"  ]["A_score"].to_numpy(dtype=float)
    controls_a = df[df["status"]=="unaffected"]["A_score"].to_numpy(dtype=float)
    cases_a    = cases_a[~np.isnan(cases_a)]
    controls_a = controls_a[~np.isnan(controls_a)]
    n_c = int(len(cases_a)); n_k = int(len(controls_a))
    d   = cohens_d(cases_a, controls_a) if (n_c>=2 and n_k>=2) else float("nan")
    if n_c>=2 and n_k>=2:
        p_perm, _    = permutation_p(cases_a, controls_a, n_perm=10000, seed=seed)
        ci_lo, ci_hi = bootstrap_d_ci(cases_a, controls_a, n_boot=2000, seed=seed)
    else:
        p_perm = float("nan"); ci_lo = float("nan"); ci_hi = float("nan")
    return {
        "analysis":          name,
        "n_cases":           n_c,
        "n_controls":        n_k,
        "case_A_mean":       float(np.mean(cases_a))    if n_c else float("nan"),
        "case_A_sd":         float(np.std(cases_a,   ddof=1)) if n_c>=2 else float("nan"),
        "control_A_mean":    float(np.mean(controls_a)) if n_k else float("nan"),
        "control_A_sd":      float(np.std(controls_a,ddof=1)) if n_k>=2 else float("nan"),
        "Cohens_d":          d,
        "permutation_p_10000": p_perm,
        "bootstrap_95CI_lo": ci_lo,
        "bootstrap_95CI_hi": ci_hi,
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="T2 GSE104942 Australian HBOC breast")
    ap.add_argument("--matrix_path", required=True)
    ap.add_argument("--panel",       required=True)
    ap.add_argument("--output_dir",  required=True)
    args = ap.parse_args()

    matrix_path = Path(args.matrix_path)
    panel_path  = Path(args.panel)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("T2 GSE104942 — Australian HBOC familial breast, Xu-538 panel")
    print("=" * 78)
    print(f"Random seed:      {RANDOM_SEED}")
    print(f"H_min(immune):    {H_MIN_IMMUNE}")
    print(f"Primary sources:  {PRIMARY_SOURCES}")
    print()

    matrix_sha = sha256_of_file(matrix_path)
    panel_sha  = sha256_of_file(panel_path)
    print(f"  matrix sha256: {matrix_sha}")
    print(f"  panel  sha256: {panel_sha}")
    print()

    panel_cpgs = load_panel(panel_path)
    print(f"Panel CpGs loaded: {len(panel_cpgs)}")

    meta, beta_df = read_series_matrix(matrix_path, panel_cpgs)

    # --- Per-sample A-scores + metadata --------------------------------------
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
        src   = m.get("source_name")
        rows.append({
            "gsm":           gsm,
            "title":         m.get("title"),
            "source_raw":    src,
            "source_class":  classify_source(src),
            "status":        parse_status(chars),
            "age":           parse_age_strict(chars),
            "n_cpgs_valid":  n_valid,
            "A_score":       a_val,
            "raw_characteristics": "|".join([c for c in chars if c is not None]),
        })
    df = pd.DataFrame(rows)

    csv_path = out_dir / "GSE104942_per_sample_A.csv"
    df.to_csv(csv_path, index=False)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV:  {csv_path}")
    print(f"  rows: {len(df)}   sha256: {csv_sha}")
    print()

    # --- Cohort composition -------------------------------------------------
    print("Cohort composition:")
    comp = df.groupby(["source_class","status"]).size().reset_index(name="n")
    for _, r in comp.iterrows():
        print(f"  {r['source_class']:<30s}  status={r['status']:<10s}  n={r['n']}")
    print()

    # --- PRIMARY analysis: blood pellet + whole peripheral blood ------------
    primary_mask = df["source_class"].isin(PRIMARY_SOURCES)
    df_primary   = df[primary_mask & df["status"].isin(("affected","unaffected"))]
    print(f"PRIMARY cohort (blood pellet + whole peripheral blood): n={len(df_primary)}")
    primary_result = run_case_control(df_primary, "primary_blood_pellet_plus_wpb")

    # --- Within-primary subgroup: blood pellet only -------------------------
    df_bp_only = df[(df["source_class"]=="blood pellet") & df["status"].isin(("affected","unaffected"))]
    bp_result  = run_case_control(df_bp_only, "blood_pellet_only")

    # --- Within-primary subgroup: WPB only ----------------------------------
    df_wpb_only = df[(df["source_class"]=="whole peripheral blood") & df["status"].isin(("affected","unaffected"))]
    wpb_result  = run_case_control(df_wpb_only, "whole_peripheral_blood_only")

    # --- SECONDARY: all 210 samples regardless of source (transparency) ----
    df_all = df[df["status"].isin(("affected","unaffected"))]
    all_result = run_case_control(df_all, "all_sources_secondary")

    # --- Report -------------------------------------------------------------
    def _report(r):
        print(f"  n_cases={r['n_cases']:<4d}  n_controls={r['n_controls']:<4d}  "
              f"d={r['Cohens_d']:+.3f}  p_perm={r['permutation_p_10000']:.4f}  "
              f"95%CI=[{r['bootstrap_95CI_lo']:+.3f}, {r['bootstrap_95CI_hi']:+.3f}]")
        print(f"    case_mean={r['case_A_mean']:.4f}(sd {r['case_A_sd']:.4f})  "
              f"ctrl_mean={r['control_A_mean']:.4f}(sd {r['control_A_sd']:.4f})")

    print()
    print("PRIMARY — blood pellet + whole peripheral blood:")
    _report(primary_result)
    print()
    print("Subgroup — blood pellet only:")
    _report(bp_result)
    print()
    print("Subgroup — whole peripheral blood only:")
    _report(wpb_result)
    print()
    print("SECONDARY — all 210 samples (includes LCLs etc., NOT primary evidence):")
    _report(all_result)
    print()

    # --- Direction check ---------------------------------------------------
    predicted_direction = "positive (d > 0) if panel generalizes"
    d_primary = primary_result["Cohens_d"]
    if math.isnan(d_primary):
        verdict = "INSUFFICIENT_N"
    elif d_primary > 0.5:
        verdict = "STRONG_POSITIVE"
    elif d_primary > 0.2:
        verdict = "MODEST_POSITIVE"
    elif d_primary > -0.2:
        verdict = "NULL"
    elif d_primary > -0.5:
        verdict = "MODEST_INVERSE"
    else:
        verdict = "STRONG_INVERSE"
    print(f"Direction verdict on PRIMARY d={d_primary:+.3f}: {verdict}")
    print("NOTE: this is a constitutional/familial test, NOT a pre-diagnostic")
    print("      test. Null here does not falsify EPIC-Italy pre-diagnostic finding.")
    print()

    # --- JSON output --------------------------------------------------------
    out_json = {
        "cohort":         "GSE104942_Joo2018_Australian_HBOC_familial_breast",
        "purpose":        ("constitutional/familial methylation discrimination test — "
                           "NOT a pre-diagnostic time-to-diagnosis test"),
        "cohort_design_note": ("GEO metadata has no age, no time-to-diagnosis, no ICD. "
                               "Status is a single-assessment 'affected/unaffected' flag."),
        "n_samples_total":    int(len(df)),
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
            {"source_class": r["source_class"], "status": r["status"], "n": int(r["n"])}
            for _, r in comp.iterrows()
        ],
        "primary_analysis":     primary_result,
        "blood_pellet_only":    bp_result,
        "whole_peripheral_blood_only": wpb_result,
        "secondary_all_sources": all_result,
        "direction_verdict_primary": verdict,
        "predicted_direction":  predicted_direction,
        "caveat": ("This cohort tests a DIFFERENT claim than EPIC-Italy. "
                   "No TtD, no pre-diagnostic windowing possible."),
    }
    json_path = out_dir / "GSE104942_case_control.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    json_sha = sha256_of_file(json_path)
    print(f"Output JSON:  {json_path}")
    print(f"  sha256: {json_sha}")
    print()
    print("T2 complete.")

if __name__ == "__main__":
    main()
