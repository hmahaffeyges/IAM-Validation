#!/usr/bin/env python3
"""
T8 — NHANES 1999-2002 DNAm public biomarker panel × Linked Mortality File
===========================================================================

Cohort
------
NHANES 1999-2000 + 2001-2002 survey cycles. DNAm assay performed on stored
whole-blood DNA for participants aged 50+ who consented. Released as
`dnmepi.sas7bdat` in July 2024 (NCHS/Duke-Liu laboratory).

What was released (and what was NOT)
-------------------------------------
The PUBLIC DNMEPI file contains COMPUTED epigenetic biomarkers per sample:
  - Epigenetic clocks:     Horvath, Hannum, SkinBlood, PhenoAge, GrimAge,
                           GrimAge2, Zhang, Lin, Weidner, VidalBralo
  - Pace-of-aging:         DunedinPoAm
  - Cell counts (DNAm-pred): CD4+, CD8+, NK, B, mono, neu
  - Mortality predictors:  GDF15, B2M, CystatinC, TIMP1, ADM, PAI1, Leptin,
                           PackYrs, CRP, logA1C (components of GrimAge)

The per-CpG β value matrix is NOT in the public release — it is in the NCHS
Research Data Center restricted-access Genetic Data Repository, requiring
DUA application (4-8 week timeline). Therefore we CANNOT compute an
Xu-538 A-score on this cohort tonight.

What this test does
-------------------
Since we can't compute our own A-score, we test a WEAKER but still relevant
hypothesis: in the NHANES cohort, do the published methylation-based
epigenetic aging biomarkers associate with subsequent cancer mortality?

If the framework's premise is correct that cumulative information-writing
drives cellular aging, then clocks that quantify biological aging signal
(PhenoAge, GrimAge, etc.) should predict cancer mortality over the 17-19
year follow-up period (NHANES draw 1999-2002 → LMF end 2019-12-31).

This does NOT validate the Xu-538 A-score specifically. It is preparatory
work: establishing that this cohort is suitable for the authorized-access
A-score analysis, and that the framework's premise holds in this
population.

Analyses performed
------------------
For each of {PhenoAge, GrimAgeMort, GrimAge2Mort, DunedinPoAm}:
  1. Age acceleration = clock_prediction - chronological_age
  2. Cox PH regression: cancer_mortality ~ age_accel + sex + race
     (using weighted or unweighted — NHANES uses complex survey design,
      but for HR estimation unweighted Cox is the standard first pass)
  3. HR per standard deviation of age acceleration
  4. HR for top decile vs bottom decile of age acceleration
  5. Sanity check: same model for all-cause mortality (should see larger HR)

Outcomes
--------
  All-cause mortality:  mortstat == 1
  Cancer mortality:     ucod_leading == 2  (malignant neoplasms, ICD-10
                                            C00-C97 grouped)
  Follow-up time:       permth_exm / 12   (months-to-death from MEC exam)

LMF files
---------
  NHANES_1999_2000_MORT_2019_PUBLIC.dat  (cycle A)
  NHANES_2001_2002_MORT_2019_PUBLIC.dat  (cycle B)
Fixed-width column layout (columns 1-indexed, inclusive):
  SEQN        1-6
  eligstat    15
  mortstat    16
  ucod_leading 17-19
  diabetes    20
  hyperten    21
  permth_int  43-45
  permth_exm  46-48

Caveats
-------
- Public-use LMF is PERTURBED for privacy. Some follow-up times and causes
  are synthetic. Vital status is NOT perturbed. Population-level HRs remain
  valid; individual records are not.
- Survey design (PSU, strata, weights) is not applied for HR estimation
  here. For a formal population-level inference we would use weighted Cox
  with the survey package; this first-pass analysis is unweighted.
- UCOD_LEADING code 2 is "Malignant neoplasms" per NCHS grouping, covering
  ICD-10 C00-C97.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# ============================================================================
# FIXED LMF COLUMN LAYOUT
# ============================================================================

LMF_COLS = [
    ("SEQN",         (1,  6),  "int"),
    ("eligstat",     (15, 15), "int"),
    ("mortstat",     (16, 16), "int"),
    ("ucod_leading", (17, 19), "int"),
    ("diabetes",     (20, 20), "int"),
    ("hyperten",     (21, 21), "int"),
    ("permth_int",   (43, 45), "int"),
    ("permth_exm",   (46, 48), "int"),
]

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_lmf(path):
    """Parse NHANES LMF fixed-width .dat file."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\r\n")
            row = {}
            for name, (s, e), typ in LMF_COLS:
                raw = line[s-1:e].strip()
                if raw == "" or raw == ".":
                    row[name] = np.nan
                else:
                    try:    row[name] = int(raw) if typ == "int" else float(raw)
                    except ValueError: row[name] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="T8 NHANES epigenetic age × cancer mortality")
    ap.add_argument("--dnmepi",   required=True)
    ap.add_argument("--demo_a",   required=True, help="DEMO.xpt (1999-2000)")
    ap.add_argument("--demo_b",   required=True, help="DEMO_B.xpt (2001-2002)")
    ap.add_argument("--lmf_a",    required=True, help="NHANES_1999_2000_MORT_2019_PUBLIC.dat")
    ap.add_argument("--lmf_b",    required=True, help="NHANES_2001_2002_MORT_2019_PUBLIC.dat")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- SHA-256 inputs ---------------------------------------------------
    print("=" * 78)
    print("T8 NHANES 1999-2002 — epigenetic age × cancer mortality")
    print("=" * 78)
    shas = {}
    for label, path in [("dnmepi", args.dnmepi), ("DEMO_A", args.demo_a),
                        ("DEMO_B", args.demo_b), ("LMF_A", args.lmf_a),
                        ("LMF_B", args.lmf_b)]:
        shas[label] = sha256_of_file(path)
        print(f"  {label:<8s} sha256: {shas[label]}")
    print()

    # ---- Load DNMEPI ------------------------------------------------------
    dnm = pd.read_sas(args.dnmepi, format="sas7bdat", encoding="latin-1")
    print(f"DNMEPI loaded: {dnm.shape[0]} rows × {dnm.shape[1]} cols")
    # Filter to rows with actual DNAm data (PhenoAge non-null is a good proxy)
    dnm = dnm.dropna(subset=["PhenoAge"]).copy()
    print(f"  with PhenoAge non-null: {len(dnm)}")

    # ---- Load demographics (both cycles) ---------------------------------
    demo_a = pd.read_sas(args.demo_a, format="xport", encoding="latin-1")
    demo_b = pd.read_sas(args.demo_b, format="xport", encoding="latin-1")
    demo = pd.concat([demo_a, demo_b], ignore_index=True, sort=False)
    # Keep only what we need: SEQN, RIDAGEYR (age at screening), RIAGENDR (sex),
    # RIDRETH1 (race/ethnicity)
    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1"]].copy()
    demo.columns = ["SEQN", "age_years", "sex", "race_eth"]
    print(f"Demographics combined: {len(demo)} rows")

    # ---- Load LMF (both cycles) ------------------------------------------
    lmf_a = read_lmf(args.lmf_a)
    lmf_b = read_lmf(args.lmf_b)
    lmf = pd.concat([lmf_a, lmf_b], ignore_index=True, sort=False)
    print(f"LMF combined: {len(lmf)} rows")
    print(f"  eligstat distribution: {lmf['eligstat'].value_counts(dropna=False).to_dict()}")
    print(f"  mortstat distribution: {lmf['mortstat'].value_counts(dropna=False).to_dict()}")
    print(f"  ucod_leading distribution: {lmf['ucod_leading'].value_counts(dropna=False).to_dict()}")
    print()

    # ---- Merge everything -------------------------------------------------
    df = dnm.merge(demo, on="SEQN", how="left").merge(lmf, on="SEQN", how="left")
    print(f"Merged (DNMEPI × DEMO × LMF): {len(df)}")
    print(f"  with mortstat non-null:       {df['mortstat'].notna().sum()}")
    print(f"  with follow-up (permth_exm):  {df['permth_exm'].notna().sum()}")
    print()

    # ---- Keep only linkage-eligible -------------------------------------
    df = df[df["eligstat"] == 1].copy()
    print(f"After eligstat=1 filter: {len(df)}")

    # ---- Build survival outcomes -----------------------------------------
    # Follow-up time in years. For those assumed alive, permth_exm is set
    # to the end-of-study period (~240 months for 2001 cohort, ~252 for 1999).
    df["follow_up_years"] = df["permth_exm"] / 12.0
    # All-cause death indicator
    df["death_any"] = (df["mortstat"] == 1).astype(int)
    # Cancer death indicator: mortstat==1 AND ucod_leading==2
    df["death_cancer"] = ((df["mortstat"] == 1) & (df["ucod_leading"] == 2)).astype(int)

    print()
    print("Outcome counts (post eligstat=1, post DNMEPI non-null):")
    print(f"  n = {len(df)}")
    print(f"  all-cause deaths:  {df['death_any'].sum()}  "
          f"({df['death_any'].mean()*100:.1f}%)")
    print(f"  cancer deaths:     {df['death_cancer'].sum()}  "
          f"({df['death_cancer'].mean()*100:.1f}%)")
    print(f"  follow-up years: median={df['follow_up_years'].median():.1f}  "
          f"max={df['follow_up_years'].max():.1f}")
    print()

    # ---- Compute age-acceleration residuals -------------------------------
    # For each epigenetic clock, age-accel = clock_age - chronological_age.
    # We fit Cox on raw age-accel (per year) and on z-score (per 1 SD).
    clocks = ["PhenoAge", "GrimAgeMort", "GrimAge2Mort", "HannumAge",
              "HorvathAge", "DunedinPoAm"]
    for c in clocks:
        if c == "DunedinPoAm":
            # Not an age; it's a rate (years/year). Use raw value.
            df[f"{c}_val"] = df[c]
            # Z-score
            df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std()
        else:
            df[f"{c}_accel"] = df[c] - df["age_years"]
            df[f"{c}_z"]     = (df[f"{c}_accel"] - df[f"{c}_accel"].mean()) / df[f"{c}_accel"].std()

    # ---- Cox PH analysis --------------------------------------------------
    results = []
    print("=" * 78)
    print("COX PROPORTIONAL HAZARDS — age-accel or rate, per 1 SD")
    print("=" * 78)

    def run_cox(df_in, predictor, outcome_col):
        cph = CoxPHFitter()
        cols = ["follow_up_years", outcome_col, predictor, "age_years", "sex"]
        sub = df_in[cols].dropna().copy()
        if len(sub) < 50 or sub[outcome_col].sum() < 5:
            return None, sub
        try:
            cph.fit(sub, duration_col="follow_up_years", event_col=outcome_col)
            return cph, sub
        except Exception as e:
            print(f"  ERROR fitting {predictor} vs {outcome_col}: {e}")
            return None, sub

    for c in clocks:
        pred = f"{c}_z"
        print(f"\n--- {c}  (z-scored per 1 SD) ---")
        for outcome_lbl, outcome_col in [("all-cause", "death_any"),
                                          ("cancer",    "death_cancer")]:
            cph, sub = run_cox(df, pred, outcome_col)
            if cph is None:
                print(f"  [{outcome_lbl}] insufficient data (n={len(sub)}, events={sub[outcome_col].sum() if len(sub) else 0})")
                continue
            hr = cph.hazard_ratios_[pred]
            ci = cph.confidence_intervals_.loc[pred].values
            ci_hr = np.exp(ci)
            pv = cph.summary.loc[pred, "p"]
            n_used = len(sub)
            n_events = int(sub[outcome_col].sum())
            print(f"  [{outcome_lbl:9s}]  n={n_used}  events={n_events:>4d}  "
                  f"HR_per_SD={hr:.3f}  95% CI=[{ci_hr[0]:.3f}, {ci_hr[1]:.3f}]  p={pv:.4g}")
            results.append({
                "clock":            c,
                "outcome":          outcome_lbl,
                "n":                n_used,
                "n_events":         n_events,
                "HR_per_SD":        float(hr),
                "CI95_HR_lo":       float(ci_hr[0]),
                "CI95_HR_hi":       float(ci_hr[1]),
                "p_value":          float(pv),
                "covariates":       ["age_years", "sex"],
            })

    # ---- Also: top-decile vs bottom-decile HR for PhenoAge & GrimAge -----
    print()
    print("=" * 78)
    print("TOP vs BOTTOM DECILE of age acceleration (informational)")
    print("=" * 78)

    for clock in ["PhenoAge", "GrimAgeMort"]:
        colz = f"{clock}_z"
        sub = df[["follow_up_years", "death_cancer", "death_any",
                  colz, "age_years", "sex"]].dropna().copy()
        if len(sub) < 100:
            print(f"  {clock}: insufficient n"); continue
        # Decile split
        sub["decile"] = pd.qcut(sub[colz], 10, labels=False, duplicates="drop")
        sub["top_vs_bot"] = sub["decile"].map(
            {0: 0, 9: 1}).where(lambda x: x.isin([0, 1]))
        sub_td = sub.dropna(subset=["top_vs_bot"])
        for outcome_lbl, outcome_col in [("all-cause", "death_any"),
                                          ("cancer",    "death_cancer")]:
            n_ev = int(sub_td[outcome_col].sum())
            if n_ev < 3:
                print(f"  {clock}  [{outcome_lbl}]:  insufficient events")
                continue
            cph = CoxPHFitter()
            try:
                cph.fit(sub_td[["follow_up_years", outcome_col, "top_vs_bot",
                                "age_years", "sex"]],
                         duration_col="follow_up_years", event_col=outcome_col)
                hr = cph.hazard_ratios_["top_vs_bot"]
                ci = cph.confidence_intervals_.loc["top_vs_bot"].values
                ci_hr = np.exp(ci)
                pv = cph.summary.loc["top_vs_bot", "p"]
                print(f"  {clock:15s}  [{outcome_lbl:9s}]  n_top={int((sub_td['top_vs_bot']==1).sum())}  "
                      f"n_bot={int((sub_td['top_vs_bot']==0).sum())}  "
                      f"events={n_ev}  "
                      f"HR_top_vs_bot={hr:.3f}  95% CI=[{ci_hr[0]:.3f}, {ci_hr[1]:.3f}]  p={pv:.4g}")
                results.append({
                    "clock":            clock,
                    "outcome":          outcome_lbl,
                    "analysis":         "top_vs_bottom_decile_age_accel",
                    "n":                int(len(sub_td)),
                    "n_events":         n_ev,
                    "HR_top_vs_bot":    float(hr),
                    "CI95_HR_lo":       float(ci_hr[0]),
                    "CI95_HR_hi":       float(ci_hr[1]),
                    "p_value":          float(pv),
                    "covariates":       ["age_years", "sex"],
                })
            except Exception as e:
                print(f"    error: {e}")

    # ---- Write JSON output -------------------------------------------------
    out = {
        "cohort":        "NHANES_1999_2002_DNMEPI_publicuse",
        "note":          ("Public-use epigenetic biomarker file × Linked Mortality "
                          "File (2019). Xu-538 panel A-score NOT computable here "
                          "because per-CpG β values are not in the public release."),
        "n_with_dnam_and_elig": int(len(df)),
        "n_all_cause_deaths":   int(df["death_any"].sum()),
        "n_cancer_deaths":      int(df["death_cancer"].sum()),
        "follow_up_years": {
            "median": float(df["follow_up_years"].median()),
            "max":    float(df["follow_up_years"].max()),
        },
        "input_sha256":    shas,
        "cox_results":     results,
    }
    json_path = out_dir / "NHANES_T8_cox_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    json_sha = sha256_of_file(json_path)
    print()
    print(f"Output JSON:  {json_path}")
    print(f"  sha256: {json_sha}")
    print()
    print("T8 complete.")

if __name__ == "__main__":
    main()
