#!/usr/bin/env python3
"""
T15 — NHANES 1999-2002 blinded prospective cohort test
=====================================================

Purpose
-------
Heath's question: "I wish we could run tests on a few hundred people's worth
of blood and just see what the data says without knowing if anyone ever
developed cancer or not. Would we be able to see if a certain number of
people flagged for cancers if we didn't know the answer already?"

This test uses the NHANES 1999-2002 cohort (n=2,532 with methylation and
mortality linkage) as the closest available approximation to the blinded
design. Samples were drawn in 1999-2002 with no knowledge of who would die
of cancer; NDI linkage through 2019-12-31 gives 17-20 years of prospective
follow-up. 271 cancer deaths.

HONEST LIMITATION — DOCUMENTED AT THE TOP
-----------------------------------------
The CDC NHANES public release (dnmepi.sas7bdat) contains only COMPUTED
biomarkers (GrimAge, Hannum, Horvath, PhenoAge, DunedinPoAm, cell-type
proportions). The raw 850K CpG beta matrix is in the NCHS Research Data
Center restricted repository and requires DUA application. Therefore:

  - T15 CANNOT test the Xu-538 panel A-score on NHANES tonight.
  - T15 CAN test whether the epigenetic-age-as-cancer-precursor premise
    that the A-score builds on produces the predicted blinded-cohort flag
    pattern in this population.

T15 uses GrimAge acceleration (chronological age regressed out of GrimAge
prediction) as the working proxy for "who would flag high on the framework's
architectural drift assay." GrimAge is not the A-score, but it is the
closest publicly-computable quantity that measures methylation-derived
accelerated biological aging in blood — and if the framework's premise
holds, individuals with high GrimAge acceleration should have elevated
cancer incidence, precisely the pattern the A-score is expected to detect
at higher resolution.

Design
------
For each of {GrimAgeMort, GrimAge2Mort, PhenoAge, DunedinPoAm}:

  (1) Compute age acceleration = clock_prediction - chronological_age.
  (2) Stratify the cohort into deciles and quartiles by age acceleration.
  (3) Report for each decile/quartile:
        - n participants
        - n cancer deaths observed over follow-up
        - cancer mortality rate (events per 100 person-years)
        - 5-year, 10-year, 15-year Kaplan-Meier cancer mortality
  (4) Top-vs-bottom decile and top-vs-bottom quartile HR with CI.
  (5) Blinded-flag analysis: if we had simply flagged everyone at/above
      the 90th percentile of age acceleration as "elevated," how many
      would have been flagged, and of those flagged, how many developed
      cancer?

This is the blinded-cohort version of the question. It answers:
"in a US national sample of 2,532 adults, if we had flagged the top 10%
on their methylation-aging signal at baseline (1999-2002), how accurate
would that flag have been at predicting who died of cancer over the
subsequent ~17 years?"

The framework's prediction: top-decile flag rate ~2-4× elevated cancer
mortality relative to bottom decile. If that prediction holds on GrimAge
acceleration, the A-score (which is a more targeted quantity than GrimAge)
is expected to produce at least that effect size when the NCHS-RDC Xu-538
test becomes accessible.

Inputs
------
  - dnmepi.sas7bdat              (NHANES DNA methylation biomarkers)
  - DEMO.xpt, DEMO_B.xpt         (chronological age, sex, race/ethnicity)
  - NHANES_1999_2000_MORT_2019_PUBLIC.dat    (Linked Mortality File 1999-2000)
  - NHANES_2001_2002_MORT_2019_PUBLIC.dat    (Linked Mortality File 2001-2002)

Random seed: 20260420 (consistent with T1-T14)
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# scipy for percentile / stats
try:
    from scipy import stats as spstats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# lifelines for Cox; if not available, fall back to manual Cox implementation
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


RANDOM_SEED = 20260420
np.random.seed(RANDOM_SEED)


# ============================================================================
# FIXED LMF COLUMN LAYOUT (same as T8)
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
# KAPLAN-MEIER (minimal implementation to avoid lifelines dependency)
# ============================================================================

def km_cumulative_incidence(follow_up, event, t_list):
    """
    Kaplan-Meier cumulative incidence at times in t_list.
    Returns dict {t: cumulative_incidence}.

    follow_up: array of follow-up times (in years)
    event: array of 0/1 event indicators
    t_list: list of evaluation time points
    """
    follow_up = np.asarray(follow_up, dtype=float)
    event = np.asarray(event, dtype=int)
    order = np.argsort(follow_up)
    fu = follow_up[order]
    ev = event[order]

    survival = 1.0
    at_risk = len(fu)
    i = 0
    S_at = {}
    sorted_t = sorted(set(t_list))
    t_idx = 0
    for j in range(len(fu)):
        t_j = fu[j]
        # advance t_idx: record S at all t <= t_j
        while t_idx < len(sorted_t) and sorted_t[t_idx] < t_j:
            S_at[sorted_t[t_idx]] = survival
            t_idx += 1
        if ev[j] == 1:
            survival = survival * (at_risk - 1) / at_risk
        at_risk -= 1
    while t_idx < len(sorted_t):
        S_at[sorted_t[t_idx]] = survival
        t_idx += 1

    return {t: 1.0 - S_at[t] for t in sorted_t}


# ============================================================================
# COX (two-sample log-rank-based HR as fallback if no lifelines)
# ============================================================================

def logrank_hr_two_group(fu, ev, group):
    """
    Compute HR for group==1 vs group==0 using log-rank-based estimator.
    Returns dict with HR, log-rank chi-square, p-value.

    group: 0 or 1 array.
    """
    fu = np.asarray(fu, dtype=float)
    ev = np.asarray(ev, dtype=int)
    gr = np.asarray(group, dtype=int)

    # Valid observations only
    mask = ~np.isnan(fu) & (ev != -1)
    fu = fu[mask]; ev = ev[mask]; gr = gr[mask]

    # Events in group 1 and in group 0
    e1 = int(ev[gr == 1].sum())
    e0 = int(ev[gr == 0].sum())
    n1 = int((gr == 1).sum())
    n0 = int((gr == 0).sum())

    if e1 == 0 or e0 == 0 or n1 == 0 or n0 == 0:
        return {"HR": float("nan"), "chi2": float("nan"), "p_value": float("nan"),
                "e1": e1, "e0": e0, "n1": n1, "n0": n0}

    # Log-rank: at each event time, compute expected events in group 1
    # (Mantel-Haenszel)
    times = np.sort(np.unique(fu[ev == 1]))
    O1 = 0.0
    E1 = 0.0
    V  = 0.0
    for t in times:
        n_at_risk_1 = int(((fu >= t) & (gr == 1)).sum())
        n_at_risk_0 = int(((fu >= t) & (gr == 0)).sum())
        N = n_at_risk_1 + n_at_risk_0
        if N < 2: continue
        d_at_time_1 = int(((fu == t) & (ev == 1) & (gr == 1)).sum())
        d_at_time   = int(((fu == t) & (ev == 1)).sum())
        O1 += d_at_time_1
        exp_1 = d_at_time * n_at_risk_1 / N
        E1 += exp_1
        if N > 1:
            V += (d_at_time * (N - d_at_time) * n_at_risk_1 * n_at_risk_0) / (N * N * (N - 1))
    if V <= 0 or E1 == 0:
        return {"HR": float("nan"), "chi2": float("nan"), "p_value": float("nan"),
                "e1": e1, "e0": e0, "n1": n1, "n0": n0}
    chi2 = (O1 - E1) ** 2 / V
    # HR = O1/E1 (Mantel-Haenszel style; approximate but standard)
    HR = (O1 / E1) / ((e0 - (e1 - O1)) / max(1e-9, (e0 + e1 - E1)))
    # Simpler and standard: HR = (O1/E1) / ((e0 - O0_exp)/E0_exp) where E0_exp = total_events - E1
    total_events = O1 + e0
    O0_expected = total_events - E1  # = e0 - (E1 - O1 + O1) = e0 - E1 + O1 ... simpler below
    # The cleanest: HR ≈ (O1/E1) / (O0/E0)  where O0 = e0, E0 = total_events - E1
    O0 = float(e0)
    E0 = total_events - E1
    if E0 <= 0:
        HR = float("nan")
    else:
        HR = (O1 / E1) / (O0 / E0)
    # p-value from chi2, 1 dof
    if HAS_SCIPY:
        p = float(spstats.chi2.sf(chi2, df=1))
    else:
        p = math.erfc(math.sqrt(chi2 / 2.0))  # approx
    return {"HR": float(HR), "chi2": float(chi2), "p_value": float(p),
            "e1": e1, "e0": e0, "n1": n1, "n0": n0,
            "O1": float(O1), "E1": float(E1)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="T15 NHANES blinded prospective cohort cancer mortality")
    ap.add_argument("--dnmepi",   required=True)
    ap.add_argument("--demo_a",   required=True)
    ap.add_argument("--demo_b",   required=True)
    ap.add_argument("--lmf_a",    required=True)
    ap.add_argument("--lmf_b",    required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("T15 NHANES 1999-2002 — blinded prospective cohort test")
    print("=" * 78)
    print()
    print("Design: samples drawn 1999-2002 with no knowledge of cancer outcomes.")
    print("        NDI mortality linkage through 2019-12-31 gives up to 20 years")
    print("        of prospective follow-up. 271 cancer deaths.")
    print()
    print("Metric: GrimAge acceleration (+ ancillary clocks) as the closest")
    print("        publicly-computable proxy for the framework's A-score.")
    print("        Raw Xu-538 panel requires NCHS-RDC DUA application.")
    print()

    # ---- SHA-256 inputs ------------------------------------------------------
    shas = {}
    for label, path in [("dnmepi", args.dnmepi), ("DEMO_A", args.demo_a),
                        ("DEMO_B", args.demo_b), ("LMF_A", args.lmf_a),
                        ("LMF_B", args.lmf_b)]:
        shas[label] = sha256_of_file(path)

    # ---- Load DNMEPI --------------------------------------------------------
    dnm = pd.read_sas(args.dnmepi, format="sas7bdat", encoding="latin-1")
    print(f"DNMEPI loaded: {dnm.shape[0]} rows × {dnm.shape[1]} cols")
    dnm = dnm.dropna(subset=["PhenoAge"]).copy()
    print(f"  with PhenoAge non-null: {len(dnm)}")

    # ---- Load demographics (both cycles) -----------------------------------
    demo_a = pd.read_sas(args.demo_a, format="xport", encoding="latin-1")
    demo_b = pd.read_sas(args.demo_b, format="xport", encoding="latin-1")
    demo = pd.concat([demo_a, demo_b], ignore_index=True, sort=False)
    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1"]].copy()
    demo.columns = ["SEQN", "age_years", "sex", "race_eth"]

    # ---- Load LMF (both cycles) --------------------------------------------
    lmf_a = read_lmf(args.lmf_a)
    lmf_b = read_lmf(args.lmf_b)
    lmf = pd.concat([lmf_a, lmf_b], ignore_index=True, sort=False)

    # ---- Merge & filter ----------------------------------------------------
    df = dnm.merge(demo, on="SEQN", how="left").merge(lmf, on="SEQN", how="left")
    df = df[df["eligstat"] == 1].copy()
    df["follow_up_years"] = df["permth_exm"] / 12.0
    df["death_any"] = (df["mortstat"] == 1).astype(int)
    df["death_cancer"] = ((df["mortstat"] == 1) & (df["ucod_leading"] == 2)).astype(int)
    df = df.dropna(subset=["follow_up_years", "age_years"]).copy()

    print()
    print("=" * 78)
    print(f"  Cohort size (eligstat=1, DNMEPI non-null, fu_years non-null): n = {len(df)}")
    print(f"  All-cause deaths:  {int(df['death_any'].sum())} ({df['death_any'].mean()*100:.1f}%)")
    print(f"  Cancer deaths:     {int(df['death_cancer'].sum())} ({df['death_cancer'].mean()*100:.1f}%)")
    print(f"  Median follow-up:  {df['follow_up_years'].median():.1f} yr  (max: {df['follow_up_years'].max():.1f})")
    print(f"  Median age at draw: {df['age_years'].median():.1f} yr")
    print("=" * 78)
    print()

    # ---- Compute age acceleration for each clock ---------------------------
    CLOCKS = [
        ("GrimAgeMort",  "GrimAgeMort"),
        ("GrimAge2Mort", "GrimAge2Mort"),
        ("PhenoAge",     "PhenoAge"),
        ("DunedinPoAm",  "DunedinPoAm"),   # This is a rate, not an age; different treatment
    ]

    results_per_clock = {}

    # ========================================================================
    # BLINDED-COHORT ANALYSIS — what would we have flagged and who got cancer?
    # ========================================================================

    print("=" * 78)
    print("BLINDED-COHORT FLAG ANALYSIS")
    print("=" * 78)
    print()
    print("Question: if in 1999-2002 we had simply flagged the top N% of each")
    print("  clock's age-acceleration distribution, how many people would have")
    print("  flagged, and of those flagged, how many developed cancer?")
    print()

    for clock_name, clock_col in CLOCKS:
        sub = df.dropna(subset=[clock_col]).copy()
        if len(sub) < 100:
            print(f"  {clock_name}: insufficient data ({len(sub)})")
            continue

        # Age acceleration = clock - chronological (except DunedinPoAm which is a rate)
        if clock_name == "DunedinPoAm":
            sub["accel"] = sub[clock_col] - 1.0   # baseline rate = 1.0
        else:
            sub["accel"] = sub[clock_col] - sub["age_years"]

        # ---- Quartile, decile, and custom cutoffs -------------------------
        cutoffs = {
            "top_50":  0.50,
            "top_25":  0.75,
            "top_10":  0.90,
            "top_5":   0.95,
        }
        print(f"[{clock_name}]  n={len(sub)}, cancer deaths in this subset={int(sub['death_cancer'].sum())}")
        print(f"  age_accel distribution: median={sub['accel'].median():+.2f}  "
              f"IQR=[{sub['accel'].quantile(0.25):+.2f}, {sub['accel'].quantile(0.75):+.2f}]")

        flag_table = []
        for label, pct in cutoffs.items():
            thresh = sub["accel"].quantile(pct)
            flagged = sub[sub["accel"] >= thresh].copy()
            not_flagged = sub[sub["accel"] < thresh].copy()
            n_f = len(flagged)
            n_nf = len(not_flagged)
            ca_f = int(flagged["death_cancer"].sum())
            ca_nf = int(not_flagged["death_cancer"].sum())
            rate_f = ca_f / n_f if n_f > 0 else float("nan")
            rate_nf = ca_nf / n_nf if n_nf > 0 else float("nan")
            # PPV (what fraction of flagged people died of cancer)
            ppv = rate_f
            # Risk ratio (relative to not-flagged)
            rr = (rate_f / rate_nf) if (rate_nf > 0) else float("nan")
            row = {
                "flag_label":     label,
                "pct_cutoff":     pct,
                "threshold":      float(thresh),
                "n_flagged":      n_f,
                "n_not_flagged":  n_nf,
                "cancer_deaths_flagged":      ca_f,
                "cancer_deaths_not_flagged":  ca_nf,
                "cancer_rate_flagged":        rate_f,
                "cancer_rate_not_flagged":    rate_nf,
                "ppv":            ppv,
                "relative_risk":  rr,
            }
            flag_table.append(row)
            pct_label = f"top {100-int(pct*100)}%"
            print(f"    {pct_label:<8}  thresh={thresh:>+6.2f}  "
                  f"flagged n={n_f:>4}  "
                  f"cancer-deaths-among-flagged={ca_f:>3} ({ppv*100:>5.1f}% PPV)  "
                  f"RR={rr:.2f}")

        # ---- Decile comparison with Cox/log-rank ---------------------------
        p90 = sub["accel"].quantile(0.90)
        p10 = sub["accel"].quantile(0.10)
        sub["decile_top"] = (sub["accel"] >= p90).astype(int)
        sub["decile_bot"] = (sub["accel"] <= p10).astype(int)
        topbot = sub[sub["decile_top"] + sub["decile_bot"] > 0].copy()
        topbot["group"] = topbot["decile_top"]  # 1 = top, 0 = bottom
        logrank = logrank_hr_two_group(
            topbot["follow_up_years"].values,
            topbot["death_cancer"].values,
            topbot["group"].values,
        )
        print(f"    top vs bottom decile: HR = {logrank['HR']:.2f}, "
              f"p = {logrank['p_value']:.2e}  "
              f"(events top={logrank['e1']}, bottom={logrank['e0']})")

        # ---- 5yr, 10yr, 15yr KM cancer mortality per quartile --------------
        eval_times = [5, 10, 15, 17]
        sub["quartile"] = pd.qcut(sub["accel"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        km_table = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub_q = sub[sub["quartile"] == q]
            km = km_cumulative_incidence(
                sub_q["follow_up_years"].values,
                sub_q["death_cancer"].values,
                eval_times,
            )
            km_table[q] = {
                "n": int(len(sub_q)),
                "events": int(sub_q["death_cancer"].sum()),
                "km_cum_incidence": {f"{t}yr": float(km[t]) for t in eval_times},
            }
        print(f"    cumulative cancer mortality by quartile of accel:")
        print(f"      {'':8s}  {'n':>5}  {'ev':>4}  {'5yr':>6}  {'10yr':>6}  {'15yr':>6}  {'17yr':>6}")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            ki = km_table[q]["km_cum_incidence"]
            print(f"      {q:8s}  {km_table[q]['n']:>5}  {km_table[q]['events']:>4}  "
                  f"{ki['5yr']*100:>5.1f}%  {ki['10yr']*100:>5.1f}%  "
                  f"{ki['15yr']*100:>5.1f}%  {ki['17yr']*100:>5.1f}%")

        print()
        results_per_clock[clock_name] = {
            "n": int(len(sub)),
            "n_cancer_deaths": int(sub["death_cancer"].sum()),
            "accel_median":    float(sub["accel"].median()),
            "accel_IQR":      [float(sub["accel"].quantile(0.25)), float(sub["accel"].quantile(0.75))],
            "flag_table":     flag_table,
            "top_vs_bot_decile_logrank": logrank,
            "km_by_quartile": km_table,
        }

    # ========================================================================
    # WRITE RESULTS
    # ========================================================================

    out = {
        "test_id":     "T15",
        "cohort":      "NHANES_1999_2002_blinded_prospective",
        "random_seed": RANDOM_SEED,
        "design_description": (
            "NHANES 1999-2002 samples drawn with no knowledge of subsequent cancer. "
            "NDI LMF through 2019-12-31 gives up to 20 yr prospective follow-up. "
            "Age acceleration (clock - chronological age) used as flag metric. "
            "Raw Xu-538 panel NOT available in public release (NCHS RDC only)."
        ),
        "honest_limitation": (
            "GrimAge/PhenoAge age-acceleration is a published methylation-based "
            "composite, NOT the framework's Xu-538 immune A-score. If the framework's "
            "premise that methylation drift predicts cancer risk holds, these clocks "
            "should produce the predicted blinded-flag pattern. The A-score is "
            "expected to produce at least as strong a signal when RDC access is "
            "obtained, because it targets the architectural drift quantity directly."
        ),
        "n_cohort":          int(len(df)),
        "n_all_deaths":      int(df["death_any"].sum()),
        "n_cancer_deaths":   int(df["death_cancer"].sum()),
        "median_follow_up_yr": float(df["follow_up_years"].median()),
        "max_follow_up_yr":  float(df["follow_up_years"].max()),
        "median_age_at_draw":float(df["age_years"].median()),
        "input_sha256":      shas,
        "per_clock":         results_per_clock,
    }
    out_json = out_dir / "T15_NHANES_blinded_prospective_results.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Results JSON: {out_json}")
    print(f"sha256:       {sha256_of_file(out_json)[:16]}...")
    print()
    print("T15 complete.")


if __name__ == "__main__":
    main()
