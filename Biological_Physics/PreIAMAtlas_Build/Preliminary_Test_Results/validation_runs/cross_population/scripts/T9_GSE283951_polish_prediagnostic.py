#!/usr/bin/env python3
"""
T9 — GSE283951 (Sokolowska 2025) Polish pre-diagnostic breast cancer cohort
=============================================================================

Cohort
------
Peripheral blood from 90 Polish women, collected from the Pomeranian Medical
University archive, analyzed on Illumina MethylationEPIC BeadChip (v1.0 B5,
GPL23976). Published August 2025 (PubMed 40418341).

Design:
  34 cases:   Polish women with breast cancer, blood drawn a study-wide mean
              of 4.29 years PRIOR to their subsequent BC diagnosis
  56 controls: Healthy Polish women with 12-year cancer-free follow-up

Critical caveats
----------------
- TtD is NOT per-sample — only the study-wide mean of 4.29 years. We cannot
  perform the canonical TtD-windowed EPIC-Italy analysis. Only pooled
  case-vs-control.
- 4.29 years pre-dx is in the middle of EPIC-Italy's range — NOT in the
  sweet-spot ≥10yr window where the d=+1.85 signal is strongest.
- Blood arsenic (As) levels measured per-sample — we should include this
  as a potential confounder in the comparison, as the study was designed
  around the As × BC × methylation interaction.

Canonical formula
-----------------
Per-sample A-score = mean over Xu-538 CpGs of [H_binary(β) / H_min(immune)]
  H_binary(β) = -β*log2(β) - (1-β)*log2(1-β); returns 0 for β in {0,1,NaN}
  H_min(immune) = 0.838889 (G-003b MCMC posterior)
  Zero-entropy CpGs are KEPT in the denominator (canonical behavior)

Citation
--------
Sokolowska KE, Antoniewski J, Sobalska-Kwapis M, Marciniak W, Strapagiel D,
Lubiński J, Wojdacz TK. Methylomes of healthy Polish women and women
diagnosed with breast cancer with blood arsenic (As) measurements.
[Manuscript in preparation / Data deposited 2025 Aug 13]
PubMed: 40418341
"""
import argparse, gzip, hashlib, json, math, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

H_MIN_IMMUNE = 0.838889
RANDOM_SEED  = 20260420
np.random.seed(RANDOM_SEED)

# ============================================================================
# CORE FUNCTIONS (canonical — matches VAL047 exactly)
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
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    s1 = float(np.std(a, ddof=1)); s2 = float(np.std(b, ddof=1))
    pooled = math.sqrt(((len(a)-1)*s1*s1 + (len(b)-1)*s2*s2) / (len(a)+len(b)-2))
    return 0.0 if pooled == 0 else float((np.mean(a) - np.mean(b)) / pooled)

def permutation_p(cases, controls, n_perm=10000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(cases, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    d_obs = cohens_d(a, b)
    if math.isnan(d_obs): return float("nan"), d_obs
    combined = np.concatenate([a, b])
    n_a = len(a)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        if abs(cohens_d(combined[:n_a], combined[n_a:])) >= abs(d_obs):
            count_ge += 1
    return float((count_ge + 1) / (n_perm + 1)), float(d_obs)

def bootstrap_ci(cases, controls, n_boot=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 1)
    a = np.asarray(cases, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    ds = []
    for _ in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        dd = cohens_d(ra, rb)
        if not math.isnan(dd): ds.append(dd)
    if not ds: return float("nan"), float("nan")
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

# ============================================================================
# LOAD PANEL
# ============================================================================
def load_panel(panel_path):
    with open(panel_path) as f:
        data = json.load(f)
    if isinstance(data, list): return set(data)
    if isinstance(data, dict):
        for key in ("cpgs", "panel", "cpg_ids", "probes"):
            if key in data and isinstance(data[key], list):
                return set(data[key])
        cpgs = set()
        for v in data.values():
            if isinstance(v, list):
                cpgs.update(x for x in v if isinstance(x, str) and x.startswith("cg"))
        if cpgs: return cpgs
    sys.exit(f"[FATAL] Cannot parse panel")

# ============================================================================
# PARSE SERIES MATRIX METADATA
# ============================================================================
def parse_series_matrix_metadata(series_matrix_path):
    """Return dict: sample_N -> {gsm, status, age, as_level, sex, tissue}."""
    opener = gzip.open if str(series_matrix_path).endswith(".gz") else open
    # Collect the header !Sample_* lines
    sample_lines = {}
    with opener(series_matrix_path, "rt") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"): break
            if not line.startswith("!Sample_"): continue
            parts = line.rstrip("\n").split("\t")
            key = parts[0]
            vals = [v.strip('"') for v in parts[1:]]
            sample_lines.setdefault(key, []).append(vals)

    # Sample_title: e.g. 'Sample_1_Polish women with BC' → sample_name='Sample_1'
    titles = sample_lines["!Sample_title"][0]
    gsms   = sample_lines["!Sample_geo_accession"][0]

    # Parse characteristics
    chars_list = sample_lines.get("!Sample_characteristics_ch1", [])

    samples = []
    for i in range(len(titles)):
        title = titles[i]
        # Extract Sample_N prefix
        m = re.match(r'(Sample_\d+)_(.+)', title)
        if m:
            sample_name = m.group(1)
            status_text = m.group(2).strip()
        else:
            sample_name = title
            status_text = ""
        rec = {"sample_name": sample_name, "gsm": gsms[i], "title": title,
               "subject_status": status_text}
        # Go through every characteristic line for this sample
        for char_line in chars_list:
            if i >= len(char_line): continue
            c = char_line[i]
            if ":" not in c: continue
            key, val = c.split(":", 1)
            key = key.strip(); val = val.strip()
            if key == "subject status":     rec["subject_status"] = val
            elif key == "age":              
                try: rec["age"] = float(val)
                except: pass
            elif key == "as":               
                try: rec["as_level"] = float(val)
                except: pass
            elif key == "Sex":              rec["sex"] = val
            elif key == "tissue":           rec["tissue"] = val
        # Classify case vs control from subject_status
        ss_low = rec.get("subject_status", "").lower()
        if "with bc" in ss_low or "breast cancer" in ss_low:
            rec["status"] = "case"
        elif "healthy" in ss_low:
            rec["status"] = "control"
        else:
            rec["status"] = "unknown"
        samples.append(rec)
    return pd.DataFrame(samples)

# ============================================================================
# STREAM-COMPUTE PER-SAMPLE A (data_table.csv.gz)
# ============================================================================
def compute_per_sample_A_streaming(data_table_path, panel_cpgs):
    """
    Stream through a GEO data_table.csv.gz and compute per-sample A-score
    (canonical formula) without loading the entire matrix.

    Returns: DataFrame with columns [sample_name, A_score, n_cpgs_used]
    """
    opener = gzip.open if str(data_table_path).endswith(".gz") else open
    with opener(data_table_path, "rt") as f:
        header = f.readline().rstrip("\n").split(",")
        # header[0] is blank or CpG identifier, header[1:] = sample names
        sample_names = header[1:]
        n_samples = len(sample_names)

        # Accumulators:
        # sum_A[i] = sum of A_score values across included panel CpGs for sample i
        # count[i] = number of included panel CpGs (non-NaN beta) for sample i
        # Canonical mean formula = sum_A[i] / count[i] (mean over ALL non-NaN betas
        # for that sample — includes zero-entropy CpGs which contribute 0 to sum_A)
        sum_A  = np.zeros(n_samples, dtype=np.float64)
        count  = np.zeros(n_samples, dtype=np.int64)

        cpgs_matched = set()
        n_lines_read = 0
        for line in f:
            n_lines_read += 1
            # Performance: only do work for panel CpGs
            first_comma = line.find(",")
            if first_comma < 0: continue
            cpg_id = line[:first_comma]
            if cpg_id not in panel_cpgs: 
                continue
            cpgs_matched.add(cpg_id)
            # Parse the rest of the line (numeric betas)
            fields = line[first_comma+1:].rstrip("\n").split(",")
            for i in range(n_samples):
                if i >= len(fields): break
                v = fields[i]
                if v == "" or v == "NA" or v == "NaN":
                    continue  # canonical .dropna() equivalent
                try:
                    b = float(v)
                except ValueError:
                    continue
                if math.isnan(b):
                    continue
                # Canonical: include even β=0 or β=1 (H_binary returns 0)
                sum_A[i] += H_binary(b) / H_MIN_IMMUNE
                count[i] += 1

    # A = sum / count per sample
    A = np.where(count > 0, sum_A / count, np.nan)
    df = pd.DataFrame({"sample_name": sample_names,
                       "A_score":     A,
                       "n_cpgs_used": count})
    print(f"  lines read: {n_lines_read}    panel CpGs matched: {len(cpgs_matched)}/{len(panel_cpgs)}")
    return df, cpgs_matched

# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series_matrix", required=True)
    ap.add_argument("--data_table",    required=True)
    ap.add_argument("--panel",         required=True)
    ap.add_argument("--output_dir",    required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("T9 GSE283951 — Polish pre-diagnostic breast cancer, Xu-538 panel")
    print("=" * 78)
    sm_sha  = sha256_of_file(args.series_matrix)
    dt_sha  = sha256_of_file(args.data_table)
    pan_sha = sha256_of_file(args.panel)
    print(f"  series_matrix sha256: {sm_sha}")
    print(f"  data_table    sha256: {dt_sha}")
    print(f"  panel         sha256: {pan_sha}")
    print(f"  H_min(immune):        {H_MIN_IMMUNE}")
    print(f"  Random seed:          {RANDOM_SEED}")
    print(f"  A-score formula:      CANONICAL (matches VAL047)")
    print()

    panel_cpgs = load_panel(args.panel)
    print(f"Panel CpGs loaded: {len(panel_cpgs)}")

    print("Parsing series matrix metadata...")
    meta = parse_series_matrix_metadata(args.series_matrix)
    print(f"  samples with metadata: {len(meta)}")
    print(f"  status distribution: {meta['status'].value_counts().to_dict()}")
    # Age range
    if "age" in meta.columns:
        print(f"  age:  min={meta['age'].min():.0f}  max={meta['age'].max():.0f}  "
              f"mean(case)={meta[meta.status=='case']['age'].mean():.1f}  "
              f"mean(ctrl)={meta[meta.status=='control']['age'].mean():.1f}")
    # As levels
    if "as_level" in meta.columns:
        print(f"  As level: mean(case)={meta[meta.status=='case']['as_level'].mean():.2f}  "
              f"mean(ctrl)={meta[meta.status=='control']['as_level'].mean():.2f}")
    print()

    print("Computing per-sample A-score (streaming, canonical formula)...")
    A_df, matched_cpgs = compute_per_sample_A_streaming(args.data_table, panel_cpgs)
    print(f"  panel CpGs matched: {len(matched_cpgs)} of {len(panel_cpgs)}")
    print(f"  per-sample A summary:")
    print(f"    mean = {A_df['A_score'].mean():.4f}   "
          f"sd = {A_df['A_score'].std():.4f}")
    print(f"    n_cpgs_used:  min={A_df['n_cpgs_used'].min()}  "
          f"max={A_df['n_cpgs_used'].max()}  "
          f"median={A_df['n_cpgs_used'].median()}")
    print()

    # Merge on sample_name
    df = meta.merge(A_df, on="sample_name", how="left")
    csv_path = out_dir / "GSE283951_per_sample_A.csv"
    df.to_csv(csv_path, index=False)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV: {csv_path}")
    print(f"  rows: {len(df)}   sha256: {csv_sha}")
    print()

    # ======================================================================
    # ANALYSIS
    # ======================================================================
    cases    = df[df.status=="case"]["A_score"].to_numpy()
    controls = df[df.status=="control"]["A_score"].to_numpy()
    case_ages    = df[df.status=="case"]["age"].to_numpy()    if "age" in df.columns else []
    control_ages = df[df.status=="control"]["age"].to_numpy() if "age" in df.columns else []

    print("=" * 78)
    print("CASE vs CONTROL ANALYSIS — pooled (mean TtD = 4.29 yr)")
    print("=" * 78)
    print(f"  n_cases:    {len(cases)}")
    print(f"  n_controls: {len(controls)}")
    print()

    d_obs = cohens_d(cases, controls)
    p_perm, _ = permutation_p(cases, controls, n_perm=10000, seed=RANDOM_SEED)
    ci_lo, ci_hi = bootstrap_ci(cases, controls, n_boot=2000, seed=RANDOM_SEED)

    delta = float(np.mean(cases) - np.mean(controls))
    print(f"  case  mean A = {np.mean(cases):.4f}  sd = {np.std(cases, ddof=1):.4f}")
    print(f"  ctrl  mean A = {np.mean(controls):.4f}  sd = {np.std(controls, ddof=1):.4f}")
    print(f"  Δ (case - control):     {delta:+.4f}")
    print(f"  Cohen's d (case - ctrl): {d_obs:+.3f}")
    print(f"  p (10k permutations):    {p_perm:.4f}")
    print(f"  95% CI (2k bootstraps):  [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print()

    # Age-matched subanalysis — simple median-split or stratify by decade
    print("=" * 78)
    print("EXPLORATORY — stratified by age decade")
    print("=" * 78)
    age_strata_results = []
    if len(case_ages) and len(control_ages):
        bins = [(40,50), (50,60), (60,70), (70,80)]
        for lo, hi in bins:
            c_in = (df.status=="case") & (df["age"]>=lo) & (df["age"]<hi)
            k_in = (df.status=="control") & (df["age"]>=lo) & (df["age"]<hi)
            c_A = df[c_in]["A_score"].to_numpy()
            k_A = df[k_in]["A_score"].to_numpy()
            if len(c_A) < 3 or len(k_A) < 3:
                print(f"  [{lo:>2}–{hi:<2}]  n_case={len(c_A):<3}  n_ctrl={len(k_A):<3}  -- skipped")
                continue
            d_s = cohens_d(c_A, k_A)
            print(f"  [{lo:>2}–{hi:<2}]  n_case={len(c_A):<3}  n_ctrl={len(k_A):<3}  "
                  f"case_mean={np.mean(c_A):.4f}  ctrl_mean={np.mean(k_A):.4f}  "
                  f"d={d_s:+.3f}")
            age_strata_results.append({
                "age_bin":  f"{lo}-{hi}",
                "n_case":   int(len(c_A)),
                "n_ctrl":   int(len(k_A)),
                "case_A_mean":   float(np.mean(c_A)),
                "control_A_mean": float(np.mean(k_A)),
                "cohens_d": float(d_s),
            })
    print()

    # As-level stratified (the study's design variable)
    print("=" * 78)
    print("EXPLORATORY — stratified by blood As level tertile")
    print("=" * 78)
    as_strata = []
    if "as_level" in df.columns:
        as_vals = df["as_level"].dropna()
        if len(as_vals) >= 9:
            t1, t2 = np.percentile(as_vals, [33, 66])
            print(f"  As tertile cuts: T1<{t1:.2f}, T2=[{t1:.2f},{t2:.2f}), T3>={t2:.2f}")
            for label, lo, hi in [("low", -float('inf'), t1),
                                   ("mid", t1, t2),
                                   ("high", t2, float('inf'))]:
                c_in = (df.status=="case") & (df["as_level"]>=lo) & (df["as_level"]<hi)
                k_in = (df.status=="control") & (df["as_level"]>=lo) & (df["as_level"]<hi)
                c_A = df[c_in]["A_score"].to_numpy()
                k_A = df[k_in]["A_score"].to_numpy()
                if len(c_A) < 3 or len(k_A) < 3:
                    print(f"  [{label:<4}]  n_case={len(c_A):<3}  n_ctrl={len(k_A):<3}  -- skipped")
                    continue
                d_s = cohens_d(c_A, k_A)
                print(f"  [{label:<4}]  n_case={len(c_A):<3}  n_ctrl={len(k_A):<3}  d={d_s:+.3f}")
                as_strata.append({
                    "as_tertile": label, "n_case": int(len(c_A)), "n_ctrl": int(len(k_A)),
                    "case_A_mean": float(np.mean(c_A)), "control_A_mean": float(np.mean(k_A)),
                    "cohens_d": float(d_s),
                })
    print()

    # Output JSON
    out = {
        "cohort":            "GSE283951_Sokolowska2025_Polish_pre_dx_BC",
        "design":            ("34 Polish women with BC, blood drawn mean 4.29 years pre-dx; "
                              "56 healthy Polish women with 12-year cancer-free follow-up. "
                              "Peripheral blood, Illumina MethylationEPIC v1.0 B5."),
        "caveats":           ("TtD is not per-sample (only study-wide mean of 4.29 yr); "
                              "can only do pooled case vs control; cannot replicate TtD windows. "
                              "4.29 yr pre-dx is outside EPIC-Italy's strongest window (>10 yr)."),
        "n_samples_total":   int(len(df)),
        "n_cases":           int(len(cases)),
        "n_controls":        int(len(controls)),
        "panel":             "xu538_breast",
        "panel_n_cpgs":      len(panel_cpgs),
        "panel_cpgs_matched": int(len(matched_cpgs)),
        "H_min_immune":      H_MIN_IMMUNE,
        "random_seed":       RANDOM_SEED,
        "A_score_formula":   "CANONICAL (matches VAL047)",
        "input_sha256":      {"series_matrix": sm_sha, "data_table": dt_sha, "panel": pan_sha},
        "output_sha256":     {"per_sample_csv": csv_sha},
        "primary_result":    {
            "n_cases":         int(len(cases)),
            "n_controls":      int(len(controls)),
            "case_A_mean":     float(np.mean(cases)),
            "case_A_sd":       float(np.std(cases, ddof=1)),
            "control_A_mean":  float(np.mean(controls)),
            "control_A_sd":    float(np.std(controls, ddof=1)),
            "delta_A":         float(delta),
            "cohens_d":        float(d_obs),
            "p_perm_10k":      float(p_perm),
            "CI95_bootstrap":  [float(ci_lo), float(ci_hi)],
        },
        "age_strata":        age_strata_results,
        "as_strata":         as_strata,
    }
    json_path = out_dir / "GSE283951_analysis.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Output JSON: {json_path}")
    print(f"  sha256: {sha256_of_file(json_path)}")
    print()
    print("T9 complete.")

if __name__ == "__main__":
    main()
