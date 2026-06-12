#!/usr/bin/env python3
"""
T5 — GSE89093 (Roos 2016) TwinsUK cancer-discordant MZ twin-pair cohort
=======================================================================

Cohort
------
92 peripheral blood samples from 46 adult female monozygotic twin-pairs,
GPL13534 (Illumina 450K), TwinsUK consortium.

Each pair contains one twin with a confirmed cancer diagnosis and her
genetically identical co-twin who is cancer-free. Blood samples are drawn
at a single time point. The TtD field ("age at diagnosis minus age at
blood sampling") is shared within each pair and indicates the timing
of blood draw relative to the affected twin's diagnosis:
  - Positive TtD  → blood drawn BEFORE diagnosis (pre-diagnostic)
  - Negative TtD  → blood drawn AFTER diagnosis (post-diagnostic)

Cohort composition (after propagating cancer location to healthy co-twin):

  Pre-diagnostic  (Before; 20 pairs / 40 samples)
    Breast:  12 pairs   TtD [0.04, 6.62] yr
    Colon:    4 pairs   TtD [2.08, 12.85] yr
    Other:    4 pairs   (melanoma, endometrium, ovary, pancreas)

  Post-diagnostic (After; 26 pairs / 52 samples)
    Breast:  14 pairs   TtD [-4.78, -0.77] yr
    Colon:    8 pairs
    Other:    4 pairs

Primary value of this cohort
-----------------------------
MZ twin pairs are **genetically identical**. Any methylation difference
between the affected twin and her healthy co-twin cannot be explained by
germline sequence variation, in-utero environment, age, sex, or early life
exposure. The paired design is the strongest possible control against
genetic and developmental confounders — stronger than unrelated case/
control matching by age, gender, and cohort.

The Roos 2016 EWAS in this cohort reported pre-diagnostic methylation
signals at SASH1, COL11A2, AXL, LINC00340, and TIMM44 that persisted
5 years before clinical diagnosis. That independent prior finding of a
pre-diagnostic signal in this exact data is a referee-relevant anchor.

Analyses performed
------------------
1. Pre-diagnostic breast paired d:  primary test (n=12 pairs)
2. Pre-diagnostic breast unpaired d: n=12 cases vs n=12 healthy co-twins
3. Pre-diagnostic breast by TtD window:  0-2y, 2-5y, 5-10y
4. Pre-diagnostic colon paired d:  small-n exploratory (n=4 pairs)
5. Post-diagnostic breast paired d:  separate biological question (n=14)
6. All-cancer pre-diagnostic pooled (n=20):  maximum-power pan-cancer test

Canonical formula
-----------------
Per-sample A-score = mean over panel CpGs of [ H_binary(β) / H_min(immune) ].
Panel = Xu-538 immune-class. H_min(immune) = 0.838889 (G-003b MCMC).
NaN β values are dropped. Zero-entropy CpGs (β = 0 or β = 1) are KEPT in
the denominator and contribute 0 to the numerator — matching the canonical
VAL047 pipeline.

Citation
--------
Roos L, van Dongen J, Bell CG, Burri A, Deloukas P, Boomsma D, Bell JT,
Spector TD. "Integrative DNA methylome analysis of pan-cancer biomarkers
in cancer discordant monozygotic twin-pairs." Clinical Epigenetics 8:7,
2016. PMID 26798410. doi:10.1186/s13148-016-0172-y.

Invariants: H_MIN_IMMUNE = 0.838889; RANDOM_SEED = 20260420.
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

# TtD windows — match EPIC-Italy exactly so cross-cohort comparison is
# meaningful. The ">10yr" window is empty in TwinsUK (max TtD = 6.62 for
# breast, 12.85 for colon), but we compute it anyway for consistency.
TTD_WINDOWS = [
    ("0-2 yr",   0.0,  2.0),
    ("2-5 yr",   2.0,  5.0),
    ("5-10 yr",  5.0, 10.0),
    (">10 yr",  10.0, 999.0),
    ("all_pre_dx", 0.0, 999.0),
]

np.random.seed(RANDOM_SEED)

# ============================================================================
# HELPERS
# ============================================================================

def H_binary(beta):
    """Binary Shannon entropy. Returns 0 for beta in {0, 1, NaN}."""
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
    """Unpaired Cohen's d with pooled SD (ddof=1). Drops NaN."""
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

def cohens_d_paired(diffs):
    """Paired Cohen's d = mean(diffs) / sd(diffs).  diffs = case - control
    within matched pairs."""
    x = np.asarray(diffs, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd == 0: return 0.0
    return float(np.mean(x) / sd)

def permutation_p_unpaired(cases, controls, n_perm=10000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(cases,    dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    d_obs = cohens_d(a, b)
    if math.isnan(d_obs): return float("nan"), d_obs
    combined = np.concatenate([a, b])
    n_a = len(a)
    count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        ca, cb = combined[:n_a], combined[n_a:]
        if abs(cohens_d(ca, cb)) >= abs(d_obs):
            count_ge += 1
    return float((count_ge + 1) / (n_perm + 1)), float(d_obs)

def permutation_p_paired(diffs, n_perm=10000, seed=RANDOM_SEED):
    """Sign-flip permutation for paired d: randomly flip sign of each diff."""
    rng = np.random.default_rng(seed)
    x = np.asarray(diffs, dtype=float); x = x[~np.isnan(x)]
    d_obs = cohens_d_paired(x)
    if math.isnan(d_obs): return float("nan"), d_obs
    count_ge = 0
    n = len(x)
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        flipped = x * signs
        dd = cohens_d_paired(flipped)
        if abs(dd) >= abs(d_obs):
            count_ge += 1
    return float((count_ge + 1) / (n_perm + 1)), float(d_obs)

def bootstrap_d_ci_unpaired(cases, controls, n_boot=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 1)
    a = np.asarray(cases,    dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    ds = []
    for _ in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        dd = cohens_d(ra, rb)
        if not math.isnan(dd): ds.append(dd)
    if not ds: return float("nan"), float("nan")
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

def bootstrap_d_ci_paired(diffs, n_boot=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed + 2)
    x = np.asarray(diffs, dtype=float); x = x[~np.isnan(x)]
    ds = []
    for _ in range(n_boot):
        rx = rng.choice(x, size=len(x), replace=True)
        dd = cohens_d_paired(rx)
        if not math.isnan(dd): ds.append(dd)
    if not ds: return float("nan"), float("nan")
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

# ============================================================================
# PANEL + MATRIX  (unchanged from canonical)
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
    sys.exit(f"[FATAL] Cannot parse panel: {type(data)}")

def read_series_matrix(path, panel_cpgs):
    print(f"[parse] {path.name} ...", flush=True)
    meta_rows = {}; data_lines = []; in_data = False; header = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_data = True
                header = next(f).rstrip("\n").split("\t")
                continue
            if line.startswith("!series_matrix_table_end"):
                in_data = False; continue
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

    if header is None: sys.exit(f"[FATAL] No matrix data block")
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

    if not data_lines: sys.exit("[FATAL] No panel CpGs found")
    cpg_ids = [row[0].strip('"') for row in data_lines]
    betas = []
    for row in data_lines:
        row_vals = []
        for v in row[1:]:
            v = v.strip('"')
            if v in ("", "NA", "NaN", "null"): row_vals.append(np.nan)
            else:
                try: row_vals.append(float(v))
                except ValueError: row_vals.append(np.nan)
        betas.append(row_vals)
    beta_df = pd.DataFrame(betas, index=cpg_ids, columns=samples)
    print(f"  samples: {len(samples)}   panel CpGs in matrix: {len(cpg_ids)}", flush=True)
    return per_sample_meta, beta_df

# ============================================================================
# PER-SAMPLE A-SCORE — CANONICAL FORMULA
# ============================================================================

def compute_per_sample_A_canonical(beta_df):
    """
    Match canonical VAL047 formula exactly:
      panel = beta_df[gsm].dropna().values       # drop NaN
      A_vals = [A_score(b) for b in panel]       # per-CpG A, KEEPING beta=0/1
      A = np.mean(A_vals)                        # mean over all non-NaN betas
    """
    results = {}
    for gsm in beta_df.columns:
        panel = beta_df[gsm].dropna().values
        if len(panel) < 10:
            results[gsm] = {"A": float("nan"), "n_cpgs": int(len(panel))}
            continue
        A_vals = [A_score(b) for b in panel]
        A = float(np.mean(A_vals))
        results[gsm] = {"A": A, "n_cpgs": int(len(panel))}
    return results

# ============================================================================
# METADATA PARSE — GSE89093 specific
# ============================================================================

def parse_key_value(char_line, key):
    """Match a single colon-separated key:value characteristic line."""
    if not char_line: return None
    pat = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+)\s*$", re.IGNORECASE)
    m = pat.match(char_line)
    return m.group(1).strip() if m else None

def parse_sample_meta(characteristics):
    """Extract all relevant fields from a sample's characteristics list."""
    out = {
        "tissue": None, "twin_pair_id": None, "age_at_blood": float("nan"),
        "gender": None, "sentrix_id": None, "cancer_status": None,
        "before_after_dx": None, "ttd_years": float("nan"),
        "cancer_location": None, "icd_info": None, "histology": None,
    }
    for c in characteristics:
        if c is None or not c.strip(): continue
        for key, outkey, cast in [
            ("tissue",                                        "tissue",        str),
            ("twin pair id",                                  "twin_pair_id",  str),
            ("age at blood sample",                           "age_at_blood",  float),
            ("gender",                                        "gender",        str),
            ("beadchip and order on beadchip (sentrix id)",   "sentrix_id",    str),
            ("cancer status",                                 "cancer_status", str),
            ("age at blood sampling vs age at diagnosis",     "before_after_dx", str),
            ("age at diagnosis minus age at blood sampling",  "ttd_years",     float),
            ("cancer location",                               "cancer_location", str),
            ("icd information",                               "icd_info",      str),
            ("histology",                                     "histology",     str),
        ]:
            v = parse_key_value(c, key)
            if v is not None:
                try:    out[outkey] = cast(v) if cast is not str else v
                except: out[outkey] = float("nan") if cast is float else v
                break
    return out

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="T5 GSE89093 TwinsUK MZ twin discordant cancer")
    ap.add_argument("--matrix_path", required=True)
    ap.add_argument("--panel",       required=True)
    ap.add_argument("--output_dir",  required=True)
    args = ap.parse_args()

    matrix_path = Path(args.matrix_path)
    panel_path  = Path(args.panel)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("T5 GSE89093 — TwinsUK MZ cancer-discordant twin-pairs, Xu-538 panel")
    print("=" * 78)
    print(f"Random seed:   {RANDOM_SEED}")
    print(f"H_min(immune): {H_MIN_IMMUNE}")
    print(f"A-score formula:  CANONICAL (keeps zero-entropy CpGs in denominator)")
    print()

    matrix_sha = sha256_of_file(matrix_path)
    panel_sha  = sha256_of_file(panel_path)
    print(f"  matrix sha256: {matrix_sha}")
    print(f"  panel  sha256: {panel_sha}")
    print()

    panel_cpgs = load_panel(panel_path)
    print(f"Panel CpGs loaded: {len(panel_cpgs)}")

    meta, beta_df = read_series_matrix(matrix_path, panel_cpgs)

    # ---- Per-sample A (CANONICAL) ----------------------------------------
    print("Computing per-sample A-scores (CANONICAL formula)...", flush=True)
    A_by_gsm = compute_per_sample_A_canonical(beta_df)

    rows = []
    for gsm in beta_df.columns:
        m = meta.get(gsm, {})
        chars = m.get("characteristics", [])
        fields = parse_sample_meta(chars)
        fields["gsm"]          = gsm
        fields["title"]        = m.get("title")
        fields["source_name"]  = m.get("source_name")
        fields["n_cpgs_used"]  = A_by_gsm[gsm]["n_cpgs"]
        fields["A_score"]      = A_by_gsm[gsm]["A"]
        fields["raw_characteristics"] = "|".join([c for c in chars if c])
        rows.append(fields)
    df = pd.DataFrame(rows)

    # Propagate cancer_location to healthy co-twin
    pair_to_loc = {}
    for _, r in df[df["cancer_status"]=="cancer-diagnosis"].iterrows():
        if r["twin_pair_id"] and isinstance(r["cancer_location"], str) and r["cancer_location"]:
            pair_to_loc[r["twin_pair_id"]] = r["cancer_location"]
    def _loc_pair(r):
        loc = r["cancer_location"]
        if isinstance(loc, str) and loc:
            return loc
        return pair_to_loc.get(r["twin_pair_id"], "")
    df["cancer_location_pair"] = df.apply(_loc_pair, axis=1)

    csv_path = out_dir / "GSE89093_per_sample_A.csv"
    df.to_csv(csv_path, index=False)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV:  {csv_path}")
    print(f"  rows: {len(df)}   sha256: {csv_sha}")

    print()
    print("n_cpgs_used summary:")
    print(f"  min={df['n_cpgs_used'].min()}  max={df['n_cpgs_used'].max()}  "
          f"median={df['n_cpgs_used'].median()}  mean={df['n_cpgs_used'].mean():.1f}")

    # ---- Composition ------------------------------------------------------
    print()
    print("Cohort composition (Before/After × status × location):")
    comp = df.groupby(["before_after_dx","cancer_status","cancer_location_pair"]).size().reset_index(name="n")
    for _, r in comp.iterrows():
        print(f"  {r['before_after_dx']:<8s}  {r['cancer_status']:<16s}  "
              f"{r['cancer_location_pair']:<20s}  n={r['n']}")
    print()

    # ============================================================================
    # HELPERS FOR SUBGROUP ANALYSIS
    # ============================================================================
    def subset_breast(bf):   return df[(df["before_after_dx"]==bf) & (df["cancer_location_pair"].str.lower()=="breast")]
    def subset_colon(bf):    return df[(df["before_after_dx"]==bf) & (df["cancer_location_pair"].str.lower()=="colon")]
    def subset_all(bf):      return df[df["before_after_dx"]==bf]

    def run_paired_analysis(sub_df, label, n_perm=10000, n_boot=2000):
        """sub_df has twin pairs. Compute paired and unpaired d, plus per-window breakdown."""
        # Pair up cases vs healthy co-twins
        pairs_df = sub_df.pivot_table(
            index="twin_pair_id",
            columns="cancer_status",
            values="A_score",
            aggfunc="first",
        )
        # Keep only complete pairs
        pairs_df = pairs_df.dropna(subset=["cancer-diagnosis", "healthy"])
        if len(pairs_df) == 0:
            return {"label": label, "error": "no complete pairs"}
        diffs = (pairs_df["cancer-diagnosis"] - pairs_df["healthy"]).values
        cases    = pairs_df["cancer-diagnosis"].values
        controls = pairs_df["healthy"].values

        # Paired d
        d_paired = cohens_d_paired(diffs)
        p_paired, _ = permutation_p_paired(diffs, n_perm=n_perm, seed=RANDOM_SEED)
        ci_lo_p, ci_hi_p = bootstrap_d_ci_paired(diffs, n_boot=n_boot, seed=RANDOM_SEED)

        # Unpaired d (for cross-cohort comparison)
        d_unpaired = cohens_d(cases, controls)
        p_unpaired, _ = permutation_p_unpaired(cases, controls, n_perm=n_perm, seed=RANDOM_SEED)
        ci_lo_u, ci_hi_u = bootstrap_d_ci_unpaired(cases, controls, n_boot=n_boot, seed=RANDOM_SEED)

        # TtD windowing — get the shared TtD per pair
        ttd_by_pair = sub_df.groupby("twin_pair_id")["ttd_years"].first()
        per_window = {}
        for wname, w_lo, w_hi in TTD_WINDOWS:
            pair_ids = ttd_by_pair[(ttd_by_pair >= w_lo) & (ttd_by_pair < w_hi)].index
            pd_sub = pairs_df.loc[pairs_df.index.isin(pair_ids)]
            if len(pd_sub) == 0:
                per_window[wname] = {"n_pairs": 0}
                continue
            w_diffs = (pd_sub["cancer-diagnosis"] - pd_sub["healthy"]).values
            w_cases = pd_sub["cancer-diagnosis"].values
            w_ctrls = pd_sub["healthy"].values
            per_window[wname] = {
                "n_pairs":       int(len(pd_sub)),
                "case_mean":     float(np.mean(w_cases)),
                "ctrl_mean":     float(np.mean(w_ctrls)),
                "delta_mean":    float(np.mean(w_diffs)),
                "d_paired":      cohens_d_paired(w_diffs),
                "d_unpaired":    cohens_d(w_cases, w_ctrls),
            }

        return {
            "label":            label,
            "n_pairs":          int(len(pairs_df)),
            "case_A_mean":      float(np.mean(cases)),
            "case_A_sd":        float(np.std(cases, ddof=1)) if len(cases)>=2 else float("nan"),
            "control_A_mean":   float(np.mean(controls)),
            "control_A_sd":     float(np.std(controls, ddof=1)) if len(controls)>=2 else float("nan"),
            "mean_within_pair_diff": float(np.mean(diffs)),
            "sd_within_pair_diff":   float(np.std(diffs, ddof=1)) if len(diffs)>=2 else float("nan"),
            "Cohens_d_paired":   d_paired,
            "p_perm_paired":     p_paired,
            "CI95_paired":       [ci_lo_p, ci_hi_p],
            "Cohens_d_unpaired": d_unpaired,
            "p_perm_unpaired":   p_unpaired,
            "CI95_unpaired":     [ci_lo_u, ci_hi_u],
            "by_TtD_window":     per_window,
        }

    # ============================================================================
    # ANALYSES
    # ============================================================================
    print("=" * 78)
    print("ANALYSIS 1 — PRE-DIAGNOSTIC BREAST (primary test; n=12 pairs expected)")
    print("=" * 78)
    r1 = run_paired_analysis(subset_breast("Before"), "pre_dx_breast")
    def _report(r):
        if "error" in r:
            print(f"  [{r['label']}] {r['error']}"); return
        print(f"  n_pairs={r['n_pairs']}")
        print(f"  case_mean_A   = {r['case_A_mean']:.4f} (sd {r['case_A_sd']:.4f})")
        print(f"  ctrl_mean_A   = {r['control_A_mean']:.4f} (sd {r['control_A_sd']:.4f})")
        print(f"  mean within-pair diff = {r['mean_within_pair_diff']:+.4f} "
              f"(sd {r['sd_within_pair_diff']:.4f})")
        print(f"  PAIRED:   d={r['Cohens_d_paired']:+.3f}  "
              f"p={r['p_perm_paired']:.4f}  95%CI=[{r['CI95_paired'][0]:+.3f}, {r['CI95_paired'][1]:+.3f}]")
        print(f"  UNPAIRED: d={r['Cohens_d_unpaired']:+.3f}  "
              f"p={r['p_perm_unpaired']:.4f}  95%CI=[{r['CI95_unpaired'][0]:+.3f}, {r['CI95_unpaired'][1]:+.3f}]")
        print(f"  By TtD window:")
        for wname, w in r["by_TtD_window"].items():
            if w["n_pairs"] == 0:
                print(f"    {wname:<12s}: n=0")
            else:
                print(f"    {wname:<12s}: n={w['n_pairs']:<3d}  "
                      f"Δ={w['delta_mean']:+.4f}  "
                      f"d_paired={w['d_paired']:+.3f}  d_unpaired={w['d_unpaired']:+.3f}")
    _report(r1)

    print()
    print("=" * 78)
    print("ANALYSIS 2 — PRE-DIAGNOSTIC COLON (exploratory; n=4 pairs expected)")
    print("=" * 78)
    r2 = run_paired_analysis(subset_colon("Before"), "pre_dx_colon")
    _report(r2)

    print()
    print("=" * 78)
    print("ANALYSIS 3 — POST-DIAGNOSTIC BREAST (separate biological question; n=14 pairs)")
    print("=" * 78)
    r3 = run_paired_analysis(subset_breast("After"), "post_dx_breast")
    _report(r3)

    print()
    print("=" * 78)
    print("ANALYSIS 4 — PRE-DIAGNOSTIC PAN-CANCER (all sites, max power; n=20 pairs)")
    print("=" * 78)
    r4 = run_paired_analysis(subset_all("Before"), "pre_dx_all_cancers")
    _report(r4)

    # ============================================================================
    # JSON OUTPUT
    # ============================================================================
    out_json = {
        "cohort":        "GSE89093_Roos2016_TwinsUK_MZ_discordant_cancer",
        "purpose":       ("pre-diagnostic and post-diagnostic case-vs-co-twin paired "
                          "comparison in genetically identical MZ twins"),
        "cohort_design": ("46 adult female MZ twin pairs, each discordant for cancer. "
                          "Peripheral blood, GPL13534 450K. Both pre- and post-dx "
                          "samples present."),
        "n_samples_total":      int(len(df)),
        "n_pairs":              46,
        "panel":                "xu538_breast",
        "panel_n_cpgs":         len(panel_cpgs),
        "panel_cpgs_in_matrix": int(beta_df.shape[0]),
        "H_min_immune":         H_MIN_IMMUNE,
        "random_seed":          RANDOM_SEED,
        "A_score_formula":      "CANONICAL (matches VAL047_tightening_fresh.py)",
        "input_sha256":         {"matrix_file": matrix_sha, "panel_file": panel_sha},
        "output_sha256":        {"per_sample_csv": csv_sha},
        "cohort_composition":   [
            {"before_after_dx": r["before_after_dx"], "cancer_status": r["cancer_status"],
             "cancer_location": r["cancer_location_pair"], "n": int(r["n"])}
            for _, r in comp.iterrows()
        ],
        "analysis_1_pre_dx_breast":     r1,
        "analysis_2_pre_dx_colon":      r2,
        "analysis_3_post_dx_breast":    r3,
        "analysis_4_pre_dx_all_cancers":r4,
    }
    json_path = out_dir / "GSE89093_paired_analysis.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2, default=str)
    json_sha = sha256_of_file(json_path)
    print()
    print(f"Output JSON:  {json_path}")
    print(f"  sha256: {json_sha}")
    print()
    print("T5 complete.")

if __name__ == "__main__":
    main()
