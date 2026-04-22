#!/usr/bin/env python3
"""
T10 — GSE37965 (Heyn 2013) UK EpiTwin discordant MZ twin pairs
================================================================

Cohort
------
15 monozygotic twin pairs discordant for breast cancer, all female, whole
blood, Illumina HumanMethylation450 BeadChip (GPL13534).  UK-based EpiTwin
study (King's College London / Bellvitge Biomedical Research Institute).
Published 2013 (Carcinogenesis 34:102-108, PMID 23054610, Esteller lab).

Design:
  30 samples = 15 MZ twin pairs
  Within each pair: 1 affected (breast cancer), 1 healthy co-twin
  All same-sex, same-platform, same-time-point collection

Metadata available per sample:
  gender (all Female), tissue (Whole blood), twin pair id, disease status
Metadata NOT available:
  Time to diagnosis (some pre-dx, some post-dx, but individual flags unclear)
  Age at blood draw
  Histology / ICD codes
  Cancer location (all breast per study design, but not per-sample-encoded)

Independence from prior cohorts
--------------------------------
GSE37965 samples are DIFFERENT individuals from GSE89093 (Roos 2016 TwinsUK).
Both are UK-registered female MZ twin pairs discordant for cancer, but
Heyn 2013 was the earlier Esteller-lab discovery cohort (n=15 pairs).
Roos 2016 was a larger follow-up (n=46 pairs) with cancer-location-specific
grouping.

Preprocessing (from Heyn 2013):
  Illumina 450K; background correction (BackgroundCorrector); quantile
  normalization of paired probe types; replaced outlier detection values
  with NA before beta computation.

Canonical formula
-----------------
Per-sample A-score = mean over Xu-538 CpGs of [H_binary(β) / H_min(immune)]
H_min(immune) = 0.838889; beta=0/1 CpGs retained in denominator
"""

import argparse, gzip, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

H_MIN_IMMUNE = 0.838889
RANDOM_SEED  = 20260420
np.random.seed(RANDOM_SEED)

# ============================================================================
# CORE (identical to T5/T9)
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
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float("nan")
    s1 = float(np.std(a, ddof=1)); s2 = float(np.std(b, ddof=1))
    pooled = math.sqrt(((len(a)-1)*s1*s1 + (len(b)-1)*s2*s2) / (len(a)+len(b)-2))
    return 0.0 if pooled == 0 else float((np.mean(a) - np.mean(b)) / pooled)

def cohens_d_paired(diffs):
    x = np.asarray(diffs, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2: return float("nan")
    sd = float(np.std(x, ddof=1))
    return 0.0 if sd == 0 else float(np.mean(x) / sd)

def permutation_p_paired(diffs, n_perm=10000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(diffs, dtype=float); x = x[~np.isnan(x)]
    d_obs = cohens_d_paired(x)
    if math.isnan(d_obs): return float("nan"), d_obs
    count_ge = 0
    for _ in range(n_perm):
        signs = rng.choice([-1,1], size=len(x))
        if abs(cohens_d_paired(x * signs)) >= abs(d_obs): count_ge += 1
    return float((count_ge+1)/(n_perm+1)), float(d_obs)

def permutation_p_unpaired(a, b, n_perm=10000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    d_obs = cohens_d(a, b)
    if math.isnan(d_obs): return float("nan"), d_obs
    combined = np.concatenate([a, b]); n_a = len(a); count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        if abs(cohens_d(combined[:n_a], combined[n_a:])) >= abs(d_obs): count_ge += 1
    return float((count_ge+1)/(n_perm+1)), float(d_obs)

def bootstrap_ci_paired(diffs, n_boot=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed+2)
    x = np.asarray(diffs, dtype=float); x = x[~np.isnan(x)]
    ds=[]
    for _ in range(n_boot):
        dd = cohens_d_paired(rng.choice(x, size=len(x), replace=True))
        if not math.isnan(dd): ds.append(dd)
    if not ds: return float("nan"), float("nan")
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

def bootstrap_ci_unpaired(a, b, n_boot=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed+3)
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    ds=[]
    for _ in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        dd = cohens_d(ra, rb)
        if not math.isnan(dd): ds.append(dd)
    if not ds: return float("nan"), float("nan")
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

# ============================================================================
# Panel loader
# ============================================================================
def load_panel(panel_path):
    with open(panel_path) as f: data = json.load(f)
    if isinstance(data, list): return set(data)
    if isinstance(data, dict):
        for key in ("cpgs","panel","cpg_ids","probes"):
            if key in data and isinstance(data[key], list): return set(data[key])
        cpgs = set()
        for v in data.values():
            if isinstance(v, list):
                cpgs.update(x for x in v if isinstance(x,str) and x.startswith("cg"))
        if cpgs: return cpgs
    sys.exit(f"[FATAL] Cannot parse panel")

# ============================================================================
# Parse combined series matrix with data table inline
# ============================================================================
def read_series_matrix(path, panel_cpgs):
    """For GSE37965 the series matrix HAS the data table inline (60MB file)."""
    meta_lines = {}
    data_lines = []; in_data = False; header = None
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
                meta_lines.setdefault(key, []).append(vals)
    if header is None: sys.exit("[FATAL] No matrix block")
    samples = [h.strip('"') for h in header[1:]]
    return samples, meta_lines, data_lines

# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series_matrix", required=True)
    ap.add_argument("--panel",         required=True)
    ap.add_argument("--output_dir",    required=True)
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("T10 GSE37965 — Heyn 2013 UK MZ twin discordant BC, Xu-538 panel")
    print("=" * 78)
    sm_sha  = sha256_of_file(args.series_matrix)
    pan_sha = sha256_of_file(args.panel)
    print(f"  series_matrix sha256: {sm_sha}")
    print(f"  panel         sha256: {pan_sha}")
    print(f"  H_min(immune):        {H_MIN_IMMUNE}")
    print(f"  Random seed:          {RANDOM_SEED}")
    print(f"  Formula:              CANONICAL")
    print()

    panel = load_panel(args.panel)
    print(f"Panel CpGs: {len(panel)}")

    samples, meta_lines, data_lines = read_series_matrix(args.series_matrix, panel)
    print(f"Samples: {len(samples)}")
    print(f"Panel CpGs matched in matrix: {len(data_lines)}")
    print()

    # Compute per-sample A canonically
    A_per_sample = {}
    count_per_sample = {}
    for gsm in samples:
        A_per_sample[gsm] = 0.0
        count_per_sample[gsm] = 0

    for row in data_lines:
        cpg = row[0].strip('"')
        for i, gsm in enumerate(samples):
            v = row[i+1].strip('"')
            if v in ("","NA","NaN","null"): continue
            try: b = float(v)
            except: continue
            if math.isnan(b): continue
            A_per_sample[gsm] += H_binary(b) / H_MIN_IMMUNE
            count_per_sample[gsm] += 1
    A = {gsm: (A_per_sample[gsm]/count_per_sample[gsm]) if count_per_sample[gsm]>0 else float("nan")
         for gsm in samples}

    # Parse metadata by zipping lines
    titles   = meta_lines["!Sample_title"][0]
    gsms     = meta_lines["!Sample_geo_accession"][0]
    char_lines = meta_lines.get("!Sample_characteristics_ch1", [])

    rows = []
    for i, gsm in enumerate(gsms):
        rec = {"gsm": gsm, "title": titles[i] if i < len(titles) else ""}
        for cl in char_lines:
            if i < len(cl):
                c = cl[i]
                if ":" in c:
                    k, v = c.split(":", 1); k = k.strip(); v = v.strip()
                    if k == "gender": rec["gender"] = v
                    elif k == "tissue": rec["tissue"] = v
                    elif k == "twin pair": rec["twin_pair"] = v
                    elif k == "disease status": rec["disease_status"] = v
        # Classify
        ds = (rec.get("disease_status") or "").lower()
        if "breast cancer" in ds or "cancer sample" in ds:
            rec["status"] = "case"
        elif "healthy" in ds:
            rec["status"] = "control"
        else:
            rec["status"] = "unknown"
        rec["A_score"] = A[gsm]
        rec["n_cpgs_used"] = count_per_sample[gsm]
        rows.append(rec)
    df = pd.DataFrame(rows)

    csv_path = out_dir / "GSE37965_per_sample_A.csv"
    df.to_csv(csv_path, index=False)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV: {csv_path}")
    print(f"  rows: {len(df)}  sha256: {csv_sha}")
    print()
    print(f"Status distribution: {df['status'].value_counts().to_dict()}")
    print(f"n_cpgs_used: min={df['n_cpgs_used'].min()}  max={df['n_cpgs_used'].max()}  "
          f"median={df['n_cpgs_used'].median()}")
    print(f"Per-sample A: mean={df['A_score'].mean():.4f}  sd={df['A_score'].std():.4f}")
    print()

    # ======================================================================
    # PAIRED ANALYSIS
    # ======================================================================
    pairs = df.pivot_table(index="twin_pair", columns="status", values="A_score", aggfunc="first")
    complete = pairs.dropna(subset=["case","control"])
    diffs = (complete["case"] - complete["control"]).values
    cases_u = complete["case"].values
    ctrls_u = complete["control"].values

    d_paired = cohens_d_paired(diffs)
    p_paired, _ = permutation_p_paired(diffs, n_perm=10000, seed=RANDOM_SEED)
    cip_lo, cip_hi = bootstrap_ci_paired(diffs, n_boot=2000, seed=RANDOM_SEED)

    d_unpaired = cohens_d(cases_u, ctrls_u)
    p_unpaired, _ = permutation_p_unpaired(cases_u, ctrls_u, n_perm=10000, seed=RANDOM_SEED)
    ciu_lo, ciu_hi = bootstrap_ci_unpaired(cases_u, ctrls_u, n_boot=2000, seed=RANDOM_SEED)

    print("=" * 78)
    print("PAIRED ANALYSIS — 15 MZ twin pairs discordant for BC")
    print("=" * 78)
    print(f"  n_pairs:         {len(complete)}")
    print(f"  case  mean A:    {np.mean(cases_u):.4f}  sd {np.std(cases_u,ddof=1):.4f}")
    print(f"  ctrl  mean A:    {np.mean(ctrls_u):.4f}  sd {np.std(ctrls_u,ddof=1):.4f}")
    print(f"  mean within-pair diff: {np.mean(diffs):+.4f}  sd {np.std(diffs,ddof=1):.4f}")
    print()
    print(f"  PAIRED:   d={d_paired:+.3f}  p={p_paired:.4f}  "
          f"95%CI=[{cip_lo:+.3f},{cip_hi:+.3f}]")
    print(f"  UNPAIRED: d={d_unpaired:+.3f}  p={p_unpaired:.4f}  "
          f"95%CI=[{ciu_lo:+.3f},{ciu_hi:+.3f}]")
    print()

    out = {
        "cohort":          "GSE37965_Heyn2013_UK_EpiTwin_MZ_discordant_BC",
        "design":          ("15 MZ twin pairs discordant for breast cancer, all female, "
                            "UK-based EpiTwin cohort. Whole blood, Illumina 450K."),
        "caveats":         ("TtD is not encoded per sample; Heyn 2013 text reports 7 pre-dx "
                            "and 8 post-dx samples but individual flags not in GEO metadata. "
                            "Analysis aggregates all 15 pairs without TtD stratification."),
        "independence":    "Independent of GSE89093 (Roos 2016 TwinsUK, which is a later cohort)",
        "n_samples":       int(len(df)),
        "n_pairs":         int(len(complete)),
        "panel":           "xu538_breast",
        "panel_cpgs":      len(panel),
        "panel_matched":   int(len(data_lines)),
        "H_min_immune":    H_MIN_IMMUNE,
        "random_seed":     RANDOM_SEED,
        "A_formula":       "CANONICAL",
        "input_sha256":    {"series_matrix": sm_sha, "panel": pan_sha},
        "output_sha256":   {"per_sample_csv": csv_sha},
        "paired_result": {
            "n_pairs":          int(len(complete)),
            "case_mean":        float(np.mean(cases_u)),
            "ctrl_mean":        float(np.mean(ctrls_u)),
            "mean_diff":        float(np.mean(diffs)),
            "Cohens_d_paired":  float(d_paired),
            "p_paired":         float(p_paired),
            "CI95_paired":      [float(cip_lo), float(cip_hi)],
            "Cohens_d_unpaired":float(d_unpaired),
            "p_unpaired":       float(p_unpaired),
            "CI95_unpaired":    [float(ciu_lo), float(ciu_hi)],
        },
    }
    json_path = out_dir / "GSE37965_analysis.json"
    with open(json_path,"w") as f: json.dump(out, f, indent=2, default=str)
    print(f"Output JSON: {json_path}")
    print(f"  sha256: {sha256_of_file(json_path)}")
    print()
    print("T10 complete.")

if __name__ == "__main__":
    main()
