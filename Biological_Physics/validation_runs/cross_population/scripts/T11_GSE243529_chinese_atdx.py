#!/usr/bin/env python3
"""
T11 — GSE243529 (Lee 2024) Singapore Chinese at-diagnosis BC case/control
==========================================================================

Cohort
------
524 ethnic Chinese individuals from Singapore:
  256 breast cancer cases (at diagnosis, recruited from genetic testing clinics)
  268 age- and ethnicity-matched non-cancer controls
Illumina Infinium MethylationEPIC BeadChip (GPL21145, EPIC v1.0).
National Cancer Centre Singapore (Ann S G Lee laboratory).
Published 2024-05-15 (Clin Epigenetics 16:66, PMID 38750495).

Design
------
This is an AT-DIAGNOSIS case-control study (not pre-diagnostic). Cases are
breast cancer patients at diagnosis, not pre-symptomatic individuals later
diagnosed with BC. Closer analog: T3 Uruguay (GSE148663).

Metadata available
------------------
Only "condition: affected" / "condition: unaffected" per sample. No age,
sex, BMI, subtype, histology, stage, or TtD. Age-matching is stated in the
paper but not encoded in GEO.

Key quirk — column-order scrambling
------------------------------------
The supplementary data_table (`GSE243529_matrix-processed.tsv.gz`) uses
internal lab IDs (HCR###, YP###, FH##, MR0###, LR0###, A####, BL###,
BS###, SG####, BM##, DI##, X###, OU###) as column headers. These lab IDs
are NOT in the natural case/control block order. The mapping from lab ID
to case/control status comes from the series_matrix.txt.gz file, which
provides two `!Sample_description` lines per sample:
  description[0] = "cancer patient peripheral blood" OR "healthy control peripheral blood"
  description[1] = <lab ID>
The second description line is the lab ID, in the SAME ORDER as
`!Sample_title` and `!Sample_geo_accession`.

Canonical formula
-----------------
Per-sample A-score = mean over Xu-538 CpGs of [H_binary(β) / H_min(immune)]
H_min(immune) = 0.838889; canonical (β=0/1 retained in denominator)

Citation
--------
Lee NY, Hum M, Tan GP, Seah AC, Ong P-Y, Kin PT, Lim CW, Samol J, Tan NC,
Law H-Y, Tan M-H, Lee S-C, Ang P, Lee ASG. "Machine learning unveils an
immune-related DNA methylation profile in germline DNA from breast cancer
patients." Clin Epigenetics 16:66 (2024). doi:10.1186/s13148-024-01674-2
"""

import argparse, gzip, hashlib, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

H_MIN_IMMUNE = 0.838889
RANDOM_SEED  = 20260420
np.random.seed(RANDOM_SEED)

# ============================================================================
# CORE
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

def permutation_p(cases, controls, n_perm=5000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(cases, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(controls, dtype=float); b = b[~np.isnan(b)]
    d_obs = cohens_d(a, b)
    if math.isnan(d_obs): return float("nan"), d_obs
    combined = np.concatenate([a, b]); n_a = len(a); count_ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        if abs(cohens_d(combined[:n_a], combined[n_a:])) >= abs(d_obs): count_ge += 1
    return float((count_ge+1)/(n_perm+1)), float(d_obs)

def bootstrap_ci(cases, controls, n_boot=1000, seed=RANDOM_SEED):
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
                cpgs.update(x for x in v if isinstance(x, str) and x.startswith("cg"))
        if cpgs: return cpgs
    sys.exit(f"[FATAL] Cannot parse panel")

# ============================================================================
# Parse series matrix: build lab_id -> (gsm, title, status)
# ============================================================================
def parse_metadata(series_matrix_path):
    opener = gzip.open if str(series_matrix_path).endswith(".gz") else open
    sample_lines = {}  # key → list[list[val]]
    with opener(series_matrix_path, "rt") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"): break
            if not line.startswith("!Sample_"): continue
            parts = line.rstrip("\n").split("\t")
            key = parts[0]
            vals = [v.strip('"') for v in parts[1:]]
            sample_lines.setdefault(key, []).append(vals)
    titles       = sample_lines["!Sample_title"][0]
    gsms         = sample_lines["!Sample_geo_accession"][0]
    descriptions = sample_lines.get("!Sample_description", [])
    chars_lines  = sample_lines.get("!Sample_characteristics_ch1", [])

    # Sample_description[0] = "cancer patient peripheral blood" / "healthy control peripheral blood"
    # Sample_description[1] = lab ID (HCR007 etc.)
    status_desc = descriptions[0] if len(descriptions) > 0 else [None]*len(titles)
    lab_ids     = descriptions[1] if len(descriptions) > 1 else [None]*len(titles)

    records = []
    for i in range(len(titles)):
        rec = {
            "gsm":     gsms[i],
            "title":   titles[i],
            "lab_id":  lab_ids[i],
            "description_status": status_desc[i],
        }
        # Parse characteristics for "condition"
        for char_line in chars_lines:
            if i < len(char_line):
                c = char_line[i]
                if ":" in c:
                    k, v = c.split(":", 1); k = k.strip(); v = v.strip()
                    if k == "condition":
                        rec["condition"] = v
        # Determine status
        cond = (rec.get("condition") or "").lower()
        desc = (rec.get("description_status") or "").lower()
        if cond == "affected" or "cancer" in desc:
            rec["status"] = "case"
        elif cond == "unaffected" or "healthy" in desc:
            rec["status"] = "control"
        else:
            rec["status"] = "unknown"
        records.append(rec)
    return pd.DataFrame(records)

# ============================================================================
# Stream-compute per-sample A-score from data_table
# ============================================================================
def compute_per_sample_A_streaming(data_table_path, panel_cpgs, has_detection_pvals=True):
    opener = gzip.open if str(data_table_path).endswith(".gz") else open
    with opener(data_table_path, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        # Identify which columns are beta vs detection p-values
        # Beta columns are those without "Detection Pval" in the name
        beta_col_indices = []    # list of column positions in each row
        sample_names     = []
        for i, h in enumerate(header):
            if i == 0: continue  # ID_REF
            if "Detection Pval" in h or h.strip() == "":
                continue
            beta_col_indices.append(i)
            sample_names.append(h.strip())
        n_samples = len(beta_col_indices)
        print(f"  Beta columns identified: {n_samples}")

        sum_A  = np.zeros(n_samples, dtype=np.float64)
        count  = np.zeros(n_samples, dtype=np.int64)
        cpgs_matched = set()
        n_lines = 0
        for line in f:
            n_lines += 1
            first_tab = line.find("\t")
            if first_tab < 0: continue
            cpg_id = line[:first_tab].strip('"')
            if cpg_id not in panel_cpgs: continue
            cpgs_matched.add(cpg_id)
            fields = line.rstrip("\n").split("\t")
            for i, col_idx in enumerate(beta_col_indices):
                if col_idx >= len(fields): continue
                v = fields[col_idx]
                if v in ("", "NA", "NaN", "null"): continue
                try: b = float(v)
                except: continue
                if math.isnan(b): continue
                sum_A[i] += H_binary(b) / H_MIN_IMMUNE
                count[i] += 1
    A = np.where(count > 0, sum_A / count, np.nan)
    df = pd.DataFrame({"lab_id": sample_names, "A_score": A, "n_cpgs_used": count})
    print(f"  lines read: {n_lines}  panel CpGs matched: {len(cpgs_matched)}/{len(panel_cpgs)}")
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
    print("T11 GSE243529 — Lee 2024 Singapore Chinese BC case/control, Xu-538")
    print("=" * 78)
    sm_sha  = sha256_of_file(args.series_matrix)
    dt_sha  = sha256_of_file(args.data_table)
    pan_sha = sha256_of_file(args.panel)
    print(f"  series_matrix sha256: {sm_sha}")
    print(f"  data_table    sha256: {dt_sha}")
    print(f"  panel         sha256: {pan_sha}")
    print(f"  H_min(immune):        {H_MIN_IMMUNE}")
    print(f"  Random seed:          {RANDOM_SEED}")
    print()

    panel = load_panel(args.panel)
    print(f"Panel CpGs: {len(panel)}")

    print("Parsing series matrix metadata...")
    meta = parse_metadata(args.series_matrix)
    print(f"  samples: {len(meta)}")
    print(f"  status: {meta['status'].value_counts().to_dict()}")
    # Verify lab_ids are non-null
    n_with_lab = meta['lab_id'].notna().sum()
    print(f"  samples with lab_id: {n_with_lab}/{len(meta)}")
    print()

    print("Computing per-sample A-score (streaming, canonical)...")
    A_df, matched_cpgs = compute_per_sample_A_streaming(args.data_table, panel)
    print(f"  A summary: mean={A_df['A_score'].mean():.4f} sd={A_df['A_score'].std():.4f}")
    print(f"  n_cpgs_used: min={A_df['n_cpgs_used'].min()} max={A_df['n_cpgs_used'].max()} "
          f"median={A_df['n_cpgs_used'].median()}")
    print()

    # Merge on lab_id
    df = meta.merge(A_df, on="lab_id", how="left")
    unmatched = df[df['A_score'].isna()]
    print(f"Samples in meta with no matching lab_id in data_table: {len(unmatched)}")
    if len(unmatched) > 0 and len(unmatched) <= 10:
        print(f"  unmatched lab_ids: {unmatched['lab_id'].tolist()}")
    print()

    csv_path = out_dir / "GSE243529_per_sample_A.csv"
    df.to_csv(csv_path, index=False)
    csv_sha = sha256_of_file(csv_path)
    print(f"Per-sample CSV: {csv_path}")
    print(f"  rows: {len(df)}  sha256: {csv_sha}")
    print()

    # ============================================================================
    # ANALYSIS
    # ============================================================================
    cases    = df[(df.status=="case") & df.A_score.notna()]["A_score"].to_numpy()
    controls = df[(df.status=="control") & df.A_score.notna()]["A_score"].to_numpy()

    print("=" * 78)
    print("CASE vs CONTROL ANALYSIS (at diagnosis)")
    print("=" * 78)
    print(f"  n_cases:    {len(cases)}")
    print(f"  n_controls: {len(controls)}")
    print()

    d_obs = cohens_d(cases, controls)
    p_perm, _ = permutation_p(cases, controls, n_perm=5000, seed=RANDOM_SEED)
    ci_lo, ci_hi = bootstrap_ci(cases, controls, n_boot=1000, seed=RANDOM_SEED)
    delta = float(np.mean(cases) - np.mean(controls))

    print(f"  case mean A  = {np.mean(cases):.4f}  sd {np.std(cases, ddof=1):.4f}")
    print(f"  ctrl mean A  = {np.mean(controls):.4f}  sd {np.std(controls, ddof=1):.4f}")
    print(f"  Δ (case-ctrl): {delta:+.4f}")
    print(f"  Cohen's d:     {d_obs:+.3f}")
    print(f"  p (5k perm):   {p_perm:.4f}")
    print(f"  95% CI (1k BS):[{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print()

    out = {
        "cohort":          "GSE243529_Lee2024_Singapore_Chinese_at_dx_BC",
        "design":          ("256 ethnic Chinese BC patients at diagnosis + 268 age-matched "
                            "healthy controls. Singapore, MethylationEPIC v1.0."),
        "caveats":         ("At-diagnosis (NOT pre-diagnostic). No per-sample age/TtD. "
                            "Lab IDs used as column headers in data_table, mapping comes "
                            "from !Sample_description[1] field of series matrix."),
        "n_samples_total":    int(len(df)),
        "n_cases":            int(len(cases)),
        "n_controls":         int(len(controls)),
        "n_unmatched":        int(len(unmatched)),
        "panel":              "xu538_breast",
        "panel_n_cpgs":       len(panel),
        "panel_matched":      int(len(matched_cpgs)),
        "H_min_immune":       H_MIN_IMMUNE,
        "random_seed":        RANDOM_SEED,
        "A_score_formula":    "CANONICAL",
        "input_sha256":       {"series_matrix": sm_sha, "data_table": dt_sha, "panel": pan_sha},
        "output_sha256":      {"per_sample_csv": csv_sha},
        "primary_result": {
            "n_cases":        int(len(cases)),
            "n_controls":     int(len(controls)),
            "case_A_mean":    float(np.mean(cases)),
            "case_A_sd":      float(np.std(cases, ddof=1)),
            "control_A_mean": float(np.mean(controls)),
            "control_A_sd":   float(np.std(controls, ddof=1)),
            "delta_A":        float(delta),
            "cohens_d":       float(d_obs),
            "p_perm":         float(p_perm),
            "CI95_bootstrap": [float(ci_lo), float(ci_hi)],
        },
    }
    json_path = out_dir / "GSE243529_analysis.json"
    with open(json_path, "w") as f: json.dump(out, f, indent=2, default=str)
    print(f"Output JSON: {json_path}")
    print(f"  sha256: {sha256_of_file(json_path)}")
    print()
    print("T11 complete.")

if __name__ == "__main__":
    main()
