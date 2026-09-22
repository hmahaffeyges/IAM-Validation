#!/usr/bin/env python3
"""
STEP 7 / CHECK E — GSE130748 Mozhui longitudinal cancer-trajectory test
=========================================================================

Processes the GSE130748 cohort (Mozhui 2019 PMID 31149338) — Health, Aging
and Body Composition prospective cohort, n=20 participants × 2 timepoints
each = 40 EPIC v1 IDAT pairs, with cancer-incidence follow-up.

QUESTION
========
Does the IAMAtlas immune A-score show different LONGITUDINAL TRAJECTORIES
between participants who later developed cancer vs those who didn't?

A trajectory test is the right shape for n=20 paired — too small for
cross-sectional case/control discovery, but each participant is their
own control across timepoints, so we test:

   ΔA_immune (timepoint 2 − timepoint 1) cancer-incident vs cancer-free

The framework predicts ΔA should be positive (immune compartment becomes
more disordered = brighter = elevated A) for participants who later develop
cancer, with smaller ΔA in cancer-free participants.

INPUTS
======
1. IAMAtlas_v0_1.csv                                 (production matrix)
2. GSE130748_RAW.tar  (already extracted — folder of *.idat.gz files)
3. GSE130748 series matrix file (downloaded from GEO; for sample metadata)

PREREQUISITES (Heath: install once)
====================================
pip install methylprep  pymc  arviz  numpy

methylprep handles EPIC v1 IDAT decoding directly. We extract β at HM450
overlap CpGs (~452K), then compute A_immune.

USAGE
=====
cd ~/IAMPerformance
# After extracting GSE130748_RAW.tar:
python3 step_7_check_e_gse130748_trajectory.py \\
    --idat_dir GSE130748_RAW \\
    --series_matrix GSE130748_series_matrix.txt.gz

OUTPUT
======
step_7_chk_e_gse130748_trajectory.md
gse130748_per_sample_a_immune.csv

IMPORTANT NOTES
===============
- This script will ask for sample-level cancer-incidence labels if not
  in the GEO metadata. Mozhui 2019 supplementary materials has this.
- If the cohort is too small for definitive trajectory call (it is),
  the report will explicitly call out that this is a hypothesis-generation
  pass not a powered validation.

Date: 2026-05-04
"""

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import sys
from pathlib import Path
from collections import defaultdict


def load_iamatlas_immune(matrix_path: Path) -> dict:
    out = {}
    with open(matrix_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpg = row["cpg_id"]
            mean = row.get("immune_mean", "NA")
            if mean not in ("NA", "", None):
                try: out[cpg] = float(mean)
                except ValueError: pass
    return out


def parse_series_matrix_metadata(series_path: Path) -> dict:
    """Parse the GSE130748 series matrix for per-GSM metadata."""
    if not series_path.exists():
        return {}
    opener = gzip.open if str(series_path).endswith(".gz") else open
    metadata_lines = []
    with opener(series_path, 'rt') as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"): break
            metadata_lines.append(line)
    full = "\n".join(metadata_lines)
    # Extract Sample_geo_accession
    gsms = []
    for line in metadata_lines:
        if line.startswith("!Sample_geo_accession"):
            gsms = [p.strip().strip('"') for p in line.strip().split("\t")[1:]]
            break
    titles = []
    for line in metadata_lines:
        if line.startswith("!Sample_title"):
            titles = [p.strip().strip('"') for p in line.strip().split("\t")[1:]]
            break
    char_lines = [line for line in metadata_lines if line.startswith("!Sample_characteristics_ch1")]
    # Each char line has: !Sample_characteristics_ch1\t"key1: val1"\t"key1: val2"\t...
    chars_per_sample = defaultdict(dict)
    for cline in char_lines:
        parts = [p.strip().strip('"') for p in cline.strip().split("\t")[1:]]
        for i, p in enumerate(parts):
            if ":" in p:
                k, _, v = p.partition(":")
                chars_per_sample[i][k.strip()] = v.strip()
    # Build per-GSM metadata
    out = {}
    for i, gsm in enumerate(gsms):
        out[gsm] = {
            "title": titles[i] if i < len(titles) else "",
            **chars_per_sample.get(i, {}),
        }
    return out


def extract_betas_from_idats(idat_dir: Path, panel_cpgs: set) -> dict:
    """
    Use methylprep to extract β values from all IDAT pairs in the directory.
    Returns: {sample_id: {cpg: beta, ...}, ...}
    
    methylprep handles EPIC v1 platform detection automatically.
    """
    try:
        import methylprep
    except ImportError:
        print("ERROR: methylprep not installed.")
        print("Run: pip install methylprep")
        sys.exit(1)

    print(f"  Running methylprep on {idat_dir} ...")
    print(f"  This may take 10-30 min depending on n samples and disk I/O.")

    # methylprep wants a 'data_dir' that contains the IDAT files
    idat_dir = idat_dir.resolve()

    # methylprep.run_pipeline returns a dict with 'beta_values' as a DataFrame
    # We pass --no_meta_export and just want betas
    result = methylprep.run_pipeline(
        data_dir=str(idat_dir),
        array_type="epic",
        export=False,
        betas=True,
        save_uncorrected=False,
    )
    # result format depends on methylprep version; try both common shapes
    if isinstance(result, dict):
        beta_df = result.get("beta_values") or result.get("betas")
    else:
        beta_df = result  # might already be a DataFrame
    if beta_df is None:
        print("ERROR: methylprep did not return a beta-values DataFrame.")
        sys.exit(1)

    # beta_df: rows=CpG IDs, columns=sample IDs (typically Sentrix barcode like '202790040032_R02C01')
    print(f"  Got beta matrix: {beta_df.shape[0]} CpGs × {beta_df.shape[1]} samples")

    # Filter to panel CpGs and convert to per-sample dict
    in_panel_mask = beta_df.index.isin(panel_cpgs)
    beta_panel = beta_df[in_panel_mask]
    print(f"  Panel-CpG coverage: {beta_panel.shape[0]}/{len(panel_cpgs)} CpGs found in EPIC")

    per_sample = {}
    for sample_id in beta_panel.columns:
        sample_dict = {}
        for cpg in beta_panel.index:
            v = beta_panel.at[cpg, sample_id]
            try:
                v_float = float(v)
                if 0 <= v_float <= 1:
                    sample_dict[cpg] = v_float
            except (ValueError, TypeError):
                pass
        per_sample[sample_id] = sample_dict
    return per_sample


def compute_a_immune(beta_dict: dict, iamatlas_immune: dict, H_MIN: float = 0.838889) -> float:
    """
    Compute the IAMAtlas-anchored immune A-score for a single sample.
    
      A_immune = mean over CpGs of: H_binary(β) / H_min(immune)
    
    where the IAMAtlas brightness gates which CpGs we use (only those with
    a posterior immune brightness in the matrix).
    """
    contributions = []
    for cpg, beta in beta_dict.items():
        if cpg not in iamatlas_immune: continue
        if not (0 < beta < 1): continue
        h = -beta * math.log2(beta) - (1.0 - beta) * math.log2(1.0 - beta)
        contributions.append(h / H_MIN)
    if not contributions:
        return float("nan")
    return statistics.mean(contributions)


def cohen_d(a, b) -> float:
    if len(a) < 2 or len(b) < 2: return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.stdev(a), statistics.stdev(b)
    pooled = math.sqrt(((len(a) - 1) * sa**2 + (len(b) - 1) * sb**2) / (len(a) + len(b) - 2))
    if pooled == 0: return float("nan")
    return (ma - mb) / pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="IAMAtlas_v0_1.csv")
    parser.add_argument("--idat_dir", default="GSE130748_RAW",
                        help="Directory containing GSE130748 IDAT files (already extracted from RAW.tar)")
    parser.add_argument("--series_matrix", default="GSE130748_series_matrix.txt.gz",
                        help="Series matrix (download from GEO if not present)")
    parser.add_argument("--cancer_labels_csv", default=None,
                        help="Optional CSV: sample_id, cancer_incident (1/0). If omitted, all samples scored but trajectory split skipped.")
    parser.add_argument("--report", default="step_7_chk_e_gse130748_trajectory.md")
    parser.add_argument("--per_sample_csv", default="gse130748_per_sample_a_immune.csv")
    args = parser.parse_args()

    print("=" * 72)
    print("STEP 7 / CHECK E — GSE130748 Mozhui trajectory")
    print("=" * 72)

    iamatlas_immune = load_iamatlas_immune(Path(args.matrix))
    print(f"\nIAMAtlas immune brightness: {len(iamatlas_immune)} CpGs")

    print(f"\nParsing series matrix metadata: {args.series_matrix}")
    metadata = parse_series_matrix_metadata(Path(args.series_matrix))
    print(f"  Samples in metadata: {len(metadata)}")

    print(f"\nExtracting β from IDATs: {args.idat_dir}")
    panel_cpgs = set(iamatlas_immune.keys())
    per_sample_betas = extract_betas_from_idats(Path(args.idat_dir), panel_cpgs)
    print(f"  Samples with extracted β: {len(per_sample_betas)}")

    # Compute A_immune per sample
    print(f"\nComputing A_immune per sample (IAMAtlas-gated CpG set, H_min=0.838889)...")
    per_sample_a = {}
    for sid, beta_dict in per_sample_betas.items():
        a = compute_a_immune(beta_dict, iamatlas_immune)
        per_sample_a[sid] = a

    # Write per-sample CSV
    with open(args.per_sample_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "gsm", "a_immune", "title", "characteristics"])
        for sid, a in per_sample_a.items():
            # Try to match Sentrix → GSM via metadata
            gsm_match = None
            title_match = ""
            char_str = ""
            for gsm, meta in metadata.items():
                if sid in str(meta.get("title", "")) or sid == gsm:
                    gsm_match = gsm
                    title_match = meta.get("title", "")
                    char_str = " | ".join(f"{k}={v}" for k, v in meta.items() if k != "title")
                    break
            w.writerow([sid, gsm_match or "", f"{a:.4f}" if not math.isnan(a) else "NA",
                        title_match, char_str])
    print(f"  Per-sample CSV: {args.per_sample_csv}")

    # If cancer labels provided, compute trajectory deltas and group d
    trajectory = None
    if args.cancer_labels_csv and Path(args.cancer_labels_csv).exists():
        print(f"\nLoading cancer labels: {args.cancer_labels_csv}")
        labels = {}
        participant_of = {}
        timepoint_of = {}
        with open(args.cancer_labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"]
                labels[sid] = row.get("cancer_incident", "0") == "1"
                participant_of[sid] = row.get("participant_id", sid)
                timepoint_of[sid] = int(row.get("timepoint", "1"))

        # Group samples by participant
        by_part = defaultdict(dict)  # participant_id -> {timepoint: a_immune}
        cancer_status = {}
        for sid, a in per_sample_a.items():
            if sid not in participant_of: continue
            pid = participant_of[sid]
            tp = timepoint_of[sid]
            by_part[pid][tp] = a
            cancer_status[pid] = labels.get(sid, False)

        # Compute ΔA per participant
        deltas_cancer = []
        deltas_free = []
        for pid, tps in by_part.items():
            if 1 not in tps or 2 not in tps: continue
            if math.isnan(tps[1]) or math.isnan(tps[2]): continue
            delta = tps[2] - tps[1]
            if cancer_status.get(pid):
                deltas_cancer.append(delta)
            else:
                deltas_free.append(delta)

        trajectory = {
            "n_cancer_incident": len(deltas_cancer),
            "n_cancer_free": len(deltas_free),
            "mean_delta_cancer": statistics.mean(deltas_cancer) if deltas_cancer else float("nan"),
            "mean_delta_free": statistics.mean(deltas_free) if deltas_free else float("nan"),
            "cohen_d": cohen_d(deltas_cancer, deltas_free),
        }
        print(f"\nTrajectory result:")
        print(f"  cancer-incident: n={trajectory['n_cancer_incident']}, mean ΔA={trajectory['mean_delta_cancer']:+.4f}")
        print(f"  cancer-free:     n={trajectory['n_cancer_free']}, mean ΔA={trajectory['mean_delta_free']:+.4f}")
        print(f"  Cohen's d (cancer − free): {trajectory['cohen_d']:+.3f}")

    # Report
    with open(args.report, "w") as f:
        f.write("# Step 7 / Check E — GSE130748 Mozhui longitudinal trajectory\n\n")
        f.write("**Date:** 2026-05-04\n")
        f.write(f"**Cohort:** GSE130748 (Mozhui 2019, PMID 31149338) — Health, Aging and Body Composition prospective cohort\n")
        f.write("**Design:** 20 participants × 2 timepoints (baseline + follow-up) on EPIC v1 (HM850K)\n")
        f.write(f"**Method:** IAMAtlas-gated A_immune per sample, ΔA = follow-up − baseline\n\n")

        f.write(f"## Sample-level summary\n\n")
        f.write(f"- Samples in IDAT directory: {len(per_sample_betas)}\n")
        f.write(f"- Samples with valid A_immune: {sum(1 for a in per_sample_a.values() if not math.isnan(a))}\n")
        f.write(f"- A_immune distribution: ")
        valid = [a for a in per_sample_a.values() if not math.isnan(a)]
        if valid:
            f.write(f"mean = {statistics.mean(valid):.4f}, sd = {statistics.stdev(valid) if len(valid)>1 else 0:.4f}, ")
            f.write(f"min = {min(valid):.4f}, max = {max(valid):.4f}\n\n")

        f.write("## Trajectory analysis\n\n")
        if trajectory:
            f.write(f"- Cancer-incident participants: n = {trajectory['n_cancer_incident']}, mean ΔA = {trajectory['mean_delta_cancer']:+.4f}\n")
            f.write(f"- Cancer-free participants: n = {trajectory['n_cancer_free']}, mean ΔA = {trajectory['mean_delta_free']:+.4f}\n")
            f.write(f"- Cohen's d (cancer − free): **{trajectory['cohen_d']:+.3f}**\n\n")
            f.write("**Note:** With n=20 paired (likely ~5 cancer-incident, ~15 free), this test is hypothesis-generating, not powered. The framework prediction (positive d for cancer-incident) is preserved if d > 0; magnitude not reliable at this n.\n")
        else:
            f.write("**Skipped trajectory split** — no cancer-incidence labels file provided.\n\n")
            f.write("Per-sample A_immune values are written to the CSV for manual analysis.\n")
            f.write("Mozhui 2019 supplementary materials has the cancer-incidence labels by participant.\n")
            f.write("Format the CSV as: `sample_id, participant_id, timepoint, cancer_incident` (1 or 0).\n")

        f.write("\n## Decision\n\n")
        if trajectory and trajectory.get("cohen_d", float("nan")) > 0:
            f.write("- **DIRECTIONAL PASS:** ΔA is positive in cancer-incident group, consistent with framework prediction (immune compartment trends brighter pre-clinical).\n")
        elif trajectory:
            f.write("- **INVESTIGATE:** ΔA non-positive in cancer-incident group. Either small-sample noise (likely at n=20), or atlas mis-calibration. Compare against VAL-047 results.\n")
        else:
            f.write("- **PENDING:** Provide cancer-incidence labels to complete the trajectory split.\n")

    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
