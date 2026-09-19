#!/usr/bin/env python3
"""
STEP 7 / CHECK C — AD cross-cohort directional scoring against IAMAtlas
========================================================================

Re-scores the AIBL (n=632) and AddNeuroMed (n=175) AD cohorts that were
processed earlier tonight, this time using IAMAtlas v0.1 immune-class
brightness as the reference rather than raw H_min(immune)=0.838889.

QUESTION
========
Does IAMAtlas-derived per-CpG brightness improve directional discrimination
over flat H_min normalization?

INPUTS
======
1. IAMAtlas_v0_1.csv                                       (production matrix)
2. val051_panel_ruleA.json                                 (7-CpG directional panel)
3. AIBL per-sample β at panel CpGs:
     repo/IAM-Validation/Biological_Physics/validation_runs/val_050_aibl/
       aibl_imm_betas.json
       aibl_manifest.json   (sample → AD/HC label)
4. AddNeuroMed per-sample β at panel CpGs:
     repo/IAM-Validation/Biological_Physics/validation_runs/val_052_addneuromed/
       addneuromed_imm_betas.json
       addneuromed_manifest.json

METHOD
======
For each sample, compute TWO directional A-scores at the panel CpGs:

  A_dir_flat:    Σ direction_i × beta_i / H_min(immune)        [tonight's method]
  A_dir_iam:     Σ direction_i × (beta_i - iamatlas_mean_i)    [IAMAtlas-anchored]

Compare AD vs HC Cohen's d for both methods. Same direction = framework
internally consistent. Higher d for IAM-anchored = matrix adds discriminative
power.

OUTPUT
======
step_7_chk_c_ad_cohort_iam_scoring.md

Date: 2026-05-04
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def load_iamatlas_immune_brightness(matrix_path: Path) -> dict:
    """Load only the immune class column to save memory."""
    out = {}
    with open(matrix_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpg = row["cpg_id"]
            mean = row.get("immune_mean", "NA")
            if mean not in ("NA", "", None):
                try:
                    out[cpg] = float(mean)
                except ValueError:
                    pass
    return out


def cohen_d(group_a, group_b) -> float:
    """Pooled-SD Cohen's d (group_a − group_b)."""
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    ma = statistics.mean(group_a)
    mb = statistics.mean(group_b)
    sa = statistics.stdev(group_a)
    sb = statistics.stdev(group_b)
    pooled = math.sqrt(((len(group_a) - 1) * sa**2 + (len(group_b) - 1) * sb**2) /
                       (len(group_a) + len(group_b) - 2))
    if pooled == 0:
        return float("nan")
    return (ma - mb) / pooled


def score_cohort(name: str, betas_json: Path, manifest_json: Path,
                 panel: list, iamatlas_immune: dict,
                 H_MIN: float = 0.838889) -> dict:
    """Score one cohort under both methods."""
    if not betas_json.exists():
        return {"name": name, "status": "MISSING_BETAS", "path": str(betas_json)}
    if not manifest_json.exists():
        return {"name": name, "status": "MISSING_MANIFEST", "path": str(manifest_json)}

    with open(betas_json) as f:
        per_sample_beta = json.load(f)
    with open(manifest_json) as f:
        manifest = json.load(f)

    # Manifest is a list of dicts. Each entry has either gsm or sentrix as ID,
    # and a disease-state field with cohort-specific naming.
    sample_label = {}
    if not isinstance(manifest, list):
        return {"name": name, "status": "MANIFEST_FORMAT_ERROR", "type": type(manifest).__name__}

    for entry in manifest:
        if not isinstance(entry, dict): continue
        # Get all plausible IDs for this sample (the betas file uses one of them as key)
        ids = []
        for k in ("sentrix", "gsm", "sample_id", "id", "Sample_geo_accession"):
            v = entry.get(k)
            if v: ids.append(v)
        # Pull label from any field that looks like disease status
        label_raw = None
        for k in ("disease status", "disease state", "disease_state", "dx",
                  "group", "label", "class", "diagnosis"):
            v = entry.get(k)
            if v:
                label_raw = str(v).lower()
                break
        if not label_raw: continue
        # Normalize to AD or HC
        if any(s in label_raw for s in ["alzheimer", "ad ", "dementia", "mci", "cognitive impair"]):
            normalized = "AD"
        elif any(s in label_raw for s in ["healthy", "control", "hc", "normal"]):
            normalized = "HC"
        else:
            continue
        for sid in ids:
            sample_label[sid] = normalized

    # Score each sample under both methods
    scores_flat_ad, scores_flat_hc = [], []
    scores_iam_ad, scores_iam_hc = [], []
    n_label_unknown = 0
    iamatlas_coverage = 0
    iamatlas_missing = 0

    for sample_id, beta_dict in per_sample_beta.items():
        # Determine label
        label = sample_label.get(sample_id)
        if not label:
            n_label_unknown += 1
            continue
        is_ad = label == "AD"
        is_hc = label == "HC"
        if not (is_ad or is_hc):
            continue

        # Compute both scores over panel
        flat_score = 0.0
        iam_score = 0.0
        n_used = 0
        for entry in panel:
            cpg = entry["cpg"]
            direction = entry["direction"]
            beta = beta_dict.get(cpg)
            if beta is None: continue
            try: beta = float(beta)
            except (ValueError, TypeError): continue
            # Flat directional A-score (the method used tonight)
            flat_score += direction * beta / H_MIN
            # IAMAtlas-anchored directional residual A-score
            atlas_mean = iamatlas_immune.get(cpg)
            if atlas_mean is not None:
                iam_score += direction * (beta - atlas_mean)
                iamatlas_coverage += 1
            else:
                iamatlas_missing += 1
            n_used += 1
        if n_used == 0: continue
        flat_score /= n_used
        iam_score /= n_used

        if is_ad:
            scores_flat_ad.append(flat_score)
            scores_iam_ad.append(iam_score)
        else:
            scores_flat_hc.append(flat_score)
            scores_iam_hc.append(iam_score)

    return {
        "name": name,
        "status": "COMPLETE",
        "n_ad": len(scores_flat_ad),
        "n_hc": len(scores_flat_hc),
        "n_label_unknown": n_label_unknown,
        "iamatlas_panel_coverage": iamatlas_coverage,
        "iamatlas_panel_missing": iamatlas_missing,
        "flat_method": {
            "AD_mean":  statistics.mean(scores_flat_ad)  if scores_flat_ad else float("nan"),
            "AD_sd":    statistics.stdev(scores_flat_ad) if len(scores_flat_ad) > 1 else 0.0,
            "HC_mean":  statistics.mean(scores_flat_hc)  if scores_flat_hc else float("nan"),
            "HC_sd":    statistics.stdev(scores_flat_hc) if len(scores_flat_hc) > 1 else 0.0,
            "cohen_d":  cohen_d(scores_flat_ad, scores_flat_hc),
        },
        "iam_method": {
            "AD_mean":  statistics.mean(scores_iam_ad)  if scores_iam_ad else float("nan"),
            "AD_sd":    statistics.stdev(scores_iam_ad) if len(scores_iam_ad) > 1 else 0.0,
            "HC_mean":  statistics.mean(scores_iam_hc)  if scores_iam_hc else float("nan"),
            "HC_sd":    statistics.stdev(scores_iam_hc) if len(scores_iam_hc) > 1 else 0.0,
            "cohen_d":  cohen_d(scores_iam_ad, scores_iam_hc),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="IAMAtlas_v0_1.csv")
    parser.add_argument("--panel", default="val051_panel_ruleA.json")
    parser.add_argument("--aibl_betas", default="aibl_imm_betas.json")
    parser.add_argument("--aibl_manifest", default="aibl_manifest.json")
    parser.add_argument("--addneuromed_betas", default="addneuromed_imm_betas.json")
    parser.add_argument("--addneuromed_manifest", default="addneuromed_manifest.json")
    parser.add_argument("--report", default="step_7_chk_c_ad_cohort_iam_scoring.md")
    args = parser.parse_args()

    print("=" * 72)
    print("STEP 7 / CHECK C — AD cross-cohort scoring vs IAMAtlas")
    print("=" * 72)

    print(f"\nLoading IAMAtlas immune column from {args.matrix}...")
    iamatlas_immune = load_iamatlas_immune_brightness(Path(args.matrix))
    print(f"  Loaded immune brightness for {len(iamatlas_immune)} CpGs")

    print(f"\nLoading val051 panel from {args.panel}...")
    with open(args.panel) as f:
        panel_obj = json.load(f)
    panel = panel_obj["cpgs"]
    print(f"  Panel: {len(panel)} CpGs (rule A directional)")

    aibl = score_cohort("AIBL", Path(args.aibl_betas), Path(args.aibl_manifest), panel, iamatlas_immune)
    print(f"\nAIBL: {aibl.get('status')}")
    if aibl["status"] == "COMPLETE":
        print(f"  n_AD={aibl['n_ad']}, n_HC={aibl['n_hc']}")
        print(f"  Flat method d = {aibl['flat_method']['cohen_d']:+.3f}")
        print(f"  IAM  method d = {aibl['iam_method']['cohen_d']:+.3f}")

    addn = score_cohort("AddNeuroMed", Path(args.addneuromed_betas), Path(args.addneuromed_manifest), panel, iamatlas_immune)
    print(f"\nAddNeuroMed: {addn.get('status')}")
    if addn["status"] == "COMPLETE":
        print(f"  n_AD={addn['n_ad']}, n_HC={addn['n_hc']}")
        print(f"  Flat method d = {addn['flat_method']['cohen_d']:+.3f}")
        print(f"  IAM  method d = {addn['iam_method']['cohen_d']:+.3f}")

    # Report
    with open(args.report, "w") as f:
        f.write("# Step 7 / Check C — AD cross-cohort IAMAtlas scoring\n\n")
        f.write("**Date:** 2026-05-04\n\n")
        f.write("Compares two scoring methods on the same val051 directional panel:\n\n")
        f.write("- **Flat method** (tonight's earlier method): `Σ direction × β / H_min(immune)`\n")
        f.write("- **IAM method** (this Check): `Σ direction × (β − IAMAtlas_immune_mean)`\n\n")
        f.write("Question: does IAMAtlas-derived per-CpG anchoring preserve or improve the AD/HC effect size?\n\n")
        for r in [aibl, addn]:
            f.write(f"## {r['name']}\n\n")
            if r["status"] != "COMPLETE":
                f.write(f"**Status:** {r['status']}\n\n")
                continue
            f.write(f"- n_AD = {r['n_ad']}, n_HC = {r['n_hc']}\n")
            f.write(f"- IAMAtlas panel-CpG coverage: {r['iamatlas_panel_coverage']} hits / {r['iamatlas_panel_missing']} missing\n\n")
            f.write("| Method | AD mean | AD sd | HC mean | HC sd | Cohen's d |\n|---|---|---|---|---|---|\n")
            for m in ["flat_method", "iam_method"]:
                d = r[m]
                f.write(f"| {m.replace('_method','').upper()} | "
                        f"{d['AD_mean']:+.4f} | {d['AD_sd']:.4f} | "
                        f"{d['HC_mean']:+.4f} | {d['HC_sd']:.4f} | "
                        f"**{d['cohen_d']:+.3f}** |\n")
            f.write("\n")
        f.write("## Decision\n\n")
        f.write("- **PASS:** Both methods give same-direction d, IAM method d ≥ 0.95 × Flat method d → IAMAtlas preserves discrimination, ready to deploy.\n")
        f.write("- **PASS+:** IAM d > Flat d → matrix adds discriminative power.\n")
        f.write("- **INVESTIGATE:** Sign flip or IAM d < 0.5 × Flat d → matrix is mis-calibrated for this disease, debug.\n")

    print(f"\nReport: {args.report}")


if __name__ == "__main__":
    main()
