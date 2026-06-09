#!/usr/bin/env python3
"""
n7_end_to_end_chain_recovery.py — L9 N7 end-to-end synthetic patient simulation

Generates a synthetic patient cohort with known injected disease signal,
runs the synthetic β matrix through the REAL CPG chain modules
(Walther IAM Deconvolver → IAMAtlas A-scoring → Mahalanobis healthy hull),
and confirms the injected signal is recovered within tolerance at each link.

This is the L9 chain-integrity test that satisfies SOP §87 — the methylome's
equivalent of Planck FFP10 end-to-end simulations.

Recovery tests run by this script:

  R1 — Walther class fraction recovery
       PASS: mean(|frac_recovered − frac_true|) ≤ 0.10 across the 8 architectural classes

  R3 — Mahalanobis case-vs-HC separation
       PASS: Cohen's d ≥ +0.5 AND (when signal_strength > 0)
             recovered_d / injected_signal_strength ≥ 0.3

  NULL — When signal_strength = 0.0
       PASS: |Cohen's d| ≤ 0.3 (chain produces near-zero separation on true null)

Run modes:
  STRONG_SIGNAL : n_case=50, n_hc=200, signal_strength=2.0, seed=7
  NULL_BASELINE : n_case=50, n_hc=200, signal_strength=0.0, seed=8

Outputs:
  ./synth_cohort_strong/    — generated cohort + per-patient chain outputs + R1/R3 results
  ./synth_cohort_null/      — same for null baseline
  ./n7_summary.json         — combined PASS/FAIL summary for both conditions
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the chain modules importable
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Biological_Physics/chain_of_custody/L9_null_suite"))
sys.path.insert(0, str(REPO_ROOT / "Biological_Physics/atlas_vault/walther_clinical_runtime/Walther_iam_deconvolver"))
sys.path.insert(0, str(REPO_ROOT / "Biological_Physics/atlas_vault/walther_clinical_runtime/A_Scoring_Module"))
sys.path.insert(0, str(REPO_ROOT / "Biological_Physics/atlas_vault/walther_clinical_runtime/Mahalanobis_healthy_reference"))

from synthetic_patient_generator import SyntheticCohort, CLASSES
from walther_iam_deconvolver import WaltherIAMDeconvolver
from iamatlas_a_scoring import score_per_celltype, load_artifact
from iamatlas_mahalanobis_scoring import MahalanobisHealthyHull

# ----- Reference paths (atlas + runtime artifacts) -----
ATLAS_CSV = Path("/home/claude/iamatlas_rebuild/IAMAtlasREBUILD.csv")
CELLTYPE_CLASS_MAP = Path("/mnt/user-data/uploads/IAMAtlasREBUILD_celltype_to_class.json")
MARKER_ARTIFACT = REPO_ROOT / "Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json"
MAHAL_REF = REPO_ROOT / "Biological_Physics/atlas_vault/pipeline_runtime_matrices/mahalanobis_healthy_reference_v0_1.json"


def cohens_d(case_vals, hc_vals):
    """Pooled-SD Cohen's d."""
    case_vals = np.asarray(case_vals, dtype=float)
    hc_vals = np.asarray(hc_vals, dtype=float)
    n1, n2 = len(case_vals), len(hc_vals)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s = np.sqrt(((n1 - 1) * case_vals.var(ddof=1) + (n2 - 1) * hc_vals.var(ddof=1)) / (n1 + n2 - 2))
    if s == 0:
        return float("nan")
    return float((case_vals.mean() - hc_vals.mean()) / s)


def run_condition(label, n_case, n_hc, signal_strength, seed, n_cpgs_subset, out_dir):
    """Generate synthetic cohort + run through real chain + compute R1, R3."""
    print(f"\n{'=' * 72}")
    print(f"CONDITION: {label}")
    print(f"  n_case={n_case}, n_hc={n_hc}, signal_strength={signal_strength}, seed={seed}")
    print(f"{'=' * 72}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # --- Phase 1: Generate synthetic cohort ----------------------------------
    print(f"\n[Phase 1] Generating synthetic cohort ({n_case + n_hc} patients, {n_cpgs_subset:,} CpGs subset)...")
    cohort = SyntheticCohort(
        n_case=n_case, n_hc=n_hc,
        disease_signal_strength=signal_strength,
        disease_panel_size=500,
        age_range=(40, 70),
        age_effect_strength=0.05,
        sex_imbalance=0.6,
        batch_count=3,
        batch_effect_strength=0.02,
        noise_sigma=0.03,
        n_cpgs=n_cpgs_subset,
        random_seed=seed,
        cohort_name=f"n7_{label.lower()}",
    )
    cohort.generate()
    cohort.export(out_dir / "generated")
    print(f"   ✓ Generated in {time.time() - t0:.1f}s")

    # --- Phase 2: Load chain modules -----------------------------------------
    print(f"\n[Phase 2] Loading chain modules...")
    t1 = time.time()
    print(f"   Loading Walther IAM Deconvolver (atlas: {ATLAS_CSV.name})...")
    walther = WaltherIAMDeconvolver(
        matrix_path=str(ATLAS_CSV),
        celltype_class_map=str(CELLTYPE_CLASS_MAP),
        n_class_markers_per_class=600,
        max_celltype_markers=4000,
        verbose=False,
    )
    print(f"     ✓ {len(walther.class_ref):,} class markers, {len(walther.celltype_ref):,} cell-type markers")

    print(f"   Loading A-scoring marker artifact (v0_2)...")
    meta, ct_markers, cmap, hmin = load_artifact(str(MARKER_ARTIFACT))
    print(f"     ✓ {len(ct_markers)} cell-types, {len(hmin)} H_min classes")

    print(f"   Loading Mahalanobis healthy reference...")
    hull = MahalanobisHealthyHull(str(MAHAL_REF))
    print(f"     ✓ {hull.n_features} features in healthy hull")
    print(f"   ✓ All modules loaded in {time.time() - t1:.1f}s")

    # --- Phase 3: Run each synthetic patient through chain --------------------
    print(f"\n[Phase 3] Running {n_case + n_hc} synthetic patients through chain...")
    t2 = time.time()

    truth_df = pd.read_csv(out_dir / "generated" / "truth_table.csv")
    beta_df = pd.read_parquet(out_dir / "generated" / "beta_matrix.parquet")
    # beta_df: rows = CpGs, columns = patient_ids

    walther_results = []
    mahal_results = []

    for i, pid in enumerate(truth_df["patient_id"]):
        # Convert patient β column to dict
        betas_dict = beta_df[pid].to_dict()

        # Stage 2: Walther deconvolution
        decon = walther.deconvolve(betas_dict, refine_celltypes=False)

        walther_row = {"patient_id": pid, "decon_status": decon.status}
        for cls in CLASSES:
            walther_row[f"frac_{cls}"] = decon.class_fractions.get(cls, 0.0)
        walther_results.append(walther_row)

        # Stage 4: A-scoring (per cell-type — feeds Mahalanobis)
        ct_scores = score_per_celltype(betas_dict, ct_markers, cmap, hmin)
        ct_score_dict = {ct: r["A"] for ct, r in ct_scores.items() if r["status"] == "OK"}

        # Stage 5: Mahalanobis
        mahal = hull.score(ct_score_dict)
        mahal_results.append({
            "patient_id": pid,
            "mahalanobis_distance": mahal["mahalanobis_distance"],
            "n_features_used": mahal["n_features_used"],
            "status": mahal["status"],
        })

        if (i + 1) % 25 == 0:
            print(f"   ...{i + 1}/{n_case + n_hc} patients chained ({time.time() - t2:.1f}s elapsed)")

    walther_df = pd.DataFrame(walther_results)
    mahal_df = pd.DataFrame(mahal_results)

    walther_df.to_csv(out_dir / "walther_class_fractions.csv", index=False)
    mahal_df.to_csv(out_dir / "mahalanobis_distances.csv", index=False)
    print(f"   ✓ Chain complete in {time.time() - t2:.1f}s")

    # --- Phase 4: Recovery tests ---------------------------------------------
    print(f"\n[Phase 4] Computing recovery tests...")

    # Merge truth with chain output
    merged = truth_df.merge(walther_df, on="patient_id").merge(mahal_df, on="patient_id")

    # R1 — Walther class fraction recovery
    r1_per_class = {}
    for cls in CLASSES:
        true_col = f"true_frac_{cls}"
        recov_col = f"frac_{cls}"
        if true_col in merged.columns:
            err = (merged[recov_col] - merged[true_col]).abs()
            r1_per_class[cls] = {
                "mean_abs_error": float(err.mean()),
                "max_abs_error": float(err.max()),
                "true_mean_frac": float(merged[true_col].mean()),
                "recovered_mean_frac": float(merged[recov_col].mean()),
            }
    r1_overall_mae = float(np.mean([v["mean_abs_error"] for v in r1_per_class.values()]))
    r1_pass = r1_overall_mae <= 0.10

    # R3 — Mahalanobis case-vs-HC Cohen's d
    case_mask = merged["arm"] == "case"
    hc_mask = merged["arm"] == "hc"
    d_case = merged.loc[case_mask, "mahalanobis_distance"].values
    d_hc = merged.loc[hc_mask, "mahalanobis_distance"].values
    r3_cohens_d = cohens_d(d_case, d_hc)

    if signal_strength > 0:
        # STRONG signal: recovered d should be positive and proportional
        r3_pass = (r3_cohens_d >= 0.5) and (r3_cohens_d / signal_strength >= 0.3)
        r3_criterion = f"d ≥ 0.5 AND d/signal_strength ({r3_cohens_d / signal_strength:.3f}) ≥ 0.3"
    else:
        # NULL: recovered d should be near zero
        r3_pass = abs(r3_cohens_d) <= 0.3
        r3_criterion = f"|d| ({abs(r3_cohens_d):.3f}) ≤ 0.3"

    # Save results
    results = {
        "condition_label": label,
        "n_case": n_case,
        "n_hc": n_hc,
        "injected_signal_strength": signal_strength,
        "random_seed": seed,
        "n_cpgs_subset": n_cpgs_subset,
        "runtime_seconds": time.time() - t0,
        "R1_walther_fraction_recovery": {
            "overall_mean_abs_error_across_classes": r1_overall_mae,
            "criterion": "mean(|frac_recovered − frac_true|) ≤ 0.10",
            "PASS": bool(r1_pass),
            "per_class": r1_per_class,
        },
        "R3_mahalanobis_case_vs_hc": {
            "case_mean_distance": float(d_case.mean()),
            "case_sd_distance": float(d_case.std(ddof=1)),
            "hc_mean_distance": float(d_hc.mean()),
            "hc_sd_distance": float(d_hc.std(ddof=1)),
            "cohens_d": r3_cohens_d,
            "criterion": r3_criterion,
            "PASS": bool(r3_pass),
        },
    }

    with open(out_dir / "recovery_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'-' * 72}")
    print(f"RECOVERY RESULTS — {label}")
    print(f"{'-' * 72}")
    print(f"  R1 Walther fraction recovery: overall MAE = {r1_overall_mae:.4f}")
    print(f"     {'✓ PASS' if r1_pass else '✗ FAIL'} ({r1_per_class})")
    for cls, v in r1_per_class.items():
        print(f"       {cls:<12} true={v['true_mean_frac']:.4f} recov={v['recovered_mean_frac']:.4f} MAE={v['mean_abs_error']:.4f}")
    print(f"  R3 Mahalanobis case-vs-HC: Cohen's d = {r3_cohens_d:+.4f}")
    print(f"     {'✓ PASS' if r3_pass else '✗ FAIL'} (criterion: {r3_criterion})")

    return results


def main():
    out_root = Path(__file__).parent

    # Use a 50k-CpG subset for speed. Strong cohort + null cohort run independently.
    n_cpgs_subset = 50_000

    strong = run_condition(
        label="STRONG_SIGNAL",
        n_case=50, n_hc=200,
        signal_strength=2.0, seed=7,
        n_cpgs_subset=n_cpgs_subset,
        out_dir=out_root / "synth_cohort_strong",
    )

    null = run_condition(
        label="NULL_BASELINE",
        n_case=50, n_hc=200,
        signal_strength=0.0, seed=8,
        n_cpgs_subset=n_cpgs_subset,
        out_dir=out_root / "synth_cohort_null",
    )

    # Combined N7 verdict — both conditions must PASS for the chain to be claimed working
    n7_passed = (
        strong["R1_walther_fraction_recovery"]["PASS"]
        and strong["R3_mahalanobis_case_vs_hc"]["PASS"]
        and null["R1_walther_fraction_recovery"]["PASS"]
        and null["R3_mahalanobis_case_vs_hc"]["PASS"]
    )

    summary = {
        "test_name": "N7_end_to_end_chain_recovery",
        "test_protocol": "Synthetic cohort generation → Walther → A-scoring → Mahalanobis → recovery verification",
        "sop_reference": "SOP §87 / cpg_null_runner.py run_N7 / synthetic_patient_generator.py",
        "session_date": "2026-06-05",
        "n_cpgs_subset_for_speed": n_cpgs_subset,
        "STRONG_SIGNAL_condition": strong,
        "NULL_BASELINE_condition": null,
        "N7_OVERALL_PASS": bool(n7_passed),
        "verdict_narrative": (
            "Full β-matrix end-to-end chain-recovery test executed for the first time. "
            "Two conditions: STRONG signal (s=2.0, expect chain to recover injected case-vs-HC "
            "Mahalanobis separation) and NULL baseline (s=0.0, expect chain to return near-zero "
            "case-vs-HC separation on true null). Both conditions must PASS for the chain to be "
            "claimed working end-to-end."
        ),
    }
    with open(out_root / "n7_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f"N7 OVERALL: {'✓ PASS' if n7_passed else '✗ FAIL'}")
    print(f"{'=' * 72}")
    print(f"  STRONG R1: {'PASS' if strong['R1_walther_fraction_recovery']['PASS'] else 'FAIL'}")
    print(f"  STRONG R3: {'PASS' if strong['R3_mahalanobis_case_vs_hc']['PASS'] else 'FAIL'}")
    print(f"  NULL   R1: {'PASS' if null['R1_walther_fraction_recovery']['PASS'] else 'FAIL'}")
    print(f"  NULL   R3: {'PASS' if null['R3_mahalanobis_case_vs_hc']['PASS'] else 'FAIL'}")


if __name__ == "__main__":
    main()
