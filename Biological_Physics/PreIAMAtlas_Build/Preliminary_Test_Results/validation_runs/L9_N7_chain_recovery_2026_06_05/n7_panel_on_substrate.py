#!/usr/bin/env python3
"""
n7_panel_on_substrate.py — N7 v2 variant with signal injected on cell-type marker CpGs

Diagnostic from first N7 run (n7_end_to_end_chain_recovery.py):
  - R1 (Walther fraction recovery): PASS both conditions
  - NULL R3 (no false positives): PASS
  - STRONG R3: FAIL — Cohen's d = -0.25 (wrong direction)

Diagnosis of STRONG R3 fail: only 21.4% of the 500 disease-panel CpGs (which were
sampled uniformly across the 22,542-CpG atlas subset) landed on the 6,802-CpG
cell-type marker substrate. With ~107 hits spread across 115 cell-types' marker
pools (~100 markers each), the Shannon entropy averaging in A-scoring dilutes the
per-cell-type signal below noise. The chain is correctly blind to signal that
doesn't reach its measurement substrate.

This script tests the chain when signal IS injected on the measurement substrate:
disease panel restricted to cell-type marker CpGs. If recovery succeeds, the
chain is validated for the kind of signal it's built to detect, and the v0.1
synthetic generator has a known limitation (random panel selection) flagged for
v0.2 (panel-on-substrate injection mode).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Biological_Physics/chain_of_custody/L9_null_suite"))
sys.path.insert(0, str(REPO_ROOT / "Biological_Physics/atlas_vault/walther_clinical_runtime/Walther_iam_deconvolver"))
sys.path.insert(0, str(REPO_ROOT / "Biological_Physics/atlas_vault/walther_clinical_runtime/A_Scoring_Module"))
sys.path.insert(0, str(REPO_ROOT / "Biological_Physics/atlas_vault/walther_clinical_runtime/Mahalanobis_healthy_reference"))

from synthetic_patient_generator import SyntheticCohort, CLASSES
from walther_iam_deconvolver import WaltherIAMDeconvolver
from iamatlas_a_scoring import score_per_celltype, load_artifact
from iamatlas_mahalanobis_scoring import MahalanobisHealthyHull

ATLAS_CSV = Path("/home/claude/iamatlas_rebuild/IAMAtlasREBUILD.csv")
CELLTYPE_CLASS_MAP = Path("/mnt/user-data/uploads/IAMAtlasREBUILD_celltype_to_class.json")
MARKER_ARTIFACT = REPO_ROOT / "Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_celltype_markers_v0_2.json"
MAHAL_REF = REPO_ROOT / "Biological_Physics/atlas_vault/pipeline_runtime_matrices/mahalanobis_healthy_reference_v0_1.json"


class MarkerSubstrateCohort(SyntheticCohort):
    """SyntheticCohort variant restricting disease-panel CpGs to cell-type marker substrate."""

    def __init__(self, marker_cpgs: set, **kwargs):
        super().__init__(**kwargs)
        self.marker_cpgs = marker_cpgs

    def _design_disease_panel(self):
        """Restrict the panel CpG candidates to the cell-type marker substrate."""
        atlas = self.atlas
        # Restrict candidates to marker CpGs that are also in the synthetic atlas subset
        candidates = atlas[atlas["cpg"].isin(self.marker_cpgs)]
        if len(candidates) < self.disease_panel_size:
            print(f"[WARN] requested {self.disease_panel_size} panel CpGs but only "
                  f"{len(candidates)} marker CpGs in synthetic subset; using all of them")
            self.disease_panel = candidates["cpg"].values
        else:
            self.disease_panel = candidates.sample(
                n=self.disease_panel_size, random_state=self.random_seed
            )["cpg"].values
        rng = np.random.default_rng(self.random_seed + 1)
        signs = rng.choice([-1, +1], size=len(self.disease_panel), p=[5.4 / (5.4 + 1), 1 / (5.4 + 1)])
        self.disease_directions = dict(zip(self.disease_panel, signs))
        print(f"[MarkerSubstrateCohort] Disease panel: {len(self.disease_panel)} CpGs ON MARKER SUBSTRATE "
              f"({(signs < 0).sum()} hypo, {(signs > 0).sum()} hyper)")


def cohens_d(case_vals, hc_vals):
    case_vals = np.asarray(case_vals, dtype=float)
    hc_vals = np.asarray(hc_vals, dtype=float)
    n1, n2 = len(case_vals), len(hc_vals)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s = np.sqrt(((n1 - 1) * case_vals.var(ddof=1) + (n2 - 1) * hc_vals.var(ddof=1)) / (n1 + n2 - 2))
    return float((case_vals.mean() - hc_vals.mean()) / s) if s > 0 else float("nan")


def main():
    out_dir = Path(__file__).parent / "synth_cohort_strong_on_substrate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CONDITION: STRONG_ON_SUBSTRATE — N7 v2 with panel-on-marker-substrate injection")
    print("=" * 72)

    # Load marker CpGs (the chain's measurement substrate)
    with open(MARKER_ARTIFACT) as f:
        art = json.load(f)
    marker_cpgs = set()
    for cpgs in art["markers_by_celltype"].values():
        marker_cpgs.update(cpgs)
    print(f"\nCell-type marker substrate: {len(marker_cpgs):,} unique CpGs across 115 cells")

    t0 = time.time()
    print(f"\n[Phase 1] Generating synthetic cohort with panel-on-substrate injection...")
    cohort = MarkerSubstrateCohort(
        marker_cpgs=marker_cpgs,
        n_case=50, n_hc=200,
        disease_signal_strength=2.0,
        disease_panel_size=500,
        age_range=(40, 70),
        age_effect_strength=0.05,
        sex_imbalance=0.6,
        batch_count=3,
        batch_effect_strength=0.02,
        noise_sigma=0.03,
        n_cpgs=50_000,
        random_seed=7,
        cohort_name="n7_strong_on_substrate",
    )
    cohort.generate()
    cohort.export(out_dir / "generated")
    print(f"   ✓ Generated in {time.time() - t0:.1f}s")

    # Confirm overlap
    disease_cpgs = set(cohort.disease_panel)
    overlap = disease_cpgs & marker_cpgs
    print(f"\n   OVERLAP check: {len(overlap)}/{len(disease_cpgs)} panel CpGs on marker substrate "
          f"= {len(overlap) / len(disease_cpgs) * 100:.1f}% (target: 100%)")

    print(f"\n[Phase 2] Loading chain modules...")
    t1 = time.time()
    walther = WaltherIAMDeconvolver(
        matrix_path=str(ATLAS_CSV),
        celltype_class_map=str(CELLTYPE_CLASS_MAP),
        n_class_markers_per_class=600,
        max_celltype_markers=4000,
        verbose=False,
    )
    meta, ct_markers, cmap, hmin = load_artifact(str(MARKER_ARTIFACT))
    hull = MahalanobisHealthyHull(str(MAHAL_REF))
    print(f"   ✓ Modules loaded in {time.time() - t1:.1f}s")

    print(f"\n[Phase 3] Running chain on {cohort.n_case + cohort.n_hc} patients...")
    t2 = time.time()
    truth_df = pd.read_csv(out_dir / "generated" / "truth_table.csv")
    beta_df = pd.read_parquet(out_dir / "generated" / "beta_matrix.parquet")

    walther_results, mahal_results = [], []
    for i, pid in enumerate(truth_df["patient_id"]):
        betas_dict = beta_df[pid].to_dict()
        decon = walther.deconvolve(betas_dict, refine_celltypes=False)
        row = {"patient_id": pid, "decon_status": decon.status}
        for cls in CLASSES:
            row[f"frac_{cls}"] = decon.class_fractions.get(cls, 0.0)
        walther_results.append(row)
        ct_scores = score_per_celltype(betas_dict, ct_markers, cmap, hmin)
        ct_score_dict = {ct: r["A"] for ct, r in ct_scores.items() if r["status"] == "OK"}
        mahal = hull.score(ct_score_dict)
        mahal_results.append({
            "patient_id": pid,
            "mahalanobis_distance": mahal["mahalanobis_distance"],
            "n_features_used": mahal["n_features_used"],
            "status": mahal["status"],
        })
        if (i + 1) % 50 == 0:
            print(f"   ...{i + 1}/250 patients ({time.time() - t2:.1f}s)")

    walther_df = pd.DataFrame(walther_results)
    mahal_df = pd.DataFrame(mahal_results)
    walther_df.to_csv(out_dir / "walther_class_fractions.csv", index=False)
    mahal_df.to_csv(out_dir / "mahalanobis_distances.csv", index=False)
    print(f"   ✓ Chain complete in {time.time() - t2:.1f}s")

    print(f"\n[Phase 4] Recovery tests...")
    merged = truth_df.merge(walther_df, on="patient_id").merge(mahal_df, on="patient_id")

    # R1 — Walther fraction recovery
    r1_per_class = {}
    for cls in CLASSES:
        err = (merged[f"frac_{cls}"] - merged[f"true_frac_{cls}"]).abs()
        r1_per_class[cls] = {
            "mean_abs_error": float(err.mean()),
            "true_mean": float(merged[f"true_frac_{cls}"].mean()),
            "recovered_mean": float(merged[f"frac_{cls}"].mean()),
        }
    r1_overall_mae = float(np.mean([v["mean_abs_error"] for v in r1_per_class.values()]))
    r1_pass = r1_overall_mae <= 0.10

    # R3 — Mahalanobis case-vs-HC
    d_case = merged.loc[merged["arm"] == "case", "mahalanobis_distance"].values
    d_hc = merged.loc[merged["arm"] == "hc", "mahalanobis_distance"].values
    r3_d = cohens_d(d_case, d_hc)
    r3_pass = (r3_d >= 0.5) and (r3_d / 2.0 >= 0.3)

    print(f"\n{'-' * 72}")
    print(f"RECOVERY RESULTS — STRONG_ON_SUBSTRATE")
    print(f"{'-' * 72}")
    print(f"  R1 Walther MAE = {r1_overall_mae:.4f} → {'✓ PASS' if r1_pass else '✗ FAIL'}")
    for cls, v in r1_per_class.items():
        print(f"     {cls:<12} true={v['true_mean']:.4f} recov={v['recovered_mean']:.4f} MAE={v['mean_abs_error']:.4f}")
    print(f"  R3 Cohen's d = {r3_d:+.4f} → {'✓ PASS' if r3_pass else '✗ FAIL'}")
    print(f"     case  mean={d_case.mean():.3f} sd={d_case.std(ddof=1):.3f}  n={len(d_case)}")
    print(f"     hc    mean={d_hc.mean():.3f} sd={d_hc.std(ddof=1):.3f}  n={len(d_hc)}")
    print(f"     d/signal_strength = {r3_d / 2.0:+.3f} (target ≥ 0.3)")

    results = {
        "condition_label": "STRONG_ON_SUBSTRATE",
        "purpose": "N7 v2 — disease panel injected ON cell-type marker substrate to test chain detectability when signal hits where the chain measures",
        "n_case": 50, "n_hc": 200,
        "injected_signal_strength": 2.0,
        "panel_substrate_overlap_pct": float(len(overlap) / len(disease_cpgs) * 100),
        "runtime_seconds": time.time() - t0,
        "R1_walther_fraction_recovery": {
            "overall_mean_abs_error": r1_overall_mae,
            "criterion": "MAE ≤ 0.10",
            "PASS": bool(r1_pass),
            "per_class": r1_per_class,
        },
        "R3_mahalanobis_case_vs_hc": {
            "case_mean_distance": float(d_case.mean()),
            "case_sd_distance": float(d_case.std(ddof=1)),
            "hc_mean_distance": float(d_hc.mean()),
            "hc_sd_distance": float(d_hc.std(ddof=1)),
            "cohens_d": r3_d,
            "criterion": "d ≥ 0.5 AND d/signal_strength ≥ 0.3",
            "PASS": bool(r3_pass),
        },
    }
    with open(out_dir / "recovery_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
