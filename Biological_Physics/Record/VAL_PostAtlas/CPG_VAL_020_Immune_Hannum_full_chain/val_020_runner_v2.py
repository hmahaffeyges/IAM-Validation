#!/usr/bin/env python3
"""val_020_runner_v2.py — Memory-conscious full-chain CPG-VAL-020.

Same canonical-module chain as v1 but with subset-to-marker-CpGs strategy
to fit in <2 GB peak memory.
"""
from __future__ import annotations
import sys, gc, time, json, math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

RUNTIME = Path("/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime")
ATLAS_CSV = "/home/claude/iamatlas_rebuild/IAMAtlasREBUILD.csv"
HC_REF = RUNTIME / "Mahalanobis_healthy_reference/mahalanobis_healthy_reference_v0_1.json"
MARKERS = RUNTIME / "Celltype_Marker/iamatlas_celltype_markers_v0_2.json"
AGE_REF = RUNTIME / "Age_Reference_Matrix_80_cells/age_reference_matrix.json"
CT_TO_CLASS = RUNTIME / "IAMAtlas_REBUILD/IAMAtlasREBUILD_celltype_to_class.json"

sys.path.insert(0, str(RUNTIME / "Walther_iam_deconvolver"))
sys.path.insert(0, str(RUNTIME / "A_Scoring_Module"))
sys.path.insert(0, str(RUNTIME / "Mahalanobis_healthy_reference"))
sys.path.insert(0, str(RUNTIME / "IAM_Cellular_Age"))

from walther_iam_deconvolver import WaltherIAMDeconvolver
from iamatlas_a_scoring import score_per_celltype, score_per_class, load_artifact
from iamatlas_mahalanobis_scoring import MahalanobisHealthyHull
from iam_cellular_age_scoring import IAMCellularAge

H_MIN = {"stem_pluri": 0.9822, "stem_adult": 0.8737, "stromal": 0.86295,
         "progenitor": 0.852216, "cycling": 0.856055, "secretory": 0.843264,
         "immune": 0.838889, "terminal": 0.7728}
CLASSES = list(H_MIN.keys())

def tier(A, hm):
    if A is None or np.isnan(A): return "UNDETERMINED"
    if A < 0.95: return "SUPPRESSED"
    if A < 1.04: return "NORMAL"
    if A < 1.07: return "ELEVATED"
    if A < 1.10: return "WARBURG_TRANSITION"
    if A < 1.12: return "SIGNIFICANTLY_ELEVATED"
    return "BREACH"

def main():
    t0 = time.time()
    print("=" * 70)
    print("CPG-VAL-020 — Hannum aging anchor reproduction (FULL CHAIN, v2)")
    print("=" * 70)

    # --- Modules ---
    print("\n--- Init canonical modules ---")
    art_meta, ct_markers, ct_to_class, h_min_by_class = load_artifact(str(MARKERS))
    print(f"  115-cell markers: {len(ct_markers)} celltypes, H_min(immune)={h_min_by_class['immune']:.6f}")

    print("  Loading Walther IAM Deconvolver...")
    walther = WaltherIAMDeconvolver(
        matrix_path=ATLAS_CSV,
        celltype_class_map=str(CT_TO_CLASS) if CT_TO_CLASS.exists() else ct_to_class,
        verbose=True)

    mah_hull = MahalanobisHealthyHull(str(HC_REF))
    print(f"  Mahalanobis HC: {mah_hull.n_features} features, n=601 pooled")

    cellage = IAMCellularAge(ref_matrix_path=str(AGE_REF), markers_artifact_path=str(MARKERS))
    print(f"  IAMCellularAge: 80-cell baseline, {sum(len(v) for v in cellage.class_markers.values())} class-marker CpGs")

    # --- Build union of needed CpGs (memory pre-flight) ---
    needed = set()
    for cpg in walther.class_ref: needed.add(cpg)
    for cpg in walther.celltype_ref: needed.add(cpg)
    for marks in ct_markers.values(): needed.update(marks)
    for cls_marks in cellage.class_markers.values(): needed.update(cls_marks)
    print(f"\n  Union of needed CpGs: {len(needed):,} (vs 473,034 full atlas → {len(needed)/473034*100:.1f}%)")

    # --- Load cohort, subset, free full array ---
    print("\n--- Load Hannum GSE40279 + subset to needed CpGs ---")
    t_load = time.time()
    npz = np.load("/tmp/geo_downloads/GSE40279_beta_matrix.npz", allow_pickle=True)
    full_betas = npz["beta"]   # float32, 473034 × 656
    full_cpgs = npz["cpgs"].tolist()
    print(f"  Full β loaded: shape={full_betas.shape}, dtype={full_betas.dtype}")

    # Build subset
    cpg_to_idx = {c: i for i, c in enumerate(full_cpgs)}
    needed_in_cohort = [c for c in needed if c in cpg_to_idx]
    subset_idx = np.array([cpg_to_idx[c] for c in needed_in_cohort])
    subset_betas = full_betas[subset_idx, :].copy()
    print(f"  Subset to needed CpGs: {subset_betas.shape}, dtype={subset_betas.dtype}, "
          f"size={subset_betas.nbytes/1e6:.0f} MB")

    # Free full array
    del full_betas, full_cpgs, cpg_to_idx, npz
    gc.collect()
    print(f"  Freed full β; load+subset elapsed: {time.time()-t_load:.0f}s")

    # --- Cohort meta ---
    meta = pd.read_csv("/tmp/geo_downloads/GSE40279_sample_meta.csv")
    meta["age"] = pd.to_numeric(meta["age (y)"], errors="coerce")
    meta["sex_at_birth"] = meta["gender"].map({"M": "M", "F": "F"}).fillna("F")
    meta["plate"] = meta["plate"].astype(str)
    n_samples = subset_betas.shape[1]
    assert len(meta) == n_samples, f"meta {len(meta)} != samples {n_samples}"
    print(f"  Meta: n={n_samples}, age {meta['age'].min():.0f}-{meta['age'].max():.0f}, "
          f"M={int((meta.sex_at_birth=='M').sum())} F={int((meta.sex_at_birth=='F').sum())}")

    # --- Per-sample chain ---
    print(f"\n--- Per-sample full chain (n={n_samples}) ---")
    t_chain = time.time()
    per_sample_rows = []
    per_celltype_rows = []
    chain_status = {}

    # Save one example for HEALPix
    healpix_example = None

    for j in range(n_samples):
        gsm = meta.iloc[j]["gsm"]
        chrono = meta.iloc[j]["age"]
        sex = meta.iloc[j]["sex_at_birth"]
        plate = meta.iloc[j]["plate"]

        # Build customer_betas dict (only valid β)
        col = subset_betas[:, j]
        customer_betas = {}
        for i, c in enumerate(needed_in_cohort):
            v = col[i]
            if not np.isnan(v) and 0.0 < v < 1.0:
                customer_betas[c] = float(v)

        # Stage 2: Walther
        decon = walther.deconvolve(customer_betas, refine_celltypes=True)
        status = decon.status
        chain_status[status] = chain_status.get(status, 0) + 1
        class_fracs = getattr(decon, "class_fractions", None) or {}
        ct_fracs = getattr(decon, "celltype_fractions", None) or {}

        # Stage 4: 115-cell A-scoring
        ct_results = score_per_celltype(customer_betas, ct_markers, ct_to_class, h_min_by_class)
        # Class-level via pooled celltype markers
        class_markers = {cls: sorted({c for ct, mks in ct_markers.items()
                                       for c in mks if ct_to_class.get(ct) == cls})
                         for cls in CLASSES}
        class_results = score_per_class(customer_betas, class_markers, h_min_by_class)

        # Stage 5: Mahalanobis
        ct_ascores = {ct: r["A"] for ct, r in ct_results.items() if r.get("A") is not None}
        mah_result = mah_hull.score(ct_ascores)

        # Stage 6: Cellular age
        cellage_result = cellage.score_patient(
            beta_dict=customer_betas,
            chronological_age=chrono if not np.isnan(chrono) else None,
            patient_id=gsm)
        # Extract per-class ages (canonical field per IAMCellularAge module)
        per_class_ages = getattr(cellage_result, "cellular_age_per_class", {}) or {}

        immune_cellage = per_class_ages.get("immune", np.nan) if isinstance(per_class_ages, dict) else np.nan
        if isinstance(immune_cellage, dict):
            immune_cellage = immune_cellage.get("cellular_age", immune_cellage.get("age", np.nan))
        try: immune_cellage = float(immune_cellage)
        except Exception: immune_cellage = np.nan
        immune_delta = (immune_cellage - chrono) if (not np.isnan(immune_cellage) and not np.isnan(chrono)) else np.nan

        # Stage 7: tier per class
        tier_per_cls = {cls: tier(class_results.get(cls, {}).get("A"), H_MIN[cls]) for cls in CLASSES}

        row = {
            "gsm": gsm, "arm": "hc", "cohort": "GSE40279",
            "age": chrono, "sex_at_birth": sex, "plate": plate,
            "ethnicity": meta.iloc[j].get("ethnicity", ""),
            "walther_status": status,
        }
        for cls in CLASSES:
            row[f"A_{cls}"] = class_results.get(cls, {}).get("A")
            row[f"tier_{cls}"] = tier_per_cls[cls]
        row["mahalanobis_d"] = mah_result.get("mahalanobis_distance")
        row["mahalanobis_status"] = mah_result.get("status")
        row["n_features_used"] = mah_result.get("n_features_used")
        row["immune_cellular_age"] = immune_cellage if not np.isnan(immune_cellage) else None
        row["immune_age_delta"] = immune_delta if not np.isnan(immune_delta) else None
        if isinstance(class_fracs, dict):
            row["frac_immune_class"] = class_fracs.get("immune")
        per_sample_rows.append(row)

        # Per-celltype (long)
        for ct in ct_markers.keys():
            r = ct_results.get(ct, {})
            per_celltype_rows.append({
                "gsm": gsm, "celltype": ct,
                "class": ct_to_class.get(ct, "unknown"),
                "A_score": r.get("A"),
                "n_markers": r.get("n_markers_matched"),
                "coverage": r.get("coverage"),
                "confidence": r.get("confidence"),
            })

        # HEALPix example: middle-aged HC sample (capture BEFORE del)
        if healpix_example is None and 50 <= chrono <= 65 and len(customer_betas) > 5000:
            healpix_example = {"gsm": gsm, "age": float(chrono), "betas": dict(customer_betas)}
        # Free customer_betas (per-sample memory hygiene)
        del customer_betas

        if (j + 1) % 50 == 0:
            mem_free = round(int(__import__('os').popen('free -m').read().split()[9]) / 1024, 1)
            elapsed = time.time() - t_chain
            rate = (j + 1) / elapsed
            eta = (n_samples - j - 1) / rate
            print(f"  {j+1}/{n_samples} ({elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s, free {mem_free}GB)")

    elapsed_chain = time.time() - t_chain
    print(f"\n  Chain complete: {n_samples} samples in {elapsed_chain/60:.1f} min")
    print(f"  Walther status: {chain_status}")

    # --- Save outputs ---
    per_sample = pd.DataFrame(per_sample_rows)
    per_ct = pd.DataFrame(per_celltype_rows)
    per_sample.to_csv("CPG_VAL_020_per_sample.csv", index=False)
    per_ct.to_csv("GSE40279_115celltype_ascores.csv", index=False)
    print(f"\n  per_sample.csv: {len(per_sample)} rows × {len(per_sample.columns)} cols")
    print(f"  115celltype_ascores.csv: {len(per_ct)} rows (long)")

    # --- Headline ---
    df = per_sample.dropna(subset=["immune_cellular_age", "age"])
    if len(df) > 10:
        r_imm, p_imm = stats.pearsonr(df["immune_cellular_age"], df["age"])
        spearman_r, _ = stats.spearmanr(df["immune_cellular_age"], df["age"])
        mae = float(np.mean(np.abs(df["immune_cellular_age"] - df["age"])))
        slope, intercept, _, _, _ = stats.linregress(df["age"], df["immune_cellular_age"])
        print(f"\n  HEADLINE: r(immune_cellular_age, chrono_age) = {r_imm:.4f} (p={p_imm:.3e}), "
              f"MAE={mae:.2f}yr, slope={slope:.4f}")
    else:
        r_imm, p_imm, spearman_r, mae, slope, intercept = (None,)*6
        print(f"\n  HEADLINE: insufficient data (n_clean={len(df)})")

    mah_v = per_sample["mahalanobis_d"].dropna()
    delt = per_sample["immune_age_delta"].dropna()
    headline = {
        "cohort": "GSE40279 Hannum 2013",
        "n_samples": int(len(per_sample)),
        "age_range": [int(per_sample["age"].min()), int(per_sample["age"].max())],
        "walther_status": chain_status,
        "headline": {
            "pearson_r_immune_cellular_age_vs_chrono": float(r_imm) if r_imm is not None else None,
            "p_value": float(p_imm) if p_imm is not None else None,
            "spearman_r": float(spearman_r) if spearman_r is not None else None,
            "MAE_years": float(mae) if mae is not None else None,
            "linear_slope": float(slope) if slope is not None else None,
            "linear_intercept": float(intercept) if intercept is not None else None,
            "prebuild_VAL_006_anchor_r": 0.9999,
            "VAL_020_vs_VAL_006_concordance_pct": float((1 - abs(r_imm - 0.9999)/0.9999) * 100) if r_imm is not None else None,
        },
        "mahalanobis": {
            "n_valid": int(len(mah_v)),
            "mean": float(mah_v.mean()) if len(mah_v) else None,
            "median": float(mah_v.median()) if len(mah_v) else None,
            "max": float(mah_v.max()) if len(mah_v) else None,
            "p25": float(mah_v.quantile(0.25)) if len(mah_v) else None,
            "p75": float(mah_v.quantile(0.75)) if len(mah_v) else None,
            "n_route_A_fire_d_ge_2": int((mah_v >= 2.0).sum()),
        },
        "tier_distribution_immune": dict(per_sample["tier_immune"].value_counts()),
        "tier_distribution_all_classes": {cls: dict(per_sample[f"tier_{cls}"].value_counts()) for cls in CLASSES},
        "immune_age_delta": {
            "n_valid": int(len(delt)),
            "mean_years": float(delt.mean()) if len(delt) else None,
            "sd_years": float(delt.std()) if len(delt) else None,
            "min_years": float(delt.min()) if len(delt) else None,
            "max_years": float(delt.max()) if len(delt) else None,
        },
        "modules_used": {
            "Stage_2_Walther": "WaltherIAMDeconvolver.deconvolve(refine_celltypes=True) against IAMAtlas REBUILD",
            "Stage_4_A_scoring": "score_per_celltype + score_per_class against iamatlas_celltype_markers_v0_2.json (115 celltypes × 100 markers)",
            "Stage_5_Mahalanobis": "MahalanobisHealthyHull(n=601 HC pooled GSE51057+GSE51032 in 115-cell A-score feature space)",
            "Stage_6_Cellular_Age": "IAMCellularAge.score_patient β_mean inversion against 80-cell baseline",
            "Stage_7_Tiers": "6-tier physics-derived v1.2 (SUPPRESSED/NORMAL/ELEVATED/WARBURG_TRANSITION/SIGNIFICANTLY_ELEVATED/BREACH)",
        },
        "frozen_anchors": {
            "H_min_immune": h_min_by_class["immune"],
            "HC_reference_n": 601,
            "atlas_csv_size_bytes": 605124914,
        },
    }
    with open("VAL_020_headline.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)
    print(f"  Saved VAL_020_headline.json")

    if healpix_example is not None:
        with open("healpix_example_sample.json", "w") as f:
            json.dump(healpix_example, f)
        print(f"  Saved healpix_example_sample.json (gsm={healpix_example['gsm']}, age={healpix_example['age']:.0f}, n_β={len(healpix_example['betas'])})")

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
