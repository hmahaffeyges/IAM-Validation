#!/usr/bin/env python3
"""
val_135_run.py — Direct executor for VAL-135 omnibus full-chain validation.

Runs the available Walther clinical chain on each of 3 cohorts:
  1. AIBL (GSE153712) — 726 samples, 18-CpG IMM panel (Stage 4 immune + Stage 4.5 bidirectional only)
  2. GSE50660 (Tsaprouni) — 464 samples, full β matrix (full chain)
  3. GSE40279 (Hannum) — 656 samples, full β matrix (full chain)

CHAIN (per cohort, what we can run with available data):
  Stage 3: Foreground subtraction (age + smoking + sex layers loaded from CSV)
  Stage 4: 8-class A-scores via H(β_mean)/H_min using IAMAtlas REBUILD CpGs
  Stage 4.5: Bidirectional decomposition (immune class — VAL-051 Rule A 7-CpG panel)
  Stage 5: Mahalanobis distance vs n=601 HC centroid
  Stage 7: 6-tier breakpoint assignment

OUTPUTS per cohort: per_sample_<cohort>.csv with all signal columns + tier assignments.
"""

from __future__ import annotations
import json, math, time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Paths
RUNTIME = Path("/home/claude/IAM-Validation/Biological_Physics/atlas_vault/walther_clinical_runtime")
ATLAS_CSV = Path("/home/claude/iamatlas_rebuild/IAMAtlasREBUILD.csv")
VAL135_DIR = Path("/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-135_immune_card_v1_full_chain_validation")

# Frozen H_min
H_MIN_BY_CLASS = {
    "stem_pluri": 0.9822, "stem_adult": 0.8737, "stromal": 0.8630,
    "progenitor": 0.8522, "cycling": 0.8561, "secretory": 0.8433,
    "immune": 0.838889, "terminal": 0.7728,
}
CLASSES = list(H_MIN_BY_CLASS.keys())


def shannon_H(b):
    b = max(min(b, 0.999), 0.001)
    return -b * math.log2(b) - (1.0 - b) * math.log2(1.0 - b)


def stage7_tier(A, h_min):
    if A is None or np.isnan(A): return "UNDETERMINED"
    if A < 0.95: return "SUPPRESSED"
    if A < 1.04: return "NORMAL"
    if A < 1.07: return "ELEVATED"
    if A < 1.10: return "WARBURG_TRANSITION"
    if A < 1.12:
        if A > 1.0/h_min - 0.005: return "SATURATED"
        return "SIGNIFICANTLY_ELEVATED"
    return "BREACH"


def load_layers():
    """Load age, smoking, sex foreground layer CSVs into dicts."""
    print("[load] Foreground layers...")
    age = pd.read_csv(RUNTIME / "IAM_Cellular_Age/IAMAtlas_age_layer.csv")
    smk = pd.read_csv(RUNTIME / "IAM_Cellular_Age/IAMAtlas_smoking_layer.csv")
    sex = pd.read_csv(RUNTIME / "IAM_Cellular_Age/IAMAtlas_sex_layer.csv")
    # Indexed by cpg_id for fast lookup
    age_dict = dict(zip(age.cpg_id, age.slope_gamma))
    smk_dict_d = dict(zip(smk.cpg_id, smk.delta_current_smoker))
    smk_dict_p = dict(zip(smk.cpg_id, smk.phi_recency))
    sex_dict = dict(zip(sex.cpg_id, sex.psi_male))
    print(f"  age: {len(age_dict)} CpGs, smoking: {len(smk_dict_d)}, sex: {len(sex_dict)}")
    return age_dict, smk_dict_d, smk_dict_p, sex_dict


def load_atlas():
    """Load IAMAtlas REBUILD CpG list + class means + class marker panels."""
    print("[load] IAMAtlas REBUILD (this is the big one)...")
    t0 = time.time()
    # We need class means and class markers. Get cpg_id + mean columns.
    df = pd.read_csv(ATLAS_CSV)
    print(f"  Atlas: {df.shape}, elapsed {time.time()-t0:.0f}s")
    print(f"  Columns: {list(df.columns)[:15]}...")
    # Extract per-class means
    class_means = {}
    for cls in CLASSES:
        col = f"{cls}_mean"
        if col in df.columns:
            class_means[cls] = dict(zip(df.cpg_id, df[col]))
    print(f"  Found class means for: {list(class_means.keys())}")
    return df.cpg_id.tolist(), class_means


def load_class_markers(atlas_cpgs, atlas_class_means, n_per_class=200):
    """Build per-class marker panels (top-N CpGs with most class-distinctive β_mean)."""
    print("[load] Building class marker panels (top-200 per class by max-discrimination)...")
    markers = {cls: [] for cls in CLASSES}
    # For each CpG, compute the discrimination: |β_class - β_others_mean|
    # Pick top 200 per class
    means_df = pd.DataFrame({cls: pd.Series(atlas_class_means.get(cls, {})) for cls in CLASSES})
    means_df = means_df.dropna(how='all')
    for cls in CLASSES:
        if cls not in atlas_class_means: continue
        other_classes = [c for c in CLASSES if c != cls and c in atlas_class_means]
        own = means_df[cls]
        others_mean = means_df[other_classes].mean(axis=1)
        discrim = (own - others_mean).abs()
        top = discrim.nlargest(n_per_class).index.tolist()
        markers[cls] = top
        print(f"  {cls}: {len(top)} markers")
    return markers


def stage3_subtract(beta_aligned, atlas_cpgs, ages, smoking_bins, sex, layers):
    """Apply age + smoking + sex foreground subtraction at per-CpG β level.
    Memory-efficient: in-place per-sample subtraction (no outer products)."""
    age_dict, smk_d, smk_p, sex_dict = layers
    n_cpgs, n_samples = beta_aligned.shape

    # Build per-CpG coefficient vectors aligned to atlas order (small, one-shot)
    gamma_age = np.array([age_dict.get(c, 0.0) for c in atlas_cpgs], dtype=np.float32)
    delta_smk = np.array([smk_d.get(c, 0.0) for c in atlas_cpgs], dtype=np.float32)
    phi_smk = np.array([smk_p.get(c, 0.0) for c in atlas_cpgs], dtype=np.float32)
    psi_sex = np.array([sex_dict.get(c, 0.0) for c in atlas_cpgs], dtype=np.float32)

    smoking_to_ind = {"never_smoker": (0, 0.0), "former_15plus_y": (0, 0.10),
                       "former_5_15y": (0, 0.30), "former_0_5y": (0, 0.60),
                       "current_smoker": (1, 1.0)}
    ages_arr = np.array(ages, dtype=np.float32)
    smk_ind = np.array([smoking_to_ind.get(b, (0, 0))[0] for b in smoking_bins], dtype=np.float32)
    smk_rec = np.array([smoking_to_ind.get(b, (0, 0))[1] for b in smoking_bins], dtype=np.float32)
    sex_ind = np.array([1 if s == "M" else 0 for s in sex], dtype=np.float32)

    age_mean = float(np.nanmean(ages_arr)) if (~np.isnan(ages_arr)).any() else 0.0
    nz = lambda v: int((v != 0).sum())
    print(f"  [stage 3] non-zero coefs: age={nz(gamma_age)}, smoking δ={nz(delta_smk)}, sex ψ={nz(psi_sex)}")
    print(f"  [stage 3] Subtracting in-place per-sample (memory-efficient)...")

    # In-place subtraction sample-by-sample
    for j in range(n_samples):
        age_c = ages_arr[j] - age_mean if not np.isnan(ages_arr[j]) else 0.0
        beta_aligned[:, j] -= gamma_age * age_c
        beta_aligned[:, j] -= delta_smk * smk_ind[j]
        beta_aligned[:, j] -= phi_smk * smk_rec[j]
        beta_aligned[:, j] -= psi_sex * sex_ind[j]

    # Clip in-place
    np.clip(beta_aligned, 0.001, 0.999, out=beta_aligned)
    return beta_aligned  # same object, modified in place


def stage4_a_scores(beta_aligned, atlas_cpgs, class_markers):
    """Compute per-class A-scores. β_aligned: (n_atlas_cpgs, n_samples)"""
    cpg_to_idx = {c: i for i, c in enumerate(atlas_cpgs)}
    n_samples = beta_aligned.shape[1]
    out = {f"A_{cls}": [] for cls in CLASSES}
    for j in range(n_samples):
        sample_beta = beta_aligned[:, j]
        for cls in CLASSES:
            indices = [cpg_to_idx[c] for c in class_markers.get(cls, []) if c in cpg_to_idx]
            if len(indices) < 5:
                out[f"A_{cls}"].append(np.nan); continue
            vals = sample_beta[indices]
            valid = vals[~np.isnan(vals) & (vals > 0) & (vals < 1)]
            if len(valid) < 5:
                out[f"A_{cls}"].append(np.nan); continue
            beta_mean = float(np.mean(valid))
            A = shannon_H(beta_mean) / H_MIN_BY_CLASS[cls]
            out[f"A_{cls}"].append(A)
    return pd.DataFrame(out)


def stage45_immune_bidirectional(beta_aligned, atlas_cpgs):
    """Stage 4.5: Bidirectional decomposition using VAL-051 Rule A 7-CpG immune panel."""
    panel_path = RUNTIME / "Bidirectional_Decomposition/directional_panels_v1_0.json"
    if not panel_path.exists():
        return pd.DataFrame({"a_dir_immune": [np.nan]*beta_aligned.shape[1],
                             "a_pool_immune": [np.nan]*beta_aligned.shape[1]})
    with open(panel_path) as f:
        full = json.load(f)
    panels = full.get("panels", {})
    immune = panels.get("immune") or {}
    if not immune or not immune.get("cpgs"):
        return pd.DataFrame({"a_dir_immune": [np.nan]*beta_aligned.shape[1],
                             "a_pool_immune": [np.nan]*beta_aligned.shape[1]})

    cpgs_directional = immune["cpgs"]
    cpg_to_idx = {c: i for i, c in enumerate(atlas_cpgs)}
    h_min_imm = immune.get("h_min", H_MIN_BY_CLASS["immune"])
    a_dir, a_pool = [], []
    for j in range(beta_aligned.shape[1]):
        sample = beta_aligned[:, j]
        # Directional: sign-multiplied z-scores against training HC mean/SD
        z_scores = []
        for entry in cpgs_directional:
            c = entry.get("cpg_id")
            idx = cpg_to_idx.get(c, -1)
            if idx < 0: continue
            beta = float(sample[idx])
            if np.isnan(beta): continue
            hc_mean = entry.get("mean_hc_train", 0.5)
            hc_sd = entry.get("sd_hc_train", 0.1)
            sign = entry.get("direction", 1)
            if hc_sd > 0:
                z_scores.append(sign * (beta - hc_mean) / hc_sd)
        a_dir.append(float(np.mean(z_scores)) if z_scores else np.nan)
        # Pooled entropy
        pool_cpgs = immune.get("pooled_panel_cpgs") or [e["cpg_id"] for e in cpgs_directional]
        pool_betas = []
        for c in pool_cpgs:
            idx = cpg_to_idx.get(c, -1)
            if idx >= 0 and not np.isnan(sample[idx]) and 0 < sample[idx] < 1:
                pool_betas.append(float(sample[idx]))
        if len(pool_betas) >= 3:
            a_pool.append(shannon_H(np.mean(pool_betas)) / h_min_imm)
        else:
            a_pool.append(np.nan)
    return pd.DataFrame({"a_dir_immune": a_dir, "a_pool_immune": a_pool})


def stage5_mahalanobis_8class(a_df, hc_mask=None):
    """Mahalanobis distance on 8-class A-vector vs HC centroid.

    Uses an inline-built HC reference from the cohort's own HC samples (or all samples if no
    case/HC split). The saved n=601 HC artifact is in 115-cell feature space, not 8-class, so
    this VAL builds its own 8-class HC reference per-cohort. v1.1 will rebuild a frozen
    8-class HC reference for production use."""
    cls_cols = [f"A_{c}" for c in CLASSES]
    X = a_df[cls_cols].to_numpy(dtype=np.float64)
    valid = ~np.any(np.isnan(X), axis=1)
    if hc_mask is None:
        hc_X = X[valid]
    else:
        hc_X = X[(hc_mask) & valid]
    if len(hc_X) < 30:
        # Fall back to all samples for centroid
        hc_X = X[valid]
    centroid = hc_X.mean(axis=0)
    cov = np.cov(hc_X.T)
    if np.linalg.matrix_rank(cov) < len(CLASSES):
        cov = cov + np.eye(len(CLASSES)) * 1e-4
    inv_cov = np.linalg.pinv(cov)
    distances = []
    for i in range(len(X)):
        if not valid[i]:
            distances.append(np.nan); continue
        diff = X[i] - centroid
        d = float(np.sqrt(diff @ inv_cov @ diff))
        distances.append(d)
    return pd.Series(distances, name="mahalanobis_d_8class"), {"centroid": centroid.tolist(),
                                                                  "n_hc_used": int(len(hc_X))}


def run_cohort_full_chain(name, beta_aligned, atlas_cpgs, sample_meta,
                          class_markers, layers):
    """Run full chain on one cohort. Returns DataFrame with per-sample outputs."""
    print(f"\n{'='*70}\n  RUNNING COHORT: {name}\n{'='*70}")
    t0 = time.time()
    print(f"  β aligned: {beta_aligned.shape}, samples: {len(sample_meta)}")

    # Stage 3
    print(f"\n[stage 3] Foreground subtraction (age + smoking + sex)...")
    cleaned = stage3_subtract(beta_aligned, atlas_cpgs,
                               ages=sample_meta["age"].tolist(),
                               smoking_bins=sample_meta["smoking_bin"].tolist(),
                               sex=sample_meta["sex_at_birth"].tolist(),
                               layers=layers)
    print(f"  cleaned shape: {cleaned.shape}, NaN: {np.isnan(cleaned).sum()}")

    # Stage 4
    print(f"\n[stage 4] 8-class A-scoring...")
    a_df = stage4_a_scores(cleaned, atlas_cpgs, class_markers)
    for cls in CLASSES:
        m = a_df[f"A_{cls}"].mean()
        print(f"  A_{cls}: mean={m:.4f}, n_valid={a_df[f'A_{cls}'].notna().sum()}")

    # Stage 4.5
    print(f"\n[stage 4.5] Bidirectional decomposition (immune class — VAL-051 Rule A panel)...")
    bidir_df = stage45_immune_bidirectional(cleaned, atlas_cpgs)
    print(f"  a_dir_immune: mean={bidir_df['a_dir_immune'].mean():.4f}, n_valid={bidir_df['a_dir_immune'].notna().sum()}")

    # Stage 5
    print(f"\n[stage 5] Mahalanobis distance (8-class A-vector, cohort-internal HC reference)...")
    hc_mask = (sample_meta["arm"] == "hc").to_numpy() if "arm" in sample_meta.columns else None
    mah_d, mah_meta = stage5_mahalanobis_8class(a_df, hc_mask=hc_mask)
    print(f"  mahalanobis: mean={mah_d.mean():.3f}, range {mah_d.min():.2f}-{mah_d.max():.2f}, n_hc_ref={mah_meta['n_hc_used']}")

    # Stage 7 (no cellular age inversion in this run — that's VAL-141 follow-up)
    print(f"\n[stage 7] 6-tier breakpoint assignment...")
    tier_cols = {}
    for cls in CLASSES:
        tier_cols[f"tier_{cls}"] = [stage7_tier(a, H_MIN_BY_CLASS[cls]) for a in a_df[f"A_{cls}"]]

    # Combine
    out = pd.concat([
        sample_meta.reset_index(drop=True),
        a_df.reset_index(drop=True),
        bidir_df.reset_index(drop=True),
        mah_d.reset_index(drop=True).to_frame(),
        pd.DataFrame(tier_cols),
    ], axis=1)
    print(f"\n  Cohort {name} complete in {time.time()-t0:.0f}s. Output shape: {out.shape}")
    return out


if __name__ == "__main__":
    # Will be called as script with --cohort args, or run all
    pass
