#!/usr/bin/env python3
"""CPG-VAL-018 sealed runner — Menarche-age effect on female immune architecture.

Pivoted from original HRT scope (HRT field absent in available cohorts).
Tests whether age at menarche predicts adult-life A_immune in HC women,
after controlling for chronological age, across GSE51057 + GSE51032.
"""
import json, hashlib, time, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/home/claude/IAM-Validation")
HERE = REPO / "Biological_Physics/validation_runs/CPG_VAL_018_Menarche_Immune"
MARKERS = REPO / "Biological_Physics/atlas_vault/walther_clinical_runtime/Celltype_Marker/iamatlas_celltype_markers_v0_2.json"

ASCORES_57 = REPO / "Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv"
ASCORES_32 = REPO / "Biological_Physics/validation_runs/foundation_cohort/GSE51032_115celltype_ascores.csv"
META_57 = REPO / "Biological_Physics/validation_runs/breast_epic_cohorts/GSE51057_EPIC_Italy/GSE51057_clinical_metadata.json"
META_32 = REPO / "Biological_Physics/validation_runs/breast_epic_cohorts/GSE51032_EPIC_Italy/GSE51032_clinical_metadata.json"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()

def partial_corr(x, y, z):
    """Partial correlation of x and y given z (single covariate)."""
    rxy, _ = stats.pearsonr(x, y)
    rxz, _ = stats.pearsonr(x, z)
    ryz, _ = stats.pearsonr(y, z)
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom == 0: return float('nan'), float('nan')
    r_partial = (rxy - rxz * ryz) / denom
    # Significance: degrees of freedom = n - 3
    n = len(x)
    if n <= 3: return r_partial, float('nan')
    t = r_partial * np.sqrt((n - 3) / (1 - r_partial**2 + 1e-30))
    p = 2 * (1 - stats.t.cdf(abs(t), df=n-3))
    return float(r_partial), float(p)

def main():
    t0 = time.time()
    print("=" * 72); print("CPG-VAL-018 — Menarche-age effect on female immune A-score"); print("=" * 72)
    
    ct2class = json.loads(MARKERS.read_text())['celltype_to_class']
    classes = sorted(set(ct2class.values()))

    def load_cohort(ascores_path, meta_path, cohort_label):
        df = pd.read_csv(ascores_path)
        with open(meta_path) as f: meta = pd.DataFrame(json.load(f))
        # Need: gsm, age, gender=F, menarche_age, arm=hc
        if 'menarche_age' not in meta.columns:
            print(f"  {meta_path.name}: no menarche_age field"); return None
        # Filter to HC women with valid menarche
        meta_hc = meta[(meta['arm']=='hc') & (meta['gender']=='F')].copy()
        meta_hc = meta_hc.dropna(subset=['age','menarche_age'])
        meta_hc = meta_hc[meta_hc['menarche_age'] > 0]  # exclude zeros/missing-coded
        # Per-class A
        for cls in classes:
            cts = [c for c in df.columns if c in ct2class and ct2class[c]==cls]
            df[f'A_{cls}'] = df[cts].mean(axis=1)
        # Join
        merged = df.merge(meta_hc[['gsm','age','menarche_age']], on='gsm', how='inner')
        merged['cohort'] = cohort_label
        return merged

    print("\n[1/4] Loading cohorts (HC women with menarche metadata)...")
    c57 = load_cohort(ASCORES_57, META_57, 'GSE51057')
    c32 = load_cohort(ASCORES_32, META_32, 'GSE51032')
    print(f"  GSE51057: n={len(c57)} HC women, menarche range [{c57['menarche_age'].min():.0f}, {c57['menarche_age'].max():.0f}], age range [{c57['age'].min():.0f}, {c57['age'].max():.0f}]")
    print(f"  GSE51032: n={len(c32)} HC women, menarche range [{c32['menarche_age'].min():.0f}, {c32['menarche_age'].max():.0f}], age range [{c32['age'].min():.0f}, {c32['age'].max():.0f}]")
    
    pooled = pd.concat([c57, c32], ignore_index=True)
    print(f"\n  POOLED: n={len(pooled)}")
    print(f"  Menarche distribution: mean={pooled['menarche_age'].mean():.2f}, median={pooled['menarche_age'].median():.1f}, range [{pooled['menarche_age'].min():.0f}, {pooled['menarche_age'].max():.0f}]")
    print(f"  Age distribution:      mean={pooled['age'].mean():.2f}, median={pooled['age'].median():.1f}, range [{pooled['age'].min():.0f}, {pooled['age'].max():.0f}]")

    # Per-cohort partial corr
    print("\n[2/4] Per-cohort partial correlation r(menarche, A_immune | age)...")
    per_cohort_results = {}
    for cn, cdf in [('GSE51057', c57), ('GSE51032', c32)]:
        r_p, p_p = partial_corr(cdf['menarche_age'].values, cdf['A_immune'].values, cdf['age'].values)
        r_raw, p_raw = stats.pearsonr(cdf['menarche_age'], cdf['A_immune'])
        # Linear regression A_immune ~ age + menarche
        from sklearn.linear_model import LinearRegression
        X = cdf[['age','menarche_age']].values
        y = cdf['A_immune'].values
        m = LinearRegression().fit(X, y)
        per_cohort_results[cn] = {
            'n': int(len(cdf)),
            'partial_r_menarche_given_age': round(r_p, 5),
            'partial_p': float(p_p),
            'raw_r_menarche_A_immune': round(float(r_raw), 5),
            'raw_p': float(p_raw),
            'slope_per_menarche_year_with_age_covariate': round(float(m.coef_[1]), 8),
            'slope_per_age_year_with_menarche_covariate': round(float(m.coef_[0]), 8),
        }
        print(f"  [{cn}] partial r(menarche, A_imm | age) = {r_p:+.4f} (p={p_p:.4f}); raw r = {r_raw:+.4f}; slope = {m.coef_[1]:+.5e} A_imm per menarche-year")

    # Pooled (cohort-fixed effect)
    print("\n[3/4] Pooled partial correlation with cohort fixed effect...")
    pooled['cohort_dummy'] = (pooled['cohort']=='GSE51057').astype(int)
    # Residualize A_immune and menarche_age against age + cohort_dummy
    from sklearn.linear_model import LinearRegression
    X_resid = pooled[['age','cohort_dummy']].values
    A_resid = pooled['A_immune'].values - LinearRegression().fit(X_resid, pooled['A_immune'].values).predict(X_resid)
    M_resid = pooled['menarche_age'].values - LinearRegression().fit(X_resid, pooled['menarche_age'].values).predict(X_resid)
    pooled_r, pooled_p = stats.pearsonr(M_resid, A_resid)
    # Full regression A_immune ~ age + cohort + menarche
    X_full = pooled[['age','cohort_dummy','menarche_age']].values
    full_m = LinearRegression().fit(X_full, pooled['A_immune'].values)
    print(f"  POOLED partial r(menarche, A_imm | age+cohort) = {pooled_r:+.4f}  p={pooled_p:.4f}  n={len(pooled)}")
    print(f"  Slope (A_immune per menarche-year, adjusted): {full_m.coef_[2]:+.5e}")

    # Per-class specificity
    print("\n[4/4] Per-class specificity check (partial r for all classes)...")
    per_class_pooled = {}
    for cls in classes:
        col = f'A_{cls}'
        vals = pooled[col].values
        mask = ~np.isnan(vals)
        if mask.sum() < 30:
            per_class_pooled[cls] = {'partial_r': None, 'partial_p': None, 'n_valid': int(mask.sum())}
            print(f"  [class] A_{cls:12s}  n_valid={mask.sum()} (insufficient — skipping)")
            continue
        X_resid_cls = X_resid[mask]
        cls_resid = vals[mask] - LinearRegression().fit(X_resid_cls, vals[mask]).predict(X_resid_cls)
        M_cls = M_resid[mask]
        r, p = stats.pearsonr(M_cls, cls_resid)
        per_class_pooled[cls] = {'partial_r': round(float(r), 5), 'partial_p': float(p), 'n_valid': int(mask.sum())}
        print(f"  [class] A_{cls:12s}  partial r = {r:+.4f}  p = {p:.4f}  n={mask.sum()}")

    # Pass conditions
    abs_pooled_r = abs(pooled_r)
    pass1 = (abs_pooled_r >= 0.10) and (pooled_p < 0.01)
    signs = [np.sign(v['partial_r_menarche_given_age']) for v in per_cohort_results.values()]
    pass2 = len(set(signs)) == 1
    abs_slope = abs(full_m.coef_[2])
    pass4_magnitude = abs_slope >= 0.002
    pass3 = pass1  # condition 3 (distinct from zero) is essentially condition 1
    valid_other_class_rs = [abs(v['partial_r']) for k,v in per_class_pooled.items() if k != 'immune' and v['partial_r'] is not None]
    pass5 = abs_pooled_r >= np.mean(valid_other_class_rs) if valid_other_class_rs else False

    if pass1 and pass2:
        overall = "PASS"
    elif (pass1 or pass2) and pass4_magnitude:
        overall = "DIRECTIONAL"
    else:
        overall = "NULL"

    # Save outputs
    HERE.mkdir(exist_ok=True)
    # Per-sample
    out_cols = ['cohort','gsm','age','menarche_age'] + [f'A_{c}' for c in classes]
    per_sample = pooled[out_cols].copy()
    for c in per_sample.select_dtypes(include=[np.number]).columns:
        per_sample[c] = per_sample[c].round(5)
    per_sample.to_csv(HERE / "CPG_VAL_018_per_sample.csv", index=False)

    results = {
        "val_id": "CPG-VAL-018",
        "title": "Menarche-age effect on female immune architecture",
        "card": "Immune universal v1.0",
        "execution_date": "2026-06-07",
        "scope_pivot_note": "Pivoted from HRT scope (HRT field not in available cohort metadata); menarche-age tested instead as cleaner reproductive-axis variable.",
        "n_total_HC_women": int(len(pooled)),
        "outcome_code": overall,
        "pooled_partial_correlation": {
            "method": "partial Pearson r(menarche_age, A_immune | age + cohort)",
            "partial_r": round(float(pooled_r), 5),
            "partial_p": float(pooled_p),
            "slope_per_menarche_year": round(float(full_m.coef_[2]), 8),
            "slope_per_age_year": round(float(full_m.coef_[0]), 8),
            "n": int(len(pooled)),
        },
        "per_cohort": per_cohort_results,
        "per_class_specificity_pooled": per_class_pooled,
        "pass_conditions": {
            "condition_1_pooled_partial_significant": {
                "criterion": "|partial r| >= 0.10 AND p < 0.01",
                "observed_r": round(float(pooled_r), 5),
                "observed_p": float(pooled_p),
                "passed": bool(pass1),
            },
            "condition_2_sign_concordance_across_cohorts": {
                "criterion": "Same-sign in both cohorts",
                "signs": {k: int(np.sign(v['partial_r_menarche_given_age'])) for k,v in per_cohort_results.items()},
                "passed": bool(pass2),
            },
            "condition_3_distinct_from_zero": {
                "criterion": "Effect direction-specific (covered by condition 1)",
                "passed": bool(pass3),
            },
            "condition_4_magnitude_per_menarche_year": {
                "criterion": "|ΔA_immune / Δ menarche year| >= 0.002",
                "observed_abs_slope": round(float(abs_slope), 8),
                "passed": bool(pass4_magnitude),
            },
            "condition_5_specificity_immune_vs_other_classes": {
                "criterion": "|partial_r_immune| >= mean(|partial_r_other_classes|)",
                "observed_abs_partial_r_immune": round(float(abs_pooled_r), 5),
                "mean_abs_partial_r_other_classes": round(float(np.mean(valid_other_class_rs)), 5) if valid_other_class_rs else None,
                "n_valid_other_classes": len(valid_other_class_rs),
                "passed": bool(pass5),
            },
        },
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str))

    (HERE / "stratified_results.json").write_text(json.dumps({
        "per_cohort_partial_correlations": per_cohort_results,
        "per_class_pooled_partial_correlations": per_class_pooled,
        "menarche_age_distribution": {
            "mean": round(float(pooled['menarche_age'].mean()), 2),
            "median": round(float(pooled['menarche_age'].median()), 1),
            "std": round(float(pooled['menarche_age'].std()), 2),
            "range": [int(pooled['menarche_age'].min()), int(pooled['menarche_age'].max())],
        }
    }, indent=2))

    # Null: shuffle menarche within cohort
    print("\n  Null: shuffle menarche_age within cohort (1000 perms)...")
    np.random.seed(42)
    null_rs = []
    for _ in range(1000):
        shuf = pooled.copy()
        shuf['menarche_age'] = shuf.groupby('cohort')['menarche_age'].transform(np.random.permutation)
        M_shuf_resid = shuf['menarche_age'].values - LinearRegression().fit(X_resid, shuf['menarche_age'].values).predict(X_resid)
        r_null, _ = stats.pearsonr(M_shuf_resid, A_resid)
        null_rs.append(r_null)
    null_rs = np.array(null_rs)
    p_null = float(np.mean(np.abs(null_rs) >= abs(pooled_r)))
    print(f"  Null mean r: {null_rs.mean():+.4f}  std: {null_rs.std():.4f}  abs>obs: p={p_null:.4f}")

    (HERE / "null_results.json").write_text(json.dumps({
        "null_type": "shuffle_menarche_age_within_cohort",
        "n_perms": 1000, "rng_seed": 42,
        "observed_pooled_partial_r": round(float(pooled_r), 5),
        "null_mean": round(float(null_rs.mean()), 5),
        "null_std": round(float(null_rs.std()), 5),
        "p_two_sided_abs": p_null,
    }, indent=2))

    (HERE / "cohort_manifest.json").write_text(json.dumps({
        "cohorts": [
            {"id": "GSE51057_EPIC_Italy", "n_HC_women_with_menarche": int(len(c57)),
             "ascores_sha256": sha256_file(ASCORES_57), "meta_sha256": sha256_file(META_57)},
            {"id": "GSE51032_EPIC_Italy", "n_HC_women_with_menarche": int(len(c32)),
             "ascores_sha256": sha256_file(ASCORES_32), "meta_sha256": sha256_file(META_32)},
        ],
        "n_total_pooled": int(len(pooled)),
        "canonical_markers_sha256": sha256_file(MARKERS),
    }, indent=2))

    print(f"\nPass-1 (|r|>=0.10, p<0.01): {pass1}")
    print(f"Pass-2 (sign concordance): {pass2}")
    print(f"Pass-3 (distinct from zero): {pass3}")
    print(f"Pass-4 (magnitude >= 0.002/yr): {pass4_magnitude}")
    print(f"Pass-5 (immune specificity): {pass5}")
    print(f"OUTCOME: {overall}")
    print(f"Elapsed: {time.time()-t0:.1f}s")
    return overall

if __name__ == "__main__":
    sys.exit(0 if main() in ("PASS","DIRECTIONAL") else 1)
