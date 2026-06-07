#!/usr/bin/env python3
"""CPG-VAL-017 — Inflammaging in pooled hull HC."""
import json, hashlib, time, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/home/claude/IAM-Validation")
HERE = REPO / "Biological_Physics/validation_runs/CPG_VAL_017_Inflammaging_Pooled_HC"
MARKERS = REPO / "Biological_Physics/atlas_vault/walther_clinical_runtime/Celltype_Marker/iamatlas_celltype_markers_v0_2.json"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()

def main():
    t0 = time.time()
    print("=" * 72); print("CPG-VAL-017 — Inflammaging in pooled hull HC"); print("=" * 72)
    
    ct2class = json.loads(MARKERS.read_text())['celltype_to_class']
    
    # Load cohorts where age metadata is available
    pooled = []
    cohorts_used = []
    
    # 1. Hannum (long format A-scores + sample meta with age)
    print("\n[1/5] Loading Hannum + age...")
    long_df = pd.read_csv(REPO / "Biological_Physics/validation_runs/CPG_VAL_020_Immune_Hannum_full_chain/GSE40279_115celltype_ascores.csv")
    h_immune = long_df[long_df['class']=='immune'].groupby('gsm')['A_score'].mean().reset_index()
    h_immune.columns = ['gsm', 'A_immune']
    h_meta = pd.read_csv("/tmp/geo_downloads/GSE40279_sample_meta.csv").rename(columns={'age (y)': 'age'})
    h = h_immune.merge(h_meta[['gsm','age','gender']], on='gsm', how='inner')
    h['cohort'] = 'GSE40279_Hannum_US'
    h['platform'] = 'HM450'
    pooled.append(h[['cohort','platform','gsm','age','gender','A_immune']])
    cohorts_used.append(('GSE40279_Hannum_US', len(h)))
    print(f"  Hannum: n={len(h)}, age range [{h['age'].min()}, {h['age'].max()}]")

    # 2. Han Chinese
    print("\n[2/5] Loading Han Chinese + age...")
    hc_df = pd.read_csv(REPO / "Biological_Physics/validation_runs/hull_expansion_phase4_asian_GSE141682/GSE141682_HanChinese_HC_115celltype_ascores_PHASE4.csv")
    imm_cts = [c for c in hc_df.columns if c in ct2class and ct2class[c]=='immune']
    hc_df['A_immune'] = hc_df[imm_cts].mean(axis=1)
    hc_df['cohort'] = 'GSE141682_HanChinese'
    hc_df['platform'] = 'EPIC'
    pooled.append(hc_df[['cohort','platform','gsm','age','gender','A_immune']])
    cohorts_used.append(('GSE141682_HanChinese', len(hc_df)))
    print(f"  Han Chinese: n={len(hc_df)}, age range [{hc_df['age'].min()}, {hc_df['age'].max()}]")


    # 3b. Tsaprouni — load age from metadata
    print("\n[3b/5] Loading Tsaprouni + age (from separate metadata)...")
    ts_meta = pd.read_csv('/tmp/geo_downloads/GSE50660_sample_meta.csv')
    ts_ascores = pd.read_csv(REPO / "Biological_Physics/validation_runs/hull_expansion_phase2_GSE50660/GSE50660_115celltype_ascores.csv")
    sample_col = 'gsm' if 'gsm' in ts_ascores.columns else ts_ascores.columns[0]
    imm_cts = [c for c in ts_ascores.columns if c in ct2class and ct2class[c]=='immune']
    ts_ascores['A_immune'] = ts_ascores[imm_cts].mean(axis=1)
    ts = ts_ascores[[sample_col, 'A_immune']].merge(ts_meta[['gsm','age','gender']], left_on=sample_col, right_on='gsm', how='inner')
    if 'gsm_x' in ts.columns: ts = ts.rename(columns={'gsm_x':'gsm'}).drop(columns=['gsm_y'], errors='ignore')
    elif sample_col != 'gsm': ts = ts.rename(columns={sample_col:'gsm'})
    ts['cohort'] = 'GSE50660_Tsaprouni_UK'; ts['platform'] = 'HM450'
    pooled.append(ts[['cohort','platform','gsm','age','gender','A_immune']])
    cohorts_used.append(('GSE50660_Tsaprouni_UK', len(ts)))
    _ts_amin, _ts_amax = ts['age'].min(), ts['age'].max()
    print(f"  Tsaprouni: n={len(ts)}, age range [{_ts_amin}, {_ts_amax}]")

    # 3c. GSE51057 foundation
    print("\n[3c/5] Loading GSE51057 EPIC-Italy + age (HC only)...")
    f57 = pd.read_csv(REPO / "Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv")
    with open(REPO / "Biological_Physics/validation_runs/breast_epic_cohorts/GSE51057_EPIC_Italy/GSE51057_clinical_metadata.json") as fh:
        m57 = pd.DataFrame(json.load(fh))
    imm_cts = [c for c in f57.columns if c in ct2class and ct2class[c]=='immune']
    f57['A_immune'] = f57[imm_cts].mean(axis=1)
    f57_hc = f57[f57['arm']=='hc'].merge(m57[['gsm','age','gender']], on='gsm', how='inner')
    f57_hc['cohort'] = 'GSE51057_EPIC_Italy'; f57_hc['platform'] = 'HM450'
    f57_hc['age'] = f57_hc['age'].astype(int)
    pooled.append(f57_hc[['cohort','platform','gsm','age','gender','A_immune']])
    cohorts_used.append(('GSE51057_EPIC_Italy', len(f57_hc)))
    _57_amin, _57_amax = f57_hc['age'].min(), f57_hc['age'].max()
    print(f"  GSE51057 HC: n={len(f57_hc)}, age range [{_57_amin}, {_57_amax}]")

    # 3d. GSE51032 foundation
    print("\n[3d/5] Loading GSE51032 EPIC-Italy + age (HC only)...")
    f32 = pd.read_csv(REPO / "Biological_Physics/validation_runs/foundation_cohort/GSE51032_115celltype_ascores.csv")
    with open(REPO / "Biological_Physics/validation_runs/breast_epic_cohorts/GSE51032_EPIC_Italy/GSE51032_clinical_metadata.json") as fh:
        m32 = pd.DataFrame(json.load(fh))
    imm_cts = [c for c in f32.columns if c in ct2class and ct2class[c]=='immune']
    f32['A_immune'] = f32[imm_cts].mean(axis=1)
    f32_hc = f32[f32['arm']=='hc'].merge(m32[['gsm','age','gender']], on='gsm', how='inner') if 'age' in m32.columns else None
    if f32_hc is not None and len(f32_hc):
        f32_hc['cohort'] = 'GSE51032_EPIC_Italy'; f32_hc['platform'] = 'HM450'
        f32_hc['age'] = f32_hc['age'].astype(int)
        pooled.append(f32_hc[['cohort','platform','gsm','age','gender','A_immune']])
        cohorts_used.append(('GSE51032_EPIC_Italy', len(f32_hc)))
        _32_amin, _32_amax = f32_hc['age'].min(), f32_hc['age'].max()
        print(f"  GSE51032 HC: n={len(f32_hc)}, age range [{_32_amin}, {_32_amax}]")
    else:
        print(f"  GSE51032: no age field in clinical metadata; skipping")


    # 3. Tsaprouni — has age but in separate metadata file. Check (original v1 path superseded by 3b above)
    # Original [3/5] Tsaprouni-from-ascore-CSV path removed; superseded by 3b above

    # 4. Foundation cohorts (GSE51057 + GSE51032) — check if age field in their metadata
    # The foundation A-score CSVs likely don't have age; would need separate metadata join
    
    # Pool everything available
    df = pd.concat(pooled, ignore_index=True)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age','A_immune'])
    df['age'] = df['age'].astype(int)
    df['age_decade'] = (df['age']//10)*10
    print(f"\n[4/5] Pooled: n={len(df)}, age range [{df['age'].min()},{df['age'].max()}]")
    print(f"  By cohort: {df['cohort'].value_counts().to_dict()}")

    # Primary regression
    r, p = stats.pearsonr(df['age'], df['A_immune'])
    slope, intercept, _, p_slope, se = stats.linregress(df['age'], df['A_immune'])
    print(f"\n  Pooled A_immune vs age: r={r:+.4f} p={p:.2e} slope={slope:+.5e}/yr")

    # Per-cohort
    per_cohort = {}
    for cn in df['cohort'].unique():
        sub = df[df['cohort']==cn]
        if len(sub) < 10: continue
        r_c, p_c = stats.pearsonr(sub['age'], sub['A_immune'])
        sl, _, _, sl_p, _ = stats.linregress(sub['age'], sub['A_immune'])
        per_cohort[cn] = {
            "n": int(len(sub)),
            "age_range": [int(sub['age'].min()), int(sub['age'].max())],
            "pearson_r": round(float(r_c), 4),
            "pearson_p": float(p_c),
            "slope_per_year": round(float(sl), 8),
            "slope_p": float(sl_p),
        }
        print(f"    [{cn}] n={len(sub)} age=[{sub['age'].min()},{sub['age'].max()}] r={r_c:+.4f} p={p_c:.2e} slope={sl:+.5e}")
    
    # Decade-medians + late-life acceleration
    decade_stats = df.groupby('age_decade').agg(
        n=('gsm','count'),
        median_A_immune=('A_immune','median'),
        mean_A_immune=('A_immune','mean'),
    ).reset_index()
    decade_stats['age_decade'] = decade_stats['age_decade'].astype(int)
    for col in decade_stats.columns:
        if decade_stats[col].dtype == 'float64':
            decade_stats[col] = decade_stats[col].round(5)
    print(f"\n  Decade medians:")
    for _, row in decade_stats.iterrows():
        print(f"    age {int(row['age_decade']):3d}s  n={int(row['n']):4d}  median A_immune={row['median_A_immune']:.4f}")
    
    # Compute pre-50 vs post-70 slopes
    early = df[df['age']<50]
    late = df[df['age']>=70]
    if len(early)>10 and len(late)>10:
        slope_e, _, _, _, _ = stats.linregress(early['age'], early['A_immune'])
        slope_l, _, _, _, _ = stats.linregress(late['age'], late['A_immune'])
        accel_ratio = abs(slope_l) / abs(slope_e) if abs(slope_e) > 0 else float('nan')
        print(f"\n  Pre-50 slope: {slope_e:+.5e}/yr (n={len(early)})")
        print(f"  Post-70 slope: {slope_l:+.5e}/yr (n={len(late)})")
        print(f"  Acceleration ratio (|slope_late|/|slope_early|): {accel_ratio:.2f}")
    else:
        slope_e = slope_l = accel_ratio = float('nan')

    # Standardized z at decades
    z_30 = z_80 = float('nan')
    s30 = df[df['age'].between(25,35)]['A_immune']
    s80 = df[df['age'].between(75,85)]['A_immune']
    if len(s30) > 5 and len(s80) > 5:
        all_mean = df['A_immune'].mean()
        all_std = df['A_immune'].std()
        z_30 = (s30.mean() - all_mean) / all_std
        z_80 = (s80.mean() - all_mean) / all_std
        print(f"  z(age~30) = {z_30:+.3f}  z(age~80) = {z_80:+.3f}  |Δz| = {abs(z_30-z_80):.3f}")

    # Pass conditions
    pass1 = (r < -0.15) and (p < 0.001)
    n_neg_cohorts = sum(1 for v in per_cohort.values() if v['pearson_r'] < 0)
    pass2 = n_neg_cohorts >= 3
    pass3 = not np.isnan(accel_ratio) and accel_ratio >= 1.5
    overall = "PASS" if (pass1 and pass2 and pass3) else (
              "DIRECTIONAL" if pass1 else "NULL")

    print(f"\nPass-1 (pooled r<-0.15, p<0.001): {pass1}")
    print(f"Pass-2 (≥3 cohorts same sign): {pass2} ({n_neg_cohorts} negative)")
    print(f"Pass-3 (late accel ratio >= 1.5): {pass3} (ratio={accel_ratio:.2f})")
    print(f"OUTCOME: {overall}")

    # Save
    HERE.mkdir(exist_ok=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].round(5)
    df.to_csv(HERE/"CPG_VAL_017_per_sample.csv", index=False)
    decade_stats.to_csv(HERE/"CPG_VAL_017_decade_medians.csv", index=False)

    (HERE/"results.json").write_text(json.dumps({
        "val_id":"CPG-VAL-017", "title":"Inflammaging in pooled hull HC",
        "card":"Immune universal v1.0", "execution_date":"2026-06-07",
        "outcome_code": overall,
        "n_pooled_samples": int(len(df)),
        "cohorts_used": cohorts_used,
        "pooled_A_immune_vs_age": {
            "pearson_r": round(float(r),5), "pearson_p": float(p),
            "slope_per_year": round(float(slope),8), "slope_p": float(p_slope),
            "slope_se": round(float(se),8),
        },
        "per_cohort": per_cohort,
        "late_life_acceleration": {
            "pre_50_slope": round(float(slope_e),8) if not np.isnan(slope_e) else None,
            "pre_50_n": int(len(early)),
            "post_70_slope": round(float(slope_l),8) if not np.isnan(slope_l) else None,
            "post_70_n": int(len(late)),
            "acceleration_ratio": round(float(accel_ratio),3) if not np.isnan(accel_ratio) else None,
        },
        "standardized_z_at_decade_anchors": {
            "z_age_30": round(float(z_30),3) if not np.isnan(z_30) else None,
            "z_age_80": round(float(z_80),3) if not np.isnan(z_80) else None,
            "abs_dz": round(abs(z_30-z_80),3) if not np.isnan(z_30) and not np.isnan(z_80) else None,
        },
        "pass_conditions": {
            "pass_1_pooled_signal": {"criterion":"r<-0.15, p<0.001", "passed": bool(pass1),
                                     "r": round(float(r),5), "p": float(p)},
            "pass_2_per_cohort_consistency": {"criterion":"≥3 cohorts neg slope",
                                              "passed": bool(pass2), "n_neg_cohorts": int(n_neg_cohorts)},
            "pass_3_late_life_acceleration": {"criterion":"|slope_late|/|slope_early| >= 1.5",
                                              "passed": bool(pass3),
                                              "observed_ratio": round(float(accel_ratio),3) if not np.isnan(accel_ratio) else None},
        },
    }, indent=2, default=str))

    (HERE/"stratified_results.json").write_text(json.dumps({
        "per_cohort": per_cohort,
        "decade_stats": decade_stats.to_dict(orient='records'),
    }, indent=2, default=str))

    # Null: shuffle age within cohort
    print("\n  Null: shuffle ages within cohort (1000 perms)...")
    np.random.seed(42)
    null_rs = []
    for _ in range(1000):
        df_shuf = df.copy()
        df_shuf['age'] = df_shuf.groupby('cohort')['age'].transform(np.random.permutation)
        r_n, _ = stats.pearsonr(df_shuf['age'], df_shuf['A_immune'])
        null_rs.append(r_n)
    null_rs = np.array(null_rs)
    null_p = float(np.mean(np.abs(null_rs) >= abs(r)))
    print(f"  Null mean r: {null_rs.mean():+.4f}  std: {null_rs.std():.4f}  abs>obs: p={null_p:.4f}")

    (HERE/"null_results.json").write_text(json.dumps({
        "null_type":"shuffle_age_within_cohort", "n_perms":1000, "rng_seed":42,
        "observed_pooled_r": round(float(r),5),
        "null_mean": round(float(null_rs.mean()),5),
        "null_std": round(float(null_rs.std()),5),
        "p_two_sided_abs": null_p,
    }, indent=2))

    (HERE/"cohort_manifest.json").write_text(json.dumps({
        "cohorts_with_age_metadata": cohorts_used,
        "pooled_n": int(len(df)),
        "age_range_pooled": [int(df['age'].min()), int(df['age'].max())],
        "input_files_sha256": {
            "Hannum_ascores": sha256_file(REPO / "Biological_Physics/validation_runs/CPG_VAL_020_Immune_Hannum_full_chain/GSE40279_115celltype_ascores.csv"),
            "HanChinese_ascores": sha256_file(REPO / "Biological_Physics/validation_runs/hull_expansion_phase4_asian_GSE141682/GSE141682_HanChinese_HC_115celltype_ascores_PHASE4.csv"),
        }
    }, indent=2))
    
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    return overall

if __name__ == "__main__":
    sys.exit(0 if main() in ("PASS","DIRECTIONAL") else 1)
