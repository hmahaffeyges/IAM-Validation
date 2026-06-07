#!/usr/bin/env python3
"""CPG-VAL-022 — Lifestyle reversibility test via smoking cessation in Tsaprouni cohort."""
import json, hashlib, time, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/home/claude/IAM-Validation")
HERE = REPO / "Biological_Physics/validation_runs/CPG_VAL_022_Smoking_Cessation_Reversibility"
MARKERS = REPO / "Biological_Physics/atlas_vault/walther_clinical_runtime/Celltype_Marker/iamatlas_celltype_markers_v0_2.json"
ASCORES = REPO / "Biological_Physics/validation_runs/hull_expansion_phase2_GSE50660/GSE50660_115celltype_ascores.csv"
META_CSV = Path("/tmp/geo_downloads/GSE50660_sample_meta.csv")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8192), b""): h.update(c)
    return h.hexdigest()

def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return float('nan')
    s = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/s) if s > 0 else float('nan')

def main():
    t0 = time.time()
    print("=" * 72); print("CPG-VAL-022 — Lifestyle reversibility: smoking cessation"); print("=" * 72)

    ct2class = json.loads(MARKERS.read_text())['celltype_to_class']
    classes = sorted(set(ct2class.values()))

    # Load data
    print("\n[1/5] Loading Tsaprouni A-scores + smoking metadata...")
    df = pd.read_csv(ASCORES)
    meta = pd.read_csv(META_CSV)
    smoking_col = [c for c in meta.columns if 'smoking' in c.lower()][0]
    meta = meta.rename(columns={smoking_col: 'smoking'})
    # Compute class means
    for cls in classes:
        cts = [c for c in df.columns if c in ct2class and ct2class[c] == cls]
        df[f'A_{cls}'] = df[cts].mean(axis=1)
    sample_col = 'gsm' if 'gsm' in df.columns else df.columns[0]
    joined = df.merge(meta[['gsm', 'smoking', 'age', 'gender']], left_on=sample_col, right_on='gsm', how='inner')
    joined = joined.dropna(subset=['smoking', 'A_immune'])
    joined['smoking'] = joined['smoking'].astype(int)
    joined['smoking_label'] = joined['smoking'].map({0: 'never', 1: 'former', 2: 'current'})
    print(f"  Joined: n={len(joined)}")
    print(f"  Smoking groups: {joined['smoking_label'].value_counts().to_dict()}")
    print(f"  Mean age by group: {joined.groupby('smoking_label')['age'].mean().round(1).to_dict()}")

    # Group A_immune statistics
    print("\n[2/5] Group A_immune statistics...")
    grp_stats = joined.groupby('smoking_label').agg(
        n=('A_immune', 'count'),
        mean_A_immune=('A_immune', 'mean'),
        median_A_immune=('A_immune', 'median'),
        std_A_immune=('A_immune', 'std'),
        mean_age=('age', 'mean'),
    ).reset_index()
    print(grp_stats.to_string(index=False))

    never = joined[joined['smoking_label'] == 'never']['A_immune'].values
    former = joined[joined['smoking_label'] == 'former']['A_immune'].values
    current = joined[joined['smoking_label'] == 'current']['A_immune'].values

    # Pairwise tests
    print("\n[3/5] Pairwise A_immune comparisons...")
    d_current_vs_never = cohens_d(current, never)
    d_former_vs_never = cohens_d(former, never)
    d_current_vs_former = cohens_d(current, former)
    t_cn, p_cn = stats.ttest_ind(current, never, equal_var=False)
    t_fn, p_fn = stats.ttest_ind(former, never, equal_var=False)
    t_cf, p_cf = stats.ttest_ind(current, former, equal_var=False)
    print(f"  current vs never:  d = {d_current_vs_never:+.4f}  p = {p_cn:.4f}")
    print(f"  former vs never:   d = {d_former_vs_never:+.4f}  p = {p_fn:.4f}")
    print(f"  current vs former: d = {d_current_vs_former:+.4f}  p = {p_cf:.4f}")

    # Reversibility ratio
    mu_never, mu_former, mu_current = never.mean(), former.mean(), current.mean()
    print(f"\n  Means: never={mu_never:.5f}  former={mu_former:.5f}  current={mu_current:.5f}")
    if abs(mu_current - mu_never) > 0:
        reversibility_ratio = abs(mu_former - mu_never) / abs(mu_current - mu_never)
        # Is former INTERMEDIATE?
        intermediate = (min(mu_never, mu_current) <= mu_former <= max(mu_never, mu_current))
    else:
        reversibility_ratio = float('nan'); intermediate = False
    print(f"  Reversibility ratio |former-never|/|current-never|: {reversibility_ratio:.3f}")
    print(f"  Former intermediate (between never and current)?: {intermediate}")
    if reversibility_ratio < 1:
        pct_back = (1 - reversibility_ratio) * 100
        print(f"  Former is {pct_back:.0f}% of the way back from current toward never")

    # Pass conditions
    pass1 = (abs(d_current_vs_never) >= 0.30) and (p_cn < 0.05)
    pass2 = intermediate
    pass3 = (not np.isnan(reversibility_ratio)) and (reversibility_ratio <= 0.7)
    
    # Age-adjusted linear model
    print("\n[4/5] Age-adjusted linear regression...")
    from sklearn.linear_model import LinearRegression
    X = pd.get_dummies(joined['smoking_label'], drop_first=False)[['former','current']].astype(float).values
    X = np.column_stack([X, joined['age'].values, (joined['gender']=='M').astype(float).values])
    y = joined['A_immune'].values
    lm = LinearRegression().fit(X, y)
    # Predict at age=55, female, for each group
    age_ref, gender_ref = 55, 0
    pred_never = lm.intercept_ + age_ref*lm.coef_[2] + gender_ref*lm.coef_[3]
    pred_former = lm.intercept_ + 1*lm.coef_[0] + age_ref*lm.coef_[2] + gender_ref*lm.coef_[3]
    pred_current = lm.intercept_ + 1*lm.coef_[1] + age_ref*lm.coef_[2] + gender_ref*lm.coef_[3]
    print(f"  Age-adjusted predictions (age=55, F):")
    print(f"    never:   {pred_never:.5f}")
    print(f"    former:  {pred_former:.5f}  (Δ from never: {pred_former-pred_never:+.5f})")
    print(f"    current: {pred_current:.5f}  (Δ from never: {pred_current-pred_never:+.5f})")
    if abs(pred_current - pred_never) > 0:
        adj_reversibility = abs(pred_former - pred_never) / abs(pred_current - pred_never)
        adj_intermediate = (min(pred_never, pred_current) <= pred_former <= max(pred_never, pred_current))
        print(f"  Age-adjusted reversibility ratio: {adj_reversibility:.3f}")
        print(f"  Age-adjusted intermediate?: {adj_intermediate}")
    else:
        adj_reversibility = float('nan'); adj_intermediate = False
    pass4 = adj_intermediate and (adj_reversibility <= 0.8 if not np.isnan(adj_reversibility) else False)

    # Per-class specificity
    print("\n[5/5] Per-class specificity (reversibility for each class)...")
    per_class_reversibility = {}
    for cls in classes:
        col = f'A_{cls}'
        if col not in joined.columns: continue
        cls_never = joined[joined['smoking_label']=='never'][col].dropna().values
        cls_former = joined[joined['smoking_label']=='former'][col].dropna().values
        cls_current = joined[joined['smoking_label']=='current'][col].dropna().values
        if len(cls_never)<2 or len(cls_former)<2 or len(cls_current)<2: continue
        mn, mf, mc = cls_never.mean(), cls_former.mean(), cls_current.mean()
        if abs(mc - mn) > 1e-10:
            cls_rev = abs(mf - mn) / abs(mc - mn)
            cls_inter = min(mn, mc) <= mf <= max(mn, mc)
        else:
            cls_rev = float('nan'); cls_inter = False
        cls_d_cn = cohens_d(cls_current, cls_never)
        per_class_reversibility[cls] = {
            'd_current_vs_never': round(cls_d_cn, 4),
            'reversibility_ratio': round(cls_rev, 3) if not np.isnan(cls_rev) else None,
            'intermediate': cls_inter,
            'group_means': {'never': round(float(mn),5), 'former': round(float(mf),5), 'current': round(float(mc),5)},
        }
        flag = "*" if cls == 'immune' else " "
        rev_str = f"{cls_rev:.3f}" if not np.isnan(cls_rev) else "n/a"
        print(f"  {flag} A_{cls:12s}  d(current-never)={cls_d_cn:+.4f}  reversibility={rev_str}  intermediate={cls_inter}")

    pass5 = sum(1 for k,v in per_class_reversibility.items() 
                if k != 'immune' and v['reversibility_ratio'] is not None 
                and v['reversibility_ratio'] > (reversibility_ratio if not np.isnan(reversibility_ratio) else 0)) <= 3

    if pass1 and pass2 and pass3:
        overall = "PASS"
    elif pass1 and pass2:
        overall = "DIRECTIONAL"
    else:
        overall = "NULL"
    print(f"\nPass-1 (smoking effect detectable): {pass1}")
    print(f"Pass-2 (former intermediate): {pass2}")
    print(f"Pass-3 (reversibility ratio <= 0.7): {pass3}")
    print(f"Pass-4 (age-adjusted intermediate): {pass4}")
    print(f"Pass-5 (immune specificity): {pass5}")
    print(f"OUTCOME: {overall}")

    # Save outputs
    HERE.mkdir(exist_ok=True)
    out_cols = [sample_col, 'smoking', 'smoking_label', 'age', 'gender'] + [f'A_{c}' for c in classes]
    per_sample = joined[[c for c in out_cols if c in joined.columns]].copy()
    for c in per_sample.select_dtypes(include=[np.number]).columns:
        per_sample[c] = per_sample[c].round(5)
    per_sample.to_csv(HERE/'CPG_VAL_022_per_sample.csv', index=False)

    (HERE/'results.json').write_text(json.dumps({
        'val_id':'CPG-VAL-022', 'title':'Lifestyle reversibility via smoking cessation',
        'card':'Immune universal v1.0', 'execution_date':'2026-06-07',
        'outcome_code': overall,
        'cohort':'GSE50660 Tsaprouni 2014 (UK, HM450, whole blood)',
        'n_samples': int(len(joined)),
        'group_n': {'never': int(len(never)), 'former': int(len(former)), 'current': int(len(current))},
        'A_immune_means': {'never': round(float(mu_never),5), 'former': round(float(mu_former),5), 'current': round(float(mu_current),5)},
        'pairwise_cohens_d': {
            'current_vs_never': round(float(d_current_vs_never),4),
            'former_vs_never':  round(float(d_former_vs_never),4),
            'current_vs_former':round(float(d_current_vs_former),4),
        },
        'pairwise_p_values': {
            'current_vs_never': float(p_cn),
            'former_vs_never':  float(p_fn),
            'current_vs_former':float(p_cf),
        },
        'reversibility': {
            'raw_ratio': round(float(reversibility_ratio),4) if not np.isnan(reversibility_ratio) else None,
            'former_intermediate': bool(intermediate),
            'percent_back_to_baseline': round(float((1-reversibility_ratio)*100),1) if not np.isnan(reversibility_ratio) else None,
        },
        'age_adjusted': {
            'predicted_never': round(float(pred_never),5),
            'predicted_former': round(float(pred_former),5),
            'predicted_current': round(float(pred_current),5),
            'adjusted_reversibility_ratio': round(float(adj_reversibility),4) if not np.isnan(adj_reversibility) else None,
            'adjusted_intermediate': bool(adj_intermediate),
        },
        'pass_conditions': {
            'p1_smoking_effect_detectable': {'criterion':'|d(current-never)|>=0.30 AND p<0.05', 'passed':bool(pass1),
                                              'observed_d': round(float(d_current_vs_never),4), 'observed_p': float(p_cn)},
            'p2_former_intermediate': {'criterion':'former mean between never and current', 'passed':bool(pass2)},
            'p3_reversibility_ratio': {'criterion':'reversibility ratio <= 0.7', 'passed':bool(pass3),
                                       'observed_ratio': round(float(reversibility_ratio),4) if not np.isnan(reversibility_ratio) else None},
            'p4_age_adjusted_intermediate': {'criterion':'age-adjusted former intermediate AND ratio <=0.8', 'passed':bool(pass4)},
            'p5_immune_specificity': {'criterion':'immune class shows reversibility stronger than majority of other classes', 'passed':bool(pass5)},
        },
        'per_class_reversibility': per_class_reversibility,
    }, indent=2, default=str))

    (HERE/'stratified_results.json').write_text(json.dumps({
        'group_summary': grp_stats.to_dict(orient='records'),
        'per_class_reversibility': per_class_reversibility,
    }, indent=2, default=str))

    # Null: shuffle smoking labels
    print("\n  Null: shuffle smoking labels (1000 perms)...")
    np.random.seed(42); null_ratios = []
    for _ in range(1000):
        s_shuf = np.random.permutation(joined['smoking_label'].values)
        n_ = joined['A_immune'].values[s_shuf=='never']
        f_ = joined['A_immune'].values[s_shuf=='former']
        c_ = joined['A_immune'].values[s_shuf=='current']
        if len(n_)<2 or len(f_)<2 or len(c_)<2: continue
        if abs(c_.mean()-n_.mean())>1e-10:
            r = abs(f_.mean()-n_.mean()) / abs(c_.mean()-n_.mean())
            null_ratios.append(r)
    null_ratios = np.array(null_ratios)
    # Under null with current's tiny n=22, the ratio is unstable. Check the d(current-never) instead.
    null_ds = []
    for _ in range(1000):
        s_shuf = np.random.permutation(joined['smoking_label'].values)
        n_ = joined['A_immune'].values[s_shuf=='never']
        c_ = joined['A_immune'].values[s_shuf=='current']
        if len(n_)>=2 and len(c_)>=2:
            null_ds.append(cohens_d(c_, n_))
    null_ds = np.array(null_ds)
    p_null_d = float(np.mean(np.abs(null_ds) >= abs(d_current_vs_never)))
    print(f"  Null distribution of d(current-never): mean={null_ds.mean():+.4f}, std={null_ds.std():.4f}")
    print(f"  Observed d={d_current_vs_never:+.4f}; p_vs_null={p_null_d:.4f}")

    (HERE/'null_results.json').write_text(json.dumps({
        'null_type':'shuffle_smoking_labels','n_perms':1000,'rng_seed':42,
        'observed_d_current_vs_never': round(float(d_current_vs_never),4),
        'null_d_mean': round(float(null_ds.mean()),5), 'null_d_std': round(float(null_ds.std()),5),
        'p_two_sided_abs': p_null_d,
    }, indent=2))

    (HERE/'cohort_manifest.json').write_text(json.dumps({
        'cohort':'GSE50660 Tsaprouni 2014','platform':'HM450','tissue':'whole blood','population':'UK',
        'n_total': int(len(joined)),
        'group_counts': {k: int(v) for k,v in joined['smoking_label'].value_counts().items()},
        'ascores_sha256': sha256_file(ASCORES),
        'metadata_sha256': sha256_file(META_CSV),
    }, indent=2))

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    return overall

if __name__ == '__main__':
    sys.exit(0 if main() in ('PASS','DIRECTIONAL') else 1)
