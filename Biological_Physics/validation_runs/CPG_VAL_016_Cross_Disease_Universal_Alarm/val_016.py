#!/usr/bin/env python3
"""CPG-VAL-016 sealed runner — Cross-disease universal alarm via A_immune class score."""
import json, hashlib, time, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/home/claude/IAM-Validation")
HERE = REPO / "Biological_Physics/validation_runs/CPG_VAL_016_Cross_Disease_Universal_Alarm"

ASCORES_AIBL = REPO / "Biological_Physics/validation_runs/ad_immune_cohorts/GSE153712_AIBL/GSE153712_115celltype_ascores.csv"
ASCORES_GSE51057 = REPO / "Biological_Physics/validation_runs/foundation_cohort/GSE51057_115celltype_ascores.csv"
ASCORES_GSE51032 = REPO / "Biological_Physics/validation_runs/foundation_cohort/GSE51032_115celltype_ascores.csv"
MARKERS = REPO / "Biological_Physics/atlas_vault/walther_clinical_runtime/Celltype_Marker/iamatlas_celltype_markers_v0_2.json"

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
    print("=" * 72); print("CPG-VAL-016 — Cross-disease universal alarm"); print("=" * 72)

    # Load class-to-celltype map from markers
    markers = json.loads(MARKERS.read_text())
    if 'celltype_to_class' in markers:
        ct2class = markers['celltype_to_class']
    else:
        # Inferred from typical structure
        ct2class = {}
        for ct, info in markers.get('celltypes', {}).items():
            ct2class[ct] = info.get('class', 'unknown')
    if not ct2class:
        print("FAIL: no celltype-class map"); sys.exit(1)
    classes = sorted(set(ct2class.values()))
    print(f"  Classes: {classes}")
    immune_celltypes = [ct for ct, c in ct2class.items() if c == 'immune']
    print(f"  N immune celltypes: {len(immune_celltypes)}")

    def cohort_class_scores(path, sample_col, case_label, hc_label='hc', cohort_label='', platform=''):
        df = pd.read_csv(path)
        if 'arm' not in df.columns:
            print(f"  {path.name}: no arm column"); return None
        # Compute per-sample per-class mean A
        out = {}
        out['cohort'] = cohort_label
        out['platform'] = platform
        # Identify columns that are celltypes
        celltypes_present = [c for c in df.columns if c in ct2class]
        results_per_class = {}
        for cls in classes:
            cls_cts = [c for c in celltypes_present if ct2class[c] == cls]
            if not cls_cts: continue
            df[f'A_{cls}'] = df[cls_cts].mean(axis=1)
        # Per-sample row with mean per class
        per_sample_rows = []
        for _, row in df.iterrows():
            r = {'sample_id': row[sample_col], 'arm': row['arm']}
            for cls in classes:
                col = f'A_{cls}'
                if col in df.columns: r[cls] = row[col]
            per_sample_rows.append(r)
        case_df = df[df['arm'] == case_label]
        hc_df = df[df['arm'] == hc_label]
        # Per-class Cohen's d
        per_class_d = {}
        for cls in classes:
            col = f'A_{cls}'
            if col not in df.columns: continue
            d = cohens_d(case_df[col], hc_df[col])
            t_stat, p_val = stats.ttest_ind(case_df[col].dropna(), hc_df[col].dropna(), equal_var=False)
            per_class_d[cls] = {
                "cohens_d": round(d, 4),
                "p_welch": float(p_val),
                "case_mean": round(float(case_df[col].mean()), 5),
                "hc_mean": round(float(hc_df[col].mean()), 5),
            }
        out['n_case'] = int(len(case_df)); out['n_hc'] = int(len(hc_df))
        out['case_arm'] = case_label
        out['per_class_d'] = per_class_d
        out['per_sample_rows'] = per_sample_rows
        return out

    print("\n[1/4] Computing per-class A-scores for AIBL (AD)...")
    aibl = cohort_class_scores(ASCORES_AIBL, 'sentrix', 'ad', cohort_label='AIBL_AD', platform='EPIC 850K')
    print(f"  AIBL: n_AD={aibl['n_case']}, n_HC={aibl['n_hc']}")
    print(f"    A_immune: d={aibl['per_class_d']['immune']['cohens_d']:+.4f}, p={aibl['per_class_d']['immune']['p_welch']:.2e}")
    for c in classes:
        if c in aibl['per_class_d']:
            print(f"    [class] A_{c:12s} d={aibl['per_class_d'][c]['cohens_d']:+.4f}")

    print("\n[2/4] Computing per-class A-scores for GSE51057 (breast)...")
    breast57 = cohort_class_scores(ASCORES_GSE51057, 'gsm', 'case', cohort_label='GSE51057_breast', platform='HM450')
    print(f"  GSE51057: n_case={breast57['n_case']}, n_HC={breast57['n_hc']}")
    print(f"    A_immune: d={breast57['per_class_d']['immune']['cohens_d']:+.4f}, p={breast57['per_class_d']['immune']['p_welch']:.2e}")

    print("\n[3/4] Computing per-class A-scores for GSE51032 (breast)...")
    breast32 = cohort_class_scores(ASCORES_GSE51032, 'gsm', 'case', cohort_label='GSE51032_breast', platform='HM450')
    print(f"  GSE51032: n_case={breast32['n_case']}, n_HC={breast32['n_hc']}")
    print(f"    A_immune: d={breast32['per_class_d']['immune']['cohens_d']:+.4f}, p={breast32['per_class_d']['immune']['p_welch']:.2e}")

    # Meta-analysis: inverse variance weighted Cohen's d
    print("\n[4/4] Cross-cohort meta-analysis...")
    cohorts = [aibl, breast57, breast32]
    immune_ds = [c['per_class_d']['immune']['cohens_d'] for c in cohorts]
    immune_ns_case = [c['n_case'] for c in cohorts]
    immune_ns_hc = [c['n_hc'] for c in cohorts]
    # SE per cohort: sqrt(1/n1 + 1/n2 + d²/(2*(n1+n2)))
    immune_ses = [np.sqrt(1/n1 + 1/n2 + d**2/(2*(n1+n2))) for d, n1, n2 in zip(immune_ds, immune_ns_case, immune_ns_hc)]
    weights = [1/se**2 for se in immune_ses]
    meta_d = sum(d * w for d, w in zip(immune_ds, weights)) / sum(weights)
    meta_se = np.sqrt(1 / sum(weights))
    meta_z = meta_d / meta_se
    meta_p = 2 * (1 - stats.norm.cdf(abs(meta_z)))
    print(f"  Meta-d: {meta_d:+.4f}  SE: {meta_se:.4f}  z={meta_z:.2f}  p={meta_p:.4e}")

    # Specificity: in each cohort, is immune effect stronger than at least 4/7 other classes?
    specificity_check = {}
    for cname, ck in [('AIBL', aibl), ('GSE51057', breast57), ('GSE51032', breast32)]:
        immune_abs_d = abs(ck['per_class_d']['immune']['cohens_d'])
        other_ds = {c: abs(v['cohens_d']) for c, v in ck['per_class_d'].items() if c != 'immune'}
        n_weaker = sum(1 for v in other_ds.values() if v < immune_abs_d)
        specificity_check[cname] = {
            "immune_abs_d": round(immune_abs_d, 4),
            "n_other_classes_weaker": n_weaker,
            "n_other_classes_total": len(other_ds),
            "other_class_abs_ds": {k: round(v, 4) for k, v in other_ds.items()},
            "immune_stronger_than_majority": n_weaker >= 4
        }
    print(f"\n  Specificity:")
    for cn, sc in specificity_check.items():
        print(f"    {cn}: immune |d|={sc['immune_abs_d']:.3f}, weaker than {sc['n_other_classes_weaker']}/{sc['n_other_classes_total']} other classes — stronger than majority?: {sc['immune_stronger_than_majority']}")

    # Pass conditions
    pass1 = (abs(aibl['per_class_d']['immune']['cohens_d']) >= 0.20 and
             (abs(breast57['per_class_d']['immune']['cohens_d']) >= 0.20 or
              abs(breast32['per_class_d']['immune']['cohens_d']) >= 0.20))
    pass2 = True  # condition allows different signs
    pass3 = abs(meta_d) >= 0.20 and meta_p < 0.01
    overall = "PASS" if (pass1 and pass3) else "DIRECTIONAL" if (pass1 or pass3) else "NULL"

    # Per-sample CSV
    HERE.mkdir(exist_ok=True)
    per_sample_all = []
    for ck in cohorts:
        for r in ck['per_sample_rows']:
            r['cohort'] = ck['cohort']
            r['platform'] = ck['platform']
            per_sample_all.append(r)
    per_sample_df = pd.DataFrame(per_sample_all)
    for col in per_sample_df.select_dtypes(include=[np.number]).columns:
        per_sample_df[col] = per_sample_df[col].round(5)
    per_sample_df.to_csv(HERE / "CPG_VAL_016_per_sample.csv", index=False)
    print(f"\n  Per-sample CSV: {per_sample_df.shape}")

    results = {
        "val_id": "CPG-VAL-016",
        "title": "Cross-disease universal alarm via A_immune",
        "card": "Immune universal v1.0",
        "execution_date": "2026-06-07",
        "outcome_code": overall,
        "n_immune_celltypes_averaged": len(immune_celltypes),
        "per_cohort_results": {
            "AIBL_AD_anchored": {
                "n_case": aibl['n_case'], "n_hc": aibl['n_hc'],
                "platform": aibl['platform'],
                "per_class_d": aibl['per_class_d'],
            },
            "GSE51057_breast": {
                "n_case": breast57['n_case'], "n_hc": breast57['n_hc'],
                "platform": breast57['platform'],
                "per_class_d": breast57['per_class_d'],
            },
            "GSE51032_breast": {
                "n_case": breast32['n_case'], "n_hc": breast32['n_hc'],
                "platform": breast32['platform'],
                "per_class_d": breast32['per_class_d'],
            },
        },
        "meta_analysis_immune_class": {
            "method": "inverse_variance_weighted_cohens_d",
            "cohort_ds": [round(d, 4) for d in immune_ds],
            "cohort_ses": [round(se, 4) for se in immune_ses],
            "meta_d": round(meta_d, 4),
            "meta_se": round(meta_se, 4),
            "meta_z": round(meta_z, 3),
            "meta_p_two_sided": float(meta_p),
        },
        "specificity_check": specificity_check,
        "pass_conditions": {
            "condition_1_immune_fires_in_both_diseases": {
                "criterion": "|d_immune| >= 0.20 in AIBL AND (in at least one of GSE51057/GSE51032)",
                "passed": bool(pass1),
                "AIBL_d": aibl['per_class_d']['immune']['cohens_d'],
                "GSE51057_d": breast57['per_class_d']['immune']['cohens_d'],
                "GSE51032_d": breast32['per_class_d']['immune']['cohens_d'],
            },
            "condition_2_direction_disease_specific_allowed": {
                "criterion": "Signs may differ across diseases (universality = shifts, not same-direction)",
                "passed": True,
            },
            "condition_3_meta_d_significant": {
                "criterion": "|d_meta| >= 0.20 AND p < 0.01",
                "passed": bool(pass3),
                "observed_d_meta": round(meta_d, 4),
                "observed_p": float(meta_p),
            },
        },
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # stratified_results.json — per cohort + per class
    (HERE / "stratified_results.json").write_text(json.dumps({
        "stratification": "by cohort and class",
        "per_cohort_per_class_d": {
            "AIBL_AD": aibl['per_class_d'],
            "GSE51057_breast": breast57['per_class_d'],
            "GSE51032_breast": breast32['per_class_d'],
        },
        "specificity_check": specificity_check,
    }, indent=2))

    # Null: random arm shuffle per cohort, recompute meta-d
    print("\n  Null test: random arm shuffle per cohort (1000 permutations)...")
    np.random.seed(42)
    null_ds = []
    for _ in range(1000):
        null_cohort_ds = []
        null_ses = []
        for ck, path, sample_col, case_label in [
            (aibl, ASCORES_AIBL, 'sentrix', 'ad'),
            (breast57, ASCORES_GSE51057, 'gsm', 'case'),
            (breast32, ASCORES_GSE51032, 'gsm', 'case'),
        ]:
            df = pd.read_csv(path)
            # Subset to case + hc
            df = df[df['arm'].isin([case_label, 'hc'])].copy()
            # Compute immune mean per sample
            imm_cts = [c for c in df.columns if c in ct2class and ct2class[c] == 'immune']
            df['_imm'] = df[imm_cts].mean(axis=1)
            # Shuffle arm labels
            df['_shuffled_arm'] = np.random.permutation(df['arm'].values)
            case = df[df['_shuffled_arm'] == case_label]['_imm']
            hc = df[df['_shuffled_arm'] == 'hc']['_imm']
            if len(case) < 2 or len(hc) < 2: continue
            d = cohens_d(case, hc)
            n1, n2 = len(case), len(hc)
            se = np.sqrt(1/n1 + 1/n2 + d**2/(2*(n1+n2)))
            null_cohort_ds.append(d); null_ses.append(se)
        if len(null_cohort_ds) == 3:
            w = [1/s**2 for s in null_ses]
            nd = sum(d*ww for d, ww in zip(null_cohort_ds, w)) / sum(w)
            null_ds.append(nd)
    null_ds = np.array(null_ds)
    p_null = float(np.mean(np.abs(null_ds) >= abs(meta_d)))
    print(f"  Null mean d_meta: {null_ds.mean():+.4f}  std: {null_ds.std():.4f}  abs>observed: p={p_null:.4f}")

    (HERE / "null_results.json").write_text(json.dumps({
        "null_type": "arm_label_shuffle_per_cohort",
        "n_permutations": 1000, "rng_seed": 42,
        "observed_meta_d": round(meta_d, 4),
        "null_mean": round(float(null_ds.mean()), 5),
        "null_std": round(float(null_ds.std()), 5),
        "p_two_sided_abs": p_null,
    }, indent=2))

    (HERE / "cohort_manifest.json").write_text(json.dumps({
        "AIBL_AD_EPIC": {"sha256": sha256_file(ASCORES_AIBL)},
        "GSE51057_breast_HM450": {"sha256": sha256_file(ASCORES_GSE51057)},
        "GSE51032_breast_HM450": {"sha256": sha256_file(ASCORES_GSE51032)},
        "canonical_markers_sha256": sha256_file(MARKERS),
        "note_on_crohns": "VAL-128 Crohn's cohort uses pre-canonical Xu-538 + Loyfer pipeline, not canonical 115-cell. Excluded from this VAL to preserve apples-to-apples comparison. Re-scoring Crohn's with canonical markers queued as separate work."
    }, indent=2))

    print(f"\nPass-1 (immune fires in both diseases): {pass1}")
    print(f"Pass-3 (meta-d significant): {pass3}")
    print(f"OUTCOME: {overall}")
    print(f"Elapsed: {time.time()-t0:.1f}s")
    return overall

if __name__ == "__main__":
    sys.exit(0 if main() in ("PASS", "DIRECTIONAL") else 1)
